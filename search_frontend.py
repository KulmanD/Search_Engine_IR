from flask import Flask, request, jsonify

import math
import pickle
import re
from collections import Counter, defaultdict
from functools import lru_cache
import os
import nltk
from nltk.corpus import stopwords
from google.cloud import storage

from inverted_index_gcp import InvertedIndex
from concurrent.futures import ThreadPoolExecutor, as_completed
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False



# gcs configuration
BUCKET_NAME = "information-retrival-ex3"

# ----------------------------
# runtime feature flags
# set env ENABLE_ANCHOR=1 to enable anchor endpoints/scoring
# default is OFF so we don't load anchor routing tables/shards unless needed.
ENABLE_ANCHOR = os.getenv("ENABLE_ANCHOR", "0").strip().lower() in {"1", "true", "yes", "on"}

# set env USE_COMBINED_SEARCH=1 to enable the experimental combined body+title+pr search.
# default is OFF because on the provided train set it hurts MAP@10 vs title-only baseline.
USE_COMBINED_SEARCH = os.getenv("USE_COMBINED_SEARCH", "0").strip().lower() in {"1", "true", "yes", "on"}

# body
BODY_BASE_DIR = "postings_gcp"  # gs://BUCKET/postings_gcp/...
BODY_INDEX = None  # InvertedIndex

# title
TITLE_BASE_DIR = "title_postings_gcp/title_postings_gcp"
TITLE_INDEX_NAME = "title_index"
TITLE_INDEX = None  # InvertedIndex

# anchor
ANCHOR_PARTS_DIR = "anchor_parts"  # gs://BUCKET/anchor_parts/part_000/
ANCHOR_PARTS_N = 60
TERM2PARTS_BLOB = "anchor_parts/term2parts_u64.pkl"
TERM2BITS = None  # dict[str,int] term -> uint64 bitset

# global artifacts
ID2TITLE_BLOB = "artifacts/id2title.pickle"
PAGERANK_BLOB = "artifacts/pagerank.pickle"
PAGEVIEW_BLOB = "artifacts/pageview_aug2021.pickle" #wasnt very usefully to us after all

ID2TITLE = None  # dict[int,str]
PAGERANK = None  # dict or list
PAGEVIEW = None  # dict or list (optional)


N_DOCS_FALLBACK = 6348910


# small helpers for runtime tuning via environment variables
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return float(default)


# tokenizer
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

try:
    english_stopwords = frozenset(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    english_stopwords = frozenset(stopwords.words("english"))

corpus_stopwords = {
    "category", "references", "also", "external", "links",
    "may", "first", "see", "history", "people", "one", "two",
    "part", "thumb", "including", "second", "following",
    "many", "however", "would", "became",
}

ALL_STOPWORDS = english_stopwords.union(corpus_stopwords)


def tokenize(text: str):
    if not text:
        return []
    tokens = [m.group() for m in RE_WORD.finditer(text.lower())]
    return [t for t in tokens if t not in ALL_STOPWORDS]


#####################################
# GCS + load once artifacts, runs once per process
def _gcs_client():
    return storage.Client()


def _gcs_download_pickle(blob_path: str):
    c = _gcs_client()
    b = c.bucket(BUCKET_NAME)
    blob = b.blob(blob_path)
    if not blob.exists():
        return None
    return pickle.loads(blob.download_as_bytes())


def _gcs_list_pickles(prefix: str):
    c = _gcs_client()
    b = c.bucket(BUCKET_NAME)
    blobs = b.list_blobs(prefix=prefix.rstrip("/") + "/")
    out = []
    for bl in blobs:
        name = bl.name
        if name.endswith(".pickle") or name.endswith(".pkl"):
            out.append(name)
    return sorted(out)

def _ensure_pagerank_loaded():
    global PAGERANK
    if PAGERANK is not None:
        return
    obj = _gcs_download_pickle(PAGERANK_BLOB)
    if obj is None:
        print(f"[startup] pagerank not found: gs://{BUCKET_NAME}/{PAGERANK_BLOB} (returning 0.0)")
        PAGERANK = {}
        return
    PAGERANK = obj
    try:
        ln = len(PAGERANK)
    except Exception:
        ln = -1
    print(f"[startup] loaded pagerank ({type(PAGERANK).__name__}), len={ln}")

def _ensure_pageview_loaded():
    """
    PageView is optional for minimum. If missing, we just return 0.
    """
    global PAGEVIEW
    if PAGEVIEW is not None:
        return
    obj = _gcs_download_pickle(PAGEVIEW_BLOB)
    if obj is None:
        print(f"[startup] pageview not found: gs://{BUCKET_NAME}/{PAGEVIEW_BLOB} (returning 0)")
        PAGEVIEW = {}
        return
    PAGEVIEW = obj
    try:
        ln = len(PAGEVIEW)
    except Exception:
        ln = -1
    # print a tiny sample of keys to detect keying scheme (int ids vs string ids vs titles)
    sample = []
    try:
        if isinstance(PAGEVIEW, dict):
            for i, k in enumerate(PAGEVIEW.keys()):
                sample.append(str(k)[:80])
                if i >= 4:
                    break
    except Exception:
        sample = []

    print(f"[startup] loaded pageview ({type(PAGEVIEW).__name__}), len={ln}, sample_keys={sample}")

def _ensure_id2title_loaded():
    global ID2TITLE
    if ID2TITLE is not None:
        return
    obj = _gcs_download_pickle(ID2TITLE_BLOB)
    if obj is None:
        print(f"[startup] id2title not found: gs://{BUCKET_NAME}/{ID2TITLE_BLOB} (using doc_id as title)")
        ID2TITLE = {}
        return
    ID2TITLE = obj
    print(f"[startup] loaded id2title entries: {len(ID2TITLE):,}")

def _ensure_term2bits_loaded():
    global TERM2BITS
    # anchor feature disabled -> never load term2parts [had high hopes for this :-( ]
    if not ENABLE_ANCHOR:
        if TERM2BITS is None:
            TERM2BITS = {}
        return
    if TERM2BITS is not None:
        return
    print(f"[startup] loading term2parts from gs://{BUCKET_NAME}/{TERM2PARTS_BLOB} ...")
    obj = _gcs_download_pickle(TERM2PARTS_BLOB)
    if obj is None:
        print(f"[startup] term2parts not found: gs://{BUCKET_NAME}/{TERM2PARTS_BLOB}")
        TERM2BITS = {}
        return
    TERM2BITS = obj
    print(f"[startup] loaded term2parts terms: {len(TERM2BITS):,}")
######################################

# loading  body/title index object from GCS to RAM
def _load_body_index():
    global BODY_INDEX
    if BODY_INDEX is not None:
        return
    # force a known body index name
    name = "index"  # gs://BUCKET/postings_gcp/index.pkl
    print(f"[startup] loading body index: gs://{BUCKET_NAME}/{BODY_BASE_DIR}/{name}.pkl")
    BODY_INDEX = InvertedIndex.read_index(BODY_BASE_DIR, name, bucket_name=BUCKET_NAME)

def _load_title_index():
    global TITLE_INDEX
    if TITLE_INDEX is not None:
        return
    print(f"[startup] loading title index: gs://{BUCKET_NAME}/{TITLE_BASE_DIR}/{TITLE_INDEX_NAME}.pkl")
    TITLE_INDEX = InvertedIndex.read_index(TITLE_BASE_DIR, TITLE_INDEX_NAME, bucket_name=BUCKET_NAME)

def _auto_detect_index_name(base_dir: str):
    """
    Detect main index pickle name (without extension) under base_dir.
    """
    pickles = _gcs_list_pickles(base_dir)
    candidates = []
    for p in pickles:
        base = p.split("/")[-1]
        if "posting_locs" in base:
            continue
        if base.endswith(".pickle"):
            candidates.append(base[:-7])
        elif base.endswith(".pkl"):
            candidates.append(base[:-4])
    return candidates[0] if candidates else None

# doc2part should have been useful assign docs to parts, or sometimes for optimizations.
# In our current frontend code, it’s not used. our routing is term -> parts, not doc -> art.
# shit idea, heavy on ram in run time
def _doc_to_title(doc_id: int) -> str:
    if not ID2TITLE:
        return str(doc_id)
    return ID2TITLE.get(doc_id, str(doc_id))


def _pr_of(doc_id: int) -> float:
    if not PAGERANK:
        return 0.0
    if isinstance(PAGERANK, dict):
        if doc_id in PAGERANK:
            return float(PAGERANK.get(doc_id, 0.0))
        # sometimes keys are strings
        return float(PAGERANK.get(str(doc_id), 0.0))
    if 0 <= doc_id < len(PAGERANK):
        return float(PAGERANK[doc_id])
    return 0.0


def _pv_of(doc_id: int) -> int:
    if not PAGEVIEW:
        return 0

    # dict based pageview
    if isinstance(PAGEVIEW, dict):
        # common case: keyed by int wiki_id
        if doc_id in PAGEVIEW:
            return int(PAGEVIEW.get(doc_id, 0))

        # sometimes keys are digit strings
        s_id = str(doc_id)
        if s_id in PAGEVIEW:
            return int(PAGEVIEW.get(s_id, 0))

        # sometimes the dict was created keyed by titles
        # fall back to title lookup (safe but slower, only used if ids not found)
        #we simple but fast boi
        title = _doc_to_title(doc_id)
        if title in PAGEVIEW:
            try:
                return int(PAGEVIEW.get(title, 0))
            except Exception:
                return 0

        return 0

    # list-based pageview (indexed by doc_id)
    if 0 <= doc_id < len(PAGEVIEW):
        return int(PAGEVIEW[doc_id])

    return 0



# debug endpoints (safe to keep; not used by graders)
@app.route("/debug_pagerank")
def debug_pagerank():
    """debug: verify pagerank loaded + key type + sample values"""
    _ensure_pagerank_loaded()

    if not PAGERANK:
        return jsonify({"loaded": False, "type": None, "len": 0})

    # dict based pagerank
    if isinstance(PAGERANK, dict):
        keys = list(PAGERANK.keys())
        sample_keys = keys[:5]
        sample = []
        for k in sample_keys:
            try:
                sample.append([str(k), float(PAGERANK.get(k, 0.0))])
            except Exception:
                sample.append([str(k), 0.0])
        return jsonify({
            "loaded": True,
            "type": "dict",
            "len": len(PAGERANK),
            "sample": sample,
            "has_int_1": (1 in PAGERANK),
            "has_str_1": ("1" in PAGERANK),
            "val_1": _pr_of(1),
            "val_5": _pr_of(5),
            "val_8": _pr_of(8),
        })
    # list-based pagerank
    try:
        ln = len(PAGERANK)
    except Exception:
        ln = -1
    v1 = float(PAGERANK[1]) if ln > 1 else None
    v5 = float(PAGERANK[5]) if ln > 5 else None
    v8 = float(PAGERANK[8]) if ln > 8 else None
    return jsonify({
        "loaded": True,
        "type": "list",
        "len": ln,
        "val_1": v1,
        "val_5": v5,
        "val_8": v8,
    })


def _iter_set_bits_u64(bits: int):
    b = bits & ((1 << 64) - 1)
    while b:
        lsb = b & -b
        i = (lsb.bit_length() - 1)
        yield i
        b ^= lsb

def _anchor_fetch_posting(part_id: int, term: str):
    """fetch posting list for (term) from one anchor shard; returns list[(doc_id, tf)] or []"""
    try:
        idx = _get_anchor_part_index(part_id)
        return idx.read_a_posting_list(f"{ANCHOR_PARTS_DIR}/part_{part_id:03d}", term, bucket_name=BUCKET_NAME)
    except Exception:
        return []

@lru_cache(maxsize=64)
def _get_anchor_part_index(part_id: int) -> InvertedIndex:
    base_dir = f"{ANCHOR_PARTS_DIR}/part_{part_id:03d}"
    return InvertedIndex.read_index(base_dir, "anchor_index", bucket_name=BUCKET_NAME)


# scoring helpers
def _get_N_docs():
    return N_DOCS_FALLBACK


def _topk(score_dict, k: int):
    if not score_dict:
        return []
    # stable: score desc, doc_id asc
    return sorted(score_dict.items(), key=lambda x: (-x[1], x[0]))[:k]

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get("query", "") # get the query from the url
    if len(query) == 0: # if the query is empty, just return an empty list immediately
        return jsonify(res)

    # BEGIN SOLUTION
    # load indexes (needed for both the baseline and the experimental combined mode)
    if TITLE_INDEX is None:
        _load_title_index()
    _ensure_id2title_loaded()

    # tokenize query
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    # default: title-only baseline
    # ------------------------------
    # we keep the combined logic below for the report/experimentation, but the title-only
    # baseline is the best-performing method on the provided train set (MAP@10 ~ 0.2167).
    if not USE_COMBINED_SEARCH:
        q_terms = set(tokens)
        match_counts = defaultdict(int)

        for t in q_terms:
            if TITLE_INDEX.df.get(t, 0) <= 0:
                continue
            try:
                pl = TITLE_INDEX.read_a_posting_list(TITLE_BASE_DIR, t, bucket_name=BUCKET_NAME)
            except Exception:
                continue
            for doc_id, _tf in pl:
                match_counts[doc_id] += 1

        if not match_counts:
            return jsonify(res)
        _ensure_pagerank_loaded()
        _ensure_pageview_loaded()

        def _key(item):
            doc_id, cnt = item
            d = int(doc_id)
            pr = _pr_of(d)
            pv = _pv_of(d)
            # primary: title match count
            # secondary: pagerank (soft log)
            # third: pageview (soft log)
            # last but not least: doc_id (stable)
            return (-cnt, -math.log1p(pr), -math.log1p(pv), d)

        ranked = sorted(match_counts.items(), key=_key)[:100]
        return jsonify([(int(doc_id), _doc_to_title(int(doc_id))) for doc_id, _c in ranked])

    # #@#@ experimental #@#@   --- > combined search
    # this logic is preserved for the report, but must be explicitly enabled with
    # USE_COMBINED_SEARCH=1.
    # make sure body index is loaded (only needed in combined mode)
    if BODY_INDEX is None:
        _load_body_index()

    # ensure helper data structures are ready
    if ENABLE_ANCHOR:
        _ensure_term2bits_loaded()
    _ensure_pagerank_loaded()

    # body tfidf cosine
    q_tf = Counter(tokens)
    N = _get_N_docs()
    q_weights = {}
    for t, tf in q_tf.items():
        df = BODY_INDEX.df.get(t, 0)
        if df <= 0:
            continue
        idf = math.log10(N / df)
        q_weights[t] = (tf / len(tokens)) * idf
    body_scores = defaultdict(float)
    body_doc_sq = defaultdict(float)
    for t, wq in q_weights.items():
        try:
            pl = BODY_INDEX.read_a_posting_list(BODY_BASE_DIR, t, bucket_name=BUCKET_NAME)
        except Exception:
            continue
        df = BODY_INDEX.df.get(t, 0)
        if df <= 0:
            continue
        idf = math.log10(N / df)
        for doc_id, tf in pl:
            wd = (1.0 + math.log10(tf)) * idf
            body_scores[doc_id] += wq * wd
            body_doc_sq[doc_id] += wd * wd
    q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
    if q_norm > 0.0:
        for doc_id in list(body_scores.keys()):
            d_norm = math.sqrt(body_doc_sq.get(doc_id, 0.0))
            if d_norm > 0.0:
                body_scores[doc_id] /= (q_norm * d_norm)
            else:
                body_scores[doc_id] = 0.0

    body_top = _topk(body_scores, 200)
    cand = set(doc_id for doc_id, _ in body_top)
    title_scores = defaultdict(float)
    q_terms = set(tokens)
    for t in q_terms:
        if TITLE_INDEX.df.get(t, 0) <= 0:
            continue
        try:
            pl = TITLE_INDEX.read_a_posting_list(TITLE_BASE_DIR, t, bucket_name=BUCKET_NAME)
        except Exception:
            continue
        for doc_id, _tf in pl:
            if doc_id in cand:
                title_scores[doc_id] += 1.0

    anchor_scores = defaultdict(float)
    if ENABLE_ANCHOR:
        anchor_tasks = []
        for t in tokens:
            bits = int(TERM2BITS.get(t, 0)) if TERM2BITS is not None else 0
            if bits == 0:
                continue
            for part_id in _iter_set_bits_u64(bits):
                if part_id >= ANCHOR_PARTS_N:
                    continue
                anchor_tasks.append((part_id, t))

        if anchor_tasks:
            max_workers = 64
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_anchor_fetch_posting, pid, term) for pid, term in anchor_tasks]
                for fut in as_completed(futs):
                    pl = fut.result()
                    for doc_id, _tf in pl:
                        anchor_scores[doc_id] += 1.0

    # weights (tunable via env vars so we can test without editing code)
    # defaults keep behavior identical unless we explicitly set our env variavbles.
    W_BODY = _env_float("W_BODY", 1.0)
    W_TITLE = _env_float("W_TITLE", 0.25)
    W_ANCHOR = _env_float("W_ANCHOR", 0.20) if ENABLE_ANCHOR else 0.0
    W_PR = _env_float("W_PR", 0.10)
    W_PV = _env_float("W_PV", 0.0)

    merged = defaultdict(float)
    for doc_id, s in body_top:
        merged[doc_id] += W_BODY * float(s)
    for doc_id, s in title_scores.items():
        merged[doc_id] += W_TITLE * float(s)
    for doc_id, s in anchor_scores.items():
        merged[doc_id] += W_ANCHOR * float(s)
    for doc_id in list(merged.keys()):
        pr = _pr_of(int(doc_id))
        if pr > 0.0:
            merged[doc_id] += W_PR * math.log1p(pr)
        if W_PV > 0.0:
            _ensure_pageview_loaded()
            pv = _pv_of(int(doc_id))
            if pv > 0:
                merged[doc_id] += W_PV * math.log1p(pv)
    top = _topk(merged, 100)
    return jsonify([(int(doc_id), _doc_to_title(int(doc_id))) for doc_id, _ in top])
    # END SOLUTION


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get("query", "")
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    # load the index that contains words from article bodies
    if BODY_INDEX is None:
        _load_body_index()
    _ensure_id2title_loaded() # load the helper that converts doc ids to titles

    tokens = tokenize(query) # split the query into tokens using our helper function
    if not tokens:
        return jsonify(res)
    q_tf = Counter(tokens)
    N = _get_N_docs()
    q_weights = {}
    for t, tf in q_tf.items():
        df = BODY_INDEX.df.get(t, 0)
        if df <= 0:
            continue
        idf = math.log10(N / df)
        q_weights[t] = (tf / len(tokens)) * idf
    if not q_weights:
        return jsonify(res)
    scores = defaultdict(float)
    doc_sq = defaultdict(float)
    for t, wq in q_weights.items():
        try:
            # fetch the list of documents that contain this word
            pl = BODY_INDEX.read_a_posting_list(BODY_BASE_DIR, t, bucket_name=BUCKET_NAME)
        except Exception:
            continue
        df = BODY_INDEX.df.get(t, 0)
        if df <= 0:
            continue
        idf = math.log10(N / df)
        # look at every dcument that has this query word
        for doc_id, tf in pl:
            wd = (1.0 + math.log10(tf)) * idf
            scores[doc_id] += wq * wd
            doc_sq[doc_id] += wd * wd

    # calculate the length of the query vector
    q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
    if q_norm <= 0.0:
        return jsonify(res)
    # normalize the scores for every document we found
    for doc_id in list(scores.keys()):
        d_norm = math.sqrt(doc_sq.get(doc_id, 0.0))
        if d_norm > 0.0:
            scores[doc_id] /= (q_norm * d_norm)
        else:
            scores[doc_id] = 0.0
    top = _topk(scores, 100)
    # return the results as a list of (id, title) tuples
    return jsonify([(int(doc_id), _doc_to_title(int(doc_id))) for doc_id, _ in top])
    # END SOLUTION

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get("query", "")
    if len(query) == 0:
        return jsonify(res)

    # ensure the title index is loaded into memory
    if TITLE_INDEX is None:
        _load_title_index()
    _ensure_id2title_loaded() # ensure the id to title mapping helper is loaded
    tokens = tokenize(query)  # tokenize
    if not tokens:     # if tokenization results in no terms, return empty list
        return jsonify(res)

    # use a set to count distinct query terms only
    q_terms = set(tokens)
    # dictionary to track how many distinct query terms each doc matches
    match_counts = defaultdict(int)
    for t in q_terms:
        # skip terms that do not appear in the title index
        if TITLE_INDEX.df.get(t, 0) <= 0:
            continue
        try:
            # fetch the posting list for the current term from the bucket
            pl = TITLE_INDEX.read_a_posting_list(TITLE_BASE_DIR, t, bucket_name=BUCKET_NAME)
        except Exception:
            continue
        # increment the match count for every document containing this term
        for doc_id, _tf in pl:
            match_counts[doc_id] += 1
    # if no documents matched any terms, return empty
    if not match_counts:
        return jsonify(res)
    # sort results: primarily by match count, secondarily by doc_id (ascending)
    ranked = sorted(match_counts.items(), key=lambda x: (-x[1], x[0]))
    # return the sorted list of (id, title) tuples
    return jsonify([(int(doc_id), _doc_to_title(int(doc_id))) for doc_id, _c in ranked])


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get("query", "")
    if len(query) == 0:
        return jsonify(res)

    # feature flag: if anchor search is disabled in settings, stop here
    if not ENABLE_ANCHOR:
        return jsonify(res)

    # BEGIN SOLUTION
    _ensure_id2title_loaded()
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    match_counts = defaultdict(int)

    # identify which index partition (part) holds each query term
    tasks = []
    for t in tokens:  # keep duplicates per staff anchor spec
        # check our lookup map to see which file parts contain this term
        bits = int(TERM2BITS.get(t, 0)) if TERM2BITS is not None else 0
        if bits == 0:
            continue
        # decode the bits to find specific partition IDs
        for part_id in _iter_set_bits_u64(bits):
            if part_id >= ANCHOR_PARTS_N:
                continue
            tasks.append((part_id, t))
    if not tasks:
        return jsonify(res)
    # fetch data from multiple files at the same time to speed up network reading
    max_workers = 16
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_anchor_fetch_posting, pid, term) for pid, term in tasks]
        for fut in as_completed(futs):
            pl = fut.result()
            # count how many query words point to this doc
            for doc_id, _tf in pl:
                match_counts[doc_id] += 1
    if not match_counts:
        return jsonify(res)
    # sort by number of matches , then by doc_id (ascending)
    ranked = sorted(match_counts.items(), key=lambda x: (-x[1], x[0]))
    return jsonify([(int(doc_id), _doc_to_title(int(doc_id))) for doc_id, _c in ranked])
    # END SOLUTION


@app.route("/get_pagerank", methods=["POST"])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if not wiki_ids:
        return jsonify(res)
    # BEGIN SOLUTION
    _ensure_pagerank_loaded() # make sure the pagerank dictionary is loaded from the pickle file
    return jsonify([_pr_of(int(i)) for i in wiki_ids]) # lookup pagerank score for each id and return as a list
    # END SOLUTION


@app.route("/get_pageview", methods=["POST"])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if not wiki_ids:
        return jsonify(res)
    # BEGIN SOLUTION
    _ensure_pageview_loaded() # ensure pageview data is loaded
    return jsonify([_pv_of(int(i)) for i in wiki_ids]) # lookup pageview count for each id and return as a list
    # END SOLUTION


def run(**options):
    app.run(**options, debug=False, use_reloader=False)


if __name__ == "__main__":
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
