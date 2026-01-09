from flask import Flask, request, jsonify

import math
import pickle
import re
import os
from collections import Counter, defaultdict

from google.cloud import storage

from inverted_index_gcp import InvertedIndex

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# ----------------------------
# gcs / artifacts configuration
# ----------------------------
# this is your existing assignment-3 bucket
BUCKET_NAME = "information-retrival-ex3"

# folder (prefix) inside the bucket that contains your posting bins + index pickle(s)
BASE_DIR = "postings_gcp"

# globals loaded once at startup
BODY_INDEX = None  # InvertedIndex
ID2TITLE = None    # dict[int,str]  (we will load in a later stage)

# total number of documents in wikipedia corpus (from course artifacts); used for idf
# we will try to infer it from the index stats if possible, otherwise fall back to this constant.
N_DOCS_FALLBACK = 6348910

def _get_N_docs():
  # best effort: infer N from df stats if possible
  # if we cannot, fall back to a known wikipedia size used in the course.
  try:
    # df values are <= N, so max(df) is a lower bound; not perfect but better than nothing
    mx = max(BODY_INDEX.df.values()) if BODY_INDEX is not None else 0
    return max(mx, N_DOCS_FALLBACK)
  except Exception:
    return N_DOCS_FALLBACK

def _ensure_id2title_loaded():
  global ID2TITLE
  if ID2TITLE is not None:
    return

  # where we uploaded it in step 1
  blob_path = "artifacts/id2title.pickle"

  client = storage.Client()
  bucket = client.bucket(BUCKET_NAME)
  blob = bucket.blob(blob_path)

  try:
    # if it doesn't exist yet, keep working with doc_id as "title"
    if not blob.exists():
      print(f"[startup] id2title not found: gs://{BUCKET_NAME}/{blob_path} (returning doc_id as title for now)")
      ID2TITLE = None
      return

    print(f"[startup] loading id2title from gs://{BUCKET_NAME}/{blob_path} ...")
    data = blob.download_as_bytes()
    ID2TITLE = pickle.loads(data)
    print(f"[startup] loaded id2title entries: {len(ID2TITLE)}")
  except Exception as e:
    print(f"[startup] failed loading id2title from gs://{BUCKET_NAME}/{blob_path}: {e}")
    ID2TITLE = None
    return

def _doc_to_title(doc_id: int):
  if ID2TITLE is None:
    return str(doc_id)
  return ID2TITLE.get(doc_id, str(doc_id))

def _topk(score_dict, k: int):
  # score_dict: doc_id -> float
  if not score_dict:
    return []
  return sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:k]

# ----------------------------
# tokenizer (assignment 3 style)
# ----------------------------
import nltk
from nltk.corpus import stopwords

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# robust: download stopwords if missing in the local env
try:
    english_stopwords = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    english_stopwords = set(stopwords.words("english"))
corpus_stopwords = {
    "category", "references", "also", "links",
    "extenal", "see", "thumb"
}

all_stopwords = english_stopwords.union(corpus_stopwords)

def tokenize(text: str):
    if not text:
        return []
    tokens = [m.group() for m in RE_WORD.finditer(text.lower())]
    return [t for t in tokens if t not in all_stopwords]
# ----------------------------
# gcs helpers: auto-detect index pickle and load it
# ----------------------------
def _gcs_list_pickles(bucket_name: str, prefix: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    # list blobs under prefix
    blobs = bucket.list_blobs(prefix=prefix + "/")
    pickles = []
    for b in blobs:
        name = b.name
        if name.endswith(".pickle") or name.endswith(".pkl"):
            pickles.append(name)
    return sorted(pickles)

def _auto_detect_index_name(bucket_name: str, base_dir: str):
    """
    find the main index pickle name (without extension) inside base_dir.
    we ignore *_posting_locs.pickle because that is only the posting locations table.
    """
    pickles = _gcs_list_pickles(bucket_name, base_dir)
    # keep only files directly under base_dir
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

def _load_body_index():
    global BODY_INDEX
    if BODY_INDEX is not None:
        return
    name = _auto_detect_index_name(BUCKET_NAME, BASE_DIR)
    if name is None:
        # keep the server up, but make it obvious what's missing
        print(f"[startup] no index pickle found in gs://{BUCKET_NAME}/{BASE_DIR}/")
        BODY_INDEX = None
        return
    print(f"[startup] loading body index: gs://{BUCKET_NAME}/{BASE_DIR}/{name}.pickle")
    BODY_INDEX = InvertedIndex.read_index(BASE_DIR, name, bucket_name=BUCKET_NAME)

# load on startup (safe: if bucket auth is missing it will print an error later)
try:
    _load_body_index()
except Exception as e:
    print(f"[startup] failed to load index from gcs: {e}")
    BODY_INDEX = None

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
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

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
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    # ensure index is loaded
    if BODY_INDEX is None:
      try:
        _load_body_index()
      except Exception as e:
        return jsonify({"error": f"body index not loaded: {e}"})

    _ensure_id2title_loaded()

    tokens = tokenize(query)
    if len(tokens) == 0:
      return jsonify(res)

    # query term frequencies
    q_tf = Counter(tokens)

    # idf + query tf-idf weights
    N = _get_N_docs()
    q_weights = {}
    for t, tf in q_tf.items():
      df = BODY_INDEX.df.get(t, 0)
      if df == 0:
        continue
      idf = math.log10(N / df)
      q_weights[t] = (tf / len(tokens)) * idf

    if len(q_weights) == 0:
      return jsonify(res)

    # accumulate dot product scores: doc_id -> sum(w_q * w_d)
    scores = defaultdict(float)
    # also track doc weight squares for cosine denom (since we don't have precomputed norms yet)
    doc_sq = defaultdict(float)

    for t, wq in q_weights.items():
      try:
        pl = BODY_INDEX.read_a_posting_list(BASE_DIR, t, bucket_name=BUCKET_NAME)
      except Exception:
        continue

      df = BODY_INDEX.df.get(t, 0)
      if df == 0:
        continue
      idf = math.log10(N / df)

      for doc_id, tf in pl:
        # normalize tf by doc length is better, but we don't have lengths from A3 artifacts here.
        # use log-tf variant to keep things stable.
        wd = (1.0 + math.log10(tf)) * idf
        scores[doc_id] += wq * wd
        doc_sq[doc_id] += wd * wd

    # cosine normalize by (||q|| * ||d||)
    q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
    if q_norm == 0.0:
      return jsonify(res)

    for doc_id in list(scores.keys()):
      d_norm = math.sqrt(doc_sq.get(doc_id, 0.0))
      if d_norm == 0.0:
        scores[doc_id] = 0.0
      else:
        scores[doc_id] = scores[doc_id] / (q_norm * d_norm)

    top = _topk(scores, 100)

    # return (wiki_id, title) tuples. for now title is doc_id string until we load id2title.
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
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

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
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
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
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
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
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
