from flask import Flask, request, jsonify

import math
import pickle
import re
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

# ----------------------------
# tokenizer (assignment 3 style)
# ----------------------------
import nltk
from nltk.corpus import stopwords

# staff-style regex used in assignment 3 gcp part
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# corpus stopwords (same set used in the course notebooks)
CORPUS_STOPWORDS = {
    "category", "references", "also", "external", "links", "may", "first", "see",
    "history", "people", "one", "two", "part", "thumb", "including", "second",
    "following", "many", "however", "would", "became"
}

def tokenize(text: str):
    # lower + regex tokenize + stopword removal
    if not text:
        return []
    english_stopwords = set(stopwords.words("english"))
    all_stopwords = english_stopwords.union(CORPUS_STOPWORDS)
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

    # debug: fetch df + posting length for the first token
    tokens = tokenize(query)
    if len(tokens) == 0:
      return jsonify(res)

    t0 = tokens[0]
    try:
      pl = BODY_INDEX.read_a_posting_list(BASE_DIR, t0, bucket_name=BUCKET_NAME)
    except Exception as e:
      return jsonify({"error": f"failed reading posting list for '{t0}': {e}"})

    # return a small debug payload for now
    return jsonify({
      "token": t0,
      "df": int(BODY_INDEX.df.get(t0, 0)),
      "posting_len": len(pl),
      "sample": pl[:5]
    })
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
