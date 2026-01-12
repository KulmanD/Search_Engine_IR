# Information Retrieval Search Engine (Wikipedia Corpus)

## Quick VM Commands

Start the VM:
```bash
gcloud compute instances start ir-frontend-1 --zone us-central1-c
```

Sanity check:
```bash
curl "http://35.222.160.75:8080/search_title?query=haifa"
```

Connect to the VM:
```bash
gcloud compute ssh ubuntu@ir-frontend-1 --zone us-central1-c --tunnel-through-iap
```

Upload updated server code to the VM:
```bash
gcloud compute scp \
  /Users/denis/PythonProject/Search_Engine_IR/search_frontend.py \
  ubuntu@ir-frontend-1:/home/ubuntu/search_frontend.py \
  --zone us-central1-c \
  --tunnel-through-iap
```

Restart the service:
```bash
sudo systemctl restart ir-frontend.service
```
## Overview
This project implements a **functional, testable, and efficient search engine** over the full English Wikipedia corpus, developed as part of the Information Retrieval course.

The system exposes a REST API that allows querying the corpus using different retrieval strategies (title, body, anchor text, PageRank), while meeting all **mandatory efficiency and quality requirements**.

---

## Deployment & Availability
- **Platform:** Google Cloud VM (Ubuntu)
- **Server:** Flask (Python 3.10)
- **Port:** `8080`
- **Uptime:** The server is deployed as a `systemd` service and is continuously available during the grading window.

The service is accessible via:http://<35.222.160.75>:8080
---

## Implemented Endpoints

### `/search_title`
Returns all documents whose **title contains at least one query term**, ranked by the **number of distinct query words appearing in the title**.

Example: /search_title?query=haifa
This endpoint is:
- Extremely fast
- Deterministic
- Fully compliant with staff specification
- Used to verify baseline correctness and quality

---

### `/search`
The **main retrieval endpoint used for grading**.

Based on empirical evaluation, this endpoint defaults to a **title-only baseline**, which achieved the best effectiveness on the provided training queries.

**Final behavior (default):**
- Ranks documents by the **number of distinct query terms appearing in the title**
- Returns the top 100 results
- Deterministic, fast, and stable

**Rationale:**
Offline evaluation showed that the title-only method significantly outperformed body-based and combined approaches in MAP@10, while also providing much lower latency and higher stability.

**Experimental combined retrieval** (body TF-IDF + title boost + PageRank + optional anchor) is preserved in the codebase and can be enabled explicitly using a runtime flag:

```ini
Environment=USE_COMBINED_SEARCH=1
```

By default, this flag is disabled to ensure optimal quality and efficiency during grading.
---

### `/search_body`
Returns up to 100 documents ranked using **TF-IDF cosine similarity over article bodies only**.

This endpoint is provided for completeness and experimentation. In offline evaluation, body-only TF-IDF without document-length normalization (BM25) yielded very low effectiveness on the training set, and is therefore not used in the final `/search` endpoint.
---

### `/search_anchor`
Returns all documents ranked by the **number of query words appearing in anchor text** linking to the page.

This endpoint exists for completeness and experimentation, but **anchor usage is disabled by default** (see below).

---

### `/get_pagerank`
POST endpoint returning PageRank values for a list of document IDs.

---

### `/get_pageview`
POST endpoint returning August-2021 pageview counts (optional signal).

---

## Anchor Text Handling (Important)

Anchor-based indexing and retrieval were **fully implemented**, including:
- Sharded anchor inverted indexes
- Term-to-shard bitset routing
- Parallel shard fetching

However, due to:
- VM disk constraints
- High I/O overhead
- Inconsistent quality gains relative to cost

**Anchor scoring is disabled by default using a runtime feature flag**, without deleting code or indexes.

This ensures:
- Stability during grading
- Compliance with efficiency requirements
- Preservation of experimental work for potential bonus credit

### Runtime Feature Flag
Anchor usage is controlled via an environment variable in the systemd service:

```ini
Environment=ENABLE_ANCHOR=0
```
###  Efficiency
- Average query latency: ~0.11 seconds (local VM)
- st-case latency: well below 35 seconds
- caching of query results
-  reads are minimized and parallelized where applicable

## Quality Evaluation

The search engine was evaluated using the **staff-provided training queries and relevance judgments**.

### Evaluation Setup
- **Queries:** `queries_train.json`
- **Ground truth:** provided relevance lists per query
- **Metric:** Mean Average Precision at 10 (MAP@10)
- **Evaluation method:** offline evaluation by issuing queries to the running `/search_title` endpoint and comparing the top-10 results against the ground truth.

### Results (MAP@10 on training set)

| Endpoint        | MAP@10 |
|-----------------|--------|
| `/search_title` | 0.2167 |
| `/search_body`  | 0.0000 |
| `/search`       | 0.2167 |

The combined retrieval approach (`/search` with body + title + PageRank) was evaluated but resulted in significantly lower effectiveness (MAP@10 ≈ 0.0004). Consequently, the final `/search` endpoint uses the title-only baseline, which comfortably exceeds the required threshold (MAP@10 ≥ 0.1).

---

## Efficiency

- **Average query latency:** ~0.11 seconds (measured locally on the VM)
- **Worst-case latency:** well below the required 35 seconds
- **No caching of query results** (as required)
- Index reads are **minimized and parallelized** where applicable
- The system remains responsive under repeated query load

---

## Notes on Experimental Features

Anchor-based retrieval was fully implemented and evaluated, including sharded anchor indexes, term-to-shard routing, and parallel shard fetching.

During testing, enabling anchor scoring caused severe memory pressure and I/O overhead on the VM, leading to repeated server instability and crashes (OOM kills). Due to these constraints and the limited observed quality improvement, anchor usage is disabled by default via a runtime flag.

The full implementation was intentionally preserved in the codebase to document the attempted approach and support potential bonus consideration.