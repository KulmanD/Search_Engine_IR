# Information Retrieval Search Engine (Wikipedia Corpus)
### for our convince the VM currently offline , to reactivate:
`gcloud compute instances start ir-frontend-1 --zone us-central1-c`
- then test by fetching simple query :
- `curl "http://35.222.160.75:8080/search_title?query=haifa"`
- or try localy `curl http://127.0.0.1:8080/search_title?query=haifa`
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
The **main retrieval endpoint**, combining multiple ranking signals:

1. **Body TF-IDF with cosine similarity** (candidate generation)
2. **Title matches** (distinct query terms)
3. **PageRank boost** (log-scaled)
4. *(Optional)* Anchor text signal (runtime-controlled)

Results are merged deterministically and ranked from best to worst.

Example: /search?query=database systems
---

### `/search_body`
Returns up to 100 documents ranked using **TF-IDF cosine similarity over article bodies only**.

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

### Result : MAP@10 = 0.2167
This score **exceeds the minimum required threshold (MAP@10 ≥ 0.1)** specified in the assignment instructions.

---

## Efficiency

- **Average query latency:** ~0.11 seconds (measured locally on the VM)
- **Worst-case latency:** well below the required 35 seconds
- **No caching of query results** (as required)
- Index reads are **minimized and parallelized** where applicable
- The system remains responsive under repeated query load

---

## Notes on Experimental Features

Anchor-based retrieval was implemented and tested, but is **disabled by default** via a runtime flag due to performance and resource considerations.  
The code and indexes were intentionally preserved to document alternative approaches and support potential bonus evaluation.