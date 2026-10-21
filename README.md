# JoySearch

Product search and personalised recommendations with FastAPI, React and reproducible offline evaluation.

## Features

- **Search:** BM25, MiniLM semantic retrieval, RRF fusion, query understanding and LambdaMART reranking across 42,994 WANDS products.
- **Recommendations:** Popularity, sparse ItemCF, history-based Two-Tower and adaptive candidate allocation for Amazon Baby Products users.
- **Evaluation:** algorithm comparisons, candidate attribution, channel ablations, user-history slices and latency profiling.
- **Traffic Lab:** fixed, exploration and UCB1 policies evaluated with held-out rating-target rewards.
- **Offline experiments:** local Qwen query rewriting and relevance labelling; GRU-based semantic product-ID generation with constrained beam search.

The React app provides Search, For You, Evaluation and Traffic Lab views. FastAPI loads model assets at startup and serves inference through typed endpoints.

## Results

Search evaluation uses 96 fixed held-out queries and the complete 42,994-product WANDS catalog. Each method returns up to 100 candidates; ranking metrics use the first 10 results.

| Search method | NDCG@10 | MRR@10 | JudgedRecall@100 | JudgmentCoverage@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.624031 | 0.782217 | 0.373704 | 0.794792 |
| Dense | 0.645170 | 0.834821 | 0.368893 | 0.825000 |
| Hybrid | 0.687759 | 0.866464 | 0.392750 | 0.857292 |
| Query Hybrid | 0.688063 | 0.866464 | 0.394610 | 0.858333 |
| LambdaMART | 0.628400 | 0.830481 | 0.392750 | 0.764583 |

Hybrid increases NDCG@10 by **10.21%** and MRR@10 by **10.77%** relative to BM25 on this cohort. Paired profiling of the exact-vector scan reduces stage p95 from **42.27 ms to 21.68 ms**, with identical candidate IDs and scores.

Recommendation evaluation uses the same 10,000 users and complete 16,574-item training catalog for every method.

| Recommendation method | HR@10 | NDCG@10 | CatalogCoverage@10 |
|---|---:|---:|---:|
| Popularity | 0.0163 | 0.007573 | 0.000905 |
| ItemCF | 0.0046 | 0.002475 | 0.820261 |
| Two-Tower | 0.0119 | 0.005768 | 0.012067 |
| RRF | 0.0158 | 0.007388 | 0.006758 |
| Fixed quota | 0.0156 | 0.007231 | 0.004948 |
| Adaptive quota | 0.0160 | 0.007470 | 0.004404 |

Popularity achieves the highest HR@10 in this comparison; ItemCF reaches the broadest catalog coverage. Two-Tower combines history embeddings with learned item representations, and the fusion policies control how channels share the candidate budget.

### Evaluation setup

- **Search:** query-ID-disjoint 60/20/20 split with seed 42. Relevance grades are Exact=2, Partial=1 and Irrelevant=0. NDCG uses gains `[0,1,3]`; unjudged results contribute zero gain. JudgedRecall and JudgmentCoverage describe the sparse relevance judgments.
- **Recommendations:** ratings of 4 or 5 define positive-preference targets. Per-user chronological leave-last-two-out produces 49,915 training events and 10,000 events for each holdout. Every method filters training positives and retains all users, including 2,023 test targets outside the training catalog.
- **Selection and reproducibility:** validation selects checkpoints and allocation policies. Fixed test cohorts support repeated regression comparisons. Reports record per-query/user results, paired differences, latency and source/configuration hashes.

See the [evaluation protocol](docs/evaluation.md), [recall diagnostics](docs/recall_evaluation.md) and [serving benchmark](docs/serving_performance.md) for definitions, experiment details and commands.

## Architecture

| Module | Responsibility |
|---|---|
| `data` | Schema validation, deduplication, deterministic splits and provenance |
| `search` | BM25, semantic encoding, fusion, query parsing, ranking and semantic-ID generation |
| `recommendation` | Popularity, sparse ItemCF, Two-Tower and candidate allocation |
| `traffic` | Finite-arm policies and offline simulation |
| `evaluation` | Metrics, evaluators, comparisons and profiling |
| `api` | Typed requests, asset lifecycle, readiness and anonymous user aliases |
| `frontend` | Search, recommendation and experiment views |
| `scripts` | Data preparation, index builds, training, evaluation and release checks |

Search uses the `wands_search` catalog; recommendations use `amazon_baby_rec`. Training and evaluation run through scripts, while HTTP handlers call the inference pipelines.

BM25 uses `k1=1.2`, `b=0.75` and repeated titles. Dense retrieval uses normalised 384-dimensional MiniLM vectors and exact cosine scoring. Hybrid combines each channel's Top100 using equal-weight RRF with `k=60`. LambdaMART reranks that pool with 13 shared features.

ItemCF builds shrinkage cosine similarities in sparse blocks and stores up to 100 neighbours per item. Two-Tower mean-pools up to 50 training-history items into a normalised 32-dimensional representation and trains with eight sampled negatives per positive. Cold users receive training-set Popularity recommendations.

## Run locally

Use Windows PowerShell, Python 3.11 in the `joysearch` Conda environment, and Node.js 24. Run these commands from the repository root:

```powershell
conda activate joysearch
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps --no-build-isolation
```

Place the datasets at:

```text
data/raw/wands/{product,query,label}.csv
data/raw/amazon/Baby_Products.csv
```

Prepare data and build the baseline models:

```powershell
python scripts/prepare_wands.py
python scripts/prepare_amazon.py --users 10000 --seed 42
python scripts/build_search.py
python scripts/build_rec.py
```

Start the API:

```powershell
python -m uvicorn joysearch.api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev -- --port 5173
```

Open the [app](http://127.0.0.1:5173) or [API docs](http://127.0.0.1:8000/docs). Vite proxies `/api` to port 8000. Prepared data and matching model assets are reused across restarts.

### Add semantic search and learned models

```powershell
python -m pip install "torch==2.9.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dense-lock.txt
python -m pip install -r requirements-rank-lock.txt
python scripts/build_dense.py
python scripts/train_ranker.py
python scripts/train_two_tower.py
```

The Dense builder retrieves the pinned MiniLM revision on first use, encodes the catalog and checkpoints every 512 products. Restart the API after building models to update the algorithm selectors.

### Evaluate and verify

```powershell
python scripts/evaluate.py --pipeline all --search-algorithms bm25 dense hybrid ltr query_hybrid --recommendation-algorithms popularity itemcf two_tower rrf quota adaptive_quota
python -m pytest -q
python -m ruff check .
python scripts/smoke.py
python scripts/smoke.py --base-url http://127.0.0.1:8000
```

Run `npm.cmd run build` from `frontend` to build the UI. With the API running, `python scripts/verify_release.py` runs the release checks. Evaluations write versioned results to `reports/run_<UTC>_<id>/`; `reports/latest.json` identifies the latest complete run.

## API

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Process health |
| `GET /readyz` | Component readiness and versions |
| `POST /api/v1/search` | Product search |
| `POST /api/v1/recommend` | Personalised recommendations |
| `GET /api/v1/users/demo` | Anonymous training-user aliases |
| `GET /api/v1/reports/summary` | Evaluation results |
| `GET /api/v1/reports/allocation` | Allocation, user slices and policy simulation |

Example request bodies:

```json
{"catalog_id":"wands_search","query":"salon chair","algorithm":"hybrid","top_k":10}
```

```json
{"catalog_id":"amazon_baby_rec","user_id":"anonymous_user_00001","algorithm":"two_tower","exclude_seen":true,"top_k":10}
```

## Technical guides

- [Architecture and module contracts](docs/architecture.md)
- [Query understanding](docs/query_understanding.md)
- [LambdaMART diagnostics](docs/ltr_diagnostics.md)
- [Candidate allocation and Traffic Lab](docs/allocation.md)
- [Local LLM experiments](docs/local_llm.md)
- [Generative retrieval](docs/generative_retrieval.md)

## Data and models

- [WANDS](https://github.com/wayfair/WANDS): product search relevance dataset, MIT licence.
- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) and [5-core ratings](https://amazon-reviews-2023.github.io/data_processing/5core.html): Baby Products preference data, used under the source terms.
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): semantic encoder, Apache-2.0.
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct): local diagnostic model, Apache-2.0.
- [LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html): LambdaMART training and model persistence.

Dataset manifests, model revisions and configuration files identify the assets used in each run.
