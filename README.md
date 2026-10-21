# JoySearch

Product search and personalised recommendations with FastAPI, React and reproducible offline evaluation.

- **Search:** BM25 over the WANDS product catalog.
- **Recommendations:** Popularity and sparse ItemCF over Amazon Baby Products ratings.
- **Evaluation:** fixed held-out splits, algorithm comparisons, latency measurements and versioned reports.

## Run locally

Windows PowerShell, the existing `joysearch` Conda environment, Python 3.11 and Node.js 24.

```powershell
cd D:\Joysearch\Joysearch
conda activate joysearch
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m uvicorn joysearch.api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd D:\Joysearch\Joysearch\frontend
npm.cmd ci
npm.cmd run dev -- --port 5173
```

Open the [application](http://127.0.0.1:5173) or [API documentation](http://127.0.0.1:8000/docs). Vite proxies `/api` to port 8000. Model artifacts load at startup; requests run inference only.

## Build from local data

Place the source files at:

```text
data/raw/wands/{product,query,label}.csv
data/raw/amazon/Baby_Products.csv
```

Then prepare, build and evaluate:

```powershell
python scripts/prepare_wands.py
python scripts/prepare_amazon.py --users 10000 --seed 42
python scripts/build_search.py
python scripts/build_rec.py
python scripts/evaluate.py --pipeline all
```

Existing prepared data and models can be reused when restarting the app. Source hashes, split membership and sampling settings are recorded under `data/processed`; model artifacts are under `artifacts`.

## Results

P0 evaluation: 96 held-out WANDS queries against 42,994 products; 10,000 Amazon users against the complete 16,574-item training catalog.

| Search algorithm | NDCG@10 | MRR@10 | JudgedRecall@100 | JudgmentCoverage@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.624031 | 0.782217 | 0.373704 | 0.794792 |

| Recommendation algorithm | HR@10 | NDCG@10 | CatalogCoverage@10 |
|---|---:|---:|---:|
| Popularity | 0.0163 | 0.007573 | 0.000905 |
| ItemCF | 0.0046 | 0.002475 | 0.820261 |

ItemCF trades lower hit rate for broader catalog coverage in this experiment. Its HR@10 difference from Popularity is −0.0117; NDCG@10 differs by −0.005098.

Each evaluation generates a local `reports/run_<UTC>_<id>/` directory with metrics, paired comparisons, per-query/user diagnostics, latency measurements and source/configuration hashes. `reports/latest.json` identifies the latest complete run. Evaluation outputs are generated locally and excluded from Git.

### Evaluation protocol

- **WANDS:** query-ID-disjoint 60/20/20 split, seed 42. Labels map to Exact=2, Partial=1, Irrelevant=0. Duplicate query/product judgments use the minimum grade: 233,448 raw rows become 231,873 unique pairs, including 14 conflicting pairs. Sparse qrels mean unjudged results contribute zero gain; JudgedRecall measures recall against judged positives.
- **Amazon:** deduplicate user/item pairs by latest timestamp, breaking timestamp ties by highest rating, then retain ratings ≥4. Sample 10,000 of 142,636 eligible users with complete positive histories. Per-user chronological leave-last-two-out yields 49,915 train, 10,000 validation and 10,000 test events. Both algorithms exclude training positives and use the same full training catalog. All 2,023 targets absent from that catalog remain misses.
- **Interpretation:** ratings represent preference; per-user temporal splits differ from a global cutoff. Search and recommendation use independent catalogs. Recommendation metadata is unavailable, so the app displays actual item IDs. Offline relevance metrics do not establish online business outcomes.
- **Latency:** load the model, warm up five requests, then measure sequential inference over the test cohort. Report p50/p95 milliseconds separately from loading, building and HTTP overhead. Repeated test runs are marked in `config.json`.

Run either evaluator independently:

```powershell
python scripts/evaluate.py --pipeline search
python scripts/evaluate.py --pipeline recommendation
```

## Architecture

| Module | Responsibility |
|---|---|
| `data` | Schema validation, deduplication, deterministic splits and provenance |
| `search` | Retrieval indexes, ranking fusion and `SearchPipeline` |
| `recommendation` | Train-only Popularity, sparse TopN ItemCF and `RecPipeline` |
| `evaluation` | Pure metric functions, independent evaluators and reports |
| `api` | Typed requests, model lifecycle, readiness and anonymous user mapping |
| `frontend` | Search, For You and Evaluation views backed by the API |
| `scripts` | Preparation, model builds, evaluation and smoke entrypoints |

BM25 uses sparse term weights, `k1=1.2`, `b=0.75`, Unicode-normalised lowercase tokens and repeated titles. ItemCF computes shrinkage cosine similarity in sparse blocks and stores up to 100 neighbours per item. Cold users receive training-set Popularity; insufficient neighbours are completed from that baseline with an explicit reason.

## API

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Process health |
| `GET /readyz` | Component readiness and versions |
| `POST /api/v1/search` | Product search |
| `POST /api/v1/recommend` | Recommendations |
| `GET /api/v1/users/demo` | Anonymous training-user aliases |
| `GET /api/v1/reports/summary` | Stored evaluation results |

```json
{"catalog_id":"wands_search","query":"salon chair","algorithm":"bm25","top_k":10}
```

```json
{"catalog_id":"amazon_baby_rec","user_id":"anonymous_user_00001","algorithm":"itemcf","exclude_seen":true,"top_k":10}
```

Invalid input returns 422. Unavailable models return 503. Result items include catalog identity, source, rank and score; scores have algorithm-specific meanings.

## Verification

```powershell
python -m pytest -q
python -m ruff check .
python scripts/smoke.py
python scripts/smoke.py --base-url http://127.0.0.1:8000
cd frontend
npm.cmd run build
```

With the API running, `python scripts/verify_release.py` records the full release checks, evaluation and ASGI/HTTP smoke output under `reports/quality/<UTC>/`. The current Starlette version emits a TestClient deprecation warning; the recorded tests pass.

For a short demo: search `salon chair`; compare ItemCF for the first two anonymous users; select an unknown user to see Popularity fallback; open Evaluation and inspect the paired algorithm results.

## Sources

- [WANDS](https://github.com/wayfair/WANDS), MIT licence. Chen et al., *WANDS: Dataset for Product Search Relevance Assessment*, ECIR 2022.
- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) and its [5-core rating-only files](https://amazon-reviews-2023.github.io/data_processing/5core.html). Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, 2024. Dataset use and redistribution follow the source terms.

Raw data, prepared interactions, model weights, user mappings and local caches stay outside Git.
