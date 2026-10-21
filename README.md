# JoySearch

Product search and personalised recommendations with FastAPI, React and reproducible offline evaluation.

- **Search:** BM25, MiniLM semantic retrieval, RRF fusion, query understanding and LambdaMART reranking over WANDS.
- **Recommendations:** Popularity, sparse ItemCF, history-based Two-Tower and candidate allocation over Amazon Baby Products ratings.
- **Traffic Lab:** cold-start slices and finite-arm policy simulation over explicit rating-target proxy rewards.
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

## Semantic search

Install the optional CPU dependencies in the same Conda environment:

```powershell
python -m pip install "torch==2.9.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dense-lock.txt
python scripts/build_dense.py
```

The builder downloads the fixed MiniLM revision on first use, encodes the catalog on CPU and saves progress every 512 products. Re-running resumes an interrupted build; an already completed matching index is reused. Query serving loads the local encoder and does not contact Hugging Face.

Select BM25, Dense or Hybrid in Search when the corresponding artifacts are ready. Dense ranks the complete catalog by exact cosine over normalized 384-dimensional vectors. Hybrid fuses each channel's Top100 using equal-weight RRF with `k=60`. Settings are in `configs/dense.json`.

```powershell
python scripts/evaluate.py --pipeline search --search-algorithms bm25 dense hybrid --search-split validation
python scripts/evaluate.py --pipeline all --search-algorithms bm25 dense hybrid
```

## Learning to rank

After building the semantic index, install LightGBM and train from the existing local data:

```powershell
python -m pip install -r requirements-rank-lock.txt
python scripts/train_ranker.py
python scripts/evaluate.py --pipeline search --search-algorithms bm25 dense hybrid ltr --search-split validation
python scripts/evaluate.py --pipeline all --search-algorithms bm25 dense hybrid ltr
```

LambdaMART reranks Hybrid's fixed Top100 pool using channel scores/ranks, lexical overlap and catalog features. The same feature extractor serves training and requests. Fit on judged retrieved candidates from training queries only, select the iteration by validation NDCG@10, then evaluate every candidate on the fixed test queries. Unjudged candidates are excluded from fitting, and known positives are never inserted into the pool.

The fixed configuration in `configs/ranker.json` uses gains `[0,1,3]`, 15 leaves, learning rate 0.05 and at most 300 trees. Validation early stopping selected 29 trees. Training used 20,125 judged candidate pairs from 284 of 288 training queries; validation used 6,534 pairs from 95 of 96 queries. Queries without judged candidates remain in full-cohort evaluation. Completed matching artifacts are reused. Restart the API to enable **LambdaMART** in the selector; the request identifier is `ltr`.

## Two-Tower recommendations

Use the CPU PyTorch installation from semantic search, or install the pinned CPU package in the same environment:

```powershell
python -m pip install "torch==2.9.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python scripts/train_two_tower.py
python scripts/evaluate.py --pipeline recommendation --recommendation-algorithms popularity itemcf two_tower --recommendation-split validation
python scripts/evaluate.py --pipeline all --search-algorithms bm25 dense hybrid ltr --recommendation-algorithms popularity itemcf two_tower
```

The user tower mean-pools learned context embeddings from up to 50 recent training items and projects them through an MLP. The item tower projects learned item embeddings into the same normalized 32-dimensional space. Scores are vector dot products over the full training catalog. There is no user-ID embedding. Unknown users receive the same explicit Popularity fallback.

Training removes each positive target from its own input history, samples eight unobserved training items per pair and rejects all of the user's training positives. Held-out labels are used only for validation selection and final evaluation; unobserved samples may still be latent positives. Single-interaction training histories cannot form a target-excluded pair, so those 897 users are excluded from fitting and retained in validation/test. The model trains on 49,018 pairs from 9,103 users.

`configs/two_tower.json` fixes one configuration. Full-catalog validation NDCG@10 selects epoch 5; training stops at epoch 9 with four-round patience, taking 21.53 seconds on CPU. Training progress and signatures are persisted locally. A completed matching model is reused; an interrupted training run restarts deterministically. Restart the API to enable **Two-Tower** in For You (`algorithm=two_tower`).

## Results

Evaluation: 96 held-out WANDS queries against 42,994 products; 10,000 Amazon users against the complete 16,574-item training catalog.

| Search algorithm | NDCG@10 | MRR@10 | JudgedRecall@100 | JudgmentCoverage@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.624031 | 0.782217 | 0.373704 | 0.794792 |
| Dense | 0.645170 | 0.834821 | 0.368893 | 0.825000 |
| Hybrid | 0.687759 | 0.866464 | 0.392750 | 0.857292 |
| Query Hybrid | 0.688063 | 0.866464 | 0.394610 | 0.858333 |
| LambdaMART | 0.628400 | 0.830481 | 0.392750 | 0.764583 |

Hybrid changes NDCG@10 by **+0.063728 (+10.21%)** relative to BM25 on this fixed test set. Dense has higher NDCG but slightly lower JudgedRecall@100 than BM25. LambdaMART changes NDCG@10 by **-0.059359 (-8.63%)** relative to Hybrid. Its lower JudgmentCoverage@10 accompanies more unjudged Top10 results. Query Hybrid adds **+0.000304 NDCG@10** and **+0.001860 JudgedRecall@100** over Hybrid: one query improves and 95 are unchanged. The default remains BM25. Retrieval settings and the ranker configuration were fixed before test evaluation; only the boosting iteration used validation. Query rules use train queries and unlabeled catalog text; their soft weight uses validation. This is a repeated fixed-test regression comparison, not a new untouched test set.

| Recommendation algorithm | HR@10 | NDCG@10 | CatalogCoverage@10 |
|---|---:|---:|---:|
| Popularity | 0.0163 | 0.007573 | 0.000905 |
| ItemCF | 0.0046 | 0.002475 | 0.820261 |
| Two-Tower | 0.0119 | 0.005768 | 0.012067 |

ItemCF trades lower hit rate for broader catalog coverage in this experiment. Its HR@10 difference from Popularity is −0.0117; NDCG@10 differs by −0.005098. Two-Tower improves HR@10 by +0.0073 and NDCG@10 by +0.003293 over ItemCF, with lower catalog coverage. Relative to Popularity, Two-Tower has HR@10 −0.0044 and NDCG@10 −0.001805. These are fixed-cohort offline comparisons; the default algorithm remains ItemCF.


Warm CPU inference over the same 96 test queries on an Intel Core i5-1240P (four encoder threads, sequential requests):

| Algorithm | p50 (ms) | p95 (ms) |
|---|---:|---:|
| bm25 | 0.910 | 3.028 |
| dense | 21.487 | 24.975 |
| hybrid | 24.273 | 30.053 |
| ltr | 30.625 | 36.320 |
| query_hybrid | 29.332 | 35.051 |

Warm recommendation inference over all 10,000 test users:

| Algorithm | p50 (ms) | p95 (ms) |
|---|---:|---:|
| popularity | 0.020 | 0.027 |
| itemcf | 1.068 | 1.190 |
| two_tower | 1.914 | 2.287 |
| rrf | 4.775 | 7.948 |
| quota | 5.442 | 9.365 |
| adaptive_quota | 5.645 | 9.675 |

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

## Recall diagnostics

Candidate-stage diagnostics are available independently of the serving benchmark:

```powershell
python scripts/evaluate_recall.py --split validation
python scripts/evaluate_recall.py --split test
python scripts/verify_release.py --recall-diagnostics
```

The diagnostic matrix compares every nonempty subset of the two search or three recommendation channels at B=100, preserving final K=10 metrics. Native ItemCF contribution excludes Popularity completion; `itemcf_served` separately preserves the existing fallback baseline. Traces retain channel ranks/scores, deduplicated raw unions, seen filtering, fixed-budget selections and final lists. Raw-union recall is reported separately from fixed-budget gains. Results stay in local `reports/run_e1_*` directories; the diagnostic pointer does not replace the complete application benchmark pointer. See [recall diagnostics](docs/recall_evaluation.md) for stage schemas, denominators and ablation rules.

On the fixed test cohorts, the diagnostic results are:

| Search channel set | CandidateJudgedRecall@100 | NDCG@10 |
|---|---:|---:|
| BM25 | 0.373704 | 0.624031 |
| Dense | 0.368893 | 0.645170 |
| BM25 + Dense RRF | 0.392750 | 0.687759 |

The untruncated search union averages 159.32 candidates and has judged recall 0.482971. It uses a larger candidate set than the B=100 comparisons.

| Recommendation channel set | CandidateHit@100 | HR@10 | NDCG@10 |
|---|---:|---:|---:|
| Popularity | 0.1006 | 0.0163 | 0.007573 |
| Native ItemCF | 0.0184 | 0.0039 | 0.002125 |
| Two-Tower | 0.0841 | 0.0119 | 0.005768 |
| All three, equal-weight RRF | 0.0815 | 0.0158 | 0.007388 |

Removing native ItemCF from the three-channel fusion changes CandidateHit@100 by +0.0117, to 0.0932. Its unique candidates do not translate into a fixed-budget gain in this experiment. Native ItemCF underfills B for 26.36% of users, averaging 85.64 candidates; the separate served baseline retains Popularity completion and reproduces HR@10=0.0046. All 2,023 unavailable test targets remain misses. The complete diagnostic run took 160.60 seconds on CPU and produced 57.10 MB of Parquet traces, without retraining or rebuilding indexes.

## Allocation and cold-start analysis

Recommendation candidates can use equal RRF, validation-selected fixed quotas or
request-level quotas based on training-history length and native ItemCF support.
Each channel retains Top100 after seen-item filtering; allocation reserves unique
slots, reallocates shortfalls and ranks the final 100 candidates with rank-based
scores. Debug responses retain requested/filled quotas and source attribution.

On the same 10,000 test users and 16,574 training items:

| Policy | CandidateHit@100 | HR@10 | NDCG@10 | CatalogCoverage@10 |
|---|---:|---:|---:|---:|
| Popularity | 0.1006 | 0.0163 | 0.007573 | 0.000905 |
| Equal RRF | 0.0815 | 0.0158 | 0.007388 | 0.006758 |
| Fixed quota `(40,20,40)` | 0.0854 | 0.0156 | 0.007231 | 0.004948 |
| Adaptive `support_only` | 0.0878 | 0.0160 | 0.007470 | 0.004404 |

Adaptive quotas add 0.0002 HR@10 over equal RRF and remain 0.0003 below
Popularity. Fixed quotas lose HR and NDCG relative to equal RRF. TailListShare@10
is zero for these fused policies: reserving candidate slots for ItemCF does not
ensure that low-frequency candidates reach Top10. Native ItemCF has substantially
broader coverage with lower hit rate. History and target-frequency slices retain
all users, including the 2,023 unavailable targets. Unknown users explicitly receive
train-only Popularity. The default remains ItemCF.

Traffic Lab compares fixed, 10% fixed exploration and UCB1 over three quota arms.
Across five declared seeds, test proxy hit rates are 0.01560, 0.01568 ± 0.00027
and 0.01570 ± 0.00026 (mean ± sample standard deviation). UCB's 0.00010 mean
difference is smaller than the seed-to-seed variation. The reward is held-out
positive-rating target inclusion, with only selected-arm feedback updating UCB.
These results are offline policy diagnostics, not exposure-log replay or online
click uplift. Top10 low-frequency share remains zero in all three simulations.

The [allocation protocol](docs/allocation.md) documents quota math, source-cache
verification, complete validation/test commands and proxy-oracle regret. The UI
reads saved reports and supports history/frequency slices. Candidate allocation
uses all three channels and does not reduce retrieval computation. No new model
training or catalog embedding build is needed for this stage.

## Query understanding

Select **Query Hybrid** to retain the original BM25/Dense channels and add a
rewrite-BM25 channel when finite aliases or unit rules apply. Final candidates
remain capped at 100. The shared parser preserves the original query, extracts
category/color/material/size and negation, and marks unsupported price, stock and
delivery requirements. API requests with `debug: true` return the `query_plan`.

Validation compared soft attribute weights 0 and 0.002 and selected 0; the larger
weight reduced NDCG. Product attributes aggregate variants, so missing fields do
not reject products and no hard filtering is performed. Human slot-quality
metrics remain unevaluated until independent annotations are available. See
[query understanding](docs/query_understanding.md) for the annotation guide,
fixed-budget ablations, saved validation selection and reproducible commands.

## Architecture

| Module | Responsibility |
|---|---|
| `data` | Schema validation, deduplication, deterministic splits and provenance |
| `search` | Retrieval indexes, fusion, shared features, offline ranker training and `SearchPipeline` |
| `recommendation` | Train-only Popularity, sparse ItemCF, history-based Two-Tower and `RecPipeline` |
| `traffic` | Finite-arm policies and offline rating-target proxy simulation |
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
| `GET /api/v1/reports/allocation` | Stored quota, cold-start slices and simulation results |

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

For a short demo: search `salon chair`; compare ItemCF and Adaptive for an anonymous
user and inspect the request's actual quota; select an unknown user to see Popularity
fallback; open Evaluation for algorithm results and Traffic Lab for history/frequency
slices and the seeded offline policy comparison.

## Sources

- [WANDS](https://github.com/wayfair/WANDS), MIT licence. Chen et al., *WANDS: Dataset for Product Search Relevance Assessment*, ECIR 2022.
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0. The fixed revision is recorded in `configs/dense.json`.
- [LightGBM Ranker](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html), used for LambdaMART fitting and native model persistence.
- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) and its [5-core rating-only files](https://amazon-reviews-2023.github.io/data_processing/5core.html). Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, 2024. Dataset use and redistribution follow the source terms.

Raw data, prepared interactions, model weights, user mappings and local caches stay outside Git.

See [architecture](docs/architecture.md) and [evaluation protocol](docs/evaluation.md) for the module contracts and metric definitions.
