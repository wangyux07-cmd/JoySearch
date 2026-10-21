# Offline evaluation

## WANDS search

Use all 42,994 products as candidates. Split query IDs 60/20/20 with seed 42; persist the exact membership. Relevance grades are Irrelevant=0, Partial=1 and Exact=2. Resolve repeated query/product judgments using the minimum grade before evaluation. The index and embedding builder read product text only.

For every query:

- **NDCG@10:** gain `2**grade - 1`, discount `log2(rank + 1)`, ideal ranking from that query's available judgments. Unjudged results contribute zero; zero ideal gain yields zero.
- **MRR@10:** reciprocal rank of the first judged grade >=1, or zero.
- **JudgedRecall@100:** distinct retrieved judged positives divided by all judged positives; zero for queries without positives.
- **JudgmentCoverage@10:** judged returned IDs divided by returned count up to ten; zero for empty results.

Macro-average over every query in the fixed split. Record empty results, missing judgments and no-positive queries. Compare BM25, Dense and Hybrid on the same QIDs, catalog and formulas. Sparse judgments constrain interpretation of relevance and recall.

Semantic retrieval uses a fixed MiniLM revision, L2-normalized 384-dimensional vectors and exact cosine over the whole catalog. Product and query encoding share one local encoder. Hybrid applies equal-weight RRF: `sum(1/(60 + rank))` over each channel's Top100. Stable ties use item IDs. Model/config choices are fixed before the held-out test evaluation; validation is separate.

### LambdaMART

Retrieve the same RRF Top100 pool for every query from the full catalog, then rerank that fixed pool. Candidate construction never reads judgments or inserts known positives. Training uses only retrieved candidates with explicit 0/1/2 judgments from training queries; unjudged rows are excluded from fitting and early stopping. Queries with no judged retrieved candidates cannot form a training group and are counted in the artifact manifest. They remain in the full-cohort evaluation.

`configs/ranker.json` fixes a single LightGBM LambdaRank configuration and gains `[0,1,3]`. Select the boosting iteration using validation **NDCG@10 on judged retrieved candidates only**, with 30-round patience. This selection proxy differs from the headline full-retrieval metric: final validation/test evaluation ranks all retrieved candidates and computes ideal DCG from all available query judgments. No test judgments select parameters or features. Repeated test use is marked as regression evaluation.

Training and serving share `SearchFeatures`: channel scores, presence and ranks, RRF score, title/category token overlap, query/title lengths and catalog review statistics. Query-class annotations and qrels are excluded from features. Persist contiguous query groups, split membership, feature order, cache checksums, model checksum and retrieval/code/data signatures. A matching completed model is reused; incompatible serving artifacts fail with a rebuild instruction.

Compare LTR with Hybrid on identical QIDs and the exact same Top100 candidate set. JudgedRecall@100 must remain equal because reranking changes order only. Save both LTR-versus-Hybrid and baseline comparisons, including negative deltas; do not infer statistical significance from point estimates.

## Amazon recommendations

Deduplicate user/item events by latest timestamp, with highest rating breaking timestamp ties, then retain rating >=4. Require at least three distinct positive items per user. Sample users deterministically while retaining complete histories. Sort by `(timestamp,item_id)`; use the final event for test, penultimate for validation and earlier events for training.

Fit Popularity and ItemCF only from training positives. Both algorithms rank the complete train-item catalog and exclude the user's training positives. A test target missing from the training catalog remains a miss.

Two-Tower uses the same cohort, candidate universe and exclusion rules. The user tower aggregates training-item context embeddings, with no user-ID embedding. A training pair removes its target before retaining the most recent 50 history items. Users with a single training positive cannot form such a pair; exclude them from fitting and record the count, while retaining them in validation/test inference. Uniform sampled negatives come only from the full training catalog and reject all of that user's training positives. Validation/test labels never determine negative sampling; unobserved draws may contain latent positives.

Select the checkpoint solely by full-catalog validation NDCG@10 over all 10,000 frozen users, including unavailable targets as misses. Batch validation computes exact target ranks, using the same history encoding and all training-item vectors; item-ID order breaks score ties. Serving and headline evaluation rank the full catalog per user with training positives excluded. Dense allocations are bounded user-batch × catalog scores, never item × item. Report sampled-negative fitting separately from full-catalog evaluation.

The fixed CPU configuration uses 32-dimensional normalized vectors, eight negatives per positive, pairwise logistic loss with temperature 0.1, Adam learning rate 0.003, at most 30 epochs and four-epoch patience. Save epoch losses, full-cohort validation metrics, selected epoch, build time, model checksums and exact data/configuration/source versions. Compare Two-Tower against both Popularity and ItemCF, including losses.

- **HR@10:** mean test-target inclusion in Top10 over all sampled eligible users.
- **NDCG@10:** mean `1/log2(rank+1)` for the single target when retrieved in Top10, otherwise zero.
- **CatalogCoverage@10:** unique recommended training items divided by the training catalog size.
- Report fallback count/rate and unavailable-target count/rate with their denominators.

Ratings are preference proxies. This per-user leave-last-two-out protocol is distinct from a global time cutoff. Search and recommendation catalogs have independent identities.

## Runtime and provenance

Load the model before measurement, warm up five requests, then time sequential inference over the complete evaluation cohort. Report p50/p95 milliseconds and sample count; exclude loading, building and HTTP overhead. Report build time separately.

Every local run stores configuration, source and artifact hashes, actual split/sample counts, code hashes, available Git SHA, package versions, CPU details, per-request quality diagnostics and algorithm comparisons. Comparisons report absolute differences and relative changes only for nonzero baselines. Repeated use of a test set is labelled in configuration.

```powershell
python scripts/evaluate.py --pipeline search --search-algorithms bm25 dense hybrid --search-split validation
python scripts/evaluate.py --pipeline all --search-algorithms bm25 dense hybrid
python scripts/train_ranker.py
python scripts/evaluate.py --pipeline all --search-algorithms bm25 dense hybrid ltr
python scripts/train_two_tower.py
python scripts/evaluate.py --pipeline recommendation --recommendation-algorithms popularity itemcf two_tower --recommendation-split validation
python scripts/evaluate.py --pipeline all --search-algorithms bm25 dense hybrid ltr --recommendation-algorithms popularity itemcf two_tower
```

Generated run directories stay under the Git-ignored `reports/` directory. A complete test run updates `reports/latest.json`; validation and single-domain runs preserve that pointer.

## Allocation and cold-start diagnostics

The [allocation protocol](allocation.md) fixes B100/K10, all held-out users, the
full Amazon training catalog and seen-item exclusion. Validation selects finite
fixed/request-level policies, then the protocol signature gates test. Source-cache
verification and per-user baseline parity protect reuse of E1 channel candidates.
Target-frequency buckets and history-length groups use training information only;
all unavailable targets remain misses. Coverage uses the full training catalog
even inside a subgroup. Reports retain signed per-user differences, quota credits,
tail list share, concentration and native underfill.

Finite-arm UCB experiments use an explicitly named held-out rating-target proxy.
Each action uses only prior selected-arm feedback. Five seeded permutations retain
all users; logs permit prefix replay. Mean/sample standard deviation, losses to the
fixed policy and contextual proxy-oracle regret are diagnostic outputs. They do
not estimate CTR, business uplift, IPS/DR or real exposure effects. Allocation and
simulation runs maintain separate pointers and preserve the complete application
benchmark pointer. Actual serving timings and 1–2/3–5/6+ history-group latency are
reported by the complete recommendation evaluator.
