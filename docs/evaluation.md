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
```

Generated run directories stay under the Git-ignored `reports/` directory. A complete test run updates `reports/latest.json`; validation and single-domain runs preserve that pointer.
