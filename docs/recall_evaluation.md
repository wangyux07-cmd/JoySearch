# Recall diagnostics

Run candidate diagnostics independently from the serving benchmark:

```powershell
python scripts/evaluate_recall.py --split validation
python scripts/evaluate_recall.py --split test
```

The primary candidate budget is **B=100**, with final lists measured at **K=10**. The configuration in `configs/recall.json` fixes one diagnostic matrix before evaluation. All nonempty subsets of BM25/Dense or Popularity/native ItemCF/Two-Tower are compared. Multi-channel lists use equal-weight RRF with offset 60 and stable item-ID ties. These recommendation fusions are offline diagnostics; the production default remains ItemCF.

## Candidate stages

Search retrieves Top100 from each channel. Recommendation retrieves raw TopD with `D=100+train_history_size`, removes all training-seen items and retains each channel's first 100 eligible items. This deterministic overfetch preserves each existing full-catalog Top100 ranking after filtering. Native ItemCF retains **strictly positive collaborative scores** and may return fewer than 100 candidates. It never includes Popularity completion. `itemcf_served` reconstructs and separately evaluates the existing positive prefix plus train-only Popularity completion; it is excluded from native-channel contribution accounting.

Merge identity is `(catalog_id,item_id)`; mixed catalogs are rejected at the evaluation boundary. Every raw candidate retains its channel, source, original rank and score. Merging preserves every contributing channel. Fusion uses eligible channel ranks rather than summing incompatible scores.

| Stage | Persisted representation |
|---|---|
| Per-channel raw TopD | `<domain>_channels.parquet`, including original rank/score and seen exclusion |
| Per-channel filtered TopB | Non-null `eligible_rank` in the same table |
| Deduplicated raw union U | One row per identity in `<domain>_candidate_stages.parquet` |
| Union after seen filtering | `in_filtered_union` |
| Union of channel TopB inputs | `in_fusion_union`, with eligible source ranks |
| Final B for every subset | Aligned `selected_variants` and `selected_ranks` lists |
| Final Top10 | `top10_variants`, or selected rank <=10 |

Aligned source/rank/score lists avoid duplicating whole candidate tables for each stage. Parquet writers flush in bounded request batches. Recommendation request identifiers are anonymous aliases only. `<domain>_channel_requests.csv` records requested depth, actual channel counts and inference time. Per-request metric CSVs record deduplication, seen filtering, per-channel truncation and final-budget losses. Missing candidates remain shortages rather than invented entries.

## Metrics and ablation

- Search CandidateJudgedRecall@100 is the fraction of that query's judged grade>=1 products in the final candidate set, macro-averaged over the full frozen split. A query without judged positives contributes zero. Final NDCG@10, MRR@10 and JudgmentCoverage@10 use the existing formulas.
- Recommendation CandidateHit@100 is target inclusion among the final candidates. HR@10 and NDCG@10 use the same target and all users. With one target per user, same-K Recall and HR are identical. `top10_given_top100_hit_rate` is total Top10 hits / total Top100 hits, or null when there are no candidate hits.
- CatalogCoverage@10 counts distinct training items in final lists / the entire training catalog. Popularity completion is counted as fallback only for the served ItemCF diagnostic; including Popularity as an explicit fusion channel is a strategy, not a hidden fallback.
- Jaccard is intersection size / union size of eligible channel TopB sets; two empty sets yield zero and are counted. Raw TopD overlap is separate.
- A channel's unique relevant products occur in its set but no other channel. Report raw and eligible unique counts separately. The unique recall fraction uses all query positives as denominator; share of relevant union uses relevant union size, with null for an empty relevant union and an explicit denominator count.
- Remove each channel from the full fusion, then apply the same RRF rule to all remaining eligible candidates up to B. Save `without - full` recall/hit differences and actual candidate counts. Unique raw contribution and fixed-budget incremental gain are distinct: a new candidate can displace another candidate.

Raw and filtered unions can exceed 100 items. Their recall is diagnostic and cannot establish a fixed-budget improvement. WANDS judgments are sparse; unjudged products are not verified irrelevant. Amazon ratings are preference proxies. All 2,023 test targets outside the training catalog remain misses in the 10,000-user headline cohort.

## Outputs and provenance

Each run creates a separate local `reports/run_e1_<UTC>_<id>/` with `recall_metrics.json`, `channel_ablation.csv`, `channel_overlap.csv`, `channel_contributions.csv`, `comparisons.csv`, per-request tables and candidate traces. Configuration, split/sample sizes, source/model/processed-data hashes and code fingerprints accompany the run. Channel inference latency excludes metric computation and trace IO; overall wall/CPU time and Parquet bytes describe diagnostic execution cost.

Validation runs are retained. Test runs are labelled fixed-test regression; they are not new untouched generalization evidence. `--pilot-limit N` is explicitly a pilot, uses actual sample denominators and never updates the latest diagnostic pointer. A complete test diagnostic updates `reports/recall_latest.json` and preserves the full application benchmark pointer `reports/latest.json`.

Reports, traces, model artifacts and work instructions stay outside Git. Reuse the prepared datasets and existing models; this diagnostic stage does not train or rebuild indexes.
