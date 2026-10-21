# LambdaMART validation diagnostics

Run the bounded diagnostic with the existing search indexes, ranker and candidate caches:

```powershell
conda activate joysearch
python scripts/diagnose_ltr.py
```

The command verifies input checksums and frozen query splits, reuses the cached Hybrid
Top100 features, and fits four fixed feature configurations on judged training rows.
It never reads test judgments or overwrites the serving model. Partial relevance remains
positive; unjudged products are excluded from fitting and retained in evaluation rankings.

The experiment uses the existing 300-round LightGBM configuration and a declared checkpoint
grid. It replays the original 30-round judged-only early stopping rule and compares it with
full-pool validation selection. Equal model scores use product ID order. All 96 validation
queries remain in aggregate metrics, including the query with no judged retrieved products.
Conventional NDCG gives unjudged results zero gain and computes IDCG from the full query qrels.

Measured validation results on WANDS (42,994 products):

| Configuration | Selected iteration | NDCG@10 | Difference from Hybrid |
|---|---:|---:|---:|
| Hybrid | — | 0.737563 | — |
| Existing LambdaMART | 29 | 0.698624 | -0.038939 |
| All features, full-pool checkpoint selection | 29 | 0.698624 | -0.038939 |
| Without rating/review features | 5 | 0.697156 | -0.040407 |
| Without text overlap/length features | 29 | 0.734749 | -0.002815 |
| Retrieval features only | 41 | 0.734500 | -0.003064 |

The full-feature replay exactly reproduces the preserved model's validation predictions.
Judged-only early stopping reports 0.854463 across 95 queries; this is a different metric
population and ranking universe from full-pool evaluation, so the two scores are not an
uplift comparison. For the full-feature model, switching checkpoint selection alone does
not change the chosen iteration on the declared grid. Other feature groups select different
checkpoints, so selection and feature effects should be assessed separately.

Only 68.0625% of validation candidates have judgments. Hybrid Top10 judgment coverage is
89.4792%, compared with 81.25% for the existing ranker. The text-feature ablation improves
measured NDCG and restores coverage to 89.1667%, but neither proves that unjudged products
are irrelevant. This remains an incomplete-qrels evaluation.

A separate Top50 truncation probe checks candidate-pool sensitivity, with a matching Top50
Hybrid comparator. It reduces the existing ranker's judged recall from 0.410767 to 0.278619;
its NDCG is 0.700929. This changed-budget probe is not a Top100 improvement.

No diagnostic variant exceeds Hybrid on validation NDCG. Serving models and defaults are
preserved, and no additional test-based selection is performed. The feature grid is a bounded
diagnosis, not an exhaustive hyperparameter search or proof of a unique cause.

Each local run writes its protocol, data/cache/source hashes, package versions, fitted model
checksums, judged-only learning curves, per-query metrics, summary and signed comparison CSV
under `reports/run_ltr_diagnostics_*`. Wall time is measured; CPU time, peak RSS and serving
latency are explicitly unmeasured. The dedicated latest pointer leaves the complete
Search/Recommendation report pointer unchanged. Reports, caches and fitted models remain local.
