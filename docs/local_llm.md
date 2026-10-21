# Local LLM diagnostics

JoySearch provides an offline CPU workflow for evidence-bound query rewriting
and relevance labeling. [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) runs locally with a pinned revision,
greedy decoding and four CPU threads. The existing Search API continues to use
its selected retrieval algorithm.

## Run

Use the existing Python environment with the optional semantic-search dependencies
installed. The first preparation downloads the pinned model snapshot (about 1 GB
of weights); subsequent inference uses local files only.

```powershell
python scripts/prepare_llm.py
python scripts/evaluate_llm.py --split validation
python scripts/evaluate_llm.py --split test --selection reports/<validation-run>/selection.json
```

Read `reports/llm_validation_latest.json` to find the validation run. Settings are
in `configs/llm.json`. Raw responses are cached under `artifacts/llm_cache/` by the
complete request, decoding limits, model checksums, runtime versions and generation
engine hash. Repeating an interrupted command reuses exact responses. Each attempt
creates a new report directory; exceeding the 1,200-second run budget leaves a
partial report and publishes no completed pointer. The 12-second generation limit
is cooperative between decoding steps, so calls can finish slightly beyond it.
Such responses are rejected explicitly.

## Query rewriting

Four variants share the same seeded 24-query diagnostic cohort and full WANDS
catalog: original Hybrid, E2 rules, LLM without documents and LLM with the original
Hybrid's first three products. This is an additional diagnostic beside the full
96-query application benchmark. Every variant retains the original BM25 and Dense
channels and fuses at most 100 candidates. One accepted, distinct rewrite can add
a BM25 channel. No old LambdaMART model is applied to the expanded candidate pool.

The model edits an E2-derived JSON template. Category, attribute, negation and
unsupported-constraint fields describe the query; product evidence does not grant
permission to invent user preferences. Documents contain clipped real titles,
classes, descriptions and features, and are treated as untrusted data. Class terms
added to a rewrite must come from a cited product. The deterministic guard checks
strict JSON, known IDs, content and brand tokens, numerical multisets, value-unit
pairs, currency, negation clauses and bounded attributes. Invalid, unsupported or
late outputs fall back to E2 rules and remain in the evaluation population.

`per_query.csv`, `comparison.csv` and `rankings.parquet` retain quality, signed
differences and actual ranks. JSON validity, typed validity, constraint preservation,
unsupported facts and fallback rates are separate. Constraint and fact checks use
typed, assessable raw proposals; their denominator is explicit and the value is
null when none qualify. These literal guards do not measure semantic intent
agreement. Human quality remains null until independent annotations exist.

## Relevance labels

Within the first 12 diagnostic queries, sample one available pair per WANDS grade
0/1/2. The query and real product fields go to the model; the gold grade stays in
the evaluator. Train-pilot, validation and test query IDs are disjoint. This
stratified sample is a diagnostic distribution rather than an estimate of natural
class prevalence. The baseline is the majority grade from training qrels only.

Accepted labels require a grade, a literal product-field quote and a matching
product ID. Invalid outputs, uncertainty and null grades abstain. The 3×4 confusion
matrix includes an abstention column. Macro-F1 counts those abstentions as false
negatives over the full sample; quadratic weighted kappa uses accepted pairs only
and reports that smaller denominator. Degenerate kappa is null. The report records
the refusal review queue and zero completed manual reviews until people review it.
Generated labels never replace WANDS qrels or train another model in this workflow.

## Cost and deployment decision

The train pilot generated eight responses: four rewrites and four available
grade-stratified labels across two queries. The frozen second prompt passed no
rewrite JSON checks and three label schema/evidence checks. Individual generations
took roughly 2.4–12 seconds on this machine, above the 1-second realtime budget;
schema validity alone does not establish label correctness.

Reports retain original generation token counts, wall time, CPU seconds, cache
hits, model load time and Windows process peak working set. Replayed cache hits
retain their original cost for comparison and are excluded from newly consumed
generation CPU. Sequential per-variant cost sums exclude loading but retain the
first cold generation; they are separate from warmed application timings and HTTP
throughput. The validation gate freezes one configuration for test diagnostics,
without promoting it to realtime serving. Source or asset changes invalidate the
gate. Both diagnostic pointers preserve the complete dual-pipeline report pointer.

## Measured held-out diagnostic

The frozen configuration evaluated 24 validation queries and 24 test queries,
with 31 and 29 grade-stratified label pairs respectively. Each ranking variant used
the same full 42,994-item catalog and B100/K10.

| Test result | Rules | LLM without documents | LLM with retrieval evidence |
|---|---:|---:|---:|
| NDCG@10 | 0.615267 | 0.615267 | 0.615267 |
| JudgedRecall@100 | 0.376341 | 0.376341 | 0.376341 |
| Syntactically valid rewrite JSON | n/a | 7/24 | 0/24 |
| Typed, schema-valid rewrites | n/a | 0/24 | 0/24 |
| Rule fallback | n/a | 24/24 | 24/24 |
| Sequential variant latency p95, ms | 58.40 | 6085.65 | 12060.88 |

The unchanged quality comes from explicit rule fallback; it is not a model gain.
Constraint preservation and unsupported-fact rates are null because no typed
rewrite was assessable. Human intent annotations remain unavailable. Test label
macro-F1 was 0.148148 over all 29 pairs, below the train-majority baseline's
0.195122. The model accepted 8 pairs and abstained on 21; accepted-only quadratic
weighted kappa was 0.0. No manual reviews were completed.

The test run generated 77 responses, consumed 33,039 input and 4,187 output tokens,
and took 434.74 seconds wall time and 1668.61 process CPU seconds. Process peak
working set was 3,415,678,976 bytes. These measurements support an offline-only
decision for this configuration; neither rewriting nor pseudo-labels are promoted
to serving or training. Validation preceded test, and no test-driven prompt tuning
was performed. The original validation run is retained beside a cache-only replay
that corrected the latency protocol description without changing outputs.
