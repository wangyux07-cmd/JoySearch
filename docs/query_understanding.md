# Query understanding

`query_hybrid` adds bounded English query rules to the WANDS search pipeline. The
default remains BM25. Original BM25, Dense, Hybrid and LambdaMART stay available.

`configs/query.json` contains the complete finite lexicon. Its development inputs
are frozen train queries and unlabeled catalog text. Unicode NFKC, case and space
normalization retain negation, brands, numbers and units. The parser returns the
original text, normalized text, category hints, color/material/size attributes,
negation scopes, at most one rewrite, unsupported requirements and rule IDs.
Dimension expressions without units retain an unknown unit; no unit is inferred.
Price, stock and delivery requirements are reported as unsupported.

Original BM25 and Dense each retrieve Top100. A rewrite adds a BM25 channel only
when a configured alias or quote-to-inch conversion changes the query. Equal RRF
with constant 60 selects the final 100 candidates. Attribute evidence can reorder
that pool using a validation-selected small additive weight. This score is a rule
match signal, not a probability. Product features aggregate variants, so no hard
filtering is performed. Missing fields, unrecognized values and explicit lexical
contradictions have separate states. A matching variant does not guarantee that
an orderable SKU satisfies the request.

## Reproduction

```powershell
python scripts/evaluate_query.py --split validation
# Read the immutable run's selection.json; set query.json soft_weight accordingly.
python scripts/train_ranker.py
python scripts/evaluate_query.py --split test --selection reports/<validation-run>/selection.json
python scripts/verify_release.py
```

The validation matrix compares original, normalization, extraction and expansion;
the only searched parameter is the declared `validation_weights`. Selection uses
full-validation expansion NDCG@10, breaking ties toward the smaller weight. Test
requires the saved selection and verifies that the protocol has not changed. All
queries use the full catalog and final budget 100. Sparse WANDS judgments are not
complete relevance labels. `per_query.csv`, `query_deltas.csv`, `comparisons.csv`
retain gains and losses. E1 channel metrics and Parquet candidate traces measure
overlap, unique relevant contributions and budget truncation. These search-only
runs update their own pointers and preserve the complete dual-pipeline pointer.

Timing uses five warmups and shared original retrieval to avoid repeated encoder
work across weights. Variant total latency includes that shared work; added
latency measures incremental fusion/extraction/rewrite work. These are sequential
domain timings, not throughput or independent HTTP latency benchmarks.

LambdaMART retains its original Hybrid Top100 pool and 13 features; it does not
consume the query expansion pool. Query code/config hashes invalidate its model
signature conservatively. Preserve the old artifact before retraining. Unjudged
rows remain excluded from fitting, and test labels never select rules or models.

## Annotation guide

Local annotation templates sample 30 frozen train queries for development and 30
validation queries for a disjoint reserved annotation cohort using seed 42. No
reserved annotation labels are available for rule development. Annotate queries without
seeing parser predictions, rankings or qrels. Record annotator identity, timestamp,
source span, canonical value and support status for product type, color, material,
size, negation and unsupported constraints. Keep brand and numeric spans intact;
do not infer absent units or product attributes. Multiple explicit values retain
their conjunction/disjunction and negative scope. Unrecognized concepts stay
unrecognized. Missing catalog fields are distinct from an explicit contradictory
field. Adjudicate independently labeled disagreements before scoring the reserved
cohort; do not edit rules using that cohort.

Exact slot-value precision/recall/F1 use independently adjudicated gold slots;
negation preservation compares scope and polarity, and unsupported-requirement
accuracy compares explicit price/stock/delivery annotations. Until human labels
exist, all these scores are null / `not_evaluable`, with zero actual annotators
and labeled samples. Generated challenges in `tests/test_query.py` verify software
behavior separately and supply no claim about human annotation accuracy.
