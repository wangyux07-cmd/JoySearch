# Generative retrieval

JoySearch includes an offline neural ID-generation experiment over the complete
WANDS catalog. A frozen MiniLM encodes the query; a 40,562-parameter GRU predicts
a 16-bit semantic address autoregressively. Catalog mapping returns real product
IDs. This is a small supervised generator, rather than an LLM rewrite or an item
embedding scan.

## Train and evaluate

Use the existing environment and previously built semantic index:

```powershell
python scripts/train_generative.py --pilot
python scripts/train_generative.py
python scripts/evaluate_generative.py --split validation
python scripts/evaluate_generative.py --split test --selection reports/<validation-run>/selection.json
python scripts/generate_ids.py --query "salon chair" --top-k 10
```

The input builder reuses existing catalog vectors and independently encodes all
query texts once. A completed matching cache is verified and reused. It refuses
changed inputs; the trainer also refuses to replace an existing published model.
Preserve prior artifacts before intentionally training a new version. Generation,
evaluation and training remain separate from HTTP handlers and the default search
pipeline.

## Catalog addresses and decoding

At each node, split catalog vectors by their largest-variance coordinate, using
item-ID order to break coordinate ties. Recursively divide at the median. Pad
singleton addresses with zero bits to the fixed depth. All 42,994 products receive
unique addresses in the 65,536-address space. This balanced partition uses only
unlabeled catalog embeddings; it is not a learned quantizer or a qrel-derived
candidate whitelist.

The network projects a 384-dimensional query vector into a 64-dimensional initial
state. A 16-dimensional token embedding and GRU predict each next bit. Teacher
forcing minimizes next-bit cross entropy. Beam search retains at most 100 prefixes
at each of 16 steps and accumulates neural log probabilities. It records expanded
branches and maps final addresses back to products. The legal-prefix variant masks
missing catalog branches after binary log-softmax, without renormalizing remaining
branches. The free variant retains unmapped addresses as explicit failures.

Prefix constraints alone do not establish generative retrieval: the trained
network supplies the token probabilities and determines beam order. Legal mapping
also does not establish relevance. Scores are sequence log probabilities, not
click/purchase likelihoods or scores directly comparable with BM25.

## Data and checkpoint policy

The 288 frozen training query IDs contain 102,242 grade>=1 positive pairs from
287 queries. Sampling selects at most 64 pairs per query: up to half the cap per
grade 2 and grade 1, then fills from the remaining positives without replacement.
Seed 42 selects 17,450 pairs. Unjudged pairs and grade-0 pairs are not supplied as
negative training labels. The encoder and catalog vectors remain frozen.

Adam uses learning rate 0.003, batch size 256, at most 12 epochs and three-epoch
patience. Every checkpoint is evaluated over all 96 validation queries with the
complete catalog and beam100; validation NDCG@10 selects the earliest best epoch.
Test judgments never choose the mapping, training pairs, epoch or fusion settings.
The training budget is checked cooperatively between batches/validation. The pilot
finished in 9.65 seconds; full fitting and validation took about 15 seconds.

The evaluator compares BM25, exact Dense, Hybrid, free generation, constrained
generation and a fixed equal-weight three-channel RRF. Every variant retains the
same query population and full catalog, with at most 100 returned candidates.
Original BM25/Dense channels remain in the fusion; the older LTR is not applied to
an expanded candidate pool. No fusion parameters are tuned on test.

## Reports and scope

Separate run directories retain source/configuration/model hashes, actual training
pairs, epoch history, generated addresses including failures, ranked products,
per-query metrics, signed comparisons and branch counts. The validation report
gates test by source and model checksums. These runs preserve the full dual-pipeline
report pointer.

Decode and scan timings share frozen cached query vectors and exclude text
encoding, model loading and HTTP. They are not end-to-end latency or measured
throughput. The text CLI performs actual local query encoding before generation.
Training wall time is recorded; training process CPU/peak RSS were not sampled.
Evaluation records its own measured CPU and process resources separately.

The model remains an offline experiment. Validation NDCG@10 was 0.100640 against
Hybrid's 0.737563; equal-weight fusion decreased NDCG to 0.720563. Both standalone
generation and fusion therefore retain their losses in the report rather than
replacing the existing search default.

The fixed test evaluation over 96 queries gave the following results:

| Variant | NDCG@10 | JudgedRecall@100 |
|---|---:|---:|
| BM25 | 0.624031 | 0.373704 |
| Dense | 0.645170 | 0.368893 |
| Hybrid | 0.687759 | 0.392750 |
| Free generation | 0.035279 | 0.016022 |
| Constrained generation | 0.035279 | 0.016022 |
| Hybrid + generation | 0.673635 | 0.338471 |

Both generators mapped all 9,600 test sequences to legal IDs; the free generator
had six unmapped sequences on validation. Constrained decode p50/p95 was
4.38/5.35 ms using cached query vectors. The fusion lost 0.014124 NDCG and 0.054279
judged recall against Hybrid at the same budget. These results support keeping the
prototype offline. Legal ID generation alone provides no evidence of a relevance
advantage.
