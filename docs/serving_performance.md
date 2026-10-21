# Serving performance

JoySearch measures model-internal stages separately from warm HTTP requests.
The benchmark uses the existing WANDS and Amazon catalogs and model artifacts.
It does not rebuild indexes or train models.

## Stage traces

Send `debug: true` to Search or Recommendation to receive `timings` alongside the
existing `trace_id`. Spans contain a stage, parent/span ID, start offset, elapsed
milliseconds and returned count when applicable. They cover query parsing,
BM25, encoding, exact scan/sort, fusion, attribute/LTR sorting, recommendation
channels, seen filtering and allocation. Nested times are inclusive; adding a
parent and its children double-counts elapsed time. HTTP client timing also includes
queueing, response construction, transport, JSON decoding and fingerprint validation.

Trace state uses a request-local context and resets on errors. Normal requests
omit timing arrays. This instrumentation lives outside model training and the
existing retrieval/configuration signatures.

## Exact scan experiment

The baseline uses the existing normalized 384-dimensional vectors, full-catalog
exact cosine and deterministic item-ID tie breaking. Profiling 24 training queries
found repeated Windows library discovery inside `threadpool_limits` to be a
substantial part of scan time. `CachedControllerDenseIndex` discovers loaded BLAS
libraries once and reuses that controller. Vectors, multiplication, full sorting
and candidate construction remain the same.

```powershell
python scripts/benchmark_local.py
```

The local benchmark compares three alternating paired repeats on all 96 validation
queries, records validation before test, then checks all 96 test queries. Both
implementations receive the same vector and Top100 budget. Every product ID and
score must agree exactly. Reports include dense NDCG/JudgedRecall, per-query scan
cost, real pipeline stage traces, artifact size, CSC structure, load duration and
Windows process resources. The existing test split is reused for regression.
Stage traces cover 96 validation queries and 96 fixed training-history users;
they do not create additional independent evaluation subjects.

The experiment remains an explicit server option. It disables LTR in that isolated
server; it does not replace the default retriever or reinterpret an older ranker's
signature. Existing BM25/CSC, exact embeddings and learned weights are retained.
ANN is deferred because the measured issue is library-discovery overhead.

The paired test run covered 96 queries × three repeats per variant over 42,994
products. Every Top100 ID and score agreed exactly. Scan/sort p50/p95 changed from
35.00/42.27 ms to 17.69/21.68 ms; encoding and HTTP were excluded from these figures.
Both variants retained dense NDCG@10 0.645170 and JudgedRecall@100 0.368893.
The existing index was reused, with 66,038,784 vector bytes. The BM25 CSC has
6,545,262 nonzeros and 78,791,300 bytes across its three arrays; it was retained.

## Controlled HTTP load

Start an isolated server, leaving the interactive application on port 8000:

```powershell
python scripts/serve_benchmark.py --port 8001 --scan-mode reference
```

After readiness succeeds, use another terminal:

```powershell
python scripts/benchmark_http.py --pilot
python scripts/benchmark_http.py
```

Stop that benchmark server and repeat with `--scan-mode cached_controller` and the
same full HTTP command. Run cells and modes sequentially to avoid competing model
loads and evaluations. Configuration is in `configs/benchmark.json`.

Each server has one process, four request workers and one intra-op/BLAS thread.
Closed-loop clients use concurrency 1/2/4/8 against the same worker budget. Each
workload has 96 real query IDs or user aliases; each cell repeats those cases to
384 requests, with two seeded repetitions. Search uses Hybrid, Recommendation
uses Adaptive. Repeating requests does not increase the number of distinct queries
or users. Response caching is off; models and OS pages are warm after preflight
and explicit warmups. Readiness reports model-loading duration, excluding
interpreter/module imports before application lifespan begins; it is not total
process launch time.

Raw request rows retain status, timeout/transport/contract errors, degradation,
empty results, latency, result fingerprint and `trace_id`. A successful response
must match its sequential preflight ranking and scores. Empty valid responses
are counted separately from HTTP errors. QPS uses measured wall time, not the
reciprocal of one request's latency. Offered attempts, HTTP completions and valid
successes share the cell-dispatch to last-completion window. Dispatch offsets are
retained in request rows, so the small setup interval before the first send is
visible. This closed-loop offered
rate is observed admission, not a configured open-loop arrival rate.

All-attempt p50/p95 include failures and timeouts. Per-cell p99 is null below the
predeclared 1,000-sample floor. CPU seconds come from the server process; current
RSS is sampled every 200 ms and at cell boundaries. The OS-reported lifetime peak
is labeled separately because it can include startup and earlier cells. Results
report both repetitions rather than implying statistical significance.
The reference mode runs before the cached-controller mode; machine drift between
those sequential runs remains a limitation of this local comparison.

Reports stay in independent `reports/run_e5_*` directories and preserve the full
dual-pipeline benchmark pointer. Each stores actual cohorts, runtime settings and
source hashes. The pilot is bounded to 90 seconds after server readiness; full
load has a 900-second admission budget checked between cells. An active cell is
allowed to finish, with configured 10-second HTTP timeouts; this is not a hard
whole-process deadline. An incomplete run retains partial logs and publishes
no completed pointer.

## HTTP observations

The measured sequence was reference → cached controller → reference again. Each
run completed 6,144 measured load requests over 16 cells; all 18,432 measured
responses passed the ranking
fingerprint check, with zero HTTP/contract errors, timeouts or degradation. The
three runs produced identical per-case rankings and scores. Counts exclude
preflight, explicit warmups and the separate stage-trace probes.

Ranges below show the two repetitions, rather than confidence intervals:

| Run | Search C1 success QPS | Search C4 success QPS | Search C4 p95, ms | Unmodified recommendation C4 success QPS |
|---|---:|---:|---:|---:|
| Reference before | 12.10–12.15 | 23.15–24.55 | 205.43–210.51 | 59.13–67.77 |
| Cached controller | 37.32–46.85 | 77.96–87.02 | 56.84–63.46 | 140.88–154.78 |
| Reference after | 30.23–30.60 | 58.90–59.25 | 77.88–78.26 | 138.27–141.57 |

The unchanged recommendation control moved substantially between the initial and
later runs. Consequently the between-run HTTP ratios cannot establish a causal
controller speedup. The paired same-vector scan experiment provides the narrower
evidence for that optimization. Concurrency 8 also increased tail latency and did
not consistently improve search throughput over concurrency 4. The default service
retains the reference implementation; the optimized variant remains explicit.
All per-cell p99 values are null under the fixed sample floor.
