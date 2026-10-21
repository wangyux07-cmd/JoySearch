# Recommendation allocation and offline policy simulation

The allocation layer combines train-only Popularity, positive-score native ItemCF
and history-based Two-Tower candidates. Their item IDs belong to the same Amazon
training catalog. Search metadata is never joined to this catalog.

## Candidate policy

Each channel retrieves `B + seen_count`, excludes seen items, deduplicates and
retains Top100. The original served ItemCF baseline includes explicit Popularity
completion; native ItemCF is also evaluated separately. Equal-weight RRF is the
fusion baseline.

Three fixed quota vectors reserve 100 unique slots in channel order Popularity,
native ItemCF, Two-Tower: `(80,0,20)`, `(60,10,30)`, `(40,20,40)`. Smooth weighted
round robin requests slots in proportion to those integers. A duplicate advances
that channel's cursor to its next unique candidate. An exhausted channel leaves
a slot for reallocation from the remaining union. Refill and final ranking use
`sum((quota_c / 100) / (60 + rank_c))`, with item ID breaking ties. Original
channel scores are retained as evidence and never added across models. Allocation
credit records the channel that filled each slot; multi-channel presence remains
separate provenance.

Two finite adaptive rules use only request-visible evidence. `history_support`
uses `(80,0,20)` when train history has at most two items or native support is
below 20, otherwise `(60,10,30)`. `support_only` uses `(80,0,20)` when native
support is below 100, otherwise `(40,20,40)`. These are hand-written rules, not a
trained gate. Validation selects the fixed vector and adaptive rule separately
using full-cohort NDCG@10, then HR@10, then declaration order. All attempts remain
in the report. Models, data, code and protocol signatures must match before test.

This changes downstream allocation after all channels have retrieved candidates;
it does not reduce channel retrieval computation. Default ItemCF remains available.
Cold users receive train-only Popularity with `cold_user`. No metadata, seller or
arrival timestamp is available locally, so frequency buckets are not new-item
cold-start claims. Out-of-training-catalog targets remain misses.

## Evaluation and reuse

```powershell
python scripts/evaluate_recall.py --split validation
python scripts/evaluate_recall.py --split test
python scripts/evaluate_allocation.py --split validation --cache reports/<E1-validation-run>
python scripts/evaluate_allocation.py --split test --cache reports/<E1-test-run> --selection reports/<E3-validation-run>/selection.json
python scripts/verify_release.py
```

The evaluator verifies E1 model artifacts and channel inference hashes, exact
training histories/counts, live/cache parity and per-user baseline reproduction.
It reuses cached candidates instead of repeating expensive retrieval. Local runs
contain fixed-budget rankings with allocation credits, per-user metrics and signed
deltas, comparison CSVs, quota contributions and slices. Coverage always divides
by the full training catalog, including when displaying a subgroup. Cache-only
allocation timings exclude retrieval and I/O. The full serving evaluator separately
measures actual inference and history-group latency.

Train history groups are 1–2, 3–5 and 6+ items. Train-frequency groups are tail
(`<=2`), mid (`3–9`), head (`>=10`) and unavailable targets. TailListShare is the
fraction of Top10 list appearances belonging to tail items. List concentration
is the sum of squared item appearance shares. These are offline lists, not real
impressions. Sparse native ItemCF can underfill; quota shortfalls and reallocations
are counted separately from the existing served fallback.

## Finite-arm simulation

The three fixed quota vectors are arms. Potential reward is whether an arm's
Top10 contains the held-out positive-rating target. Fixed uses the validation
winner; fixed exploration chooses a random arm with probability 0.1; UCB1 first
tries each arm then maximizes `mean_reward + sqrt(2*log(total_pulls)/arm_pulls)`.
Only the chosen arm's reward updates state. Each of five declared seeds permutes
the same request population and starts fresh policy state. Logs retain request
order, anonymous user alias, action, observed proxy reward and previous selected-arm
statistics. No user target is available to the action selector.

The contextual proxy oracle takes the maximum potential reward for each request.
Its cumulative reward minus the policy reward is a diagnostic proxy regret;
it is not stationary-arm regret or a causal business estimate. Full potential
outcomes are available only because this is an offline rating-target experiment.
It is neither exposure-log replay nor IPS/DR evaluation. Summary tables preserve
mean/sample-standard-deviation, fixed-policy differences, coverage, tail share
and concentration even when exploration loses.

`rrf`, `quota` and `adaptive_quota` are optional recommendation API algorithms.
The latter two become ready only after complete validation publishes a matching
selection artifact. Debug results expose source ranks and allocation credit. The
Traffic Lab reads saved test reports; opening the page does not run a simulation.
