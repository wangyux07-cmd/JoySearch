"""Rank-based candidate allocation; scores are not preference probabilities."""

from dataclasses import replace

from joysearch.common import Candidate
from joysearch.data.wands import file_hash

CHANNELS = ("popularity", "itemcf_native", "two_tower")


def allocation_signature(root):
    paths = [
        "configs/allocation.json",
        *[
            f"src/joysearch/recommendation/{name}.py"
            for name in ["pipeline", "fusion", "popularity", "itemcf", "two_tower"]
        ],
        *[
            f"artifacts/recommendation/{name}"
            for name in ["manifest.json", "model.json", "similarity.npz"]
        ],
        "artifacts/recommendation_two_tower/manifest.json",
        "artifacts/recommendation_two_tower/model.pt",
    ]
    return {p: file_hash(root / p) for p in paths}


def filter_channels(raw, seen, budget):
    result = {}
    for channel in CHANNELS:
        found, rows = set(), []
        for c in raw[channel]:
            if c.catalog_id != "amazon_baby_rec":
                raise ValueError(f"Allocation received a foreign catalog from {channel}")
            if c.item_id not in found and c.item_id not in seen:
                rows.append(replace(c, rank=len(rows) + 1))
            found.add(c.item_id)
        result[channel] = rows[:budget]
    return result


def adaptive_quotas(config, rule_name, history_length, native_count):
    rule = config["adaptive_rules"][rule_name]
    sparse = (
        history_length <= rule["short_history_max"] or native_count < rule["native_support_min"]
    )
    name = rule["sparse"] if sparse else rule["supported"]
    return dict(zip(CHANNELS, config["fixed_quotas"][name])), name


def ranked_union(rankings, weights, rrf_k):
    scores, evidence = {}, {}
    for channel in CHANNELS:
        for rank, c in enumerate(rankings[channel], 1):
            if c.catalog_id != "amazon_baby_rec":
                raise ValueError("Recommendation fusion requires the Amazon catalog")
            scores[c.item_id] = scores.get(c.item_id, 0) + weights[channel] / (rrf_k + rank)
            evidence.setdefault(c.item_id, {})[channel] = {"rank": rank, "score": c.score}
    order = sorted(scores, key=lambda item: (-scores[item], item))
    return [
        Candidate(
            "amazon_baby_rec",
            item,
            scores[item],
            "fusion",
            rank,
            {"sources": list(evidence[item]), "channels": evidence[item]},
        )
        for rank, item in enumerate(order, 1)
    ]


def allocate(rankings, quotas=None, budget=100, rrf_k=60):
    """Reserve unique channel slots, refill exhausted slots, then rank the selected pool."""
    if quotas is None:
        rows = ranked_union(rankings, dict.fromkeys(CHANNELS, 1), rrf_k)[:budget]
        return rows, {"method": "equal_rrf", "budget": budget, "selected_count": len(rows)}
    if (
        set(quotas) != set(CHANNELS)
        or any(type(v) is not int or v < 0 for v in quotas.values())
        or sum(quotas.values()) != budget
    ):
        raise ValueError(
            "Quotas must contain nonnegative integer slots for all channels summing to budget"
        )
    ranked = ranked_union(rankings, {c: quotas[c] / budget for c in CHANNELS}, rrf_k)
    selected, credited, current = set(), {}, dict.fromkeys(CHANNELS, 0)
    pointers, filled = dict.fromkeys(CHANNELS, 0), dict.fromkeys(CHANNELS, 0)
    duplicate_skips, exhausted_slots = 0, 0
    for _ in range(budget):
        for channel in CHANNELS:
            current[channel] += quotas[channel]
        channel = max(CHANNELS, key=lambda c: current[c])
        current[channel] -= budget
        candidates = rankings[channel]
        while (
            pointers[channel] < len(candidates)
            and candidates[pointers[channel]].item_id in selected
        ):
            pointers[channel] += 1
            duplicate_skips += 1
        if pointers[channel] == len(candidates):
            exhausted_slots += 1
            continue
        item = candidates[pointers[channel]].item_id
        pointers[channel] += 1
        selected.add(item)
        credited[item] = channel
        filled[channel] += 1
    refilled = 0
    for c in ranked:
        if len(selected) == budget:
            break
        if c.item_id not in selected:
            selected.add(c.item_id)
            credited[c.item_id] = "reallocated"
            refilled += 1
    result = [
        replace(c, rank=i, evidence={**c.evidence, "quota_credit": credited[c.item_id]})
        for i, c in enumerate((c for c in ranked if c.item_id in selected), 1)
    ]
    return result, {
        "method": "quota_weighted_rrf",
        "budget": budget,
        "requested_quotas": quotas,
        "filled_quotas": filled,
        "duplicate_skips": duplicate_skips,
        "exhausted_slots": exhausted_slots,
        "reallocated_slots": refilled,
        "selected_count": len(result),
    }
