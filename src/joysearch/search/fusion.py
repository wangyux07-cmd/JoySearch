from joysearch.common import Candidate


def reciprocal_rank_fusion(rankings, k, rrf_k=60):
    """Equal-weight RRF preserves channel ranks instead of summing incompatible scores."""
    scores, evidence = {}, {}
    for channel, candidates in rankings.items():
        for rank, candidate in enumerate(candidates, 1):
            if candidate.catalog_id != "wands_search":
                raise ValueError("Search fusion accepts only WANDS candidates")
            item = candidate.item_id
            scores[item] = scores.get(item, 0.0) + 1 / (rrf_k + rank)
            evidence.setdefault(item, {})[channel] = rank
    ordered = sorted(scores, key=lambda item: (-scores[item], item))[:k]
    return [
        Candidate(
            "wands_search",
            item,
            scores[item],
            "hybrid",
            rank,
            {"ranks": evidence[item], "sources": sorted(evidence[item])},
        )
        for rank, item in enumerate(ordered, 1)
    ]
