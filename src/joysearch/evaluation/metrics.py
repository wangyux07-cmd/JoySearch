import math


def ndcg(ranking, qrels, k=10):
    def dcg(grades):
        return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))

    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    actual = dcg([qrels.get(item, 0) for item in ranking[:k]])
    return actual / ideal if ideal else 0.0


def reciprocal_rank(ranking, qrels, k=10):
    return next(
        (1 / rank for rank, item in enumerate(ranking[:k], 1) if qrels.get(item, 0) >= 1), 0.0
    )


def judged_recall(ranking, qrels, k=100):
    relevant = {item for item, grade in qrels.items() if grade >= 1}
    return len(relevant.intersection(ranking[:k])) / len(relevant) if relevant else 0.0


def judgment_coverage(ranking, qrels, k=10):
    top = ranking[:k]
    return sum(item in qrels for item in top) / len(top) if top else 0.0


def hit_rate(ranking, target, k=10):
    return float(target in ranking[:k])


def jaccard(left, right):
    left, right = set(left), set(right)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def conditional_hit_rate(top_k_hits, candidate_hits):
    return top_k_hits / candidate_hits if candidate_hits else None
