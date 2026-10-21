import math

import pytest

from joysearch.evaluation.metrics import (
    hit_rate,
    judged_recall,
    judgment_coverage,
    ndcg,
    reciprocal_rank,
)


def test_hand_computed_metrics():
    ranking = ["unknown", "b", "a"]
    qrels = {"a": 2, "b": 1, "irrelevant": 0}
    expected = (1 / math.log2(3) + 3 / math.log2(4)) / (3 + 1 / math.log2(3))
    assert ndcg(ranking, qrels) == pytest.approx(expected)
    assert reciprocal_rank(ranking, qrels) == 0.5
    assert judged_recall(ranking, qrels) == 1
    assert judgment_coverage(ranking, qrels) == pytest.approx(2 / 3)
    assert hit_rate(ranking, "b") == 1
    assert ndcg(ranking, {"b": 1}) == pytest.approx(1 / math.log2(3))
    assert ndcg([], {}) == reciprocal_rank([], {}) == judged_recall([], {}) == 0
    assert judgment_coverage([], qrels) == hit_rate([], "b") == 0
