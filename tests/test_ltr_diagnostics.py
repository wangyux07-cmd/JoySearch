import numpy as np
import pandas as pd

from joysearch.evaluation.ltr_diagnostics import score_pool


def test_full_pool_retains_unknowns_and_full_qrel_idcg():
    pool = pd.DataFrame(
        {
            "query_id": ["q"] * 11,
            "product_id": [f"u{i:02}" for i in range(10)] + ["relevant"],
            "grade": [np.nan] * 10 + [2.0],
        }
    )
    labels = {"q": {"relevant": 2, "outside_pool": 2}}
    full = score_pool(pool, [1.0] * 10 + [0.0], labels, variant="full").iloc[0]
    judged = score_pool(pool.tail(1), [0.0], labels, variant="judged").iloc[0]
    assert full.ndcg_at_10 == full.judgment_coverage_at_10 == 0
    assert full.judged_recall_at_100 == 0.5
    assert 0 < judged.ndcg_at_10 < 1
    assert len(pool) == 11 and pool.grade.isna().sum() == 10


def test_equal_scores_use_item_id_and_query_macro_population():
    pool = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2"],
            "product_id": ["b", "a", "c"],
            "grade": [2, 0, np.nan],
        }
    )
    rows = score_pool(pool, [1, 1, 1], {"q1": {"a": 0, "b": 2}, "q2": {}}, variant="tie")
    assert rows.query_id.tolist() == ["q1", "q2"]
    assert rows.iloc[0].mrr_at_10 == 0.5
    assert rows.iloc[1].ndcg_at_10 == 0
