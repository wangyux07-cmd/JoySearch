import numpy as np
import pandas as pd
import pytest

from joysearch.common import ROOT, read_json
from joysearch.search.features import FEATURE_NAMES
from joysearch.search.ranker import LambdaMARTRanker, query_groups, retrieval_signature


def test_query_groups_reject_leakage_and_interleaving():
    frame = pd.DataFrame({"query_id": ["train-a", "train-a", "train-b"]})
    assert query_groups(frame, ["train-a", "train-b"]) == [2, 1]
    with pytest.raises(ValueError, match="split"):
        query_groups(frame, ["train-a"])
    with pytest.raises(ValueError, match="contiguous"):
        query_groups(pd.DataFrame({"query_id": ["a", "b", "a"]}), ["a", "b"])


def test_real_ranker_feature_parity_pool_and_persistence():
    pytest.importorskip("lightgbm")
    from joysearch.search.pipeline import SearchPipeline

    folder = ROOT / "artifacts/search_ltr"
    if not (folder / "manifest.json").exists():
        pytest.skip("Optional LambdaMART model has not been trained")
    model = SearchPipeline.load(ROOT)
    splits = read_json(ROOT / "data/processed/wands/splits.json")
    queries = pd.read_parquet(ROOT / "data/processed/wands/queries.parquet").set_index("query_id")
    cache = pd.read_parquet(folder / "train.parquet")
    qid = splits["train"][0]
    query = queries.loc[qid, "query"]
    pool, rankings = model.hybrid_candidates(query)
    online = model.features.transform(query, pool, rankings, model.candidate_k)
    offline = cache[cache.query_id.eq(qid)]
    assert offline.product_id.tolist() == [c.item_id for c in pool]
    np.testing.assert_array_equal(online.to_numpy(), offline[FEATURE_NAMES].to_numpy())
    ranked = model.search(query, 100, "ltr")
    assert {c.item_id for c in ranked} == {c.item_id for c in pool}
    assert model.search(query, 10, "ltr") == ranked[:10]
    reloaded = LambdaMARTRanker.load(ROOT, retrieval_signature(ROOT))
    assert reloaded.rerank(pool, online, 100) == ranked
    manifest = model.ranker.manifest
    for split in ["train", "validation"]:
        rows = pd.read_parquet(folder / f"{split}.parquet")
        assert set(rows.query_id) <= set(splits[split])
        assert not set(rows.query_id) & set(splits["test"])
        assert manifest["counts"][split]["judged_rows"] == int(rows.grade.notna().sum())
        assert (
            sum(query_groups(rows.dropna(subset=["grade"]), splits[split]))
            == rows.grade.notna().sum()
        )
    with pytest.raises(ValueError, match="changed"):
        LambdaMARTRanker.load(ROOT, {})


def test_real_ltr_api_contract():
    pytest.importorskip("lightgbm")
    from fastapi.testclient import TestClient

    from joysearch.api.main import create_app

    if not (ROOT / "artifacts/search_ltr/manifest.json").exists():
        pytest.skip("Optional LambdaMART model has not been trained")
    with TestClient(create_app()) as client:
        assert client.get("/readyz").json()["ltr_ready"]
        response = client.post(
            "/api/v1/search",
            json={
                "query": "salon chair",
                "algorithm": "ltr",
                "debug": True,
            },
        )
        assert response.status_code == 200
        assert response.json()["model_version"] == "lambdamart-v1"
        results = response.json()["results"]
        assert len({r["item_id"] for r in results}) == 10
        assert all(set(r["evidence"]["feature_values"]) == set(FEATURE_NAMES) for r in results)
        assert all(r["evidence"]["reranker"] == "lambdamart-v1" for r in results)
