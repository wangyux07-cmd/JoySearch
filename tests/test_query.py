import pandas as pd
import pytest

from joysearch.common import ROOT
from joysearch.search.query import CatalogAttributes, QueryUnderstanding, normalize


def test_synthetic_constraints_negation_dimensions_and_bounded_rewrite():
    parser = QueryUnderstanding()
    query = "  MOEN ＢＬＫ chair 18x18 without red or leather under £100 delivery tomorrow "
    plan = parser.parse(query)
    assert plan.original_query == query
    assert normalize(query).startswith("moen blk chair 18x18")
    assert plan.category_hints == ["chair"]
    assert plan.attributes["size"] == ["18x18"]
    assert not plan.attributes["color"]
    assert {n["value"] for n in plan.negations} >= {"red", "leather"}
    assert plan.unsupported_constraints == ["price", "delivery"]
    assert len(plan.rewrites) == 1
    assert "moen black chair 18x18 without red or leather under £100" in plan.rewrites[0]
    assert parser.parse('MOEN 48" vanity').rewrites == ["moen 48 inch vanity"]
    assert parser.parse("non slip tile").negations == []
    assert parser.parse("without 48.5 inch red cushions").attributes["size"] == []
    assert parser.parse("without unobtainium").negations[0]["slot"] == "scope"
    assert parser.parse("faux leather chairs").attributes["material"] == ["faux leather"]
    assert parser.parse("unknownbrand zzz").slot_status["color"] == "unrecognized"


def test_missing_unknown_contradiction_and_soft_only():
    parser = QueryUnderstanding()
    products = pd.DataFrame(
        {
            "product_id": ["a", "b", "c", "d"],
            "product_features": ["", "color:ultraviolet", "color:red", "color:blue|color:red"],
        }
    )
    catalog = CatalogAttributes(products, parser)
    plan = parser.parse("blue chair")
    assert catalog.compare(plan, "a")[0]["color"] == "field_missing"
    assert catalog.compare(plan, "b")[0]["color"] == "unrecognized"
    assert catalog.compare(plan, "c")[0]["color"] == "contradiction"
    assert catalog.compare(plan, "d")[0]["color"] == "match"
    assert catalog.compare(parser.parse("chair without red"), "c")[1] == -1


def test_real_query_budget_channel_preservation_and_api():
    pytest.importorskip("sentence_transformers")
    from fastapi.testclient import TestClient

    from joysearch.api.main import create_app
    from joysearch.search.pipeline import SearchPipeline

    if not (ROOT / "artifacts/search_dense/manifest.json").exists():
        pytest.skip("Optional dense index not built")
    model = SearchPipeline.load(ROOT, enable_ltr=False)
    query = "blk 18x18 seat cushions"
    original, base, _ = model.query_candidates(query, "original")
    assert original == model.search(query, 100, "hybrid")
    expanded, channels, plan = model.query_candidates(query, rankings=base)
    assert channels["bm25"] == base["bm25"] and channels["dense"] == base["dense"]
    assert "rewrite_bm25" in channels and plan.rewrites
    from joysearch.search.fusion import reciprocal_rank_fusion

    offline_pool = reciprocal_rank_fusion(channels, 100, model.rrf_k)
    assert model.catalog_attributes.rerank(
        plan, offline_pool, model.query_understanding.config["soft_weight"]
    ) == expanded
    assert len(expanded) == len({c.item_id for c in expanded}) == 100
    assert model.search(query, 10, "query_hybrid") == expanded[:10]
    assert all(not c.evidence["hard_filtered"] for c in expanded)
    assert "rewrite_bm25" not in model.query_candidates("salon chair")[1]
    with TestClient(create_app()) as client:
        assert "query_hybrid" in client.get("/readyz").json()["search_algorithms"]
        body = {"query": query + " under £100", "algorithm": "query_hybrid", "debug": True}
        response = client.post("/api/v1/search", json=body)
        assert response.status_code == 200, response.text
        assert (
            response.json()["query_plan"]
            == model.query_understanding.parse(body["query"]).to_dict()
        )
        assert "price" in response.json()["query_plan"]["unsupported_constraints"]
        body["debug"] = False
        assert "query_plan" not in client.post("/api/v1/search", json=body).json()
