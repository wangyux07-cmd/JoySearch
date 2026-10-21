import numpy as np
import pandas as pd
import pytest

from joysearch.common import Candidate
from joysearch.search.bm25 import BM25Index
from joysearch.search.dense import DenseIndex
from joysearch.search.fusion import reciprocal_rank_fusion
from joysearch.search.pipeline import SearchPipeline


class SyntheticEncoder:
    def encode(self, texts):
        return np.asarray([[1, 0] for _ in texts], dtype=np.float32)


def test_cosine_ranking_reload_and_catalog_validation(tmp_path):
    pytest.importorskip("threadpoolctl")
    manifest = {
        "version": "dense-v1",
        "catalog_id": "wands_search",
        "normalization": "l2",
        "dimension": 2,
        "data_manifest": {"sources": {}},
    }
    dense = DenseIndex(
        np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32),
        ["a", "b", "c"],
        manifest,
        SyntheticEncoder(),
    )
    dense.save(tmp_path)
    loaded = DenseIndex.load(tmp_path, encoder=SyntheticEncoder())
    assert [c.item_id for c in loaded.retrieve("synthetic query", 3)] == ["a", "b", "c"]
    assert [c.score for c in loaded.retrieve("synthetic query", 3)] == [1, 1, 0]
    assert loaded.retrieve(" ", 3) == []
    products = pd.DataFrame({"product_id": ["a", "b", "x"], "product_name": ["desk"] * 3})
    for field in ["product_class", "category hierarchy", "product_features", "product_description"]:
        products[field] = ""
    bm25 = BM25Index.build(products, {"sources": {}})
    with pytest.raises(ValueError, match="same ordered"):
        SearchPipeline(bm25, dense)
    bad = np.asarray([[2, 0], [1, 0], [0, 1]], dtype=np.float32)
    bad_path = tmp_path / "invalid"
    DenseIndex(bad, ["a", "b", "c"], manifest, SyntheticEncoder()).save(bad_path)
    with pytest.raises(ValueError, match="L2 normalized"):
        DenseIndex.load(bad_path, encoder=SyntheticEncoder())


def test_rrf_uses_ranks_and_reports_channel_evidence():
    def candidate(item, score, rank):
        return Candidate("wands_search", item, score, "test", rank)

    rankings = {
        "bm25": [candidate("a", 999, 1), candidate("b", 0.01, 2)],
        "dense": [candidate("b", 0.9, 1), candidate("c", 0.8, 2)],
    }
    results = reciprocal_rank_fusion(rankings, 3)
    assert [c.item_id for c in results] == ["b", "a", "c"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[0].evidence == {"ranks": {"bm25": 2, "dense": 1}, "sources": ["bm25", "dense"]}
    assert len({c.item_id for c in results}) == 3
    assert reciprocal_rank_fusion({}, 10) == []


def test_resume_rejects_changed_input_before_model_load(tmp_path):
    pytest.importorskip("huggingface_hub")
    from joysearch.common import write_json
    from scripts.build_dense import build

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    write_json(source / "manifest.json", {"dataset": "wands_search"})
    pd.DataFrame({"product_id": ["synthetic-item"]}).to_parquet(source / "products.parquet")
    write_json(output / "build_state.json", {"signature": {"products_sha256": "old-input"}})
    with pytest.raises(ValueError, match="changed input/config"):
        build(source, output, {})


def test_real_dense_and_hybrid_api():
    from fastapi.testclient import TestClient

    from joysearch.api.main import create_app
    from joysearch.common import ROOT

    if not (ROOT / "artifacts/search_dense/manifest.json").exists():
        pytest.skip("Optional dense index has not been built")
    with TestClient(create_app()) as client:
        readiness = client.get("/readyz").json()
        assert readiness["dense_ready"]
        assert readiness["search_algorithms"][:3] == ["bm25", "dense", "hybrid"]
        for algorithm in ["dense", "hybrid"]:
            response = client.post(
                "/api/v1/search",
                json={"query": "salon chair", "algorithm": algorithm, "debug": True},
            )
            assert response.status_code == 200
            results = response.json()["results"]
            assert len({item["item_id"] for item in results}) == 10
            assert all(item["catalog_id"] == "wands_search" and item["title"] for item in results)
            if algorithm == "hybrid":
                for item in results:
                    ranks = item["evidence"]["ranks"]
                    assert item["score"] == pytest.approx(
                        sum(1 / (60 + rank) for rank in ranks.values())
                    )
                    assert set(item["sources"]) == set(ranks)
