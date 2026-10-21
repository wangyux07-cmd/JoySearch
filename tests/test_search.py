import pandas as pd

from joysearch.common import ROOT
from joysearch.search.bm25 import BM25Index


def test_bm25_ranking_and_persistence(tmp_path):
    products = pd.DataFrame(
        {"product_id": ["b", "a", "c"], "product_name": ["wooden desk", "wooden desk", "chair"]}
    )
    for field in ["product_class", "category hierarchy", "product_features", "product_description"]:
        products[field] = None
    index = BM25Index.build(products)
    index.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert [c.item_id for c in loaded.retrieve("WOODEN", 3)] == ["a", "b"]
    assert loaded.retrieve("xyznotaword", 3) == []
    assert loaded.retrieve("wooden", 3) == index.retrieve("wooden", 3)
    assert all(c.catalog_id == "wands_search" for c in loaded.retrieve("desk", 3))


def test_real_wands_reload():
    index = BM25Index.load(ROOT / "artifacts/search")
    results = index.retrieve("salon chair", 10)
    assert len(results) == 10
    assert {c.item_id for c in results} <= set(index.item_ids)
