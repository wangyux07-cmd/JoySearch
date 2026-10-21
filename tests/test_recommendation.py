import pandas as pd
import pytest

from joysearch.common import ROOT, RecommendationContext
from joysearch.recommendation.pipeline import RecPipeline


def test_itemcf_train_only_exclusion_and_reload(tmp_path):
    train = pd.DataFrame(
        [("u", "a"), ("u", "b"), ("v", "a"), ("v", "c"), ("w", "b"), ("w", "d")],
        columns=["user_id", "parent_asin"],
    )
    train["split"] = "train"
    model = RecPipeline.build(train)
    model.save(tmp_path)
    loaded = RecPipeline.load(tmp_path)
    result, _ = loaded.recommend(RecommendationContext("u"), 2)
    assert {c.item_id for c in result} == {"c", "d"}
    assert result == model.recommend(RecommendationContext("u"), 2)[0]
    cold, reason = loaded.recommend(RecommendationContext("unknown"), 2)
    assert reason == "cold_user" and all(c.source == "popularity" for c in cold)
    assert all(c.catalog_id == "amazon_baby_rec" for c in result)
    train.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="train-only"):
        RecPipeline.build(train)


def test_real_recommendations():
    model = RecPipeline.load(ROOT / "artifacts/recommendation")
    users = list(model.aliases.values())[:2]
    results = [model.recommend(RecommendationContext(u), 10)[0] for u in users]
    assert [c.item_id for c in results[0]] != [c.item_id for c in results[1]]
    for user, candidates in zip(users, results):
        seen = {model.popularity.items[i] for i in model.popularity.histories[user]}
        assert not seen.intersection(c.item_id for c in candidates)
        assert len(candidates) == 10
