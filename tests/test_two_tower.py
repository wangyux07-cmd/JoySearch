import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from joysearch.common import ROOT, RecommendationContext, read_json
from joysearch.evaluation.metrics import ndcg

torch = pytest.importorskip("torch")

from joysearch.recommendation.popularity import Popularity
from joysearch.recommendation.two_tower import TwoTower, TwoTowerNetwork, padded_histories
from joysearch.recommendation.two_tower_training import sample_negatives, validation_metrics


def test_target_excluded_context_and_train_only_negative_sampling():
    context = padded_histories([[0, 1, 2], [3, 4]], 2, 6, targets=[1, 4])
    assert context.tolist() == [[0, 2], [3, 6]]
    known = sparse.csr_matrix([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0]], dtype=bool)
    rows = np.array([0, 1, 0, 1])
    negatives = sample_negatives(np.random.default_rng(42), rows, known, 20).numpy()
    assert not np.asarray(known[np.repeat(rows, 20), negatives.ravel()]).any()
    # Held-out labels are not an input to the sampler; unobserved draws are proxies.
    assert set(negatives[0]) <= {3, 4, 5}
    with pytest.raises(ValueError, match="unobserved"):
        sample_negatives(np.random.default_rng(42), np.array([0]), sparse.csr_matrix([[1, 1]]), 1)


def test_validation_uses_serving_history_and_full_catalog_target_ranks():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    popularity = Popularity(
        [f"item-{i:02d}" for i in range(20)], [1] * 20, {"u": [0, 1], "v": [2, 3]}
    )
    model = TwoTower(TwoTowerNetwork(20, 8), popularity, {"config": {"max_history": 2}})
    validation = pd.DataFrame({"user_id": ["u", "v"], "parent_asin": ["item-05", "unavailable"]})
    metrics = validation_metrics(model, validation, 2)
    candidates, _ = model.recommend(RecommendationContext("u"), 10)
    assert metrics["hr_at_10"] == int("item-05" in [c.item_id for c in candidates]) / 2
    assert metrics["ndcg_at_10"] == pytest.approx(
        ndcg([c.item_id for c in candidates], {"item-05": 1}) / 2
    )
    assert metrics["users"] == 2 and metrics["unavailable_targets"] == 1
    assert not {"item-00", "item-01"} & {c.item_id for c in candidates}
    cold, reason = model.recommend(RecommendationContext("unknown"), 10)
    assert reason == "cold_user" and all(c.source == "popularity" for c in cold)


def test_real_two_tower_persistence_personalisation_and_api():
    from fastapi.testclient import TestClient

    from joysearch.api.main import create_app
    from joysearch.recommendation.pipeline import RecPipeline

    folder = ROOT / "artifacts/recommendation_two_tower"
    if not (folder / "manifest.json").exists():
        pytest.skip("Optional Two-Tower model has not been trained")
    model = RecPipeline.load(ROOT / "artifacts/recommendation")
    manifest = read_json(folder / "manifest.json")
    events = pd.read_parquet(ROOT / "data/processed/amazon/events.parquet")
    train = events[events.split.eq("train")]
    eligible = train.groupby("user_id").size().gt(1)
    assert manifest["training_examples"] == int(train.user_id.isin(eligible[eligible].index).sum())
    assert manifest["best_epoch"] <= manifest["config"]["epochs"]
    users = list(model.aliases.values())[:2]
    rankings = [model.recommend(RecommendationContext(u), 10, "two_tower")[0] for u in users]
    assert [c.item_id for c in rankings[0]] != [c.item_id for c in rankings[1]]
    reloaded = RecPipeline.load(ROOT / "artifacts/recommendation")
    assert reloaded.recommend(RecommendationContext(users[0]), 10, "two_tower")[0] == rankings[0]
    mismatched = Popularity(
        model.popularity.items[::-1], model.popularity.counts, model.popularity.histories
    )
    with pytest.raises(ValueError, match="catalog/history/features changed"):
        TwoTower.load(folder, mismatched, model.manifest)
    for user, candidates in zip(users, rankings):
        seen = {model.popularity.items[i] for i in model.popularity.histories[user]}
        assert not seen & {c.item_id for c in candidates}
        assert len(candidates) == 10 and all(c.source == "two_tower" for c in candidates)
    with TestClient(create_app(enable_dense=False)) as client:
        assert client.get("/readyz").json()["two_tower_ready"]
        response = client.post(
            "/api/v1/recommend",
            json={
                "algorithm": "two_tower",
                "user_id": next(iter(model.aliases)),
            },
        )
        assert response.status_code == 200
        assert response.json()["model_version"] == "two-tower-v1"
        assert [r["item_id"] for r in response.json()["results"]] == [
            c.item_id for c in rankings[0]
        ]
        cold = client.post("/api/v1/recommend", json={"algorithm": "two_tower"}).json()
        assert cold["fallback_reason"] == "cold_user"
