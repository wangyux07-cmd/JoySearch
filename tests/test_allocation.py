import numpy as np
import pytest

from joysearch.common import ROOT, Candidate, RecommendationContext, read_json
from joysearch.recommendation.fusion import CHANNELS, allocate, filter_channels
from joysearch.traffic.policy import AllocationPolicy
from joysearch.traffic.simulator import simulate


def test_quota_dedup_reallocation_weighted_ranking_and_catalog_boundary():
    def rows(items, source):
        return [
            Candidate("amazon_baby_rec", item, 1.0, source, i) for i, item in enumerate(items, 1)
        ]

    rankings = dict(
        zip(CHANNELS, [rows("abc", "popularity"), rows("a", "itemcf"), rows("bde", "two_tower")])
    )
    quotas = dict(zip(CHANNELS, [2, 1, 1]))
    ranked, diagnostic = allocate(rankings, quotas, 4)
    assert [c.item_id for c in ranked] == list("abcd")
    assert ranked[0].score == pytest.approx(0.75 / 61)
    assert diagnostic["filled_quotas"] == dict(zip(CHANNELS, [2, 0, 1]))
    assert diagnostic["exhausted_slots"] == diagnostic["reallocated_slots"] == 1
    assert ranked[-1].evidence["quota_credit"] == "reallocated"
    assert ranked[0].evidence["sources"] == ["popularity", "itemcf_native"]
    assert allocate(rankings, quotas, 4) == (ranked, diagnostic)
    filtered = filter_channels(rankings, {"a"}, 4)
    assert not filtered["itemcf_native"]
    assert all(c.item_id != "a" for values in filtered.values() for c in values)
    with pytest.raises(ValueError, match="Quotas"):
        allocate(rankings, dict(zip(CHANNELS, [2, 1, 2])), 4)
    rankings["popularity"] = [Candidate("wands_search", "a", 1, "bm25", 1)]
    with pytest.raises(ValueError, match="catalog"):
        allocate(rankings)


def test_ucb_selected_feedback_and_reproducible_proxy_simulation():
    rng = np.random.default_rng(42)
    policy = AllocationPolicy("ucb1", 2)
    assert policy.choose(rng) == 0
    policy.update(0, 1)
    assert policy.counts.tolist() == [1, 0] and policy.rewards.tolist() == [1, 0]
    assert policy.choose(rng) == 1
    policy.update(1, 0)
    assert policy.choose(rng) == 0
    config = {
        "seeds": [7, 42],
        "policies": ["fixed", "fixed_exploration", "ucb1"],
        "exploration_rate": 0.1,
    }
    args = (
        ["u", "v"],
        [[1, 0], [0, 1]],
        [[["a"], ["b"]], [["a"], ["b"]]],
        ["popular", "balanced"],
        0,
        config,
        {"a": 3, "b": 1},
        2,
    )
    summaries, logs = simulate(*args)
    assert (summaries, logs) == simulate(*args)
    assert all(s["contextual_proxy_regret"] == 2 - s["proxy_reward_sum"] for s in summaries)
    for seed in config["seeds"]:
        for kind in config["policies"]:
            counts = {"popular": 0, "balanced": 0}
            for row in [r for r in logs if r["seed"] == seed and r["policy"] == kind]:
                assert row["selected_arm_previous_count"] == counts[row["arm"]]
                counts[row["arm"]] += 1


def test_real_allocation_shared_candidate_path_and_cold_api():
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from joysearch.api.main import create_app
    from joysearch.evaluation.allocation_eval import policy_candidates
    from joysearch.recommendation.pipeline import RecPipeline

    if not (ROOT / "artifacts/recommendation_fusion/selection.json").exists():
        pytest.skip("Run complete allocation validation before testing selected policies")
    model = RecPipeline.load(ROOT / "artifacts/recommendation")
    user = next(iter(model.aliases.values()))
    context = RecommendationContext(user)
    channels, length = model.recall_channels(context)
    variants = policy_candidates(channels, length, model.allocation["config"])
    for algorithm, offline in [
        ("rrf", "rrf"),
        ("quota", "quota:" + model.allocation["fixed_policy"]),
        ("adaptive_quota", "adaptive:" + model.allocation["adaptive_policy"]),
    ]:
        ranking, _ = model.recommend(context, 100, algorithm)
        assert [(c.item_id, c.score) for c in ranking] == [
            (c.item_id, c.score) for c in variants[offline][0]
        ]
        assert model.recommend(context, 10, algorithm)[0] == ranking[:10]
        assert len(ranking) == len({c.item_id for c in ranking}) == 100
        assert all(c.catalog_id == "amazon_baby_rec" for c in ranking)
        for unknown in [
            RecommendationContext(None),
            RecommendationContext("missing", ("invalid-item",)),
        ]:
            candidates, reason = model.recommend(unknown, 10, algorithm)
            assert reason == "cold_user" and all(c.source == "popularity" for c in candidates)
    with TestClient(create_app(enable_dense=False)) as client:
        assert "adaptive_quota" in client.get("/readyz").json()["recommendation_algorithms"]
        response = client.post(
            "/api/v1/recommend",
            json={
                "algorithm": "adaptive_quota",
                "user_id": next(iter(model.aliases)),
                "debug": True,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["allocation"]["budget"] == 100
        assert len(response.json()["results"]) == 10
        cold = client.post("/api/v1/recommend", json={"algorithm": "quota"}).json()
        assert cold["fallback_reason"] == "cold_user" and cold["degraded"]
    assert (
        read_json(ROOT / "artifacts/recommendation_fusion/selection.json")["scope"]
        == "complete_fixed_cohort"
    )
