import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from joysearch.common import ROOT, Candidate, RecommendationContext, read_json
from joysearch.evaluation.metrics import conditional_hit_rate, jaccard
from joysearch.evaluation.recall_eval import (
    UNION_SCHEMA,
    candidate_variants,
    prepare_channels,
    request_metrics,
    served_itemcf,
    trace_rows,
)
from joysearch.recommendation.pipeline import RecPipeline


def candidate(item, rank=1, catalog="wands_search", source="test"):
    return Candidate(catalog, item, 10.0 / rank, source, rank)


def test_budget_dedup_sources_seen_and_ablation_eviction(tmp_path):
    raw = {
        "a": [candidate("seen"), candidate("x", 2), candidate("x", 3), candidate("z", 4)],
        "b": [candidate("z"), candidate("y", 2)],
    }
    eligible = prepare_channels(raw, {"seen"}, 2, "wands_search")
    assert [c.item_id for c in eligible["a"]] == ["x", "z"]
    config = {"budget": 2, "display_k": 1, "rrf_k": 60}
    variants, _ = candidate_variants(eligible, config)
    assert [c.item_id for c in variants["a+b"]] == ["z", "x"]
    assert variants["a+b"][0].evidence["sources"] == ["a", "b"]
    rows, overlap, contribution = request_metrics(
        "q", raw, eligible, variants, {"seen"}, {"y": 2}, config, False
    )
    full = next(r for r in rows if r["variant"] == "a+b")
    assert full["raw_union_recall"] == 1 and full["candidate_recall"] == 0
    assert full["selected_count"] == 2 and full["seen_filter_loss"] == 1
    assert overlap[0]["jaccard"] == pytest.approx(1 / 3)
    assert contribution[1]["eligible_unique_relevant_count"] == 1
    channels, stages = trace_rows("q", raw, eligible, variants, {"seen"}, 1)
    z = next(r for r in stages if r["item_id"] == "z")
    assert z["sources"] == ["a", "b"] and z["raw_ranks"] == [4, 1]
    assert z["top10_variants"] == ["a+b", "b"]
    path = tmp_path / "stages.parquet"
    pq.write_table(pa.Table.from_pylist(stages, schema=UNION_SCHEMA), path)
    assert pq.read_table(path).to_pylist() == stages
    assert all(r["excluded_seen"] for r in channels if r["item_id"] == "seen")
    with pytest.raises(ValueError, match="Mixed catalog"):
        prepare_channels(
            {"a": [candidate("x", catalog="amazon_baby_rec")]}, set(), 2, "wands_search"
        )


def test_candidate_hit_conversion_null_and_shortage():
    assert jaccard(["a", "a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert jaccard([], []) == 0
    assert conditional_hit_rate(2, 5) == 0.4
    assert conditional_hit_rate(0, 0) is None
    c = candidate("a", catalog="amazon_baby_rec", source="itemcf")
    config = {"budget": 100, "display_k": 10, "rrf_k": 60}
    eligible = {"popularity": [], "itemcf_native": [c], "two_tower": []}
    variants, _ = candidate_variants(eligible, config, True)
    rows, _, _ = request_metrics(
        "anonymous_user", eligible, eligible, variants, set(), {"missing": 1}, config, True
    )
    assert all(r["candidate_shortage"] == 1 and r["candidate_recall"] == 0 for r in rows)


def test_native_itemcf_and_completion_preserve_real_baseline():
    model = RecPipeline.load(ROOT / "artifacts/recommendation")
    config = read_json(ROOT / "configs/recall.json")
    for user in list(model.aliases.values())[:3]:
        seen = {model.popularity.items[i] for i in model.popularity.histories[user]}
        context = RecommendationContext(user, exclude_seen=False)
        depth = config["budget"] + len(seen)
        raw = {
            "itemcf_native": model.itemcf.retrieve_native(context, depth),
            "popularity": model.popularity.recommend(context, depth),
        }
        kept = prepare_channels(raw, seen, 100, "amazon_baby_rec")
        served = served_itemcf(kept["itemcf_native"], kept["popularity"], 100)
        expected, _ = model.recommend(RecommendationContext(user), 100, "itemcf")
        assert served == expected
        assert not seen & {c.item_id for c in served}
        assert all(c.score > 0 and c.source == "itemcf" for c in kept["itemcf_native"])
    assert model.itemcf.retrieve_native(RecommendationContext("unknown"), 100) == []
    assert model.recommend(RecommendationContext("unknown"), 10)[1] == "cold_user"


def test_real_search_diagnostic_fusion_matches_served_hybrid():
    from joysearch.search.pipeline import SearchPipeline

    model = SearchPipeline.load(ROOT, enable_ltr=False)
    raw = {
        "bm25": model.index.retrieve("salon chair", 100),
        "dense": model.dense.retrieve("salon chair", 100),
    }
    eligible = prepare_channels(raw, set(), 100, "wands_search")
    variants, _ = candidate_variants(eligible, read_json(ROOT / "configs/recall.json"))
    expected = model.search("salon chair", 100, "hybrid")
    assert [c.item_id for c in variants["bm25+dense"]] == [c.item_id for c in expected]
    assert [c.score for c in variants["bm25+dense"]] == pytest.approx([c.score for c in expected])
