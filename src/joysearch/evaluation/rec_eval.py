import time

import pandas as pd

from joysearch.common import RecommendationContext, read_json
from joysearch.evaluation.metrics import hit_rate, ndcg
from joysearch.evaluation.search_eval import latency_summary
from joysearch.recommendation.pipeline import RecPipeline


def evaluate(source, artifact, algorithms=("popularity", "itemcf"), split="test"):
    started = time.perf_counter()
    model = RecPipeline.load(artifact, enable_two_tower="two_tower" in algorithms)
    manifest = read_json(source / "manifest.json")
    if model.manifest["data_manifest"]["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("Recommendation model/data split mismatch; rebuild model")
    events = pd.read_parquet(source / "events.parquet")
    test = events[events.split == split].sort_values("user_id")
    train = events[events.split == "train"]
    expected = {
        user: {str(item) for item in group.parent_asin} for user, group in train.groupby("user_id")
    }
    actual = {
        user: {str(model.popularity.items[i]) for i in history}
        for user, history in model.popularity.histories.items()
    }
    if expected != actual or set(test.user_id) != set(actual):
        raise ValueError("Recommendation model histories/cohort differ from exact train split")
    if set(train.parent_asin) != set(model.popularity.items):
        raise ValueError("Recommendation candidate universe differs from full train catalog")
    train_items = set(model.popularity.items)
    unseen = int((~test.parent_asin.isin(train_items)).sum())
    aliases = {user: alias for alias, user in model.aliases.items()}
    algorithm_reports, latency, rows = {}, {}, []
    for algorithm in algorithms:
        if algorithm not in model.algorithms:
            raise ValueError(f"Requested recommendation algorithm unavailable: {algorithm}")
        for user in test.user_id.iloc[:5]:
            model.recommend(RecommendationContext(user), 10, algorithm)
        hits, discounted, fallbacks, recommended, times = 0, 0.0, 0, set(), []
        for event in test.itertuples(index=False):
            tick = time.perf_counter()
            candidates, reason = model.recommend(
                RecommendationContext(event.user_id), 10, algorithm
            )
            times.append((time.perf_counter() - tick) * 1000)
            ranking = [c.item_id for c in candidates]
            hit = hit_rate(ranking, event.parent_asin)
            gain = ndcg(ranking, {event.parent_asin: 1})
            hits += int(hit)
            discounted += gain
            fallbacks += int(reason is not None)
            recommended.update(ranking)
            rows.append(
                {
                    "user_alias": aliases[event.user_id],
                    "algorithm": algorithm,
                    "hr_at_10": hit,
                    "ndcg_at_10": gain,
                    "unavailable_target": int(event.parent_asin not in train_items),
                    "fallback_reason": reason,
                }
            )
        algorithm_reports[algorithm] = {
            "hr_at_10": hits / len(test),
            "ndcg_at_10": discounted / len(test),
            "catalog_coverage_at_10": len(recommended) / len(train_items),
            "hit_count": hits,
            "ndcg_sum": discounted,
            "unique_recommended_items": len(recommended),
            "fallback_count": fallbacks,
            "fallback_rate": fallbacks / len(test),
        }
        latency[algorithm] = latency_summary(times)
    report = {
        "status": "completed",
        "dataset": "amazon_baby_rec",
        "split": split,
        "seed": manifest["seed"],
        "k": 10,
        "sample_count": len(test),
        "eligible_count": manifest["eligible_count"],
        "users_evaluated": len(test),
        "catalog_size": len(train_items),
        "candidate_universe": "train_items_full_catalog",
        "exclude_seen": "train positives",
        "unavailable_test_item_count": unseen,
        "unavailable_test_item_rate": unseen / len(test),
        "unavailable_target_count": unseen,
        "unavailable_target_rate": unseen / len(test),
        "algorithms": algorithm_reports,
        "model_version": model.version,
        "config": {
            "baseline": model.manifest,
            "two_tower": model.two_tower.manifest if "two_tower" in algorithms else None,
        },
        "duration_seconds": time.perf_counter() - started,
        "limitations": [
            "rating >= 4 is positive rating preference, not clicks/purchases",
            "Deterministic sampled user cohort; complete histories retained",
            "Per-user leave-last-two-out is not global temporal evaluation",
            "Unavailable targets count as misses; full-catalog evaluation without sampled negatives",
            "No recommendation metadata; separate from WANDS catalog",
        ],
    }
    return report, latency, pd.DataFrame(rows)
