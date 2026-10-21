import hashlib
import time

import numpy as np
import pandas as pd

from joysearch.common import write_json
from joysearch.data.wands import file_hash


def split_events(events, users=10000, seed=42):
    required = {"user_id", "parent_asin", "rating", "timestamp"}
    if not required <= set(events):
        raise ValueError(f"Amazon missing columns: {required - set(events)}")
    if events[list(required)].isna().any().any():
        raise ValueError("Amazon has missing IDs, ratings or timestamps")
    events = events.copy()
    for key in ["user_id", "parent_asin"]:
        events[key] = events[key].astype(str)
        if events[key].str.strip().eq("").any():
            raise ValueError(f"Amazon empty {key}")
    events["rating"] = pd.to_numeric(events.rating, errors="raise")
    events["timestamp"] = pd.to_numeric(events.timestamp, errors="raise")
    if not events.rating.between(1, 5).all() or not np.isfinite(events.timestamp).all():
        raise ValueError("Amazon invalid rating/timestamp")
    # Latest event wins; equal timestamps retain the highest rating deterministically.
    unique = events.sort_values(["user_id", "parent_asin", "timestamp", "rating"]).drop_duplicates(
        ["user_id", "parent_asin"], keep="last"
    )
    positive = unique[unique.rating >= 4].copy()
    sizes = positive.groupby("user_id").size()
    eligible = sorted(sizes[sizes >= 3].index)
    if users and len(eligible) > users:
        eligible = sorted(
            np.random.default_rng(seed).choice(eligible, users, replace=False).tolist()
        )
    if not eligible:
        raise ValueError("Amazon contains no users with >=3 distinct positive items")
    cohort = (
        positive[positive.user_id.isin(eligible)]
        .sort_values(["user_id", "timestamp", "parent_asin"])
        .reset_index(drop=True)
    )
    reverse = cohort.groupby("user_id").cumcount(ascending=False)
    cohort["split"] = np.where(reverse == 0, "test", np.where(reverse == 1, "validation", "train"))
    assert not cohort.duplicated(["user_id", "parent_asin"]).any()
    assert cohort.groupby("user_id").size().min() >= 3
    assert (cohort[cohort.split == "train"].groupby("user_id").size() >= 1).all()
    return cohort, len(sizes[sizes >= 3]), len(events) - len(unique)


def prepare(source, output, users=10000, seed=42):
    started = time.perf_counter()
    events = pd.read_csv(source, dtype={"user_id": str, "parent_asin": str})
    cohort, eligible, duplicates = split_events(events, users, seed)
    output.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(output / "events.parquet", index=False)
    user_ids = sorted(cohort.user_id.unique())
    write_json(output / "users.json", user_ids)
    train = cohort[cohort.split == "train"]
    train_items = set(train.parent_asin)
    test = cohort[cohort.split == "test"]
    manifest = {
        "dataset": "amazon_baby_rec",
        "version": "amazon-v1",
        "seed": seed,
        "sources": {source.name: {"sha256": file_hash(source), "bytes": source.stat().st_size}},
        "raw_events": len(events),
        "duplicates_removed": duplicates,
        "deduplication": "latest timestamp; highest rating breaks equal timestamp ties; before positive filter",
        "positive_policy": "rating >= 4; positive rating preference, not clicks/purchases",
        "eligible_count": eligible,
        "sample_count": len(user_ids),
        "sample_policy": "seeded user sampling, complete positive histories retained",
        "cohort_sha256": hashlib.sha256("\n".join(user_ids).encode()).hexdigest(),
        "split_sizes": {str(k): int(v) for k, v in cohort.split.value_counts().items()},
        "split_policy": "per-user chronological (timestamp,item_id) leave-last-two-out; not global time split",
        "split_membership": "events.parquet contains exact user/item/timestamp/split membership",
        "split_sha256": file_hash(output / "events.parquet"),
        "catalog_size": len(train_items),
        "candidate_universe": "train_items_full_catalog",
        "unavailable_test_items": int((~test.parent_asin.isin(train_items)).sum()),
        "excluded_seen": "train positives only",
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(output / "manifest.json", manifest)
    return manifest
