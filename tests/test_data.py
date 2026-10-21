import pandas as pd
import pytest

from joysearch.data.amazon import split_events


def test_chronological_dedup_before_positive_filter():
    # Synthetic events exercise latest negative, timestamp ties and equal-time item ordering.
    events = pd.DataFrame(
        [
            ("u", "a", 5, 1),
            ("u", "a", 1, 7),
            ("u", "b", 5, 2),
            ("u", "c", 4, 3),
            ("u", "d", 4, 3),
            ("u", "d", 5, 3),
            ("v", "a", 5, 1),
        ],
        columns=["user_id", "parent_asin", "rating", "timestamp"],
    )
    cohort, eligible, removed = split_events(events)
    assert eligible == 1 and removed == 2
    assert cohort.parent_asin.tolist() == ["b", "c", "d"]
    assert cohort.split.tolist() == ["train", "validation", "test"]
    assert not cohort.duplicated(["user_id", "parent_asin"]).any()
    with pytest.raises(ValueError, match="missing"):
        split_events(events.drop(columns="timestamp"))


def test_real_split_integrity():
    from joysearch.common import ROOT, read_json

    base = ROOT / "data/processed"
    splits = read_json(base / "wands/splits.json")
    groups = [set(splits[key]) for key in ["train", "validation", "test"]]
    assert len(set.union(*groups)) == sum(map(len, groups)) == 480
    events = pd.read_parquet(base / "amazon/events.parquet")
    assert not events.duplicated(["user_id", "parent_asin"]).any()
    assert events.groupby("user_id").size().min() >= 3
    for _, group in events.groupby("user_id", sort=False):
        ordered = group.sort_values(["timestamp", "parent_asin"])
        assert ordered.split.iloc[-2:].tolist() == ["validation", "test"]
        assert ordered.split.iloc[:-2].eq("train").all()
