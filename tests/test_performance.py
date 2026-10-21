import asyncio

import numpy as np
import pytest

from joysearch.evaluation.http_benchmark import cell_summary
from joysearch.evaluation.process_stats import process_stats
from joysearch.search.dense import DenseIndex
from joysearch.search.exact_scan import CachedControllerDenseIndex
from joysearch.telemetry import RequestTrace, timed


def test_exact_cached_controller_preserves_scores_ties_and_budget():
    original = DenseIndex(
        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32), ["b", "a", "c"], {}, None
    )
    cached = CachedControllerDenseIndex(original)
    for vector in [np.array([1.0, 0.0], dtype=np.float32), np.zeros(2, dtype=np.float32)]:
        for k in [0, 1, 2, 3]:
            assert original.rank_vector(vector, k) == cached.rank_vector(vector, k)
    assert [c.item_id for c in cached.rank_vector(np.zeros(2, dtype=np.float32), 3)] == [
        "a",
        "b",
        "c",
    ]


def test_request_trace_isolation_nested_counts_and_error_cleanup():
    class Service:
        def inner(self):
            return [1, 2]

        def outer(self):
            return self.inner()

    service = Service()
    timed(service, "inner", "inner")
    timed(service, "outer", "outer")

    async def task(enabled):
        with RequestTrace(enabled) as trace:
            await asyncio.sleep(0)
            assert service.outer() == [1, 2]
        return trace.spans

    async def run():
        return await asyncio.gather(task(True), task(False), task(True))

    a, b, c = asyncio.run(run())
    assert not b and len(a) == len(c) == 2
    assert a[1]["parent_id"] == c[1]["parent_id"] == 0
    assert a[1]["output_count"] == 2
    with pytest.raises(ValueError), RequestTrace(True):
        raise ValueError("expected boundary failure")
    assert service.outer() == [1, 2]


def test_http_denominators_failure_empty_and_p99_sample_floor():
    rows = [
        {
            "latency_ms": 10,
            "success": True,
            "status_code": 200,
            "error": None,
            "degraded": False,
            "result_count": 0,
        },
        {
            "latency_ms": 20,
            "success": True,
            "status_code": 200,
            "error": None,
            "degraded": True,
            "result_count": 10,
        },
        {
            "latency_ms": 100,
            "success": False,
            "status_code": None,
            "error": "timeout",
            "degraded": False,
            "result_count": None,
        },
        {
            "latency_ms": 30,
            "success": False,
            "status_code": 503,
            "error": "http_error",
            "degraded": False,
            "result_count": None,
        },
    ]
    stats = cell_summary(rows, 2, 1000)
    assert (stats["offered_qps"], stats["completed_qps"], stats["success_qps"]) == (2, 1.5, 1)
    assert stats["timeout_rate"] == 0.25 and stats["error_rate"] == 0.5
    assert stats["degraded_rate_successful"] == 0.5 and stats["empty_success_count"] == 1
    assert stats["p99_ms"] is None
    assert process_stats()["rss_bytes"] > 0


def test_real_debug_stage_contract_and_default_parity():
    from fastapi.testclient import TestClient

    from joysearch.api.main import create_app
    from joysearch.common import ROOT

    if not (ROOT / "artifacts/search_dense/manifest.json").exists():
        pytest.skip("Build local semantic index first")
    with TestClient(create_app(enable_ltr=False)) as client:
        for path, payload, expected_stage in [
            (
                "/api/v1/search",
                {"query": "salon chair", "algorithm": "query_hybrid"},
                "search.parse",
            ),
            (
                "/api/v1/recommend",
                {"user_id": "anonymous_user_00001", "algorithm": "adaptive_quota"},
                "rec.filter",
            ),
        ]:
            plain = client.post(path, json=payload).json()
            debug = client.post(path, json={**payload, "debug": True}).json()
            assert "timings" not in plain
            assert expected_stage in {s["stage"] for s in debug["timings"]}
            assert all(s["elapsed_ms"] >= 0 for s in debug["timings"])
            assert [(r["item_id"], r["score"]) for r in plain["results"]] == [
                (r["item_id"], r["score"]) for r in debug["results"]
            ]
