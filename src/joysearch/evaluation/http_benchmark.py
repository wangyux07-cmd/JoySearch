"""Closed-loop real HTTP load, separate from model-internal timings and quality."""

import asyncio
import hashlib
import json
import os
import platform
import random
import time
from datetime import UTC, datetime

import httpx
import numpy as np
import pandas as pd

from joysearch.common import read_json, write_json
from joysearch.evaluation.process_stats import process_stats
from joysearch.evaluation.report import code_manifest


def result_signature(body):
    content = [(r["catalog_id"], r["item_id"], r["rank"], r["score"]) for r in body["results"]]
    return hashlib.sha256(json.dumps(content).encode()).hexdigest()


def cell_summary(rows, elapsed, p99_min_samples):
    values = [r["latency_ms"] for r in rows]
    successful = [r for r in rows if r["success"]]
    received = sum(r["status_code"] is not None for r in rows)
    return {
        "offered_count": len(rows),
        "completed_http_count": received,
        "success_count": len(successful),
        "duration_seconds": elapsed,
        "offered_qps": len(rows) / elapsed,
        "completed_qps": received / elapsed,
        "success_qps": len(successful) / elapsed,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)) if len(values) >= p99_min_samples else None,
        "p99_status": "measured"
        if len(values) >= p99_min_samples
        else "not_evaluable: below predeclared sample minimum",
        "timeout_rate": sum(r["error"] == "timeout" for r in rows) / len(rows),
        "error_rate": 1 - len(successful) / len(rows),
        "degraded_rate_successful": sum(r["degraded"] for r in successful) / len(successful)
        if successful
        else None,
        "empty_success_count": sum(r["result_count"] == 0 for r in successful),
        "latency_population": "All attempts, including errors/timeouts; end-to-end client elapsed",
        "qps_window": "Same monotonic first-send to last-completion closed-loop window; offered is attempted arrivals, not a configured open-loop arrival rate",
    }


def workloads(root, count, seed):
    source = root / "data/processed/wands"
    qids = read_json(source / "splits.json")["test"]
    queries = pd.read_parquet(source / "queries.parquet").set_index("query_id")["query"]
    model = read_json(root / "artifacts/recommendation/model.json")
    aliases = random.Random(seed).sample(sorted(model["aliases"]), count)
    return {
        "search": [
            {
                "request_id": qid,
                "path": "/api/v1/search",
                "payload": {"query": queries[qid], "algorithm": "hybrid", "top_k": 10},
            }
            for qid in qids[:count]
        ],
        "recommendation": [
            {
                "request_id": alias,
                "path": "/api/v1/recommend",
                "payload": {"user_id": alias, "algorithm": "adaptive_quota", "top_k": 10},
            }
            for alias in aliases
        ],
    }


async def run(root, base_url, pilot=False):
    config = read_json(root / "configs/benchmark.json")
    frozen = code_manifest(root)
    folder = root / "reports" / ("run_e5_http_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    folder.mkdir(parents=True)
    started = time.perf_counter()
    cases = workloads(root, config["unique_requests_per_workload"], config["seed"])
    timeout = config["timeout_seconds"]
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        trust_env=False,
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
    ) as client:
        response = await client.get("/readyz")
        response.raise_for_status()
        ready = response.json()
        runtime = ready["runtime"]
        if (runtime["worker_threads"], runtime["inference_threads"], runtime["response_cache"]) != (
            config["worker_threads"],
            config["inference_threads"],
            "off",
        ):
            raise ValueError(
                "Server thread/cache settings differ; launch scripts/serve_benchmark.py"
            )
        write_json(
            folder / "inputs.json",
            {
                "base_url": base_url,
                "ready": ready,
                "config": config,
                "code": frozen,
                "pilot": pilot,
                "hardware": {
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "logical_cpus": os.cpu_count(),
                },
                "cases": cases,
                "cohort_note": "96 distinct test query IDs and 96 train-history user aliases; repeated requests do not add new users/queries or quality evidence",
                "load_note": "Server loading excluded; readyz records startup duration; no response cache; OS/model already warm after preflight",
            },
        )
        expected, preflight = {}, []
        for domain, samples in cases.items():
            for case in samples:
                response = await client.post(case["path"], json=case["payload"])
                response.raise_for_status()
                body = response.json()
                expected[(domain, case["request_id"])] = result_signature(body)
            for case in samples[: config["warmup_requests"]]:
                response = await client.post(case["path"], json=case["payload"])
                response.raise_for_status()
            for case in samples[:12]:
                response = await client.post(case["path"], json={**case["payload"], "debug": True})
                response.raise_for_status()
                body = response.json()
                preflight.append(
                    {
                        "domain": domain,
                        "request_id": case["request_id"],
                        "trace_id": body["trace_id"],
                        "timings": body["timings"],
                    }
                )
        write_json(folder / "stage_traces.json", preflight)
        all_rows, summaries = [], []
        repeats = 1 if pilot else config["repeats"]
        concurrencies = [1, 4] if pilot else config["concurrency"]
        request_count = 32 if pilot else config["requests_per_cell"]
        for repeat in range(repeats):
            for domain, samples in cases.items():
                order = list(concurrencies)
                random.Random(config["seed"] + repeat).shuffle(order)
                for concurrency in order:
                    if time.perf_counter() - started > (
                        90 if pilot else config["run_budget_seconds"]
                    ):
                        write_json(folder / "status.json", {"status": "partial_budget_exhausted"})
                        raise RuntimeError(f"HTTP budget exceeded; partial run: {folder}")
                    sequence = [samples[i % len(samples)] for i in range(request_count)]
                    random.Random(config["seed"] + repeat).shuffle(sequence)
                    tasks = iter(enumerate(sequence))
                    rows, rss_samples = [], []
                    before = process_stats(runtime["pid"])
                    tick = time.perf_counter()
                    stopped = asyncio.Event()

                    async def monitor(stopped=stopped, rss_samples=rss_samples):
                        while not stopped.is_set():
                            rss_samples.append(process_stats(runtime["pid"])["rss_bytes"])
                            try:
                                await asyncio.wait_for(stopped.wait(), timeout=0.2)
                            except TimeoutError:
                                pass

                    async def worker(
                        tasks=tasks,
                        repeat=repeat,
                        domain=domain,
                        concurrency=concurrency,
                        tick=tick,
                        rows=rows,
                    ):
                        for index, case in tasks:
                            begin = time.perf_counter()
                            row = {
                                "repeat": repeat,
                                "domain": domain,
                                "concurrency": concurrency,
                                "request_index": index,
                                "request_id": case["request_id"],
                                "status_code": None,
                                "error": None,
                                "success": False,
                                "degraded": False,
                                "result_count": None,
                                "trace_id": None,
                                "start_offset_seconds": begin - tick,
                            }
                            try:
                                response = await client.post(case["path"], json=case["payload"])
                                row["status_code"] = response.status_code
                                if response.status_code == 200:
                                    body = response.json()
                                    row.update(
                                        trace_id=body.get("trace_id"),
                                        degraded=body.get("degraded", False),
                                        result_count=len(body.get("results", [])),
                                        server_latency_ms=body.get("latency_ms"),
                                    )
                                    signature = result_signature(body)
                                    row["result_signature"] = signature
                                    row["success"] = (
                                        bool(row["trace_id"])
                                        and signature == expected[(domain, case["request_id"])]
                                    )
                                    if not row["success"]:
                                        row["error"] = "contract_or_ranking_mismatch"
                                else:
                                    row["error"] = "http_error"
                            except httpx.TimeoutException:
                                row["error"] = "timeout"
                            except httpx.TransportError:
                                row["error"] = "transport_error"
                            except (json.JSONDecodeError, KeyError, TypeError):
                                row["error"] = "invalid_response_contract"
                            row["latency_ms"] = (time.perf_counter() - begin) * 1000
                            rows.append(row)

                    monitor_task = asyncio.create_task(monitor())
                    await asyncio.gather(*(worker() for _ in range(concurrency)))
                    elapsed = time.perf_counter() - tick
                    stopped.set()
                    await monitor_task
                    after = process_stats(runtime["pid"])
                    summary = {
                        "repeat": repeat,
                        "domain": domain,
                        "concurrency": concurrency,
                        **cell_summary(rows, elapsed, config["p99_min_samples"]),
                        "server_cpu_seconds": after["cpu_seconds"] - before["cpu_seconds"],
                        "observed_peak_rss_bytes": max(
                            [before["rss_bytes"], after["rss_bytes"], *rss_samples]
                        ),
                        "server_lifetime_peak_rss_bytes": after["process_lifetime_peak_rss_bytes"],
                        "rss_protocol": "200ms samples/current boundaries; lifetime peak is cumulative, not per-cell",
                    }
                    summaries.append(summary)
                    all_rows.extend(rows)
                    pd.DataFrame(all_rows).to_csv(folder / "requests.csv", index=False)
                    pd.DataFrame(summaries).to_csv(folder / "cells.csv", index=False)
                    print(
                        json.dumps(
                            {
                                k: summary[k]
                                for k in [
                                    "domain",
                                    "repeat",
                                    "concurrency",
                                    "success_qps",
                                    "p95_ms",
                                    "error_rate",
                                ]
                            }
                        ),
                        flush=True,
                    )
        if code_manifest(root) != frozen:
            raise ValueError("Source changed during HTTP benchmark")
        write_json(
            folder / "metrics.json",
            {
                "status": "completed",
                "pilot": pilot,
                "runtime": runtime,
                "cells": summaries,
                "wall_seconds": time.perf_counter() - started,
                "request_count": len(all_rows),
                "unique_case_count": sum(len(v) for v in cases.values()),
                "all_successful": all(r["success"] for r in all_rows),
                "source": frozen,
            },
        )
        if not pilot:
            write_json(
                root / "reports" / f"performance_http_{runtime['scan_mode']}_latest.json",
                {"run": folder.name},
            )
    return folder
