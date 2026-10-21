"""Exact scan equivalence and stage profiling on frozen real-data populations."""

import platform
import random
import time
from datetime import UTC, datetime

import pandas as pd

from joysearch.common import RecommendationContext, read_json, write_json
from joysearch.data.wands import file_hash
from joysearch.evaluation.metrics import judged_recall, ndcg
from joysearch.evaluation.process_stats import process_stats
from joysearch.evaluation.report import code_manifest
from joysearch.recommendation.pipeline import RecPipeline
from joysearch.search.exact_scan import CachedControllerDenseIndex
from joysearch.search.pipeline import SearchPipeline
from joysearch.telemetry import RequestTrace, instrument


def run(root):
    started = time.perf_counter()
    folder = root / "reports" / ("run_e5_local_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    folder.mkdir(parents=True)
    source = root / "data/processed/wands"
    splits = read_json(source / "splits.json")
    config = read_json(root / "configs/benchmark.json")
    frozen = code_manifest(root)
    assets = read_json(root / "reports/run_20261021T154234Z_c33ef5/config.json")[
        "model_artifact_sha256"
    ]
    if any(file_hash(root / path) != digest for path, digest in assets.items()):
        raise ValueError("Baseline model artifacts changed before profiling")
    write_json(
        folder / "inputs.json",
        {
            "code": frozen,
            "config": config,
            "splits": splits,
            "assets": assets,
            "hardware": platform.platform(),
            "protocol": "5 warmups; 3 paired alternating scan repeats per query; identical vectors, exact full catalog, Top100; test set reused for regression, no new untouched-generalization claim",
        },
    )
    loading = time.perf_counter()
    search = SearchPipeline.load(root, enable_ltr=False)
    rec = RecPipeline.load(root / "artifacts/recommendation")
    load_seconds = time.perf_counter() - loading
    optimized = CachedControllerDenseIndex(search.dense)
    instrument(search, rec)
    queries = pd.read_parquet(source / "queries.parquet").set_index("query_id")["query"]
    qrels = pd.read_parquet(source / "qrels.parquet")
    judgments = {qid: dict(zip(g.product_id, g.grade)) for qid, g in qrels.groupby("query_id")}
    users = random.Random(42).sample(sorted(rec.popularity.histories), 96)
    for qid, user in zip(splits["train"][:5], users[:5], strict=True):
        search.search(queries[qid], 10, "hybrid")
        rec.recommend(RecommendationContext(user), 10, "adaptive_quota")
    stages, rows, ranks, summaries = [], [], [], {}
    for split in ["validation", "test"]:
        for qi, qid in enumerate(splits[split]):
            if time.perf_counter() - started > config["run_budget_seconds"]:
                raise RuntimeError(f"Local benchmark budget exceeded; partial run: {folder}")
            vector = search.dense.encoder.encode([queries[qid]])[0]
            outputs = {}
            for repeat in range(3):
                pairs = [("reference", search.dense), ("cached_controller", optimized)]
                if (qi + repeat) % 2:
                    pairs.reverse()
                for name, model in pairs:
                    tick = time.perf_counter()
                    outputs[name] = model.rank_vector(vector, 100)
                    rows.append(
                        {
                            "split": split,
                            "query_id": qid,
                            "repeat": repeat,
                            "variant": name,
                            "latency_ms": (time.perf_counter() - tick) * 1000,
                        }
                    )
                assert [(c.item_id, c.score) for c in outputs["reference"]] == [
                    (c.item_id, c.score) for c in outputs["cached_controller"]
                ], f"Exact scan differs: {qid}"
            for name, pool in outputs.items():
                ids = [c.item_id for c in pool]
                ranks.extend(
                    {
                        "split": split,
                        "query_id": qid,
                        "variant": name,
                        "rank": c.rank,
                        "item_id": c.item_id,
                        "score": c.score,
                    }
                    for c in pool
                )
                rows.append(
                    {
                        "split": split,
                        "query_id": qid,
                        "repeat": -1,
                        "variant": name,
                        "ndcg_at_10": ndcg(ids, judgments.get(qid, {})),
                        "judged_recall_at_100": judged_recall(ids, judgments.get(qid, {})),
                    }
                )
            if split == "validation":
                with RequestTrace(True) as trace:
                    search.search(queries[qid], 10, "query_hybrid")
                stages.extend(
                    {"request_id": qid, "domain": "search", **span} for span in trace.spans
                )
                with RequestTrace(True) as trace:
                    rec.recommend(RecommendationContext(users[qi]), 10, "adaptive_quota")
                stages.extend(
                    {"request_id": str(qi), "domain": "recommendation", **span}
                    for span in trace.spans
                )
        frame = pd.DataFrame(rows)
        subset = frame[frame.split.eq(split)]
        summaries[split] = {
            name: {
                "sample_count": len(splits[split]),
                "scan_sample_count": int(group.latency_ms.notna().sum()),
                "scan_p50_ms": float(group.latency_ms.median()),
                "scan_p95_ms": float(group.latency_ms.quantile(0.95)),
                "ndcg_at_10": float(group.ndcg_at_10.mean()),
                "judged_recall_at_100": float(group.judged_recall_at_100.mean()),
                "exact_ids_and_scores_equal": True,
            }
            for name, group in subset.groupby("variant")
        }
        pd.DataFrame(rows).to_csv(folder / "paired_scans.csv", index=False)
        pd.DataFrame(ranks).to_parquet(folder / "rankings.parquet", index=False)
        if split == "validation":
            write_json(
                folder / "selection.json",
                {
                    "scope": "No default promotion; fixed exact-scan experiment",
                    "validation": summaries["validation"],
                    "code": frozen,
                },
            )
        if code_manifest(root) != frozen:
            raise ValueError("Source changed during local benchmark")
    pd.DataFrame(stages).to_csv(folder / "stages.csv", index=False)
    stage_frame = pd.DataFrame(stages)
    stage_frame.groupby("stage").elapsed_ms.agg(["count", "median", "mean", "max"]).to_csv(
        folder / "stage_summary.csv"
    )
    matrix = search.index.weights
    metrics = {
        "status": "completed",
        "phases": summaries,
        "load_seconds": load_seconds,
        "wall_seconds": time.perf_counter() - started,
        "process": process_stats(),
        "catalog_sizes": {
            "search": len(search.index.item_ids),
            "recommendation": len(rec.popularity.items),
        },
        "bm25_csc": {
            "shape": list(matrix.shape),
            "nnz": matrix.nnz,
            "canonical": matrix.has_canonical_format,
            "indptr_dtype": str(matrix.indptr.dtype),
            "indices_dtype": str(matrix.indices.dtype),
            "data_dtype": str(matrix.data.dtype),
            "array_bytes": matrix.indptr.nbytes + matrix.indices.nbytes + matrix.data.nbytes,
        },
        "dense_vector_bytes": search.dense.embeddings.nbytes,
        "artifact_disk_bytes": {
            directory: sum(
                p.stat().st_size for p in (root / "artifacts" / directory).rglob("*") if p.is_file()
            )
            for directory in [
                "search",
                "search_dense",
                "recommendation",
                "recommendation_two_tower",
            ]
        },
        "build": "Existing immutable artifacts reused; no build/re-encoding/training performed",
        "ann": "not_run: profile identifies library enumeration overhead; exact vectors retained",
        "stage_protocol": "Inclusive nested spans with parent IDs; do not sum parent and child elapsed time",
    }
    write_json(folder / "metrics.json", metrics)
    write_json(root / "reports/performance_local_latest.json", {"run": folder.name})
    return folder
