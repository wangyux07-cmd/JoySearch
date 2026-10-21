"""Frozen-budget query ablations with validation selection and E1 recall diagnostics."""

import platform
import time
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd

from joysearch.common import read_json, write_json
from joysearch.data.wands import file_hash
from joysearch.evaluation.metrics import judged_recall, judgment_coverage, ndcg, reciprocal_rank
from joysearch.evaluation.recall_eval import (
    CHANNEL_SCHEMA,
    UNION_SCHEMA,
    candidate_variants,
    prepare_channels,
    request_metrics,
    trace_rows,
)
from joysearch.evaluation.report import code_manifest
from joysearch.evaluation.search_eval import latency_summary
from joysearch.search.fusion import reciprocal_rank_fusion
from joysearch.search.pipeline import SearchPipeline
from joysearch.search.ranker import retrieval_signature

METRICS = ["ndcg_at_10", "mrr_at_10", "judged_recall_at_100", "judgment_coverage_at_10"]


def protocol_signature(root):
    config = read_json(root / "configs/query.json")
    config.pop("soft_weight")
    return {
        "query_config_without_selected_weight": config,
        "sources": {
            p: file_hash(root / p)
            for p in [
                "src/joysearch/search/query.py",
                "src/joysearch/search/pipeline.py",
                "src/joysearch/evaluation/query_eval.py",
                "data/processed/wands/queries.parquet",
                "data/processed/wands/qrels.parquet",
                "data/processed/wands/splits.json",
                "data/processed/wands/manifest.json",
                "artifacts/search/manifest.json",
                "artifacts/search_dense/manifest.json",
            ]
        },
    }


def run(root, split="validation", selection=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    started, cpu = time.perf_counter(), time.process_time()
    config = read_json(root / "configs/query.json")
    signature = protocol_signature(root)
    selected = None
    if split == "test":
        if selection is None:
            raise ValueError("Test requires --selection from a completed full validation run")
        selected = read_json(selection)
        if selected["protocol_signature"] != signature or selected["split"] != "validation":
            raise ValueError("Query protocol changed after validation; run validation again")
        if config["soft_weight"] != selected["selected_weight"]:
            raise ValueError("Serving soft_weight must equal the validation-selected weight")
    weights = (
        config["validation_weights"] if split == "validation" else [selected["selected_weight"]]
    )
    run_id = "run_e2_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    folder = root / "reports" / run_id
    folder.mkdir(parents=True)
    source = root / "data/processed/wands"
    splits, manifest = read_json(source / "splits.json"), read_json(source / "manifest.json")
    if splits != manifest["query_splits"] or any(
        set(splits[a]) & set(splits[b])
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("Frozen WANDS splits changed or overlap")
    model = SearchPipeline.load(root, enable_ltr=False)
    if model.index.manifest["data_manifest"]["sources"] != manifest["sources"]:
        raise ValueError("Search index/source mismatch")
    queries = pd.read_parquet(source / "queries.parquet").set_index("query_id")["query"]
    qrels = pd.read_parquet(source / "qrels.parquet")
    judgments = {qid: dict(zip(g.product_id, g.grade)) for qid, g in qrels.groupby("query_id")}
    write_json(
        folder / "frozen_inputs.json",
        {
            "protocol_signature": signature,
            "config": config,
            "code_sha256": code_manifest(root),
            "retrieval_signature": retrieval_signature(root),
            "query_ids": splits[split],
            "selection": str(selection) if selection else None,
            "selection_sha256": file_hash(selection) if selection else None,
            "split": split,
            "budget": 100,
            "display_k": 10,
            "latency_protocol": "5 warmups; sequential shared base retrieval; variant elapsed includes shared base; added latency excludes it",
            "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        },
    )
    for qid in splits[split][:5]:
        model.query_candidates(queries[qid])
    rows, plans, recall_rows, overlaps, contributions = [], [], [], [], []
    diagnostic_config = {"budget": 100, "display_k": 10, "rrf_k": model.rrf_k}
    with (
        pq.ParquetWriter(folder / "channel_candidates.parquet", CHANNEL_SCHEMA) as channel_writer,
        pq.ParquetWriter(folder / "union_candidates.parquet", UNION_SCHEMA) as union_writer,
    ):
        for qid in splits[split]:
            query, judged = queries[qid], judgments.get(qid, {})
            tick = time.perf_counter()
            original, base, plan = model.query_candidates(query, "original")
            base_ms = (time.perf_counter() - tick) * 1000
            plans.append({"query_id": qid, **plan.to_dict()})
            variants = {"original": (original, base_ms, 0.0)}
            tick = time.perf_counter()
            normalized, _, _ = model.query_candidates(query, "normalize", plan=plan)
            variants["normalize"] = (normalized, (time.perf_counter() - tick) * 1000, None)
            tick = time.perf_counter()
            channels = dict(base)
            if plan.rewrites:
                channels["rewrite_bm25"] = model.index.retrieve(plan.rewrites[0], 100)
            expanded = reciprocal_rank_fusion(channels, 100, model.rrf_k)
            expansion_ms = (time.perf_counter() - tick) * 1000
            for weight in weights:
                for mode, pool, extra_ms in [
                    ("extraction", original, 0.0),
                    ("expansion", expanded, expansion_ms),
                ]:
                    tick = time.perf_counter()
                    ranked = model.catalog_attributes.rerank(plan, pool, weight)
                    added = extra_ms + (time.perf_counter() - tick) * 1000
                    variants[f"{mode}:{weight:g}"] = (ranked, base_ms + added, added)
            for name, (candidates, elapsed, added) in variants.items():
                ranking = [c.item_id for c in candidates]
                if len(ranking) > 100 or len(ranking) != len(set(ranking)):
                    raise ValueError(f"Invalid fixed-budget candidates for {qid}/{name}")
                rows.append(
                    {
                        "query_id": qid,
                        "variant": name,
                        "ndcg_at_10": ndcg(ranking, judged),
                        "mrr_at_10": reciprocal_rank(ranking, judged),
                        "judged_recall_at_100": judged_recall(ranking, judged),
                        "judgment_coverage_at_10": judgment_coverage(ranking, judged),
                        "latency_ms": elapsed,
                        "added_latency_ms": added,
                        "candidate_count": len(ranking),
                    }
                )
            # An inactive optional channel abstains; all ablations retain the full cohort.
            channels.setdefault("rewrite_bm25", [])
            eligible = prepare_channels(channels, set(), 100, "wands_search")
            fused, _ = candidate_variants(eligible, diagnostic_config)
            r, o, c = request_metrics(
                qid, channels, eligible, fused, set(), judged, diagnostic_config, False
            )
            recall_rows.extend(r)
            overlaps.extend(o)
            contributions.extend(c)
            channel_trace, union_trace = trace_rows(qid, channels, eligible, fused, set(), 10)
            channel_writer.write_table(pa.Table.from_pylist(channel_trace, schema=CHANNEL_SCHEMA))
            union_writer.write_table(pa.Table.from_pylist(union_trace, schema=UNION_SCHEMA))
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "per_query.csv", index=False)
    write_json(folder / "query_plans.json", plans)
    for name, values in [
        ("recall_per_query", recall_rows),
        ("channel_overlap", overlaps),
        ("channel_contribution", contributions),
    ]:
        pd.DataFrame(values).to_csv(folder / f"{name}.csv", index=False)
    summaries, differences = {}, []
    baseline = frame[frame.variant.eq("original")].set_index("query_id")
    for name, group in frame.groupby("variant", sort=False):
        summaries[name] = {key: float(group[key].mean()) for key in METRICS}
        summaries[name]["latency"] = latency_summary(group.latency_ms)
        summaries[name]["latency"]["protocol"] = (
            "Shared sequential base retrieval; see frozen_inputs.json"
        )
        added = group.added_latency_ms.dropna()
        summaries[name]["added_latency"] = latency_summary(added) if len(added) else None
        for row in group.itertuples():
            differences.append(
                {
                    "query_id": row.query_id,
                    "variant": name,
                    **{
                        f"delta_{key}": getattr(row, key) - baseline.loc[row.query_id, key]
                        for key in METRICS
                    },
                }
            )
    differences = pd.DataFrame(differences)
    differences.to_csv(folder / "query_deltas.csv", index=False)
    comparison = []
    for name, metrics in summaries.items():
        for key in METRICS:
            delta = metrics[key] - summaries["original"][key]
            values = differences[differences.variant.eq(name)][f"delta_{key}"]
            comparison.append(
                {
                    "variant": name,
                    "metric": key,
                    "baseline": summaries["original"][key],
                    "value": metrics[key],
                    "delta": delta,
                    "wins": int((values > 1e-12).sum()),
                    "losses": int((values < -1e-12).sum()),
                    "ties": int((values.abs() <= 1e-12).sum()),
                }
            )
    pd.DataFrame(comparison).to_csv(folder / "comparisons.csv", index=False)
    recall_frame = pd.DataFrame(recall_rows)
    counts = recall_frame.groupby("variant", sort=False).request_id.nunique()
    if not counts.eq(len(splits[split])).all():
        raise ValueError(
            "Channel ablations must retain every held-out query, including inactive rewrites"
        )
    channel_summary = recall_frame.groupby("variant", sort=False).mean(numeric_only=True)
    channel_summary["sample_count"] = counts
    channel_summary.to_csv(folder / "channel_ablation.csv")
    if split == "validation":
        winner = max(weights, key=lambda w: (summaries[f"expansion:{w:g}"]["ndcg_at_10"], -w))
        write_json(
            folder / "selection.json",
            {
                "split": split,
                "sample_count": len(splits[split]),
                "protocol_signature": signature,
                "selected_weight": winner,
                "selection_rule": "Expansion full-validation NDCG@10; ties prefer smaller weight",
                "all_attempts": [f"expansion:{w:g}" for w in weights],
            },
        )
    write_json(
        folder / "annotation_metrics.json",
        {
            "status": "not_evaluable",
            "reason": "No independent human annotations supplied",
            "annotator_count": 0,
            "human_labeled_sample_count": 0,
            "slot_precision": None,
            "slot_recall": None,
            "slot_f1": None,
            "negation_constraint_preservation": None,
            "unsupported_constraint_accuracy": None,
            "synthetic_challenges": "Separately tested in tests/test_query.py; not real-data accuracy",
        },
    )
    result = {
        "status": "completed",
        "run_id": run_id,
        "split": split,
        "sample_count": len(splits[split]),
        "catalog_size": len(model.index.item_ids),
        "metrics": summaries,
        "rewrite_query_count": sum(bool(p["rewrites"]) for p in plans),
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu,
        "limitations": [
            "Sparse qrels; unjudged is not a verified negative",
            "Human parser quality not evaluable",
            "Soft attributes include aggregated variants; no hard filtering",
            "Query score is not calibrated confidence",
        ],
    }
    write_json(folder / "metrics.json", result)
    write_json(root / f"reports/query_{split}_latest.json", {"run_id": run_id})
    print({"run_id": run_id, "split": split, "metrics": summaries}, flush=True)
    return result
