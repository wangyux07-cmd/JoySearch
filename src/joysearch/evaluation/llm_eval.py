"""Query-disjoint, fixed-budget CPU LLM diagnostics with explicit abstentions."""

import ctypes
import json
import os
import platform
import random
import time
from collections import Counter
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd

from joysearch.common import read_json, write_json
from joysearch.data.wands import file_hash
from joysearch.evaluation.metrics import judged_recall, judgment_coverage, ndcg, reciprocal_rank
from joysearch.evaluation.report import code_manifest
from joysearch.evaluation.search_eval import latency_summary
from joysearch.search.fusion import reciprocal_rank_fusion
from joysearch.search.llm import (
    label_messages,
    product_document,
    rewrite_messages,
    validate_label,
    validate_rewrite,
)
from joysearch.search.local_model import LocalGenerator
from joysearch.search.pipeline import SearchPipeline

RANK_METRICS = ["ndcg_at_10", "mrr_at_10", "judged_recall_at_100", "judgment_coverage_at_10"]


def diagnostic_latency(values):
    summary = latency_summary(values)
    summary["protocol"] = (
        "Sequential diagnostic; load excluded; no warmup discarded; first generation cold; "
        "cache replay retains original measured generation cost; not HTTP latency"
    )
    return summary


def classification_metrics(gold, predictions):
    """Abstentions count as false negatives; kappa uses accepted pairs only."""
    matrix = np.zeros((3, 4), dtype=int)
    for actual, predicted in zip(gold, predictions, strict=True):
        matrix[actual, 3 if predicted is None else predicted] += 1
    if not len(gold):
        raise ValueError("Cannot evaluate an empty label cohort")
    f1 = []
    for grade in range(3):
        tp = matrix[grade, grade]
        denominator = matrix[grade, :].sum() + matrix[:, grade].sum()
        f1.append(float(2 * tp / denominator) if denominator else 0.0)
    accepted = matrix[:, :3]
    count = int(accepted.sum())
    kappa = None
    if count:
        weights = (np.arange(3)[:, None] - np.arange(3)[None, :]) ** 2 / 4
        expected = np.outer(accepted.sum(axis=1), accepted.sum(axis=0)) / count
        denominator = float((weights * expected).sum())
        if denominator:
            kappa = float(1 - (weights * accepted).sum() / denominator)
    return {
        "sample_count": len(gold),
        "class_counts": matrix.sum(axis=1).tolist(),
        "confusion_gold_rows_predicted_0_1_2_abstain_columns": matrix.tolist(),
        "macro_f1_full_cohort_abstentions_as_fn": float(np.mean(f1)),
        "per_class_f1": f1,
        "accepted_count": count,
        "abstention_count": len(gold) - count,
        "quadratic_weighted_kappa_accepted_only": kappa,
        "kappa_null_reason": "no accepted pairs or zero expected disagreement"
        if kappa is None
        else None,
        "accepted_accuracy": float(np.trace(accepted) / count) if count else None,
        "manual_review_completed": 0,
        "manual_review_queue_count": len(gold) - count,
    }


def protocol_signature(root):
    paths = [
        "configs/llm.json",
        "configs/query.json",
        "configs/dense.json",
        "src/joysearch/search/llm.py",
        "src/joysearch/search/local_model.py",
        "src/joysearch/search/query.py",
        "src/joysearch/search/pipeline.py",
        "src/joysearch/search/bm25.py",
        "src/joysearch/search/dense.py",
        "src/joysearch/search/fusion.py",
        "src/joysearch/evaluation/llm_eval.py",
        "src/joysearch/evaluation/metrics.py",
        "scripts/evaluate_llm.py",
        "data/processed/wands/products.parquet",
        "data/processed/wands/queries.parquet",
        "data/processed/wands/qrels.parquet",
        "data/processed/wands/splits.json",
        "data/processed/wands/manifest.json",
        "artifacts/search/manifest.json",
        "artifacts/search_dense/manifest.json",
        "artifacts/models/Qwen2.5-0.5B-Instruct/manifest.json",
    ]
    return {path: file_hash(root / path) for path in paths}


def process_memory():
    if os.name != "nt":
        return {"status": "not_measured", "reason": "Windows working-set sampler unavailable"}

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in [
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            ]
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "status": "measured",
        "rss_bytes": counters.WorkingSetSize,
        "process_lifetime_peak_rss_bytes": counters.PeakWorkingSetSize,
    }


def ranked_candidates(model, base, rewrites, query):
    channels = dict(base)
    if rewrites and rewrites[0].strip().lower() != query.strip().lower():
        channels["rewrite_bm25"] = model.index.retrieve(rewrites[0], 100)
    return reciprocal_rank_fusion(channels, 100, model.rrf_k)


def rewrite_metrics(records):
    typed = [r for r in records if r["schema_valid"]]
    preservation = [
        r
        for r in typed
        if not any(
            e.startswith("constraint_") or e == "unsupported_constraint_dropped"
            for e in r["errors"]
        )
    ]
    unsupported = [
        r
        for r in typed
        if any(
            e.startswith("unsupported_")
            and e != "unsupported_constraint_dropped"
            or e in {"unverified_evidence_id", "missing_evidence_id"}
            for e in r["errors"]
        )
    ]
    return {
        "sample_count": len(records),
        "json_valid_rate": sum(r["json_valid"] for r in records) / len(records),
        "typed_json_valid_rate": len(typed) / len(records),
        "constraint_preservation_typed_only": len(preservation) / len(typed) if typed else None,
        "unsupported_fact_rate_typed_only": len(unsupported) / len(typed) if typed else None,
        "typed_assessable_count": len(typed),
        "fallback_rate": sum(bool(r["errors"]) for r in records) / len(records),
        "fallback_reasons": dict(Counter(e for r in records for e in r["errors"])),
        "guard_scope": "Literal content/brand token, numeric multiset, value-unit pairs, currency, negation, finite attributes and cited class-term support; not semantic intent accuracy",
        "human_same_intent_quality": None,
        "human_quality_status": "not_evaluable",
        "human_quality_reason": "No independent human intent annotations collected",
        "human_annotators": 0,
        "human_sample_count": 0,
    }


def run(root, split="validation", selection=None):
    started, cpu = time.perf_counter(), time.process_time()
    config = read_json(root / "configs/llm.json")
    signature = protocol_signature(root)
    if split == "test":
        if selection is None:
            raise ValueError("Test requires --selection from completed validation")
        gate = read_json(selection)
        if (
            gate["status"] != "completed"
            or gate["split"] != "validation"
            or gate["protocol_signature"] != signature
        ):
            raise ValueError("Protocol differs from completed validation; do not tune on test")
    source = root / "data/processed/wands"
    splits = read_json(source / "splits.json")
    manifest = read_json(source / "manifest.json")
    if splits != manifest["query_splits"] or any(
        set(splits[a]) & set(splits[b])
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("Frozen WANDS query splits changed or overlap")
    qids = random.Random(config["seed"]).sample(splits[split], config["diagnostic_query_count"])
    label_qids = qids[: config["label_query_count"]]
    folder = (
        root
        / "reports"
        / ("run_e4_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6])
    )
    folder.mkdir(parents=True)
    write_json(
        folder / "frozen_inputs.json",
        {
            "split": split,
            "query_ids": qids,
            "label_query_ids": label_qids,
            "config": config,
            "protocol_signature": signature,
            "code_sha256": code_manifest(root),
            "selection": str(selection) if selection else None,
            "selection_sha256": file_hash(selection) if selection else None,
            "scope": "Seeded 24-query diagnostic, full 42994-item catalog; not full 96-query headline",
            "budget": 100,
            "display_k": 10,
            "pid": os.getpid(),
            "label_sampling": "One pair per available gold grade per query, stratified diagnostic; gold never in model messages",
            "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        },
    )
    print(str(folder), flush=True)
    generator = LocalGenerator(root)
    model = SearchPipeline.load(root, enable_ltr=False)
    if model.query_understanding.config["soft_weight"] != 0:
        raise ValueError("This diagnostic freezes the E2-selected zero attribute weight")
    if model.index.manifest["data_manifest"]["sources"] != manifest["sources"]:
        raise ValueError("Search index/source mismatch")
    queries = pd.read_parquet(source / "queries.parquet").set_index("query_id")["query"]
    products = model.index.products.fillna("").set_index("product_id", drop=False)
    qrels = pd.read_parquet(source / "qrels.parquet")
    judgments = {qid: dict(zip(g.product_id, g.grade)) for qid, g in qrels.groupby("query_id")}
    majority = int(qrels[qrels.query_id.isin(splits["train"])].grade.value_counts().idxmax())
    rows, records, labels, traces = [], [], [], []

    def generate(messages, limit):
        if time.perf_counter() - started > config["run_budget_seconds"]:
            write_json(
                folder / "metrics.json",
                {
                    "status": "partial_budget_exhausted",
                    "completed_queries": len(rows) // 4,
                    "resume": "Rerun the same command; exact raw-response cache is reused. No completed pointer published.",
                },
            )
            raise RuntimeError(f"LLM run budget exhausted; partial results at {folder}")
        return generator.generate(messages, limit)

    def save():
        write_json(folder / "raw_records.json", records)
        write_json(folder / "label_predictions_and_gold.json", labels)
        pd.DataFrame(rows).to_csv(folder / "per_query.csv", index=False)
        pd.DataFrame(traces).to_parquet(folder / "rankings.parquet", index=False)

    for qi, qid in enumerate(qids):
        query = queries[qid]
        tick = time.perf_counter()
        original, base, plan = model.query_candidates(query, "original")
        base_ms = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        rules = ranked_candidates(model, base, plan.rewrites, query)
        rules_ms = (time.perf_counter() - tick) * 1000
        variants = {"original": (original, base_ms), "rules_only": (rules, base_ms + rules_ms)}
        docs = [
            product_document(products.loc[c.item_id].to_dict(), config)
            for c in original[: config["evidence_k"]]
        ]
        for name, evidence in [("llm_plain", []), ("llm_grounded", docs)]:
            response = generate(
                rewrite_messages(query, evidence, model.query_understanding),
                config["rewrite_max_new_tokens"],
            )
            tick = time.perf_counter()
            proposal, errors = validate_rewrite(
                query, response, evidence, model.query_understanding
            )
            rewrites = plan.rewrites if errors else proposal.rewrites
            pool = rules if errors else ranked_candidates(model, base, rewrites, query)
            guard_ms = (time.perf_counter() - tick) * 1000
            try:
                json.loads(response["raw"])
                valid_json = True
            except json.JSONDecodeError:
                valid_json = False
            records.append(
                {
                    "task": "rewrite",
                    "query_id": qid,
                    "mode": name,
                    "response": response,
                    "schema_valid": proposal is not None,
                    "json_valid": valid_json,
                    "errors": errors,
                    "proposal": proposal.model_dump() if proposal else None,
                    "applied_rewrites": rewrites,
                    "fallback": bool(errors),
                    "guard_and_extra_retrieval_ms": guard_ms,
                }
            )
            variants[name] = (
                pool,
                base_ms + response["wall_ms"] + guard_ms + (rules_ms if errors else 0),
            )
            save()
        for name, (pool, elapsed) in variants.items():
            ranking = [c.item_id for c in pool]
            if (
                len(ranking) != len(set(ranking))
                or len(ranking) > 100
                or any(c.catalog_id != "wands_search" for c in pool)
            ):
                raise ValueError(f"Invalid fixed-budget candidates: {qid}/{name}")
            judged = judgments.get(qid, {})
            rows.append(
                {
                    "query_id": qid,
                    "variant": name,
                    "ndcg_at_10": ndcg(ranking, judged),
                    "mrr_at_10": reciprocal_rank(ranking, judged),
                    "judged_recall_at_100": judged_recall(ranking, judged),
                    "judgment_coverage_at_10": judgment_coverage(ranking, judged),
                    "latency_ms": elapsed,
                    "candidate_count": len(ranking),
                }
            )
            traces.extend(
                {"query_id": qid, "variant": name, "product_id": pid, "rank": rank}
                for rank, pid in enumerate(ranking, 1)
            )
        if qid in label_qids:
            for grade in [0, 1, 2]:
                group = qrels[qrels.query_id.eq(qid) & qrels.grade.eq(grade)].sort_values(
                    "product_id"
                )
                if group.empty:
                    continue
                pair = group.iloc[
                    random.Random(config["seed"] + qi * 3 + grade).randrange(len(group))
                ]
                document = product_document(products.loc[pair.product_id].to_dict(), config)
                response = generate(label_messages(query, document), config["label_max_new_tokens"])
                proposal, errors = validate_label(response, document)
                predicted = (
                    proposal.grade if proposal and not errors and not proposal.uncertain else None
                )
                records.append(
                    {
                        "task": "label",
                        "query_id": qid,
                        "product_id": pair.product_id,
                        "response": response,
                        "errors": errors,
                        "proposal": proposal.model_dump() if proposal else None,
                    }
                )
                labels.append(
                    {
                        "query_id": qid,
                        "product_id": pair.product_id,
                        "gold": grade,
                        "predicted": predicted,
                        "train_majority_prediction": majority,
                        "cache_key": response["cache_key"],
                    }
                )
                save()
        save()
        print(
            json.dumps(
                {
                    "query": qi + 1,
                    "of": len(qids),
                    "labels": len(labels),
                    "wall_seconds": round(time.perf_counter() - started, 2),
                }
            ),
            flush=True,
        )
    frame = pd.DataFrame(rows)
    summaries = {
        name: {
            **{k: float(group[k].mean()) for k in RANK_METRICS},
            "latency": diagnostic_latency(group.latency_ms),
        }
        for name, group in frame.groupby("variant", sort=False)
    }
    baseline = frame[frame.variant.eq("rules_only")].set_index("query_id")
    differences = []
    for row in rows:
        differences.append(
            {
                "query_id": row["query_id"],
                "variant": row["variant"],
                **{
                    k + "_delta_vs_rules": row[k] - baseline.loc[row["query_id"], k]
                    for k in RANK_METRICS
                },
            }
        )
    pd.DataFrame(differences).to_csv(folder / "comparison.csv", index=False)
    gold = [r["gold"] for r in labels]
    generation = [r["response"] for r in records]
    metrics = {
        "status": "completed",
        "split": split,
        "query_count": len(qids),
        "catalog_size": len(products),
        "scope": "Seeded diagnostic only; no full-headline or human semantic-quality claim",
        "candidate_universe": "All WANDS catalog product IDs; original BM25/Dense channels retained",
        "seed": config["seed"],
        "budget": 100,
        "k": 10,
        "queries_without_positive_qrels": sum(
            not any(g >= 1 for g in judgments.get(qid, {}).values()) for qid in qids
        ),
        "judgment_policy": "Unjudged items contribute zero gain; sparse judgments do not establish true corpus recall",
        "latency_protocol": "Sequential shared original retrieval; load excluded; no warmup discarded, first generation includes cold execution; per-variant sums retain original generation costs on cache replay; not HTTP latency",
        "rankings": summaries,
        "rewrites": {
            mode: rewrite_metrics([r for r in records if r.get("mode") == mode])
            for mode in ["llm_plain", "llm_grounded"]
        },
        "labels": {
            "llm": classification_metrics(gold, [r["predicted"] for r in labels]),
            "train_majority": classification_metrics(gold, [majority] * len(gold)),
        },
        "generation": {
            "requests": len(generation),
            "cache_hits": sum(r["cache_hit"] for r in generation),
            "input_tokens": sum(r["input_tokens"] for r in generation),
            "output_tokens": sum(r["output_tokens"] for r in generation),
            "measured_original_generation_latency": diagnostic_latency(
                [r["wall_ms"] for r in generation]
            ),
            "original_cpu_seconds": sum(r["cpu_seconds"] for r in generation),
            "new_generation_cpu_seconds": sum(
                r["cpu_seconds"] for r in generation if not r["cache_hit"]
            ),
            "note": "Cache hits retain original measured cost; modeled sequential variant latency is not HTTP latency or QPS",
        },
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu,
            "model_load_seconds": generator.load_seconds,
            **process_memory(),
        },
        "realtime_eligible": False,
        "decision": "Offline diagnostics only: train pilot exceeds 1000 ms realtime budget; defaults unchanged",
        "protocol_signature": signature,
    }
    if protocol_signature(root) != signature:
        raise ValueError(f"Protocol changed during evaluation; results not published: {folder}")
    write_json(folder / "metrics.json", metrics)
    if split == "validation":
        write_json(
            folder / "selection.json",
            {
                "status": "completed",
                "split": split,
                "protocol_signature": signature,
                "decision": "Freeze one configuration for diagnostic test only; no realtime promotion or quality selection",
                "run": str(folder),
            },
        )
    write_json(root / "reports" / f"llm_{split}_latest.json", {"run": folder.name})
    return folder
