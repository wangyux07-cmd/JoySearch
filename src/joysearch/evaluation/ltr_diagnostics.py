"""Validation-only diagnosis of judged-only LambdaMART model selection."""

import importlib.metadata
import platform
import subprocess
import time
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, record_evaluation

from joysearch.common import read_json, write_json
from joysearch.data.wands import file_hash
from joysearch.evaluation.metrics import judged_recall, judgment_coverage, ndcg, reciprocal_rank
from joysearch.evaluation.report import code_manifest
from joysearch.search.features import FEATURE_NAMES
from joysearch.search.ranker import LambdaMARTRanker, query_groups, retrieval_signature

CHECKPOINTS = [1, 5, 10, 20, 29, 50, 100, 200, 300]
FEATURE_GROUPS = {
    "all": FEATURE_NAMES,
    "without_catalog": [
        f for f in FEATURE_NAMES if f not in {"log_review_count", "average_rating"}
    ],
    "without_text": [
        f
        for f in FEATURE_NAMES
        if f not in {"title_overlap", "category_overlap", "query_length", "title_length"}
    ],
    "retrieval_only": FEATURE_NAMES[:7],
}


def score_pool(pool, scores, qrels, *, variant, iteration=0, depth=100):
    """Keep unjudged candidates in rankings; only conventional metrics assign zero gain."""
    rows = []
    for qid, group in pool.assign(prediction=scores).groupby("query_id", sort=False):
        group = group.head(depth)
        order = np.lexsort((group.product_id.to_numpy(), -group.prediction.to_numpy()))
        ranking = group.product_id.to_numpy()[order].tolist()
        labels = qrels[qid]
        rows.append(
            {
                "query_id": qid,
                "variant": variant,
                "iteration": iteration,
                "candidate_depth": depth,
                "ndcg_at_10": ndcg(ranking, labels),
                "mrr_at_10": reciprocal_rank(ranking, labels),
                "judged_recall_at_100": judged_recall(ranking, labels),
                "judgment_coverage_at_10": judgment_coverage(ranking, labels),
                "candidate_judgment_coverage": float(group.grade.notna().mean()),
            }
        )
    return pd.DataFrame(rows)


def run(root):
    started = time.perf_counter()
    source = root / "data/processed/wands"
    assets = root / "artifacts/search_ltr"
    splits = read_json(source / "splits.json")
    manifest = read_json(assets / "manifest.json")
    cache = read_json(assets / "candidate_cache.json")
    signature = retrieval_signature(root)
    old = LambdaMARTRanker.load(root, signature)
    if cache["signature"] != manifest["training_signature"]:
        raise ValueError("LTR candidate cache training signature mismatch")
    input_paths = [assets / f"{s}.parquet" for s in ["train", "validation"]]
    if cache["sha256"] != {p.name: file_hash(p) for p in input_paths}:
        raise ValueError("LTR candidate cache checksum mismatch")
    if manifest["training_signature"]["files"] != {
        n: file_hash(source / n)
        for n in ["queries.parquet", "qrels.parquet", "splits.json", "manifest.json"]
    }:
        raise ValueError("LTR cached training data no longer matches source")
    if any(
        set(splits[a]) & set(splits[b])
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("Query splits overlap")
    pools = [pd.read_parquet(p) for p in input_paths]
    for split, pool in zip(["train", "validation"], pools):
        query_groups(pool, splits[split])
        if set(pool.query_id) != set(splits[split]):
            raise ValueError("Candidate cache does not cover the complete split")
    training, validation = [p.dropna(subset=["grade"]) for p in pools]
    groups = [
        query_groups(f, splits[s]) for f, s in zip([training, validation], ["train", "validation"])
    ]
    # Restrict to validation immediately; test labels never enter a diagnostic calculation.
    labels = pd.read_parquet(
        source / "qrels.parquet", filters=[("query_id", "in", splits["validation"])]
    )
    qrels = {q: dict(zip(g.product_id, g.grade.astype(int))) for q, g in labels.groupby("query_id")}
    run_id = (
        "run_ltr_diagnostics_"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid4().hex[:6]
    )
    folder = root / "reports" / run_id
    folder.mkdir(parents=True)
    protocol = {
        "split": "validation",
        "seed": 42,
        "sample_count": len(splits["validation"]),
        "eligible_count": len(splits["validation"]),
        "catalog_size": read_json(root / "artifacts/search/manifest.json")["catalog_size"],
        "candidate_universe": "Frozen full-catalog BM25/Dense RRF Top100 cache",
        "k": 10,
        "candidate_budget": 100,
        "checkpoints": CHECKPOINTS,
        "feature_groups": FEATURE_GROUPS,
        "model_config": manifest["config"]["model"],
        "selection": "Diagnostic only: full validation NDCG@10, earlier checkpoint breaks ties; no promotion",
        "pool_probe": "Top50 truncation of the same cached Hybrid pool; diagnostic only, not a fair B100 comparison",
        "labels": "Train judged grades 0/1/2 only; unjudged excluded from fitting, Partial remains positive",
        "metric_policy": "Full-qrel IDCG; unjudged zero gain for conventional metrics, never training negatives",
        "test_access": False,
        "production_model_sha256": manifest["model_sha256"],
        "data_hashes": manifest["training_signature"]["files"],
        "cache_hashes": cache["sha256"],
        "code": code_manifest(root),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "lightgbm": importlib.metadata.version("lightgbm"),
        "latency": None,
        "latency_status": "not_measured",
        "latency_reason": "Offline cache diagnostics; no serving change",
    }
    write_json(folder / "protocol.json", protocol)
    frames = [score_pool(pools[1], pools[1].rrf_score, qrels, variant="hybrid")]
    old_scores = old.booster.predict(pools[1][FEATURE_NAMES], num_threads=1)
    old_iteration = manifest["best_iteration"]
    frames.append(
        score_pool(pools[1], old_scores, qrels, variant="existing_ltr", iteration=old_iteration)
    )
    frames.append(score_pool(pools[1], pools[1].rrf_score, qrels, variant="hybrid_top50", depth=50))
    frames.append(
        score_pool(
            pools[1],
            old_scores,
            qrels,
            variant="existing_ltr_top50",
            iteration=old_iteration,
            depth=50,
        )
    )
    fits = {}
    for name, features in FEATURE_GROUPS.items():
        fit_started = time.perf_counter()
        model = LGBMRanker(**manifest["config"]["model"])
        history = {}
        model.fit(
            training[features],
            training.grade.astype(int),
            group=groups[0],
            eval_set=[(validation[features], validation.grade.astype(int))],
            eval_group=[groups[1]],
            eval_at=[10],
            callbacks=[record_evaluation(history)],
        )
        curve = history["valid_0"]["ndcg@10"]
        pd.DataFrame(
            {"iteration": np.arange(1, len(curve) + 1), "judged_only_ndcg_at_10": curve}
        ).to_csv(folder / f"{name}_judged_curve.csv", index=False)
        if name == "all":
            replay = model.booster_.predict(
                pools[1][features], num_iteration=old_iteration, num_threads=1
            )
            if not np.array_equal(replay, old_scores):
                raise ValueError("Fixed-config replay differs from preserved production model")
        # Replay the original patience rule, rather than selecting with hindsight over 300 rounds.
        best = 0
        for i in range(1, len(curve)):
            if curve[i] > curve[best]:
                best = i
            if i - best >= manifest["config"]["early_stopping_rounds"]:
                break
        checkpoints = sorted(set(CHECKPOINTS + [best + 1]))
        for iteration in checkpoints:
            scores = model.booster_.predict(
                pools[1][features], num_iteration=iteration, num_threads=1
            )
            frames.append(score_pool(pools[1], scores, qrels, variant=name, iteration=iteration))
        fits[name] = {
            "judged_only_early_stop_iteration": best + 1,
            "judged_only_ndcg_at_10": curve[best],
            "judged_only_queries": len(groups[1]),
            "full_validation_queries": len(splits["validation"]),
            "fit_and_diagnostic_seconds": time.perf_counter() - fit_started,
            "features": features,
        }
        model.booster_.save_model(str(folder / f"{name}.txt"))
        fits[name]["model_sha256"] = file_hash(folder / f"{name}.txt")
        print(f"Completed {name}: judged-only checkpoint {best + 1}", flush=True)
    per_query = pd.concat(frames, ignore_index=True)
    per_query.to_csv(folder / "per_query.csv", index=False)
    metrics = [
        "ndcg_at_10",
        "mrr_at_10",
        "judged_recall_at_100",
        "judgment_coverage_at_10",
        "candidate_judgment_coverage",
    ]
    summary = (
        per_query.groupby(["variant", "iteration", "candidate_depth"], sort=False)[metrics]
        .mean()
        .reset_index()
    )
    baseline = summary[summary.variant == "hybrid"].iloc[0]
    for metric in metrics:
        summary[f"delta_{metric}_vs_hybrid"] = summary[metric] - baseline[metric]
    summary.to_csv(folder / "summary.csv", index=False)
    comparisons = []
    for row in summary.to_dict("records"):
        reference = (
            summary[summary.variant == "hybrid_top50"].iloc[0]
            if row["candidate_depth"] == 50
            else baseline
        )
        for metric in metrics:
            before, after = float(reference[metric]), row[metric]
            comparisons.append(
                {
                    "dataset": "wands_search",
                    "split": "validation",
                    "metric": metric,
                    "baseline_algorithm": reference.variant,
                    "candidate_algorithm": f"{row['variant']}@{row['iteration']}",
                    "baseline_value": before,
                    "candidate_value": after,
                    "delta": after - before,
                    "delta_pct": 100 * (after - before) / abs(before) if before else None,
                    "comparability_note": f"Same 96 queries and Top{row['candidate_depth']} pool; unjudged retained",
                }
            )
    pd.DataFrame(comparisons).to_csv(folder / "comparisons.csv", index=False)
    for name, fit in fits.items():
        selected = (
            summary[summary.variant == name]
            .sort_values(["ndcg_at_10", "iteration"], ascending=[False, True])
            .iloc[0]
        )
        fit["full_validation_selected_iteration"] = int(selected.iteration)
        fit["full_validation_ndcg_at_10"] = float(selected.ndcg_at_10)
    if file_hash(assets / "model.txt") != manifest["model_sha256"]:
        raise ValueError("Production LTR model changed during diagnostics")
    result = {
        "status": "completed",
        "run_id": run_id,
        "protocol": protocol,
        "fits": fits,
        "duration_seconds": time.perf_counter() - started,
        "production_model_preserved": True,
        "existing_model_prediction_replay_exact": True,
        "resource_measurement": {
            "cpu_seconds": None,
            "peak_rss_bytes": None,
            "status": "not_measured",
            "reason": "Bounded cache-only diagnostic; wall time measured separately",
        },
        "promotion": "none; validation diagnosis only",
        "training_rows": len(training),
        "validation_judged_rows": len(validation),
        "validation_unjudged_rows": int(pools[1].grade.isna().sum()),
        "metrics": summary.to_dict("records"),
    }
    write_json(folder / "results.json", result)
    write_json(root / "reports/ltr_diagnostics_latest.json", {"run_id": run_id})
    return result
