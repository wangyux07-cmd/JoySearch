import time

import numpy as np
import pandas as pd

from joysearch.common import read_json
from joysearch.evaluation.metrics import judged_recall, judgment_coverage, ndcg, reciprocal_rank
from joysearch.search.pipeline import SearchPipeline


def latency_summary(values):
    return {
        "sample_count": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "protocol": "Loaded model; 5 warmup requests; sequential full test cohort; domain inference only",
    }


def evaluate(source, artifact, algorithm="bm25", split="test", model=None):
    started = time.perf_counter()
    model = model or SearchPipeline.load(artifact.parent.parent, enable_dense=algorithm != "bm25")
    manifest = read_json(source / "manifest.json")
    if model.index.manifest["data_manifest"]["sources"] != manifest["sources"]:
        raise ValueError("Search model/data source mismatch; rebuild index")
    splits = read_json(source / "splits.json")
    if splits != manifest["query_splits"]:
        raise ValueError("Search query split differs from frozen manifest")
    queries = pd.read_parquet(source / "queries.parquet").set_index("query_id")["query"]
    qrels = pd.read_parquet(source / "qrels.parquet")
    judgments = {
        qid: dict(zip(group.product_id, group.grade)) for qid, group in qrels.groupby("query_id")
    }
    for qid in splits[split][:5]:
        model.search(queries[qid], 100, algorithm)
    rows, latencies, top_items = [], [], set()
    for qid in splits[split]:
        tick = time.perf_counter()
        ranking = [c.item_id for c in model.search(queries[qid], 100, algorithm)]
        latencies.append((time.perf_counter() - tick) * 1000)
        judged = judgments.get(qid, {})
        top_items.update(ranking[:10])
        rows.append(
            {
                "query_id": qid,
                "algorithm": algorithm,
                "ndcg_at_10": ndcg(ranking, judged),
                "mrr_at_10": reciprocal_rank(ranking, judged),
                "judged_recall_at_100": judged_recall(ranking, judged),
                "judgment_coverage_at_10": judgment_coverage(ranking, judged),
                "judged_relevant_count": sum(g >= 1 for g in judged.values()),
                "unjudged_top10_count": sum(i not in judged for i in ranking[:10]),
                "returned_top10_count": len(ranking[:10]),
                "empty_result": int(not ranking),
            }
        )
    frame = pd.DataFrame(rows)
    metric_keys = ["ndcg_at_10", "mrr_at_10", "judged_recall_at_100", "judgment_coverage_at_10"]
    report = {
        "status": "completed",
        "dataset": "wands_search",
        "algorithm": algorithm,
        "split": split,
        "seed": manifest["seed"],
        "k": 10,
        "recall_k": 100,
        "sample_count": len(frame),
        "eligible_count": len(splits[split]),
        "catalog_size": len(model.index.item_ids),
        "candidate_universe": "all_wands_products",
        "metrics": {key: float(frame[key].mean()) for key in metric_keys},
        "catalog_coverage_at_10": len(top_items) / len(model.index.item_ids),
        "unique_top10_items": len(top_items),
        "judged_relevant_count": int(frame.judged_relevant_count.sum()),
        "no_positive_qrel_queries": int(frame.judged_relevant_count.eq(0).sum()),
        "empty_result_queries": int(frame.empty_result.sum()),
        "query_coverage": float(1 - frame.empty_result.mean()),
        "unjudged_top10_count": int(frame.unjudged_top10_count.sum()),
        "returned_top10_count": int(frame.returned_top10_count.sum()),
        "model_version": model.versions[algorithm],
        "config": model.model_config(algorithm),
        "duration_seconds": time.perf_counter() - started,
        "limitations": [
            "Incomplete qrels; unjudged retrieved items contribute zero gain",
            "JudgedRecall is not full corpus recall",
            "Conflicting duplicate qrels resolve to minimum grade before evaluation",
        ],
    }
    return report, latency_summary(latencies), frame
