"""Held-out full-catalog neural ID generation and fixed-budget fusion diagnostics."""

import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from joysearch.common import read_json, write_json
from joysearch.data.wands import file_hash
from joysearch.evaluation.metrics import judged_recall, judgment_coverage, ndcg, reciprocal_rank
from joysearch.evaluation.process_stats import process_stats
from joysearch.evaluation.report import code_manifest
from joysearch.search.fusion import reciprocal_rank_fusion
from joysearch.search.generative import GenerativeRetriever
from joysearch.search.pipeline import SearchPipeline

METRICS = {
    "ndcg_at_10": ndcg,
    "mrr_at_10": reciprocal_rank,
    "judged_recall_at_100": judged_recall,
    "judgment_coverage_at_10": judgment_coverage,
}


def run(root, split="validation", selection=None):
    started, cpu = time.perf_counter(), time.process_time()
    frozen = code_manifest(root)
    model = GenerativeRetriever.load(root)
    config = model.manifest["config"]
    if split == "test":
        if selection is None:
            raise ValueError("Test requires completed validation --selection")
        gate = read_json(selection)
        if gate.get("status") != "completed" or gate.get("split") != "validation":
            raise ValueError("Test requires a completed validation selection")
        if gate["code"] != frozen or gate["model_sha256"] != file_hash(
            root / "artifacts/search_generative/manifest.json"
        ):
            raise ValueError("Generative source/model changed after validation")
    folder = root / "reports" / ("run_e6_eval_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    folder.mkdir(parents=True)
    pipeline = SearchPipeline.load(root, enable_ltr=False)
    cache = np.load(root / "artifacts/generative_inputs/queries.npz", allow_pickle=False)
    vectors = dict(zip(cache["query_ids"].tolist(), cache["vectors"], strict=True))
    qids = read_json(root / "data/processed/wands/splits.json")[split]
    queries = pd.read_parquet(root / "data/processed/wands/queries.parquet").set_index("query_id")[
        "query"
    ]
    qrels = pd.read_parquet(root / "data/processed/wands/qrels.parquet")
    judged = {
        q: dict(zip(g.product_id, g.grade))
        for q, g in qrels[qrels.query_id.isin(qids)].groupby("query_id")
    }
    catalog = set(model.item_ids)
    if catalog != set(pipeline.index.item_ids):
        raise ValueError("Generative and baseline catalog universes differ")
    write_json(
        folder / "inputs.json",
        {
            "split": split,
            "query_ids": qids,
            "config": config,
            "code": frozen,
            "model": model.manifest,
            "selection": str(selection) if selection else None,
            "selection_sha256": file_hash(selection) if selection else None,
            "catalog_size": len(catalog),
            "candidate_universe": "All WANDS product IDs, no qrel-based narrowing",
            "latency_protocol": "5 train-query warmups; frozen cached MiniLM query vectors shared by all learned retrievers; encode/load excluded; sequential algorithm-specific time; not text/HTTP latency or QPS",
            "judgment_policy": "Unjudged contributes zero gain under incomplete qrels; sparse judgments are not corpus-wide relevance truth; fixed test reused for regression",
        },
    )
    for qid in read_json(root / "data/processed/wands/splits.json")["train"][:5]:
        model.decode(vectors[qid])
        pipeline.dense.rank_vector(vectors[qid], 100)
        pipeline.index.retrieve(queries[qid], 100)
    rows, traces, generated, diagnostics = [], [], [], []
    for qid in qids:
        variants, durations = {}, {}
        tick = time.perf_counter()
        variants["bm25"] = pipeline.index.retrieve(queries[qid], 100)
        durations["bm25"] = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        variants["dense"] = pipeline.dense.rank_vector(vectors[qid], 100)
        durations["dense"] = (time.perf_counter() - tick) * 1000
        base = {"bm25": variants["bm25"], "dense": variants["dense"]}
        tick = time.perf_counter()
        variants["hybrid"] = reciprocal_rank_fusion(base, 100, config["rrf_k"])
        durations["hybrid"] = (
            durations["bm25"] + durations["dense"] + (time.perf_counter() - tick) * 1000
        )
        for constrained, name in [(False, "generative_raw"), (True, "generative")]:
            tick = time.perf_counter()
            pool, diagnostic = model.decode(vectors[qid], constrained)
            durations[name] = (time.perf_counter() - tick) * 1000
            variants[name] = pool
            generated.extend(
                {"query_id": qid, "variant": name, "rank": rank, **item}
                for rank, item in enumerate(diagnostic["generated"], 1)
            )
            diagnostics.append(
                {
                    "query_id": qid,
                    "variant": name,
                    **{
                        k: v
                        for k, v in diagnostic.items()
                        if k not in {"generated", "constraint_policy"}
                    },
                }
            )
        tick = time.perf_counter()
        variants["hybrid_generative"] = reciprocal_rank_fusion(
            {**base, "generative": variants["generative"]}, 100, config["rrf_k"]
        )
        durations["hybrid_generative"] = (
            durations["bm25"]
            + durations["dense"]
            + durations["generative"]
            + (time.perf_counter() - tick) * 1000
        )
        positives = {pid for pid, grade in judged.get(qid, {}).items() if grade >= 1}
        original = {c.item_id for c in variants["hybrid"]}
        gen_ids = {c.item_id for c in variants["generative"]}
        for name, pool in variants.items():
            ranking = [c.item_id for c in pool]
            if (
                len(ranking) > 100
                or len(ranking) != len(set(ranking))
                or not set(ranking) <= catalog
            ):
                raise ValueError(f"Invalid candidate budget/universe: {qid}/{name}")
            rows.append(
                {
                    "query_id": qid,
                    "variant": name,
                    **{k: f(ranking, judged.get(qid, {})) for k, f in METRICS.items()},
                    "candidate_count": len(ranking),
                    "latency_ms": durations[name],
                    "novel_judged_relevant_vs_hybrid": len((set(ranking) - original) & positives),
                    "lost_judged_relevant_vs_hybrid": len((original - set(ranking)) & positives),
                }
            )
            traces.extend(
                {
                    "query_id": qid,
                    "variant": name,
                    "product_id": c.item_id,
                    "rank": c.rank,
                    "score": c.score,
                }
                for c in pool
            )
        diagnostics.append(
            {
                "query_id": qid,
                "variant": "incremental_pool",
                "generator_unique_not_in_hybrid": len(gen_ids - original),
                "generator_judged_relevant_not_in_hybrid": len((gen_ids - original) & positives),
                "judged_relevant_lost_after_fusion": len(
                    (original - {c.item_id for c in variants["hybrid_generative"]}) & positives
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "per_query.csv", index=False)
    pd.DataFrame(traces).to_parquet(folder / "rankings.parquet", index=False)
    pd.DataFrame(generated).to_parquet(folder / "generated_codes.parquet", index=False)
    pd.DataFrame(diagnostics).to_csv(folder / "decode_diagnostics.csv", index=False)
    reference = frame[frame.variant.eq("hybrid")].set_index("query_id")
    differences = [
        {
            "query_id": r["query_id"],
            "variant": r["variant"],
            **{key + "_delta": r[key] - reference.loc[r["query_id"], key] for key in METRICS},
        }
        for r in rows
    ]
    pd.DataFrame(differences).to_csv(folder / "comparison.csv", index=False)
    summaries = {
        name: {
            **{key: float(group[key].mean()) for key in METRICS},
            "sample_count": len(group),
            "latency_p50_ms": float(group.latency_ms.median()),
            "latency_p95_ms": float(group.latency_ms.quantile(0.95)),
            "candidate_count_mean": float(group.candidate_count.mean()),
        }
        for name, group in frame.groupby("variant", sort=False)
    }
    code_frame = pd.DataFrame(generated)
    validity = {
        name: {
            "generated_count": len(group),
            "mapped_count": int(group.product_id.notna().sum()),
            "legal_id_rate": float(group.product_id.notna().mean()),
            "mapping_failure_rate": float(group.product_id.isna().mean()),
            "note": "Constrained legality follows complete catalog prefix masking; it does not demonstrate relevance",
        }
        for name, group in code_frame.groupby("variant")
    }
    if code_manifest(root) != frozen:
        raise ValueError("Sources changed during generative evaluation")
    metrics = {
        "status": "completed",
        "split": split,
        "seed": config["seed"],
        "query_count": len(qids),
        "catalog_size": len(catalog),
        "candidate_budget": 100,
        "k": 10,
        "metrics": summaries,
        "validity": validity,
        "semantic_id_collisions": 0,
        "mapping_coverage": len(model.code_to_item) / len(catalog),
        "queries_without_positive_qrels": sum(
            not any(g >= 1 for g in judged.get(q, {}).values()) for q in qids
        ),
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu,
        "process": process_stats(),
        "source": frozen,
    }
    write_json(folder / "metrics.json", metrics)
    if split == "validation":
        write_json(
            folder / "selection.json",
            {
                "status": "completed",
                "split": split,
                "code": frozen,
                "model_sha256": file_hash(root / "artifacts/search_generative/manifest.json"),
                "decision": "Evaluate frozen neural model and fixed fusion without default promotion",
            },
        )
    write_json(root / "reports" / f"generative_{split}_latest.json", {"run": folder.name})
    return folder
