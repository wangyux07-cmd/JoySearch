import importlib.metadata
import itertools
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from joysearch.common import Candidate, RecommendationContext, read_json, write_json
from joysearch.data.wands import file_hash
from joysearch.evaluation.metrics import (
    conditional_hit_rate,
    hit_rate,
    jaccard,
    judgment_coverage,
    ndcg,
    reciprocal_rank,
)
from joysearch.evaluation.report import code_manifest
from joysearch.evaluation.search_eval import latency_summary
from joysearch.recommendation.pipeline import RecPipeline
from joysearch.search.pipeline import SearchPipeline


def prepare_channels(raw, seen, budget, catalog_id):
    """Deduplicate identities, exclude train-seen IDs, and retain each channel TopB."""
    eligible = {}
    for channel, candidates in raw.items():
        unique, rows = set(), []
        for c in candidates:
            if c.catalog_id != catalog_id:
                raise ValueError(f"Mixed catalog in recall channel {channel}: {c.catalog_id}")
            identity = (c.catalog_id, c.item_id)
            if identity not in unique and c.item_id not in seen:
                rows.append(replace(c, rank=len(rows) + 1))
            unique.add(identity)
        eligible[channel] = rows[:budget]
    return eligible


def fuse_channels(rankings, budget, rrf_k):
    scores, evidence = {}, {}
    for channel, candidates in rankings.items():
        for rank, c in enumerate(candidates, 1):
            identity = (c.catalog_id, c.item_id)
            scores[identity] = scores.get(identity, 0) + 1 / (rrf_k + rank)
            evidence.setdefault(identity, {})[channel] = {"rank": rank, "score": c.score}
    order = sorted(scores, key=lambda key: (-scores[key], key))[:budget]
    return [
        Candidate(
            catalog,
            item,
            scores[(catalog, item)],
            "rrf",
            rank,
            {"sources": sorted(evidence[(catalog, item)]), "channels": evidence[(catalog, item)]},
        )
        for rank, (catalog, item) in enumerate(order, 1)
    ]


def served_itemcf(native, popularity, budget):
    """Reconstruct the existing positive-score prefix and popularity completion."""
    result = list(native[:budget])
    seen = {c.item_id for c in result}
    for c in popularity:
        if len(result) == budget:
            break
        if c.item_id not in seen:
            result.append(replace(c, rank=len(result) + 1))
            seen.add(c.item_id)
    return result


def candidate_variants(eligible, config, recommendation=False):
    variants, timings = {}, {}
    channels = list(eligible)
    for size in range(1, len(channels) + 1):
        for names in itertools.combinations(channels, size):
            tick = time.perf_counter()
            name = "+".join(names)
            variants[name] = (
                eligible[names[0]]
                if size == 1
                else fuse_channels(
                    {n: eligible[n] for n in names}, config["budget"], config["rrf_k"]
                )
            )
            timings[name] = (time.perf_counter() - tick) * 1000
    if recommendation:
        tick = time.perf_counter()
        variants["itemcf_served"] = served_itemcf(
            eligible["itemcf_native"], eligible["popularity"], config["budget"]
        )
        timings["itemcf_served"] = (time.perf_counter() - tick) * 1000
    return variants, timings


def identities(candidates):
    return {c.item_id for c in candidates}


def request_metrics(request_id, raw, eligible, variants, seen, qrels, config, recommendation):
    positive = {item for item, grade in qrels.items() if grade >= 1}
    raw_sets = {name: identities(rows) for name, rows in raw.items()}
    sets = {name: identities(rows) for name, rows in eligible.items()}
    union = set().union(*sets.values())
    target = next(iter(qrels)) if recommendation else None
    pool_metric = (
        (lambda items: float(target in items))
        if recommendation
        else (lambda items: len(items & positive) / len(positive) if positive else 0.0)
    )
    rows, overlaps, contributions = [], [], []
    for name, selected in variants.items():
        ranking = [c.item_id for c in selected]
        subset = name.split("+") if name != "itemcf_served" else ["itemcf_native", "popularity"]
        subset_raw = set().union(*(raw_sets[n] for n in subset))
        subset_union = set().union(*(sets[n] for n in subset))
        row = {
            "request_id": request_id,
            "variant": name,
            "selected_count": len(selected),
            "candidate_shortage": int(len(selected) < config["budget"]),
            "raw_union_count": len(subset_raw),
            "filtered_union_count": len(subset_raw - seen),
            "fusion_union_count": len(subset_union),
            "deduplication_loss": sum(len(raw[n]) for n in subset) - len(subset_raw),
            "seen_filter_loss": len(subset_raw & seen),
            "channel_truncation_loss": len(subset_raw - seen) - len(subset_union),
            "budget_truncation_loss": len(subset_union) - len(selected),
            "candidate_recall": pool_metric(set(ranking)),
            "raw_union_recall": pool_metric(subset_raw),
            "filtered_union_recall": pool_metric(subset_raw - seen),
            "fusion_union_recall": pool_metric(subset_union),
            "ndcg_at_10": ndcg(ranking, qrels, config["display_k"]),
            "judged_positive_count": len(positive),
        }
        if recommendation:
            row.update(
                hr_at_10=hit_rate(ranking, target, config["display_k"]),
                fallback=int(
                    name == "itemcf_served"
                    and any(c.source == "popularity" for c in selected[: config["display_k"]])
                ),
            )
        else:
            row.update(
                mrr_at_10=reciprocal_rank(ranking, qrels, config["display_k"]),
                judgment_coverage_at_10=judgment_coverage(ranking, qrels, config["display_k"]),
            )
        rows.append(row)
    for left, right in itertools.combinations(sets, 2):
        overlaps.append(
            {
                "request_id": request_id,
                "left": left,
                "right": right,
                "jaccard": jaccard(sets[left], sets[right]),
                "raw_jaccard": jaccard(raw_sets[left], raw_sets[right]),
                "both_empty": int(not (sets[left] | sets[right])),
            }
        )
    for channel, channel_items in sets.items():
        others = set().union(*(s for name, s in sets.items() if name != channel))
        raw_others = set().union(*(s for name, s in raw_sets.items() if name != channel))
        unique = (channel_items - others) & positive
        raw_unique = (raw_sets[channel] - raw_others) & positive
        contributions.append(
            {
                "request_id": request_id,
                "channel": channel,
                "raw_unique_relevant_count": len(raw_unique),
                "eligible_unique_relevant_count": len(unique),
                "unique_recall_fraction": len(unique) / len(positive) if positive else 0.0,
                "unique_share_of_union_relevant": len(unique) / len(union & positive)
                if union & positive
                else None,
            }
        )
    return rows, overlaps, contributions


CHANNEL_SCHEMA = pa.schema(
    [
        ("request_id", pa.string()),
        ("catalog_id", pa.string()),
        ("channel", pa.string()),
        ("item_id", pa.string()),
        ("source", pa.string()),
        ("raw_rank", pa.int32()),
        ("raw_score", pa.float64()),
        ("eligible_rank", pa.int32()),
        ("excluded_seen", pa.bool_()),
    ]
)
UNION_SCHEMA = pa.schema(
    [
        ("request_id", pa.string()),
        ("catalog_id", pa.string()),
        ("item_id", pa.string()),
        ("sources", pa.list_(pa.string())),
        ("raw_ranks", pa.list_(pa.int32())),
        ("raw_scores", pa.list_(pa.float64())),
        ("eligible_sources", pa.list_(pa.string())),
        ("eligible_ranks", pa.list_(pa.int32())),
        ("excluded_seen", pa.bool_()),
        ("in_filtered_union", pa.bool_()),
        ("in_fusion_union", pa.bool_()),
        ("selected_variants", pa.list_(pa.string())),
        ("selected_ranks", pa.list_(pa.int32())),
        ("top10_variants", pa.list_(pa.string())),
    ]
)


def trace_rows(request_id, raw, eligible, variants, seen, display_k):
    channel_rows, union = [], {}
    eligible_ranks = {
        name: {c.item_id: i for i, c in enumerate(rows, 1)} for name, rows in eligible.items()
    }
    positions = defaultdict(dict)
    for name, candidates in variants.items():
        for rank, c in enumerate(candidates, 1):
            positions[(c.catalog_id, c.item_id)][name] = rank
    for channel, candidates in raw.items():
        for rank, c in enumerate(candidates, 1):
            channel_rows.append(
                {
                    "request_id": request_id,
                    "catalog_id": c.catalog_id,
                    "channel": channel,
                    "item_id": c.item_id,
                    "source": c.source,
                    "raw_rank": rank,
                    "raw_score": c.score,
                    "eligible_rank": eligible_ranks[channel].get(c.item_id),
                    "excluded_seen": c.item_id in seen,
                }
            )
            union.setdefault((c.catalog_id, c.item_id), {}).setdefault(channel, (rank, c.score))
    union_rows = []
    for (catalog, item), evidence in sorted(union.items()):
        sources = sorted(evidence)
        kept = [name for name in sources if item in eligible_ranks[name]]
        names = sorted(positions[(catalog, item)])
        union_rows.append(
            {
                "request_id": request_id,
                "catalog_id": catalog,
                "item_id": item,
                "sources": sources,
                "raw_ranks": [evidence[s][0] for s in sources],
                "raw_scores": [evidence[s][1] for s in sources],
                "eligible_sources": kept,
                "eligible_ranks": [eligible_ranks[s][item] for s in kept],
                "excluded_seen": item in seen,
                "in_filtered_union": item not in seen,
                "in_fusion_union": bool(kept),
                "selected_variants": names,
                "selected_ranks": [positions[(catalog, item)][n] for n in names],
                "top10_variants": [n for n in names if positions[(catalog, item)][n] <= display_k],
            }
        )
    return channel_rows, union_rows


def aggregate(frame, top_items, domain, split, config, catalog_size):
    variants = {}
    for name, group in frame.groupby("variant", sort=False):
        metrics = {
            "candidate_hit_at_100"
            if domain == "recommendation"
            else "candidate_judged_recall_at_100": float(group.candidate_recall.mean()),
            "ndcg_at_10": float(group.ndcg_at_10.mean()),
            "raw_union_recall": float(group.raw_union_recall.mean()),
            "filtered_union_recall": float(group.filtered_union_recall.mean()),
            "fusion_union_recall": float(group.fusion_union_recall.mean()),
            "candidate_shortage_rate": float(group.candidate_shortage.mean()),
        }
        if domain == "recommendation":
            metrics.update(
                hr_at_10=float(group.hr_at_10.mean()),
                catalog_coverage_at_10=len(top_items[name]) / catalog_size,
                fallback_rate=float(group.fallback.mean()),
                candidate_hit_count=int(group.candidate_recall.sum()),
                top10_hit_count=int(group.hr_at_10.sum()),
                top10_given_top100_hit_rate=conditional_hit_rate(
                    int(group.hr_at_10.sum()), int(group.candidate_recall.sum())
                ),
            )
        else:
            metrics.update(
                mrr_at_10=float(group.mrr_at_10.mean()),
                judgment_coverage_at_10=float(group.judgment_coverage_at_10.mean()),
            )
        variants[name] = {
            "metrics": metrics,
            "sample_count": len(group),
            "mean_candidate_count": float(group.selected_count.mean()),
            "min_candidate_count": int(group.selected_count.min()),
            "mean_stage_counts": {
                key: float(group[key].mean())
                for key in [
                    "raw_union_count",
                    "filtered_union_count",
                    "fusion_union_count",
                    "deduplication_loss",
                    "seen_filter_loss",
                    "channel_truncation_loss",
                    "budget_truncation_loss",
                ]
            },
        }
    channels = config[f"{domain}_channels"]
    full = "+".join(channels)
    metric = (
        "candidate_hit_at_100" if domain == "recommendation" else "candidate_judged_recall_at_100"
    )
    ablations = []
    for channel in channels:
        without = "+".join(n for n in channels if n != channel)
        baseline, candidate = [variants[n]["metrics"][metric] for n in [full, without]]
        ablations.append(
            {
                "domain": domain,
                "split": split,
                "budget": config["budget"],
                "removed_channel": channel,
                "full_variant": full,
                "without_variant": without,
                "metric": metric,
                "full_value": baseline,
                "without_value": candidate,
                "delta_without_minus_full": candidate - baseline,
                "delta_pct": 100 * (candidate - baseline) / baseline if baseline else None,
                "without_mean_count": variants[without]["mean_candidate_count"],
                "without_shortage_rate": variants[without]["metrics"]["candidate_shortage_rate"],
            }
        )
    return variants, ablations


def evaluate_domain(root, folder, config, domain, split, limit=None):
    started = time.perf_counter()
    is_rec = domain == "recommendation"
    data = root / "data/processed" / ("amazon" if is_rec else "wands")
    manifest = read_json(data / "manifest.json")
    if is_rec:
        if file_hash(data / "events.parquet") != manifest["split_sha256"]:
            raise ValueError("Recall evaluation Amazon split hash mismatch")
        model = RecPipeline.load(root / "artifacts/recommendation")
        if model.two_tower is None:
            raise ValueError("Recall diagnostics require the completed Two-Tower model")
        events = pd.read_parquet(data / "events.parquet")
        heldout = events[events.split.eq(split)].sort_values("user_id")
        expected = {
            u: set(g.parent_asin) for u, g in events[events.split.eq("train")].groupby("user_id")
        }
        actual = {
            u: {model.popularity.items[i] for i in h} for u, h in model.popularity.histories.items()
        }
        if expected != actual or set(heldout.user_id) != set(actual):
            raise ValueError("Recall evaluation histories/cohort differ from frozen train split")
        if set(model.popularity.items) != set().union(*expected.values()):
            raise ValueError("Recall evaluation train catalog differs from baseline")
        aliases = {user: alias for alias, user in model.aliases.items()}
        requests = [
            (aliases[r.user_id], r.user_id, {r.parent_asin: 1}) for r in heldout.itertuples()
        ]
        catalog_id, catalog_size = "amazon_baby_rec", len(model.popularity.items)
        unavailable = int((~heldout.parent_asin.isin(model.popularity.items)).sum())
    else:
        model = SearchPipeline.load(root, enable_ltr=False)
        splits = read_json(data / "splits.json")
        if (
            splits != manifest["query_splits"]
            or model.index.manifest["data_manifest"]["sources"] != manifest["sources"]
        ):
            raise ValueError("Recall evaluation WANDS data/split mismatch")
        queries = pd.read_parquet(data / "queries.parquet").set_index("query_id")["query"]
        qrels = pd.read_parquet(data / "qrels.parquet")
        qrels = qrels[qrels.query_id.isin(splits[split])]
        judgments = {qid: dict(zip(g.product_id, g.grade)) for qid, g in qrels.groupby("query_id")}
        requests = [(qid, queries[qid], judgments.get(qid, {})) for qid in splits[split]]
        catalog_id, catalog_size, unavailable = "wands_search", len(model.index.item_ids), None
    eligible_count = len(requests)
    requests = requests[:limit] if limit else requests
    if is_rec:
        unavailable = sum(
            next(iter(qrels)) not in model.itemcf.item_map for _, _, qrels in requests
        )

    def retrieve(value):
        raw, timing = {}, {}
        seen = (
            {model.popularity.items[i] for i in model.popularity.histories[value]}
            if is_rec
            else set()
        )
        depth = config["budget"] + len(seen)
        for name in config[f"{domain}_channels"]:
            tick = time.perf_counter()
            if not is_rec:
                raw[name] = (model.index if name == "bm25" else model.dense).retrieve(value, depth)
            else:
                context = RecommendationContext(value, exclude_seen=False)
                if name == "popularity":
                    raw[name] = model.popularity.recommend(context, depth)
                elif name == "itemcf_native":
                    raw[name] = model.itemcf.retrieve_native(context, depth)
                else:
                    raw[name] = model.two_tower.recommend(context, depth)[0]
            timing[name] = (time.perf_counter() - tick) * 1000
        return raw, timing, seen, depth

    for _, value, _ in requests[:5]:
        retrieve(value)
    rows, overlaps, contributions, channel_requests = [], [], [], []
    channel_buffer, union_buffer = [], []
    top_items = defaultdict(set)
    with (
        pq.ParquetWriter(
            folder / f"{domain}_channels.parquet", CHANNEL_SCHEMA, compression="zstd"
        ) as channel_writer,
        pq.ParquetWriter(
            folder / f"{domain}_candidate_stages.parquet", UNION_SCHEMA, compression="zstd"
        ) as union_writer,
    ):
        for index, (request_id, value, qrels) in enumerate(requests, 1):
            raw, timing, seen, depth = retrieve(value)
            eligible = prepare_channels(raw, seen, config["budget"], catalog_id)
            variants, fusion_timing = candidate_variants(eligible, config, is_rec)
            a, b, c = request_metrics(
                request_id, raw, eligible, variants, seen, qrels, config, is_rec
            )
            for row in a:
                row["fusion_ms"] = fusion_timing[row["variant"]]
            rows.extend(a)
            overlaps.extend(b)
            contributions.extend(c)
            for name, candidates in variants.items():
                top_items[name].update(c.item_id for c in candidates[: config["display_k"]])
            for name, candidates in raw.items():
                channel_requests.append(
                    {
                        "request_id": request_id,
                        "channel": name,
                        "requested_depth": depth,
                        "raw_count": len(candidates),
                        "eligible_count": len(eligible[name]),
                        "latency_ms": timing[name],
                    }
                )
            a, b = trace_rows(request_id, raw, eligible, variants, seen, config["display_k"])
            channel_buffer.extend(a)
            union_buffer.extend(b)
            if index % config["trace_batch_requests"] == 0 or index == len(requests):
                channel_writer.write_table(
                    pa.Table.from_pylist(channel_buffer, schema=CHANNEL_SCHEMA)
                )
                union_writer.write_table(pa.Table.from_pylist(union_buffer, schema=UNION_SCHEMA))
                channel_buffer.clear()
                union_buffer.clear()
            if index % 1000 == 0:
                print(f"{domain}: {index}/{len(requests)} candidate traces persisted", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / f"{domain}_per_request.csv", index=False)
    pd.DataFrame(channel_requests).to_csv(folder / f"{domain}_channel_requests.csv", index=False)
    pd.DataFrame(contributions).to_csv(folder / f"{domain}_channel_contributions.csv", index=False)
    pd.DataFrame(overlaps).to_csv(folder / f"{domain}_overlap_per_request.csv", index=False)
    variants, ablations = aggregate(frame, top_items, domain, split, config, catalog_size)
    overlap = (
        pd.DataFrame(overlaps)
        .groupby(["left", "right"], as_index=False)
        .agg(
            jaccard=("jaccard", "mean"),
            raw_jaccard=("raw_jaccard", "mean"),
            sample_count=("request_id", "size"),
            both_empty_count=("both_empty", "sum"),
        )
    )
    overlap["domain"], overlap["split"] = domain, split
    contribution = (
        pd.DataFrame(contributions)
        .groupby("channel", as_index=False)
        .agg(
            raw_unique_relevant_count=("raw_unique_relevant_count", "sum"),
            eligible_unique_relevant_count=("eligible_unique_relevant_count", "sum"),
            macro_unique_recall_fraction=("unique_recall_fraction", "mean"),
            macro_unique_share_of_union_relevant=("unique_share_of_union_relevant", "mean"),
            unique_share_denominator_requests=("unique_share_of_union_relevant", "count"),
            sample_count=("request_id", "size"),
        )
    )
    contribution["domain"], contribution["split"] = domain, split
    latency = {
        name: latency_summary(g.latency_ms.tolist())
        for name, g in pd.DataFrame(channel_requests).groupby("channel")
    }
    for timing in latency.values():
        timing["protocol"] = (
            f"Loaded models; 5 warmups; sequential {split} cohort; raw TopD channel inference; excludes metrics and trace IO"
        )
    return (
        {
            "status": "completed",
            "domain": domain,
            "split": split,
            "catalog_id": catalog_id,
            "catalog_size": catalog_size,
            "sample_count": len(requests),
            "eligible_count": eligible_count,
            "budget": config["budget"],
            "k": config["display_k"],
            "variants": variants,
            "unavailable_target_count": unavailable,
            "unavailable_target_rate": unavailable / len(requests) if is_rec else None,
            "channel_latency": latency,
            "duration_seconds": time.perf_counter() - started,
        },
        ablations,
        overlap,
        contribution,
    )


def run(root, split="validation", limit=None):
    config = read_json(root / "configs/recall.json")
    if config["budget"] != 100 or config["display_k"] != 10:
        raise ValueError("Primary recall diagnostics require B=100 and K=10")
    folder = (
        root
        / "reports"
        / ("run_e1_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6])
    )
    folder.mkdir(parents=True)
    started, cpu_started = time.perf_counter(), time.process_time()
    config.update(
        split=split,
        scope="pilot" if limit else "complete_fixed_cohort",
        pilot_limit=limit,
        test_reuse=split == "test",
        git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        git_status=subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        ).splitlines(),
        code=code_manifest(root),
        platform=platform.platform(),
        packages={
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "pyarrow", "torch"]
        },
        command=f"conda run -n joysearch python scripts/evaluate_recall.py --split {split}"
        + (f" --pilot-limit {limit}" if limit else ""),
    )
    write_json(folder / "config.json", config)
    data = {
        domain: read_json(root / f"data/processed/{name}/manifest.json")
        for domain, name in [("search", "wands"), ("recommendation", "amazon")]
    }
    for domain, name in [("search", "wands"), ("recommendation", "amazon")]:
        for filename, metadata in data[domain]["sources"].items():
            if file_hash(root / f"data/raw/{name}/{filename}") != metadata["sha256"]:
                raise ValueError(f"Recall diagnostics source changed: {name}/{filename}")
    write_json(folder / "data_manifest.json", data)
    write_json(
        folder / "processed_data_sha256.json",
        {
            p.relative_to(root).as_posix(): file_hash(p)
            for name in ["wands", "amazon"]
            for p in sorted((root / "data/processed" / name).glob("*"))
            if p.is_file()
        },
    )
    artifacts = ["search", "search_dense", "recommendation", "recommendation_two_tower"]
    model_files = {
        p.relative_to(root).as_posix(): file_hash(p)
        for name in artifacts
        for p in sorted((root / "artifacts" / name).glob("*"))
        if p.is_file()
        and p.name not in ["build_state.json", "training_progress.json", "best_state.pt"]
    }
    write_json(folder / "model_artifacts.json", model_files)
    reports, ablations, overlaps, contributions = {}, [], [], []
    for domain in ["search", "recommendation"]:
        report, ablation, overlap, contribution = evaluate_domain(
            root, folder, config, domain, split, limit
        )
        reports[domain] = report
        ablations.extend(ablation)
        overlaps.append(overlap)
        contributions.append(contribution)
    for filename, frame in [
        ("channel_ablation.csv", pd.DataFrame(ablations)),
        ("channel_overlap.csv", pd.concat(overlaps)),
        ("channel_contributions.csv", pd.concat(contributions)),
    ]:
        frame["run_id"], frame["seed"] = folder.name, config["seed"]
        frame["sample_count"] = frame.domain.map({d: r["sample_count"] for d, r in reports.items()})
        frame.to_csv(folder / filename, index=False)
    comparisons = []
    for domain, report in reports.items():
        baseline_names = (
            ["popularity", "itemcf_served", "two_tower"] if domain == "recommendation" else ["bm25"]
        )
        keys = (
            ["candidate_hit_at_100", "hr_at_10", "ndcg_at_10", "catalog_coverage_at_10"]
            if domain == "recommendation"
            else [
                "candidate_judged_recall_at_100",
                "ndcg_at_10",
                "mrr_at_10",
                "judgment_coverage_at_10",
            ]
        )
        for baseline_name in baseline_names:
            for name, variant in report["variants"].items():
                if name == baseline_name:
                    continue
                for key in keys:
                    baseline = report["variants"][baseline_name]["metrics"][key]
                    candidate = variant["metrics"][key]
                    comparisons.append(
                        {
                            "domain": domain,
                            "split": split,
                            "metric": key,
                            "baseline_algorithm": baseline_name,
                            "candidate_algorithm": name,
                            "baseline_value": baseline,
                            "candidate_value": candidate,
                            "delta": candidate - baseline,
                            "delta_pct": 100 * (candidate - baseline) / baseline
                            if baseline
                            else None,
                            "budget": config["budget"],
                            "sample_count": report["sample_count"],
                            "candidate_mean_count": variant["mean_candidate_count"],
                            "comparability_note": "Same frozen cohort, catalog, seen filter and B; native ItemCF may underfill, actual counts reported",
                        }
                    )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["run_id"], comparison_frame["seed"] = folder.name, config["seed"]
    comparison_frame.to_csv(folder / "comparisons.csv", index=False)
    result = {
        "status": "completed",
        "run_id": folder.name,
        "split": split,
        "scope": config["scope"],
        "budget": config["budget"],
        "k": config["display_k"],
        "seed": config["seed"],
        "domains": reports,
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "trace_disk_bytes": sum(p.stat().st_size for p in folder.glob("*.parquet")),
        "limitations": [
            "WANDS sparse qrels: unjudged is not verified irrelevant; judged recall only",
            "Amazon ratings proxy preference; full train catalog and all users retained",
            "Raw unions have a larger budget and cannot establish fixed-B uplift",
            "Recall@K equals HR@K for one recommendation target",
            "Diagnostic RRF fusions are offline comparisons, not production policy changes",
            "Per-user leave-last-two-out is not a global temporal split; repeated test is regression",
        ],
    }
    write_json(folder / "recall_metrics.json", result)
    lines = [
        f"# E1 召回诊断 — {folder.name}",
        "",
        f"split={split}；scope={config['scope']}；B=100；K=10。",
        "",
        "原始并集与固定预算结果分开。ItemCF原生正分不含热门补齐；itemcf_served保留原对外兜底。",
        "",
    ]
    for domain, report in reports.items():
        lines += [
            f"## {domain}",
            f"{report['sample_count']} requests / {report['catalog_size']} items",
            "",
            "| variant | candidate recall/hit@100 | NDCG@10 | raw union recall | shortage |",
            "|---|---|---|---|---|",
        ]
        for name, variant in report["variants"].items():
            m = variant["metrics"]
            recall = m.get("candidate_hit_at_100", m.get("candidate_judged_recall_at_100"))
            lines.append(
                f"| {name} | {recall:.8f} | {m['ndcg_at_10']:.8f} | {m['raw_union_recall']:.8f} | {m['candidate_shortage_rate']:.6f} |"
            )
    lines += ["", *result["limitations"], "", "```powershell", config["command"], "```"]
    (folder / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    if split == "test" and limit is None:
        write_json(
            root / "reports/recall_latest.json", {"status": "verified", "run_id": folder.name}
        )
    print(f"Recall report: {folder}", flush=True)
    return folder
