"""Same-cohort allocation experiments reusing verified E1 channel candidates."""

import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from joysearch.common import Candidate, RecommendationContext, read_json, write_json
from joysearch.data.wands import file_hash
from joysearch.evaluation.metrics import hit_rate, ndcg
from joysearch.evaluation.recall_eval import served_itemcf
from joysearch.evaluation.report import code_manifest
from joysearch.recommendation.fusion import (
    CHANNELS,
    adaptive_quotas,
    allocate,
    allocation_signature,
)
from joysearch.recommendation.pipeline import RecPipeline
from joysearch.traffic.simulator import simulate

METRICS = [
    "candidate_hit_at_100",
    "hr_at_10",
    "ndcg_at_10",
    "catalog_coverage_at_10",
    "tail_list_share_at_10",
    "list_concentration_hhi",
    "mean_train_frequency_at_10",
]
TRACE_SCHEMA = pa.schema(
    [
        ("user_alias", pa.string()),
        ("policy", pa.string()),
        ("item_ids", pa.list_(pa.string())),
        ("quota_credits", pa.list_(pa.string())),
        ("requested_quotas", pa.list_(pa.int32())),
        ("filled_quotas", pa.list_(pa.int32())),
        ("reallocated_slots", pa.int32()),
    ]
)


def evaluation_signature(root):
    return {
        "allocation": allocation_signature(root),
        "evaluation": {
            p: file_hash(root / p)
            for p in [
                "src/joysearch/evaluation/allocation_eval.py",
                "src/joysearch/evaluation/metrics.py",
                "src/joysearch/evaluation/recall_eval.py",
                "src/joysearch/traffic/policy.py",
                "src/joysearch/traffic/simulator.py",
                "data/processed/amazon/events.parquet",
                "data/processed/amazon/manifest.json",
            ]
        },
    }


def cached_requests(path):
    current, rankings = None, {c: [] for c in CHANNELS}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32768):
        for row in batch.to_pylist():
            alias = row["request_id"]
            if current is not None and current != alias:
                yield current, rankings
                rankings = {c: [] for c in CHANNELS}
            current = alias
            if row["catalog_id"] != "amazon_baby_rec":
                raise ValueError("Candidate cache contains a foreign catalog")
            if row["eligible_rank"] is not None:
                if row["excluded_seen"] or not 1 <= row["eligible_rank"] <= 100:
                    raise ValueError("Candidate cache violates seen filtering or budget")
                rankings[row["channel"]].append(
                    Candidate(
                        row["catalog_id"],
                        row["item_id"],
                        row["raw_score"],
                        row["source"],
                        row["eligible_rank"],
                    )
                )
    if current is not None:
        yield current, rankings


def policy_candidates(rankings, history_length, config):
    tick = time.perf_counter()
    variants = {
        "popularity": (rankings["popularity"], {}),
        "itemcf_native": (rankings["itemcf_native"], {}),
        "two_tower": (rankings["two_tower"], {}),
        "itemcf": (
            served_itemcf(rankings["itemcf_native"], rankings["popularity"], config["budget"]),
            {},
        ),
        "rrf": allocate(rankings, budget=config["budget"], rrf_k=config["rrf_k"]),
    }
    variants["rrf"][1]["allocation_ms"] = (time.perf_counter() - tick) * 1000
    for name, values in config["fixed_quotas"].items():
        tick = time.perf_counter()
        variants["quota:" + name] = allocate(
            rankings, dict(zip(CHANNELS, values)), config["budget"], config["rrf_k"]
        )
        variants["quota:" + name][1]["allocation_ms"] = (time.perf_counter() - tick) * 1000
    for name in config["adaptive_rules"]:
        tick = time.perf_counter()
        quotas, routed = adaptive_quotas(
            config, name, history_length, len(rankings["itemcf_native"])
        )
        candidates, diagnostic = allocate(rankings, quotas, config["budget"], config["rrf_k"])
        variants["adaptive:" + name] = (
            candidates,
            {**diagnostic, "route": routed, "allocation_ms": (time.perf_counter() - tick) * 1000},
        )
    return variants


def target_bucket(target, frequencies, config):
    count = frequencies.get(target)
    if count is None:
        return "unavailable"
    if count <= config["tail_frequency_max"]:
        return "tail"
    return "head" if count >= config["head_frequency_min"] else "mid"


def counter_statistics(counts, frequencies, tail_max):
    total = sum(counts.values())
    return {
        "catalog_coverage_at_10": len(counts) / len(frequencies),
        "tail_list_share_at_10": sum(n for i, n in counts.items() if frequencies[i] <= tail_max)
        / total
        if total
        else 0.0,
        "list_concentration_hhi": sum((n / total) ** 2 for n in counts.values()) if total else 0.0,
        "mean_train_frequency_at_10": sum(n * frequencies[i] for i, n in counts.items()) / total
        if total
        else 0.0,
        "offline_list_appearances": total,
    }


def run(root, split, cache, selection=None, pilot_limit=None):
    if pilot_limit is not None and pilot_limit <= 0:
        raise ValueError("Pilot limit must be positive")
    started, cpu = time.perf_counter(), time.process_time()
    config = read_json(root / "configs/allocation.json")
    signature = evaluation_signature(root)
    if split == "test":
        if selection is None or pilot_limit:
            raise ValueError(
                "Full test requires a completed validation --selection; pilots use validation"
            )
        frozen = read_json(selection)
        if (
            frozen["split"] != "validation"
            or frozen["scope"] != "complete_fixed_cohort"
            or frozen["evaluation_signature"] != signature
        ):
            raise ValueError("Allocation protocol changed after validation; rerun validation")
    run_id = "run_e3_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    folder = root / "reports" / run_id
    folder.mkdir(parents=True)
    cache_config = read_json(cache / "config.json")
    if (
        cache_config["split"] != split
        or cache_config["scope"] != "complete_fixed_cohort"
        or cache_config["budget"] != 100
    ):
        raise ValueError(
            "Allocation requires complete E1 Top100 candidates from the requested split"
        )
    artifacts = read_json(cache / "model_artifacts.json")
    for path, checksum in artifacts.items():
        if path.startswith("artifacts/recommendation") and file_hash(root / path) != checksum:
            raise ValueError(f"Cached recommendation asset changed: {path}")
    original_code = cache_config["code"]["files"]
    for name in ["popularity", "itemcf", "two_tower"]:
        path = f"src/joysearch/recommendation/{name}.py"
        if file_hash(root / path) != original_code[path]:
            raise ValueError(f"Cached channel inference changed: {path}")
    model = RecPipeline.load(root / "artifacts/recommendation", enable_allocation=False)
    events = pd.read_parquet(root / "data/processed/amazon/events.parquet")
    data_manifest = read_json(root / "data/processed/amazon/manifest.json")
    if (
        file_hash(root / "data/processed/amazon/events.parquet") != data_manifest["split_sha256"]
        or model.manifest["data_manifest"]["split_sha256"] != data_manifest["split_sha256"]
    ):
        raise ValueError("Amazon data/model split hashes disagree")
    train = events[events.split.eq("train")]
    actual = {
        u: {str(model.popularity.items[i]) for i in h}
        for u, h in model.popularity.histories.items()
    }
    expected = {u: set(g.parent_asin) for u, g in train.groupby("user_id")}
    if actual != expected:
        raise ValueError("Allocation requires exact train histories")
    frequencies = dict(zip(model.popularity.items, model.popularity.counts.astype(int)))
    if frequencies != train.parent_asin.value_counts().to_dict():
        raise ValueError("Train frequency buckets disagree with model counts")
    user_to_alias = {u: a for a, u in model.aliases.items()}
    targets = {
        user_to_alias[r.user_id]: r.parent_asin for r in events[events.split.eq(split)].itertuples()
    }
    if len(targets) != len(model.aliases):
        raise ValueError("Held-out cohort must retain all train users")
    write_json(
        folder / "frozen_inputs.json",
        {
            "split": split,
            "scope": "pilot" if pilot_limit else "complete_fixed_cohort",
            "evaluation_signature": signature,
            "config": config,
            "code": code_manifest(root),
            "cache_run": cache.name,
            "cache_sha256": file_hash(cache / "recommendation_channels.parquet"),
            "selection": str(selection) if selection else None,
            "selection_sha256": file_hash(selection) if selection else None,
            "candidate_universe": "full train catalog; seen train items excluded; B100/K10",
            "inference": "Reuse verified E1 channel cache; allocation_ms excludes retrieval and IO",
        },
    )
    rows, traces, appearances = [], [], defaultdict(Counter)
    aliases_seen, simulation_aliases, rewards, simulation_lists = set(), [], [], []
    arms = list(config["fixed_quotas"])
    with pq.ParquetWriter(
        folder / "policy_rankings.parquet", TRACE_SCHEMA, compression="zstd"
    ) as trace_writer:
        for index, (alias, rankings) in enumerate(
            cached_requests(cache / "recommendation_channels.parquet")
        ):
            if pilot_limit and index >= pilot_limit:
                break
            if alias not in targets or alias in aliases_seen:
                raise ValueError(f"Duplicate or unexpected cache user: {alias}")
            aliases_seen.add(alias)
            user = model.aliases[alias]
            history = model.popularity.histories[user]
            seen = {str(model.popularity.items[i]) for i in history}
            for candidates in rankings.values():
                if (
                    len({c.item_id for c in candidates}) != len(candidates)
                    or [c.rank for c in candidates] != list(range(1, len(candidates) + 1))
                    or seen & {c.item_id for c in candidates}
                ):
                    raise ValueError(f"Invalid eligible cache for {alias}")
            if index < 3:
                online, _ = model.recall_channels(RecommendationContext(user), 100)
                for channel in CHANNELS:
                    if [(c.item_id, c.score) for c in online[channel]] != [
                        (c.item_id, c.score) for c in rankings[channel]
                    ]:
                        raise ValueError(f"Live/cache channel mismatch for {alias}/{channel}")
            tick = time.perf_counter()
            variants = policy_candidates(rankings, len(history), config)
            allocation_ms = (time.perf_counter() - tick) * 1000
            history_bucket = next(
                (f"le_{n}" for n in config["history_bucket_maxima"] if len(history) <= n),
                f"gt_{config['history_bucket_maxima'][-1]}",
            )
            target = targets[alias]
            bucket = target_bucket(target, frequencies, config)
            for name, (candidates, diagnostic) in variants.items():
                ranking = [c.item_id for c in candidates]
                if (
                    len(ranking) > 100
                    or len(set(ranking)) != len(ranking)
                    or set(ranking) - frequencies.keys()
                    or seen & set(ranking)
                ):
                    raise ValueError(f"Invalid allocated ranking for {alias}/{name}")
                row = {
                    "user_alias": alias,
                    "policy": name,
                    "history_length": len(history),
                    "history_bucket": history_bucket,
                    "target_bucket": bucket,
                    "unavailable_target": int(bucket == "unavailable"),
                    "candidate_hit_at_100": hit_rate(ranking, target, 100),
                    "hr_at_10": hit_rate(ranking, target),
                    "ndcg_at_10": ndcg(ranking, {target: 1}),
                    "candidate_count": len(ranking),
                    "candidate_shortage": int(len(ranking) < 100),
                    "native_support": len(rankings["itemcf_native"]),
                    "fallback": int(
                        name == "itemcf" and any(c.source == "popularity" for c in candidates[:10])
                    ),
                    "reallocated_slots": diagnostic.get("reallocated_slots", 0),
                    "allocation_ms": diagnostic.get("allocation_ms"),
                    "all_variants_allocation_ms": allocation_ms,
                    "route": diagnostic.get("route"),
                    **{
                        f"requested_{c}": diagnostic.get("requested_quotas", {}).get(c, 0)
                        for c in CHANNELS
                    },
                    **{
                        f"filled_{c}": diagnostic.get("filled_quotas", {}).get(c, 0)
                        for c in CHANNELS
                    },
                    **{
                        f"top10_credit_{c}": sum(
                            x.evidence.get("quota_credit") == c for x in candidates[:10]
                        )
                        for c in [*CHANNELS, "reallocated"]
                    },
                }
                rows.append(row)
                for axis, value in [
                    ("overall", "all"),
                    ("history", history_bucket),
                    ("target", bucket),
                ]:
                    appearances[(name, axis, value)].update(ranking[:10])
                traces.append(
                    {
                        "user_alias": alias,
                        "policy": name,
                        "item_ids": ranking,
                        "quota_credits": [c.evidence.get("quota_credit", "") for c in candidates],
                        "requested_quotas": [
                            diagnostic.get("requested_quotas", {}).get(c, 0) for c in CHANNELS
                        ],
                        "filled_quotas": [
                            diagnostic.get("filled_quotas", {}).get(c, 0) for c in CHANNELS
                        ],
                        "reallocated_slots": diagnostic.get("reallocated_slots", 0),
                    }
                )
            simulation_aliases.append(alias)
            rewards.append(
                [
                    hit_rate([c.item_id for c in variants["quota:" + name][0]], target)
                    for name in arms
                ]
            )
            simulation_lists.append(
                [[c.item_id for c in variants["quota:" + name][0][:10]] for name in arms]
            )
            if index % 32 == 31:
                trace_writer.write_table(pa.Table.from_pylist(traces, schema=TRACE_SCHEMA))
                traces.clear()
            if (index + 1) % 2000 == 0:
                print(f"{split}: {index + 1}/{len(targets)} allocation requests", flush=True)
        if traces:
            trace_writer.write_table(pa.Table.from_pylist(traces, schema=TRACE_SCHEMA))
    if not pilot_limit and aliases_seen != targets.keys():
        raise ValueError("Allocation cache omitted held-out users")
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "per_user.csv", index=False)
    metrics, slices = {}, []
    for name, group in frame.groupby("policy", sort=False):
        for axis, column in [
            ("overall", None),
            ("history", "history_bucket"),
            ("target", "target_bucket"),
        ]:
            groups = [("all", group)] if column is None else group.groupby(column, sort=False)
            for value, subset in groups:
                result = {
                    "policy": name,
                    "axis": axis,
                    "bucket": value,
                    "sample_count": len(subset),
                    **{
                        k: float(subset[k].mean())
                        for k in [
                            "candidate_hit_at_100",
                            "hr_at_10",
                            "ndcg_at_10",
                            "candidate_shortage",
                            "fallback",
                            "unavailable_target",
                        ]
                    },
                    "mean_reallocated_slots": float(subset.reallocated_slots.mean()),
                    "allocation_p50_ms": float(subset.allocation_ms.median())
                    if subset.allocation_ms.notna().any()
                    else None,
                    "allocation_p95_ms": float(subset.allocation_ms.quantile(0.95))
                    if subset.allocation_ms.notna().any()
                    else None,
                    **counter_statistics(
                        appearances[(name, axis, value)], frequencies, config["tail_frequency_max"]
                    ),
                }
                slices.append(result)
                if axis == "overall":
                    metrics[name] = result
    pd.DataFrame(slices).to_csv(folder / "slices.csv", index=False)
    write_json(folder / "slices.json", slices)
    quota_columns = [c for c in frame if c.startswith(("requested_", "filled_", "top10_credit_"))]
    frame.groupby("policy", sort=False)[quota_columns].mean().to_csv(
        folder / "quota_contributions.csv"
    )
    # Reproduce E1 baselines per user, rather than accepting cache provenance alone.
    e1 = pd.read_csv(cache / "recommendation_per_request.csv")
    for name, old_name in {
        "popularity": "popularity",
        "itemcf_native": "itemcf_native",
        "two_tower": "two_tower",
        "itemcf": "itemcf_served",
        "rrf": "popularity+itemcf_native+two_tower",
    }.items():
        fresh = frame[frame.policy.eq(name)].set_index("user_alias")
        old = e1[e1.variant.eq(old_name)].set_index("request_id").loc[fresh.index]
        for new_key, old_key in [
            ("candidate_hit_at_100", "candidate_recall"),
            ("hr_at_10", "hr_at_10"),
            ("ndcg_at_10", "ndcg_at_10"),
        ]:
            if not (fresh[new_key] - old[old_key]).abs().lt(1e-12).all():
                raise ValueError(f"E1 baseline differs for {name}/{new_key}")
    comparisons = []
    for baseline in ["popularity", "rrf"]:
        for name in metrics:
            for key in METRICS:
                base, value = metrics[baseline][key], metrics[name][key]
                comparisons.append(
                    {
                        "dataset": "amazon_baby_rec",
                        "split": split,
                        "metric": key,
                        "baseline_algorithm": baseline,
                        "candidate_algorithm": name,
                        "baseline_value": base,
                        "candidate_value": value,
                        "delta": value - base,
                        "delta_pct": 100 * (value - base) / abs(base) if base else None,
                        "comparability_note": "Same full held-out cohort, train catalog, seen exclusion, B100/K10",
                    }
                )
    pd.DataFrame(comparisons).to_csv(folder / "comparisons.csv", index=False)
    for name in metrics:
        group = frame[frame.policy.eq(name)].set_index("user_alias")
        base = frame[frame.policy.eq("popularity")].set_index("user_alias")
        frame.loc[frame.policy.eq(name), "delta_hr_vs_popularity"] = (
            group.hr_at_10 - base.hr_at_10
        ).to_numpy()
        frame.loc[frame.policy.eq(name), "delta_ndcg_vs_popularity"] = (
            group.ndcg_at_10 - base.ndcg_at_10
        ).to_numpy()
    frame.to_csv(folder / "per_user.csv", index=False)
    if split == "validation":

        def best(prefix):
            names = [name for name in metrics if name.startswith(prefix)]
            return max(
                names, key=lambda name: (metrics[name]["ndcg_at_10"], metrics[name]["hr_at_10"])
            )

        frozen = {
            "split": split,
            "scope": "pilot" if pilot_limit else "complete_fixed_cohort",
            "validation_run": run_id,
            "signature": allocation_signature(root),
            "evaluation_signature": signature,
            "config": config,
            "fixed_policy": best("quota:").split(":", 1)[1],
            "adaptive_policy": best("adaptive:").split(":", 1)[1],
        }
        write_json(folder / "selection.json", frozen)
    fixed_arm = arms.index(frozen["fixed_policy"])
    summaries, logs = simulate(
        simulation_aliases,
        rewards,
        simulation_lists,
        arms,
        fixed_arm,
        config["simulation"],
        frequencies,
        config["tail_frequency_max"],
    )
    pd.DataFrame(logs).to_parquet(folder / "simulation_actions.parquet", index=False)
    write_json(folder / "simulation_seeds.json", summaries)
    simulation_summary = (
        pd.DataFrame(summaries)
        .groupby("policy", sort=False)
        .agg(
            seed_count=("seed", "size"),
            mean_proxy_hit_rate=("proxy_hit_rate", "mean"),
            std_proxy_hit_rate=("proxy_hit_rate", "std"),
            mean_proxy_regret=("contextual_proxy_regret", "mean"),
            mean_catalog_coverage=("catalog_coverage_at_10", "mean"),
            mean_tail_list_share=("tail_list_share_at_10", "mean"),
            mean_list_hhi=("list_concentration_hhi", "mean"),
        )
    )
    simulation_summary["delta_proxy_hit_vs_fixed"] = (
        simulation_summary.mean_proxy_hit_rate
        - simulation_summary.loc["fixed", "mean_proxy_hit_rate"]
    )
    simulation_summary.to_csv(folder / "simulation_summary.csv")
    write_json(
        folder / "simulation_summary.json", simulation_summary.reset_index().to_dict("records")
    )
    cold_checks = {}
    cold_model = RecPipeline(
        model.popularity,
        model.itemcf.similarity,
        model.aliases,
        model.manifest,
        model.two_tower,
        frozen,
    )
    for algorithm in ["itemcf", "rrf", "quota", "adaptive_quota"]:
        candidates, reason = cold_model.recommend(RecommendationContext(None), 10, algorithm)
        cold_checks[algorithm] = reason == "cold_user" and candidates == model.popularity.recommend(
            RecommendationContext(None), 10
        )
    if not all(cold_checks.values()):
        raise ValueError("Cold-user train-popularity fallback failed")
    result = {
        "status": "completed",
        "run_id": run_id,
        "split": split,
        "scope": "pilot" if pilot_limit else "complete_fixed_cohort",
        "sample_count": len(aliases_seen),
        "catalog_size": len(frequencies),
        "metrics": metrics,
        "unavailable_target_count": sum(targets[a] not in frequencies for a in aliases_seen),
        "cold_user_checks": cold_checks,
        "item_bucket_counts": dict(
            Counter(target_bucket(i, frequencies, config) for i in frequencies)
        ),
        "selected_fixed_policy": frozen["fixed_policy"],
        "selected_adaptive_policy": frozen["adaptive_policy"],
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu,
        "limitations": [
            "Positive ratings are preference proxies, not clicks or purchases",
            "Quota allocation does not reduce channel retrieval work",
            "Simulation uses counterfactual rating-target proxy outcomes; not IPS/DR or actual exposure replay",
            "Contextual proxy oracle is a diagnostic upper bound, not observed feedback",
            "Frequency buckets are train-only; low-frequency is not new-item cold start",
            "No Amazon metadata or item arrival timestamps are available locally",
        ],
    }
    write_json(folder / "metrics.json", result)
    if not pilot_limit:
        write_json(root / f"reports/allocation_{split}_latest.json", {"run_id": run_id})
        if split == "validation":
            write_json(root / "artifacts/recommendation_fusion/selection.json", frozen)
    print(
        {
            "run_id": run_id,
            "fixed": frozen["fixed_policy"],
            "adaptive": frozen["adaptive_policy"],
            "metrics": {
                k: {m: v[m] for m in ["candidate_hit_at_100", "hr_at_10", "ndcg_at_10"]}
                for k, v in metrics.items()
            },
        },
        flush=True,
    )
    return result
