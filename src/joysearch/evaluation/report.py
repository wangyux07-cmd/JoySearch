import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd

from joysearch.common import read_json, write_json
from joysearch.data.wands import file_hash
from joysearch.evaluation import rec_eval, search_eval
from joysearch.search.pipeline import SearchPipeline


def code_manifest(root):
    files = sorted(
        [
            *root.glob("src/**/*.py"),
            *root.glob("scripts/*.py"),
            *root.glob("frontend/src/*"),
            *root.glob("configs/*"),
            *root.glob("frontend/*.json"),
            *root.glob("frontend/*.ts"),
            root / "requirements-lock.txt",
            root / "requirements-dense-lock.txt",
            root / "requirements-rank-lock.txt",
            root / "pyproject.toml",
        ]
    )
    hashes = {
        str(p.relative_to(root)).replace("\\", "/"): file_hash(p) for p in files if p.is_file()
    }
    digest = hashlib.sha256(str(sorted(hashes.items())).encode()).hexdigest()
    return {"sha256": digest, "files": hashes}


def run(root, pipeline="all", search_algorithms=("bm25",), search_split="test"):
    if search_split != "test" and pipeline != "search":
        raise ValueError("Validation split is available only with --pipeline search")
    if (
        not search_algorithms
        or search_algorithms[0] != "bm25"
        or len(set(search_algorithms)) != len(search_algorithms)
    ):
        raise ValueError("Search comparison must start with bm25 and contain no duplicates")
    run_id = "run_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    folder = root / "reports" / run_id
    folder.mkdir(parents=True)
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--short"], cwd=root, capture_output=True, text=True, check=False
    )
    repeated = (root / "reports/latest.json").exists()
    config = {
        "seed": 42,
        "pipeline": pipeline,
        "search_algorithms": list(search_algorithms),
        "search_split": search_split,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "git_sha": git.stdout.strip() if git.returncode == 0 else None,
        "git_sha_reason": None if git.returncode == 0 else "Repository has no HEAD commit",
        "git_status": status.stdout.splitlines(),
        "code": code_manifest(root),
        "test_reuse": repeated and search_split == "test",
        "test_reuse_note": "Regression reproducibility rerun; not untouched generalisation"
        if repeated
        else "First fixed-parameter test evaluation; no test-based model selection",
        "packages": {
            p: importlib.metadata.version(p)
            for p in [
                "joysearch",
                "numpy",
                "scipy",
                "pandas",
                "pyarrow",
                "fastapi",
                "pydantic",
                "uvicorn",
                "pytest",
                "ruff",
                "httpx",
            ]
        },
        "command": (
            "conda run -n joysearch python scripts/evaluate.py --pipeline "
            + pipeline
            + " --search-algorithms "
            + " ".join(search_algorithms)
            + " --search-split "
            + search_split
        ),
    }
    artifact_directories = []
    if "ltr" in search_algorithms:
        artifact_directories.append(root / "artifacts/search_ltr")
        config["packages"]["lightgbm"] = importlib.metadata.version("lightgbm")
    if pipeline in ["all", "search"]:
        artifact_directories.append(root / "artifacts/search")
        if any(algorithm != "bm25" for algorithm in search_algorithms):
            artifact_directories.append(root / "artifacts/search_dense")
            dense_manifest = read_json(root / "artifacts/search_dense/manifest.json")
            artifact_directories.append(root / dense_manifest["model_path"])
    if pipeline in ["all", "recommendation"]:
        artifact_directories.append(root / "artifacts/recommendation")
    config["model_artifact_sha256"] = {
        p.relative_to(root).as_posix(): file_hash(p)
        for directory in artifact_directories
        for p in sorted(directory.rglob("*"))
        if p.is_file()
        and ".cache" not in p.parts
        and p.name
        not in ["build_state.json", "candidate_cache.json", "train.parquet", "validation.parquet"]
    }
    if any(algorithm != "bm25" for algorithm in search_algorithms):
        config["packages"].update(
            {
                name: importlib.metadata.version(name)
                for name in [
                    "torch",
                    "sentence-transformers",
                    "transformers",
                    "tokenizers",
                    "huggingface-hub",
                ]
            }
        )
    if search_split == "validation":
        config["test_reuse_note"] = (
            "Validation-only protocol check; fixed parameters, no hyperparameter search"
        )
    data = {
        name: read_json(root / f"data/processed/{directory}/manifest.json")
        for name, directory in [("search", "wands"), ("recommendation", "amazon")]
        if pipeline in ["all", name]
    }
    if any(manifest["seed"] != config["seed"] for manifest in data.values()):
        raise ValueError("P0 evaluation requires seed 42 for both frozen splits")
    # Validate source/processed integrity once at the evaluation boundary.
    directories = {"search": "wands", "recommendation": "amazon"}
    for domain, manifest in data.items():
        directory = directories[domain]
        for name, metadata in manifest["sources"].items():
            if file_hash(root / f"data/raw/{directory}/{name}") != metadata["sha256"]:
                raise ValueError(f"Raw source changed since preparation: {name}")
    events = root / "data/processed/amazon/events.parquet"
    if "recommendation" in data and file_hash(events) != data["recommendation"]["split_sha256"]:
        raise ValueError("Amazon processed event split hash mismatch")
    write_json(folder / "config.json", config)
    write_json(folder / "data_manifest.json", data)
    not_run = {"status": "not_run", "reason": f"CLI selected pipeline={pipeline}", "metrics": None}
    search, rec, latency = not_run.copy(), not_run.copy(), {}
    if pipeline in ["all", "search"]:
        model = SearchPipeline.load(root, enable_dense=any(a != "bm25" for a in search_algorithms))
        algorithm_reports, search_latency, frames = {}, {}, []
        for algorithm in search_algorithms:
            if algorithm not in model.algorithms:
                raise ValueError(f"Requested search algorithm unavailable: {algorithm}")
            metrics, timing, rows = search_eval.evaluate(
                root / "data/processed/wands",
                root / "artifacts/search",
                algorithm,
                search_split,
                model,
            )
            algorithm_reports[algorithm] = metrics
            search_latency[algorithm] = timing
            frames.append(rows)
        search = {**algorithm_reports["bm25"], "algorithms": algorithm_reports}
        latency["search"] = search_latency
        pd.concat(frames, ignore_index=True).to_csv(folder / "search_per_query.csv", index=False)
    if pipeline in ["all", "recommendation"]:
        rec, latency["recommendation"], rows = rec_eval.evaluate(
            root / "data/processed/amazon", root / "artifacts/recommendation"
        )
        rows.to_csv(folder / "recommendation_per_user.csv", index=False)
    if search["status"] == "completed":
        for report in search["algorithms"].values():
            assert all(0 <= value <= 1 for value in report["metrics"].values())
    if rec["status"] == "completed":
        for metrics in rec["algorithms"].values():
            assert all(
                0 <= metrics[key] <= 1
                for key in ["hr_at_10", "ndcg_at_10", "catalog_coverage_at_10", "fallback_rate"]
            )
    comparisons = []
    if search["status"] == "completed":
        for algorithm in search_algorithms[1:]:
            for key, baseline in search["metrics"].items():
                candidate = search["algorithms"][algorithm]["metrics"][key]
                comparisons.append(
                    {
                        "dataset": "wands_search",
                        "split": search_split,
                        "metric": key,
                        "baseline_algorithm": "bm25",
                        "candidate_algorithm": algorithm,
                        "baseline_value": baseline,
                        "candidate_value": candidate,
                        "delta": candidate - baseline,
                        "delta_pct": 100 * (candidate - baseline) / abs(baseline)
                        if baseline
                        else None,
                        "comparability_note": "Same frozen query IDs, complete WANDS catalog, qrels and formulas",
                    }
                )
    if rec["status"] == "completed":
        for key in ["hr_at_10", "ndcg_at_10", "catalog_coverage_at_10", "fallback_rate"]:
            baseline, candidate = [rec["algorithms"][a][key] for a in ["popularity", "itemcf"]]
            comparisons.append(
                {
                    "dataset": "amazon_baby_rec",
                    "split": "test",
                    "metric": key,
                    "baseline_algorithm": "popularity",
                    "candidate_algorithm": "itemcf",
                    "baseline_value": baseline,
                    "candidate_value": candidate,
                    "delta": candidate - baseline,
                    "delta_pct": 100 * (candidate - baseline) / abs(baseline) if baseline else None,
                    "comparability_note": "Same users, train universe, seen filter and metric; "
                    "zero baseline makes relative delta undefined"
                    if baseline == 0
                    else "Same users, full train catalog, train-positive exclusion",
                }
            )
    if search["status"] == "completed" and {"hybrid", "ltr"} <= set(search_algorithms):
        for key, baseline in search["algorithms"]["hybrid"]["metrics"].items():
            candidate = search["algorithms"]["ltr"]["metrics"][key]
            comparisons.append(
                {
                    "dataset": "wands_search",
                    "split": search_split,
                    "metric": key,
                    "baseline_algorithm": "hybrid",
                    "candidate_algorithm": "ltr",
                    "baseline_value": baseline,
                    "candidate_value": candidate,
                    "delta": candidate - baseline,
                    "delta_pct": 100 * (candidate - baseline) / abs(baseline) if baseline else None,
                    "comparability_note": "Same frozen queries, qrels and exact RRF Top100 pool; reranking only",
                }
            )
    columns = [
        "dataset",
        "split",
        "metric",
        "baseline_algorithm",
        "candidate_algorithm",
        "baseline_value",
        "candidate_value",
        "delta",
        "delta_pct",
        "comparability_note",
    ]
    pd.DataFrame(comparisons, columns=columns).to_csv(folder / "comparisons.csv", index=False)
    write_json(folder / "search_metrics.json", search)
    write_json(folder / "recommendation_metrics.json", rec)
    write_json(folder / "latency.json", latency)
    lines = [f"# JoySearch 实测报告 — {run_id}", ""]
    if "search" in data:
        manifest = data["search"]
        lines.append(
            f"WANDS：{manifest['catalog_size']} 商品、{manifest['queries']} 查询，"
            f"split={manifest['split_sizes']}。BM25 k1=1.2，b=.75，标题重复两次。"
        )
    if "recommendation" in data:
        manifest = data["recommendation"]
        lines.append(
            f"Amazon：{manifest['raw_events']} 条原始评分，{manifest['sample_count']} 用户，"
            f"split={manifest['split_sizes']}，{manifest['catalog_size']} train items。"
            "ItemCF Top100、shrinkage=10。"
        )
    lines += [
        "",
        f"执行状态：search={search['status']}，recommendation={rec['status']}。",
        (
            "seed=42；WANDS 按 query-ID 60/20/20；Amazon 最新评分去重后 rating>=4，"
            "每用户按 (timestamp,item_id) 留最后两条。验证／测试不参与训练。"
        ),
        "",
        "| 算法 | 指标 | 实测值 |",
        "|---|---|---|",
    ]
    if search["status"] == "completed":
        for algorithm, report in search["algorithms"].items():
            lines.extend(
                f"| {algorithm} | {key} | {value:.8f} |" for key, value in report["metrics"].items()
            )
    if rec["status"] == "completed":
        for algorithm, values in rec["algorithms"].items():
            lines.extend(
                f"| {algorithm} | {key} | {values[key]:.8f} |"
                for key in ["hr_at_10", "ndcg_at_10", "catalog_coverage_at_10", "fallback_rate"]
            )
        lines += [
            "",
            (
                f"测试目标不在训练库：{rec['unavailable_test_item_count']}/{rec['users_evaluated']}，"
                "全部保留在指标分母。"
            ),
            "",
            "| ItemCF − Popularity | 绝对差值 | 相对变化 % |",
            "|---|---|---|",
        ]
        lines.extend(
            f"| {r['metric']} | {r['delta']:+.8f} | {r['delta_pct']} |"
            for r in comparisons
            if r["dataset"] == "amazon_baby_rec"
        )
    if search["status"] == "completed" and len(search_algorithms) > 1:
        lines += [
            "",
            "搜索对照：同一固定查询与完整商品库，正负差值均保存于 comparisons.csv。",
            "MiniLM 固定 revision、128 tokens、384维归一化向量；RRF k=60，每路 Top100，等权。",
            "召回配置固定；LTR 仅用 train 标注拟合、validation judged-only NDCG@10 早停，test 不调参。",
        ]
    if search["status"] == "completed" and len(search_algorithms) > 1:
        lines += ["", "| 搜索候选 / 基线 | 指标 | 绝对差 | 相对变化 % |", "|---|---|---|---|"]
        for row in comparisons:
            if row["dataset"] == "wands_search":
                lines.append(
                    f"| {row['candidate_algorithm']} / {row['baseline_algorithm']} | {row['metric']} | {row['delta']:+.8f} | {row['delta_pct']} |"
                )
    lines += [
        "",
        (
            "延迟单位 ms，模型先加载、5 次预热后按完整测试集顺序测量；只含领域推理，"
            "不含 HTTP/网络、加载或训练。"
        ),
        "```json",
        json.dumps(latency, indent=2),
        "```",
        "",
        "构建耗时在 metrics.config.build_seconds；运行环境、代码和数据哈希在 config/data_manifest。",
        "",
        (
            "限制：WANDS 稀疏标注，未标注项按 0 gain，JudgedRecall 不是全库真实召回率；"
            "重复冲突 qrels 固定取最低 grade。Amazon 为正向评分代理，非点击／购买，"
            "采用抽样用户、每用户留后法，非全局时间切分；两个 catalog 无共同身份，"
            "推荐没有商品元数据；无线上业务提升结论。"
        ),
        f"测试集重复使用：{config['test_reuse_note']}",
        "",
        "复现：",
        "```powershell",
        config["command"],
        "```",
    ]
    (folder / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    if pipeline == "all" and search_split == "test":
        write_json(root / "reports/latest.json", {"status": "verified", "run_id": run_id})
    print(f"Report: {folder}")
    if search["status"] == "completed":
        print("Search:", search["metrics"])
    if rec["status"] == "completed":
        print("Recommendation:", rec["algorithms"])
    return folder
