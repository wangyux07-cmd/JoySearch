import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime

from joysearch.common import ROOT, read_json, write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_algorithms = (
        ["bm25", "dense", "hybrid"]
        if (ROOT / "artifacts/search_dense/manifest.json").exists()
        else ["bm25"]
    )
    if (ROOT / "artifacts/search_ltr/manifest.json").exists():
        default_algorithms.append("ltr")
    parser.add_argument(
        "--search-algorithms",
        nargs="+",
        choices=["bm25", "dense", "hybrid", "ltr", "query_hybrid"],
        default=default_algorithms,
    )
    default_rec = ["popularity", "itemcf"]
    if (ROOT / "artifacts/recommendation_two_tower/manifest.json").exists():
        default_rec.append("two_tower")
    parser.add_argument(
        "--recommendation-algorithms",
        nargs="+",
        choices=["popularity", "itemcf", "two_tower"],
        default=default_rec,
    )
    if "dense" in default_algorithms:
        default_algorithms.append("query_hybrid")
    parser.add_argument("--recall-diagnostics", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "reports/quality" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder.mkdir(parents=True)
    commands = [
        ([sys.executable, "-m", "pytest", "-q"], ROOT, "pytest"),
        ([sys.executable, "-m", "ruff", "check", "."], ROOT, "ruff"),
        (["npm.cmd", "run", "build"], ROOT / "frontend", "frontend_build"),
        (
            [
                sys.executable,
                "scripts/evaluate.py",
                "--pipeline",
                "all",
                "--search-algorithms",
                *args.search_algorithms,
                "--recommendation-algorithms",
                *args.recommendation_algorithms,
            ],
            ROOT,
            "real_evaluations",
        ),
        ([sys.executable, "scripts/smoke.py"], ROOT, "api_smoke"),
        (
            [sys.executable, "scripts/smoke.py", "--base-url", "http://127.0.0.1:8000"],
            ROOT,
            "http_smoke",
        ),
    ]
    if args.recall_diagnostics:
        commands.insert(
            3,
            (
                [sys.executable, "scripts/evaluate_recall.py", "--split", "test"],
                ROOT,
                "recall_diagnostics",
            ),
        )
    results = []
    for command, cwd, name in commands:
        started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        (folder / f"{name}.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        row = {
            "check": name,
            "command": subprocess.list2cmdline(command),
            "cwd": str(cwd),
            "exit_code": result.returncode,
            "duration_seconds": time.perf_counter() - started,
            "output": str((folder / f"{name}.txt").relative_to(ROOT)),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    write_json(
        folder / "results.json",
        {
            "checks": results,
            "latest_metrics": read_json(ROOT / "reports/latest.json"),
            "latest_recall": read_json(ROOT / "reports/recall_latest.json")
            if args.recall_diagnostics and (ROOT / "reports/recall_latest.json").exists()
            else None,
        },
    )
    write_json(
        ROOT / "reports/quality/latest.json",
        {
            "folder": str(folder.relative_to(ROOT)),
            "pass": all(r["exit_code"] == 0 for r in results),
        },
    )
    raise SystemExit(0 if all(r["exit_code"] == 0 for r in results) else 1)
