"""Isolated single-process server with an explicit total worker/thread budget."""

import argparse

import uvicorn

from joysearch.api.main import create_app
from joysearch.common import ROOT, read_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--scan-mode", choices=["reference", "cached_controller"], default="reference"
    )
    args = parser.parse_args()
    config = read_json(ROOT / "configs/benchmark.json")
    uvicorn.run(
        create_app(
            worker_threads=config["worker_threads"],
            inference_threads=config["inference_threads"],
            scan_mode=args.scan_mode,
            enable_ltr=False,
        ),
        host="127.0.0.1",
        port=args.port,
        access_log=False,
    )
