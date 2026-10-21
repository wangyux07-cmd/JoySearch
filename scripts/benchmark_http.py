import argparse
import asyncio

from joysearch.common import ROOT
from joysearch.evaluation.http_benchmark import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    print(asyncio.run(run(ROOT, args.base_url, args.pilot)), flush=True)
