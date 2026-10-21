import argparse

from joysearch.common import ROOT
from joysearch.evaluation.report import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["all", "search", "recommendation"], default="all")
    parser.add_argument(
        "--search-algorithms", nargs="+", choices=["bm25", "dense", "hybrid"], default=["bm25"]
    )
    parser.add_argument("--search-split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    run(ROOT, args.pipeline, args.search_algorithms, args.search_split)
