import argparse
from pathlib import Path

from joysearch.common import ROOT
from joysearch.evaluation.allocation_eval import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--pilot-limit", type=int)
    args = parser.parse_args()
    run(ROOT, args.split, args.cache, args.selection, args.pilot_limit)
