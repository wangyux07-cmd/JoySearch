import argparse

from joysearch.common import ROOT
from joysearch.evaluation.recall_eval import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--pilot-limit", type=int)
    args = parser.parse_args()
    if args.pilot_limit is not None and args.pilot_limit < 1:
        parser.error("--pilot-limit must be positive")
    run(ROOT, args.split, args.pilot_limit)
