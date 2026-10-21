import argparse
from pathlib import Path

from joysearch.common import ROOT
from joysearch.evaluation.query_eval import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    run(ROOT, args.split, args.selection)
