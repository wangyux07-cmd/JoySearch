import argparse

from joysearch.common import ROOT
from joysearch.evaluation.report import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["all", "search", "recommendation"], default="all")
    args = parser.parse_args()
    run(ROOT, args.pipeline)
