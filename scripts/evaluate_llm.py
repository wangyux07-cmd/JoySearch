import argparse
from pathlib import Path

from joysearch.common import ROOT
from joysearch.evaluation.llm_eval import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bounded local-LLM diagnostics; not serving")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    print(run(ROOT, args.split, args.selection))
