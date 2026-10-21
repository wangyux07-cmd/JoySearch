import argparse
from pathlib import Path

from joysearch.common import ROOT
from joysearch.data.wands import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/wands")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/wands")
    args = parser.parse_args()
    print(prepare(args.source, args.output))
