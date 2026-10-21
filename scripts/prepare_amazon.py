import argparse
from pathlib import Path

from joysearch.common import ROOT
from joysearch.data.amazon import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/amazon/Baby_Products.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/amazon")
    parser.add_argument("--users", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.users < 0:
        parser.error("--users must be >= 0 (0 means all eligible users)")
    print(prepare(args.source, args.output, args.users, args.seed))
