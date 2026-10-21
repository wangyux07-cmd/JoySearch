import argparse

from joysearch.common import ROOT
from joysearch.search.generative_training import fit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    print(fit(ROOT, args.pilot), flush=True)
