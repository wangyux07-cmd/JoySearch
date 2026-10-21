import argparse
import json

from joysearch.common import ROOT
from joysearch.search.dense import DenseIndex
from joysearch.search.generative import GenerativeRetriever

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, choices=range(1, 101), default=10)
    args = parser.parse_args()
    if not args.query.strip() or len(args.query) > 500:
        parser.error("--query must contain 1..500 characters and non-whitespace text")
    dense = DenseIndex.load(ROOT / "artifacts/search_dense")
    model = GenerativeRetriever.load(ROOT, encoder=dense.encoder)
    print(
        json.dumps(
            [
                {
                    "product_id": c.item_id,
                    "rank": c.rank,
                    "log_probability": c.score,
                    "semantic_code": c.evidence["semantic_code"],
                }
                for c in model.retrieve(args.query, args.top_k)
            ],
            indent=2,
        )
    )
