import pandas as pd

from joysearch.common import ROOT, read_json
from joysearch.search.bm25 import BM25Index

if __name__ == "__main__":
    source = ROOT / "data/processed/wands"
    index = BM25Index.build(
        pd.read_parquet(source / "products.parquet"), read_json(source / "manifest.json")
    )
    index.save(ROOT / "artifacts/search")
    print(index.manifest)
    print(index.retrieve("salon chair", 3))
