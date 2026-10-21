import hashlib
import time

import numpy as np
import pandas as pd

from joysearch.common import write_json

GRADES = {"Irrelevant": 0, "Partial": 1, "Exact": 2}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(source, output, seed=42):
    started = time.perf_counter()
    products = pd.read_csv(source / "product.csv", sep="\t", dtype={"product_id": str})
    queries = pd.read_csv(source / "query.csv", sep="\t", dtype={"query_id": str})
    labels = pd.read_csv(source / "label.csv", sep="\t", dtype={"query_id": str, "product_id": str})
    required = {
        "product_id",
        "product_name",
        "product_class",
        "category hierarchy",
        "product_features",
        "product_description",
    }
    if not required <= set(products) or not {"query_id", "query"} <= set(queries):
        raise ValueError("WANDS product/query schema does not match required columns")
    if not {"query_id", "product_id", "label"} <= set(labels):
        raise ValueError("WANDS label schema does not match required columns")
    for frame, key in [(products, "product_id"), (queries, "query_id")]:
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"WANDS invalid or duplicate {key}")
    if queries["query"].isna().any():
        raise ValueError("WANDS missing query text")
    if not labels.label.isin(GRADES).all():
        raise ValueError(f"Unknown WANDS labels: {set(labels.label) - GRADES.keys()}")
    if not labels.product_id.isin(products.product_id).all():
        raise ValueError("WANDS labels reference unknown products")
    if not labels.query_id.isin(queries.query_id).all():
        raise ValueError("WANDS labels reference unknown queries")
    raw_judgments = len(labels)
    labels["grade"] = labels.label.map(GRADES)
    conflicts = int(labels.groupby(["query_id", "product_id"]).grade.nunique().gt(1).sum())
    labels = labels.groupby(["query_id", "product_id"], as_index=False).grade.min()
    ids = np.random.default_rng(seed).permutation(sorted(queries.query_id))
    first, second = int(len(ids) * 0.6), int(len(ids) * 0.8)
    splits = {
        name: sorted(part.tolist())
        for name, part in zip(
            ["train", "validation", "test"], [ids[:first], ids[first:second], ids[second:]]
        )
    }
    normalized = queries.assign(normalized=queries["query"].str.strip().str.lower())
    duplicate_text = normalized[normalized.normalized.duplicated(False)].to_dict("records")
    output.mkdir(parents=True, exist_ok=True)
    products.to_parquet(output / "products.parquet", index=False)
    queries.to_parquet(output / "queries.parquet", index=False)
    labels.to_parquet(output / "qrels.parquet", index=False)
    write_json(output / "splits.json", splits)
    manifest = {
        "dataset": "wands_search",
        "version": "wands-v1",
        "seed": seed,
        "sources": {
            name: {"sha256": file_hash(source / name), "bytes": (source / name).stat().st_size}
            for name in ["product.csv", "query.csv", "label.csv"]
        },
        "catalog_size": len(products),
        "queries": len(queries),
        "judgments": len(labels),
        "raw_judgments": raw_judgments,
        "duplicate_pairs_removed": raw_judgments - len(labels),
        "conflicting_pairs": conflicts,
        "qrel_resolution": "minimum grade per query/product pair; fixed before evaluation",
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "query_splits": splits,
        "relevance_mapping": GRADES,
        "duplicate_normalized_query_text": duplicate_text,
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(output / "manifest.json", manifest)
    return manifest
