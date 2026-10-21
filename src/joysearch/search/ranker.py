import importlib.metadata
import time

import numpy as np
import pandas as pd

from joysearch.common import Candidate, read_json, write_json
from joysearch.data.wands import file_hash
from joysearch.search.features import FEATURE_NAMES


def retrieval_signature(root):
    paths = [
        "configs/dense.json",
        "artifacts/search/manifest.json",
        "artifacts/search/weights.npz",
        "artifacts/search/vocabulary.json",
        "artifacts/search/products.parquet",
        "artifacts/search_dense/manifest.json",
        "artifacts/search_dense/item_ids.json",
        "artifacts/search_dense/embeddings.npy",
        *[
            f"src/joysearch/search/{name}.py"
            for name in ["bm25", "dense", "fusion", "features", "pipeline"]
        ],
    ]
    return {path: file_hash(root / path) for path in paths}


def query_groups(frame, allowed_qids):
    """Reject leakage and noncontiguous query groups before handing rows to LightGBM."""
    qids = frame.query_id.astype(str)
    if frame.empty or not set(qids) <= set(allowed_qids):
        raise ValueError("Training rows must belong to the requested nonempty query split")
    starts = qids.ne(qids.shift())
    if qids[starts].duplicated().any():
        raise ValueError("Query groups must be contiguous")
    groups = frame.groupby(starts.cumsum(), sort=False).size().tolist()
    if sum(groups) != len(frame):
        raise ValueError("Query group sizes do not cover feature rows")
    return groups


class LambdaMARTRanker:
    version = "lambdamart-v1"

    def __init__(self, booster, manifest):
        self.booster = booster
        self.manifest = manifest

    @classmethod
    def load(cls, root, signature):
        from lightgbm import Booster

        folder = root / "artifacts/search_ltr"
        manifest = read_json(folder / "manifest.json")
        if (
            manifest["version"] != cls.version
            or manifest["feature_names"] != FEATURE_NAMES
            or manifest["retrieval_signature"] != signature
        ):
            raise ValueError("LTR retrieval/features changed; run scripts/train_ranker.py")
        if manifest["model_sha256"] != file_hash(folder / "model.txt"):
            raise ValueError("LTR model checksum mismatch")
        booster = Booster(model_file=str(folder / "model.txt"))
        if booster.feature_name() != FEATURE_NAMES:
            raise ValueError("LTR model feature order mismatch")
        return cls(booster, manifest)

    def rerank(self, candidates, features, k):
        if not candidates:
            return []
        scores = self.booster.predict(features, num_threads=1)
        ids = np.asarray([c.item_id for c in candidates])
        order = np.lexsort((ids, -scores))[:k]
        return [
            Candidate(
                "wands_search",
                candidates[i].item_id,
                float(scores[i]),
                "ltr",
                rank,
                {
                    **candidates[i].evidence,
                    "reranker": self.version,
                    "hybrid_rank": candidates[i].rank,
                    "feature_values": features.iloc[i].to_dict(),
                },
            )
            for rank, i in enumerate(order, 1)
        ]


def train(root):
    from lightgbm import LGBMRanker, early_stopping

    from joysearch.search.pipeline import SearchPipeline

    started = time.perf_counter()
    source, output = root / "data/processed/wands", root / "artifacts/search_ltr"
    config = read_json(root / "configs/ranker.json")
    manifest = read_json(source / "manifest.json")
    splits = read_json(source / "splits.json")
    if splits != manifest["query_splits"]:
        raise ValueError("Training split differs from frozen WANDS manifest")
    if any(
        set(splits[a]) & set(splits[b])
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("WANDS query splits overlap")
    signature = retrieval_signature(root)
    training_signature = {
        "retrieval": signature,
        "config": config,
        "files": {
            name: file_hash(source / name)
            for name in ["queries.parquet", "qrels.parquet", "splits.json", "manifest.json"]
        },
        "training_code": file_hash(root / "src/joysearch/search/ranker.py"),
        "lightgbm": importlib.metadata.version("lightgbm"),
    }
    final = output / "manifest.json"
    if final.exists() and read_json(final)["training_signature"] == training_signature:
        LambdaMARTRanker.load(root, signature)
        print("Matching trained LambdaMART artifact reused", flush=True)
        return read_json(final)
    cache = output / "candidate_cache.json"
    cache_paths = [output / f"{split}.parquet" for split in ["train", "validation"]]
    cache_valid = cache.exists() and all(p.exists() for p in cache_paths)
    cached = read_json(cache) if cache_valid else {}
    cache_valid = (
        cache_valid
        and cached["signature"] == training_signature
        and cached["sha256"] == {p.name: file_hash(p) for p in cache_paths}
    )
    if not cache_valid:
        model = SearchPipeline.load(root, enable_ltr=False)
        if model.index.manifest["data_manifest"]["sources"] != manifest["sources"]:
            raise ValueError("LTR index/source data mismatch")
        queries = pd.read_parquet(source / "queries.parquet").set_index("query_id")["query"]
        qrels = pd.read_parquet(source / "qrels.parquet")
        # Restrict labels before candidate construction; test judgments never enter fitting.
        qrels = qrels[qrels.query_id.isin(splits["train"] + splits["validation"])]
        labels = {(r.query_id, r.product_id): r.grade for r in qrels.itertuples()}
        for split, path in zip(["train", "validation"], cache_paths):
            frames = []
            for qid in splits[split]:
                candidates, rankings = model.hybrid_candidates(queries[qid])
                frame = model.features.transform(
                    queries[qid],
                    candidates,
                    rankings,
                    model.candidate_k,
                )
                frame["query_id"] = qid
                frame["product_id"] = [c.item_id for c in candidates]
                frame["grade"] = [labels.get((qid, c.item_id), np.nan) for c in candidates]
                frames.append(frame)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
            print(f"Cached {split}: {len(frames)} retrieved query pools", flush=True)
        write_json(
            cache,
            {
                "signature": training_signature,
                "sha256": {p.name: file_hash(p) for p in cache_paths},
            },
        )
    frames, groups, counts = [], [], {}
    for split, path in zip(["train", "validation"], cache_paths):
        pool = pd.read_parquet(path)
        query_groups(pool, splits[split])
        frame = pool.dropna(subset=["grade"]).copy()
        group = query_groups(frame, splits[split])
        frames.append(frame)
        groups.append(group)
        counts[split] = {
            "queries": len(splits[split]),
            "judged_queries": len(group),
            "candidate_rows": len(pool),
            "judged_rows": len(frame),
            "unjudged_rows_excluded": len(pool) - len(frame),
            "label_counts": frame.grade.astype(int).value_counts().sort_index().to_dict(),
            "qids": list(splits[split]),
        }
    training, validation = frames
    ranker = LGBMRanker(**config["model"])
    ranker.fit(
        training[FEATURE_NAMES],
        training.grade.astype(int),
        group=groups[0],
        eval_set=[(validation[FEATURE_NAMES], validation.grade.astype(int))],
        eval_group=[groups[1]],
        eval_at=[10],
        callbacks=[early_stopping(config["early_stopping_rounds"], first_metric_only=True)],
    )
    ranker.booster_.save_model(str(output / "model.txt"))
    result = {
        "version": LambdaMARTRanker.version,
        "catalog_id": "wands_search",
        "feature_names": FEATURE_NAMES,
        "retrieval_signature": signature,
        "training_signature": training_signature,
        "config": config,
        "counts": counts,
        "candidate_policy": "RRF Top100 from full-catalog BM25/Dense Top100 each; no qrel injection",
        "label_policy": "Fit/evaluate early stopping on judged retrieved candidates only; gains [0,1,3]",
        "selection_policy": "Single fixed config; validation judged-only NDCG@10 early stopping; no test tuning",
        "best_iteration": ranker.best_iteration_,
        "validation_judged_only": ranker.best_score_,
        "feature_importance_gain": dict(
            zip(FEATURE_NAMES, ranker.booster_.feature_importance("gain").tolist())
        ),
        "model_sha256": file_hash(output / "model.txt"),
        "build_seconds": time.perf_counter() - started,
    }
    write_json(final, result)
    print(
        f"Saved LambdaMART: iteration={ranker.best_iteration_}, "
        f"judged rows train={len(training)}, validation={len(validation)}",
        flush=True,
    )
    return result
