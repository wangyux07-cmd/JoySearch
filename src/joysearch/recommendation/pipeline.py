import time

import pandas as pd
from scipy import sparse

from joysearch.common import read_json, write_json
from joysearch.recommendation.itemcf import ItemCF, build_similarity
from joysearch.recommendation.popularity import Popularity


class RecPipeline:
    version = "rec-v1"

    def __init__(self, popularity, similarity, aliases, manifest, two_tower=None):
        self.popularity = popularity
        self.itemcf = ItemCF(similarity, popularity)
        self.aliases = aliases
        self.manifest = manifest
        self.two_tower = two_tower
        self.algorithms = ["popularity", "itemcf"] + (["two_tower"] if two_tower else [])
        self.versions = {"popularity": self.version, "itemcf": self.version}
        if two_tower:
            self.versions["two_tower"] = two_tower.version

    @classmethod
    def build(cls, train, data_manifest=None, top_n=100, shrinkage=10):
        started = time.perf_counter()
        if not train.split.eq("train").all():
            raise ValueError("Recommendation build requires train-only interactions")
        if train.duplicated(["user_id", "parent_asin"]).any():
            raise ValueError("Recommendation train has duplicate user/item pairs")
        users, items = sorted(train.user_id.unique()), sorted(train.parent_asin.unique())
        item_map = {item: i for i, item in enumerate(items)}
        histories = {
            u: [item_map[i] for i in group.parent_asin]
            for u, group in train.groupby("user_id", sort=True)
        }
        counts = train.groupby("parent_asin").size().reindex(items).tolist()
        similarity = build_similarity(train, users, items, top_n, shrinkage)
        aliases = {f"anonymous_user_{i:05d}": user for i, user in enumerate(users, 1)}
        manifest = {
            "version": cls.version,
            "catalog_id": "amazon_baby_rec",
            "top_n": top_n,
            "shrinkage": shrinkage,
            "users": len(users),
            "catalog_size": len(items),
            "train_events": len(train),
            "similarity_nonzero": similarity.nnz,
            "candidate_universe": "train_items_full_catalog",
            "zero_score_tie_break": "train popularity descending, item_id ascending",
            "data_manifest": data_manifest,
            "build_seconds": time.perf_counter() - started,
        }
        return cls(Popularity(items, counts, histories), similarity, aliases, manifest)

    def save(self, output):
        output.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(output / "similarity.npz", self.itemcf.similarity)
        write_json(
            output / "model.json",
            {
                "items": self.popularity.items.tolist(),
                "counts": self.popularity.counts.tolist(),
                "histories": self.popularity.histories,
                "aliases": self.aliases,
            },
        )
        write_json(output / "manifest.json", self.manifest)

    @classmethod
    def load(cls, output, enable_two_tower=True):
        manifest = read_json(output / "manifest.json")
        if manifest["version"] != cls.version or manifest["catalog_id"] != "amazon_baby_rec":
            raise ValueError(f"Incompatible recommendation artifact: {output}")
        model = read_json(output / "model.json")
        similarity = sparse.load_npz(output / "similarity.npz")
        if similarity.shape != (len(model["items"]), len(model["items"])):
            raise ValueError(f"Recommendation artifact dimensions disagree: {output}")
        popularity = Popularity(model["items"], model["counts"], model["histories"])
        two_tower = None
        path = output.parent / "recommendation_two_tower"
        if enable_two_tower and (path / "manifest.json").exists():
            from joysearch.recommendation.two_tower import TwoTower

            two_tower = TwoTower.load(path, popularity, manifest)
        return cls(popularity, similarity, model["aliases"], manifest, two_tower)

    def recommend(self, context, k=10, algorithm="itemcf"):
        if algorithm == "popularity":
            reason = None if context.user_id in self.popularity.histories else "cold_user"
            return self.popularity.recommend(context, k), reason
        if algorithm == "itemcf":
            return self.itemcf.recommend(context, k)
        if algorithm == "two_tower" and self.two_tower is not None:
            return self.two_tower.recommend(context, k)
        raise ValueError(f"Unknown recommendation algorithm: {algorithm}")


def train_from_path(source, output):
    events = pd.read_parquet(source / "events.parquet")
    pipeline = RecPipeline.build(
        events[events.split == "train"], read_json(source / "manifest.json")
    )
    pipeline.save(output)
    return pipeline.manifest
