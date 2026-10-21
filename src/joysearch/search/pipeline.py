from joysearch.common import read_json
from joysearch.search.bm25 import BM25Index
from joysearch.search.dense import DenseIndex
from joysearch.search.fusion import reciprocal_rank_fusion


class SearchPipeline:
    def __init__(self, index: BM25Index, dense=None, rrf_k=60, candidate_k=100):
        self.index = index
        self.dense = dense
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        if dense is not None:
            if index.item_ids.tolist() != dense.item_ids.tolist():
                raise ValueError("BM25 and dense indexes must share the same ordered WANDS catalog")
            if (
                index.manifest["data_manifest"]["sources"]
                != dense.manifest["data_manifest"]["sources"]
            ):
                raise ValueError("BM25 and dense indexes use different data versions")
        self.metadata = (
            index.products[["product_id", "product_name", "product_class"]]
            .set_index("product_id")
            .fillna("")
            .to_dict("index")
        )
        self.version = index.version
        self.algorithms = ["bm25", "dense", "hybrid"] if dense is not None else ["bm25"]
        self.versions = {"bm25": index.version, "dense": DenseIndex.version, "hybrid": "rrf-v1"}

    @classmethod
    def load(cls, root, enable_dense=True):
        index = BM25Index.load(root / "artifacts/search")
        dense_path = root / "artifacts/search_dense"
        # A final manifest is published only after the offline build completes.
        if enable_dense and (dense_path / "manifest.json").exists():
            dense = DenseIndex.load(dense_path)
            config = read_json(root / "configs/dense.json")
            return cls(index, dense, config["rrf_k"], config["candidate_k"])
        return cls(index)

    def model_config(self, algorithm):
        if algorithm == "bm25":
            return self.index.manifest
        if algorithm == "dense":
            return self.dense.manifest
        return {
            "version": "rrf-v1",
            "rrf_k": self.rrf_k,
            "candidate_k": self.candidate_k,
            "weights": {"bm25": 1, "dense": 1},
            "channels": {"bm25": self.index.manifest, "dense": self.dense.manifest},
        }

    def search(self, query, k=10, algorithm="bm25"):
        if algorithm not in self.algorithms:
            raise ValueError(f"Search algorithm unavailable: {algorithm}")
        if algorithm == "bm25":
            return self.index.retrieve(query, k)
        if algorithm == "dense":
            return self.dense.retrieve(query, k)
        depth = max(self.candidate_k, k)
        rankings = {
            "bm25": self.index.retrieve(query, depth),
            "dense": self.dense.retrieve(query, depth),
        }
        return reciprocal_rank_fusion(rankings, k, self.rrf_k)
