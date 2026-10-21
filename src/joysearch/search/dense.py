from pathlib import Path

import numpy as np

from joysearch.common import ROOT, Candidate, read_json, write_json
from joysearch.data.wands import file_hash

TEXT_FIELDS = [
    "product_name",
    "product_class",
    "category hierarchy",
    "product_features",
    "product_description",
]


def product_texts(products):
    return [
        ". ".join(value for value in row if value)
        for row in products[TEXT_FIELDS].fillna("").astype(str).itertuples(index=False, name=None)
    ]


class SentenceEncoder:
    """Local-only CPU encoder; downloads belong to the offline build CLI."""

    def __init__(self, path: Path, max_seq_length=128, threads=4):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(threads)
        self.model = SentenceTransformer(
            str(path), device="cpu", local_files_only=True, trust_remote_code=False
        )
        self.model.max_seq_length = max_seq_length
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size=64):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)


class DenseIndex:
    version = "dense-v1"

    def __init__(self, embeddings, item_ids, manifest, encoder):
        self.embeddings = embeddings
        self.item_ids = np.asarray(item_ids, dtype=str)
        self.manifest = manifest
        self.encoder = encoder

    def rank_vector(self, vector, k):
        from threadpoolctl import threadpool_limits

        # A single score vector is O(items), never an item-by-item matrix.
        with threadpool_limits(limits=1, user_api="blas"):
            scores = self.embeddings @ vector
        order = np.lexsort((self.item_ids, -scores))[:k]
        return [
            Candidate("wands_search", str(self.item_ids[i]), float(scores[i]), "dense", rank)
            for rank, i in enumerate(order, 1)
        ]

    def retrieve(self, query, k):
        if not query.strip():
            return []
        return self.rank_vector(self.encoder.encode([query])[0], k)

    def save(self, output):
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "embeddings.npy", self.embeddings)
        write_json(output / "item_ids.json", self.item_ids.tolist())
        write_json(output / "manifest.json", self.manifest)

    @classmethod
    def load(cls, output, encoder=None):
        manifest = read_json(output / "manifest.json")
        if (
            manifest["version"] != cls.version
            or manifest["catalog_id"] != "wands_search"
            or manifest["normalization"] != "l2"
        ):
            raise ValueError(f"Incompatible dense artifact: {output}")
        embeddings = np.load(output / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        items = read_json(output / "item_ids.json")
        if embeddings.shape != (len(items), manifest["dimension"]):
            raise ValueError(f"Dense artifact dimensions disagree: {output}")
        if len(set(items)) != len(items) or not np.isfinite(embeddings).all():
            raise ValueError(f"Dense artifact has duplicate IDs or nonfinite embeddings: {output}")
        if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4):
            raise ValueError(f"Dense embeddings must be L2 normalized: {output}")
        if encoder is None:
            model_path = ROOT / manifest["model_path"]
            for name, digest in manifest["model_files_sha256"].items():
                if file_hash(model_path / name) != digest:
                    raise ValueError(f"Dense encoder file hash mismatch: {model_path / name}")
            encoder = SentenceEncoder(
                ROOT / manifest["model_path"], manifest["max_seq_length"], manifest["torch_threads"]
            )
        return cls(embeddings, items, manifest, encoder)
