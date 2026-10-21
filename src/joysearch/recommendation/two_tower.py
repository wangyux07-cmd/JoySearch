from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from joysearch.common import Candidate, read_json
from joysearch.data.wands import file_hash


def padded_histories(histories, max_history, padding_idx, targets=None):
    """Use the same recent-history tensor in fitting, validation and serving."""
    result = np.full((len(histories), max_history), padding_idx, dtype=np.int64)
    for row, history in enumerate(histories):
        values = history if targets is None else [i for i in history if i != targets[row]]
        values = values[-max_history:]
        result[row, : len(values)] = values
    return torch.from_numpy(result)


class TwoTowerNetwork(nn.Module):
    def __init__(self, item_count, dimension):
        super().__init__()
        self.padding_idx = item_count
        self.context_embedding = nn.Embedding(item_count + 1, dimension, padding_idx=item_count)
        self.item_embedding = nn.Embedding(item_count, dimension)
        self.user_projection = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.ReLU(),
            nn.Linear(dimension, dimension),
        )
        self.item_projection = nn.Linear(dimension, dimension)

    def encode_users(self, histories):
        mask = histories.ne(self.padding_idx)
        pooled = self.context_embedding(histories).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return F.normalize(self.user_projection(pooled), dim=-1)

    def encode_items(self, items):
        return F.normalize(self.item_projection(self.item_embedding(items)), dim=-1)


class TwoTower:
    version = "two-tower-v1"

    def __init__(self, network, popularity, manifest):
        self.network = network.eval()
        self.popularity = popularity
        self.manifest = manifest
        self.item_map = {item: i for i, item in enumerate(popularity.items)}
        self.max_history = manifest["config"]["max_history"]
        with torch.inference_mode():
            self.item_vectors = network.encode_items(torch.arange(len(popularity.items)))

    @classmethod
    def load(cls, output, popularity, base_manifest):
        manifest = read_json(output / "manifest.json")
        if (
            manifest["version"] != cls.version
            or manifest["catalog_id"] != "amazon_baby_rec"
            or manifest["items"] != popularity.items.tolist()
            or manifest["data_split_sha256"] != base_manifest["data_manifest"]["split_sha256"]
            or manifest["base_model_sha256"]
            != file_hash(output.parent / "recommendation/model.json")
            or manifest["inference_code_sha256"] != file_hash(Path(__file__))
        ):
            raise ValueError(
                "Two-Tower catalog/history/features changed; run scripts/train_two_tower.py"
            )
        if manifest["model_sha256"] != file_hash(output / "model.pt"):
            raise ValueError("Two-Tower model checksum mismatch")
        torch.set_num_threads(manifest["config"]["threads"])
        network = TwoTowerNetwork(len(popularity.items), manifest["config"]["dimension"])
        network.load_state_dict(
            torch.load(output / "model.pt", map_location="cpu", weights_only=True)
        )
        if not all(torch.isfinite(p).all() for p in network.parameters()):
            raise ValueError("Two-Tower contains nonfinite weights")
        return cls(network, popularity, manifest)

    def score_histories(self, histories):
        batch = padded_histories(histories, self.max_history, len(self.popularity.items))
        with torch.inference_mode():
            return self.network.encode_users(batch) @ self.item_vectors.T

    def recommend(self, context, k):
        history = list(self.popularity.histories.get(context.user_id, []))
        history = list(
            dict.fromkeys(
                history + [self.item_map[i] for i in context.recent_item_ids if i in self.item_map]
            )
        )
        if not history:
            return self.popularity.recommend(context, k), "cold_user"
        scores = self.score_histories([history])[0].numpy()
        if context.exclude_seen:
            scores[history] = -np.inf
        order = np.lexsort((self.popularity.items, -scores))
        order = order[np.isfinite(scores[order])][:k]
        return [
            Candidate(
                "amazon_baby_rec",
                str(self.popularity.items[i]),
                float(scores[i]),
                "two_tower",
                rank,
            )
            for rank, i in enumerate(order, 1)
        ], None
