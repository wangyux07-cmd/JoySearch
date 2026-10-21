"""Query-conditioned next-bit model and constrained neural beam decoding."""

import numpy as np
import torch
from torch import nn

from joysearch.common import Candidate, read_json
from joysearch.data.wands import file_hash
from joysearch.search.semantic_ids import prefix_tree


class BinaryGenerator(nn.Module):
    def __init__(self, query_dimension, hidden_size, token_embedding_size):
        super().__init__()
        self.query_projection = nn.Linear(query_dimension, hidden_size)
        self.embedding = nn.Embedding(3, token_embedding_size)
        self.decoder = nn.GRU(token_embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, vectors, tokens):
        initial = torch.tanh(self.query_projection(vectors)).unsqueeze(0)
        output, _ = self.decoder(self.embedding(tokens), initial)
        return self.output(output)


class GenerativeRetriever:
    def __init__(self, network, item_ids, codes, depth, beam_size=100, encoder=None):
        self.network = network.eval()
        self.item_ids = list(item_ids)
        self.codes = np.asarray(codes)
        self.depth = depth
        self.beam_size = beam_size
        self.encoder = encoder
        self.children, self.leaves = prefix_tree(self.codes, depth)
        self.code_to_item = {int(code): i for i, code in enumerate(self.codes)}

    @torch.inference_mode()
    def decode(self, vector, constrained=True):
        hidden = torch.tanh(
            self.network.query_projection(torch.as_tensor(vector).reshape(1, -1))
        ).unsqueeze(0)
        tokens = torch.full((1,), 2, dtype=torch.long)
        nodes, codes, scores = (
            np.array([0]),
            np.array([0], dtype=np.int64),
            np.array([0.0], dtype=np.float32),
        )
        expanded = 0
        for _ in range(self.depth):
            output, state = self.network.decoder(self.network.embedding(tokens)[:, None], hidden)
            log_probs = self.network.output(output[:, 0]).log_softmax(-1).numpy()
            values = scores[:, None] + log_probs
            if constrained:
                values[self.children[nodes] < 0] = -np.inf
            flat = np.flatnonzero(np.isfinite(values.ravel()))
            expanded += len(flat)
            next_codes = np.repeat(codes * 2, 2) + np.tile([0, 1], len(codes))
            order = flat[np.lexsort((next_codes[flat], -values.ravel()[flat]))[: self.beam_size]]
            parent, bit = order // 2, order % 2
            hidden = state[:, torch.as_tensor(parent), :]
            tokens = torch.as_tensor(bit, dtype=torch.long)
            codes, scores = next_codes[order], values.ravel()[order]
            nodes = self.children[nodes[parent], bit] if constrained else nodes
            if not len(order):
                break
        generated, mapped = [], []
        for code, score in zip(codes.tolist(), scores.tolist(), strict=True):
            item = self.code_to_item.get(code)
            generated.append(
                {
                    "code": code,
                    "score": score,
                    "product_id": self.item_ids[item] if item is not None else None,
                }
            )
            if item is not None:
                mapped.append((self.item_ids[item], score, code))
        mapped.sort(key=lambda row: (-row[1], row[0]))
        candidates = [
            Candidate(
                "wands_search",
                pid,
                score,
                "generative",
                rank,
                {"semantic_code": code, "constrained": constrained},
            )
            for rank, (pid, score, code) in enumerate(mapped, 1)
        ]
        return candidates, {
            "generated": generated,
            "expanded_branches": expanded,
            "generated_count": len(generated),
            "mapped_count": len(mapped),
            "failed_mapping_count": len(generated) - len(mapped),
            "constraint_policy": "Mask absent catalog prefixes after full binary log-softmax; do not renormalize legal branches",
        }

    def retrieve(self, query, k=10):
        if self.encoder is None:
            raise ValueError("Query encoder is required for text retrieval")
        return self.decode(self.encoder.encode([query])[0])[0][:k]

    @classmethod
    def load(cls, root, encoder=None):
        folder = root / "artifacts/search_generative"
        manifest = read_json(folder / "manifest.json")
        for path, digest in manifest["files"].items():
            if file_hash(folder / path) != digest:
                raise ValueError(f"Generative artifact checksum mismatch: {path}")
        for path, digest in manifest["protocol_signature"].items():
            if file_hash(root / path) != digest:
                raise ValueError(f"Generative training inputs/source changed: {path}")
        config = manifest["config"]
        torch.set_num_threads(config["threads"])
        network = BinaryGenerator(
            manifest["query_dimension"], config["hidden_size"], config["token_embedding_size"]
        )
        network.load_state_dict(
            torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        )
        mapping = read_json(folder / "semantic_ids.json")
        model = cls(
            network,
            mapping["item_ids"],
            mapping["codes"],
            config["depth"],
            config["beam_size"],
            encoder,
        )
        model.manifest = manifest
        return model
