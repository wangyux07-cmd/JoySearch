import numpy as np
from scipy import sparse

from joysearch.common import Candidate


def build_similarity(train, users, items, top_n=100, shrinkage=10, block_size=256):
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {item: i for i, item in enumerate(items)}
    matrix = sparse.csr_matrix(
        (
            np.ones(len(train), dtype=np.float64),
            (train.user_id.map(user_map), train.parent_asin.map(item_map)),
        ),
        shape=(len(users), len(items)),
    )
    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    item_users = matrix.T.tocsr()
    rows, columns, values = [], [], []
    for start in range(0, len(items), block_size):
        block = (item_users[start : start + block_size] @ matrix).tocsr()
        for local in range(block.shape[0]):
            item = start + local
            begin, end = block.indptr[local : local + 2]
            indices, cooccurrence = block.indices[begin:end], block.data[begin:end]
            keep = indices != item
            indices, cooccurrence = indices[keep], cooccurrence[keep]
            similarity = cooccurrence / np.sqrt(frequencies[item] * frequencies[indices])
            similarity *= cooccurrence / (cooccurrence + shrinkage)
            best = np.lexsort((indices, -similarity))[:top_n]
            rows.extend([item] * len(best))
            columns.extend(indices[best].tolist())
            values.extend(similarity[best].tolist())
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(items), len(items)))


class ItemCF:
    version = "itemcf-v1"

    def __init__(self, similarity, popularity):
        self.similarity = similarity
        self.popularity = popularity
        self.item_map = {item: i for i, item in enumerate(popularity.items)}

    def recommend(self, context, k):
        history = self.popularity.histories.get(context.user_id, [])
        recent = [self.item_map[i] for i in context.recent_item_ids if i in self.item_map]
        history = sorted(set(history) | set(recent))
        if not history:
            return self.popularity.recommend(context, k), "cold_user"
        scores = np.asarray(self.similarity[history].sum(axis=0)).ravel()
        if context.exclude_seen:
            scores[history] = -np.inf
        # All train items are eligible; zero-score ties use train popularity then item ID.
        order = np.lexsort((self.popularity.items, -self.popularity.counts, -scores))
        order = order[np.isfinite(scores[order])][:k]
        candidates = [
            Candidate(
                "amazon_baby_rec",
                str(self.popularity.items[i]),
                float(scores[i]),
                "itemcf" if scores[i] > 0 else "popularity",
                rank,
            )
            for rank, i in enumerate(order, 1)
        ]
        if any(c.source == "popularity" for c in candidates):
            # Completion scores are train counts, never fabricated relevance estimates.
            candidates = [
                Candidate(
                    c.catalog_id,
                    c.item_id,
                    c.score
                    if c.source == "itemcf"
                    else float(self.popularity.counts[self.item_map[c.item_id]]),
                    c.source,
                    c.rank,
                )
                for c in candidates
            ]
            return candidates, "insufficient_neighbors"
        return candidates, None
