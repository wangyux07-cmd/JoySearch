"""Experimental exact retriever reusing an already discovered BLAS controller."""

import numpy as np
from threadpoolctl import ThreadpoolController

from joysearch.common import Candidate
from joysearch.search.dense import DenseIndex


class CachedControllerDenseIndex(DenseIndex):
    def __init__(self, index):
        super().__init__(index.embeddings, index.item_ids, index.manifest, index.encoder)
        self.controller = ThreadpoolController()

    def rank_vector(self, vector, k):
        with self.controller.limit(limits=1, user_api="blas"):
            scores = self.embeddings @ vector
        order = np.lexsort((self.item_ids, -scores))[:k]
        return [
            Candidate("wands_search", str(self.item_ids[i]), float(scores[i]), "dense", rank)
            for rank, i in enumerate(order, 1)
        ]
