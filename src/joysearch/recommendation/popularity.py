import numpy as np

from joysearch.common import Candidate


class Popularity:
    version = "popularity-v1"

    def __init__(self, items, counts, histories):
        self.items = np.asarray(items, dtype=str)
        self.counts = np.asarray(counts, dtype=np.float64)
        self.histories = histories
        self.order = np.lexsort((self.items, -self.counts))

    def recommend(self, context, k):
        seen = set(self.histories.get(context.user_id, [])) if context.exclude_seen else set()
        order = []
        for index in self.order:
            if index not in seen:
                order.append(int(index))
                if len(order) == k:
                    break
        return [
            Candidate(
                "amazon_baby_rec", str(self.items[i]), float(self.counts[i]), "popularity", rank
            )
            for rank, i in enumerate(order, 1)
        ]
