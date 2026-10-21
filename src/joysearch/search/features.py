import numpy as np
import pandas as pd

from joysearch.search.bm25 import tokenize

FEATURE_NAMES = [
    "bm25_score",
    "dense_score",
    "bm25_present",
    "dense_present",
    "bm25_rank",
    "dense_rank",
    "rrf_score",
    "title_overlap",
    "category_overlap",
    "query_length",
    "title_length",
    "log_review_count",
    "average_rating",
]


class SearchFeatures:
    """Catalog-only features shared by training and serving; never accepts qrels."""

    def __init__(self, products):
        self.text = {}
        for row in products.fillna("").to_dict("records"):
            self.text[row["product_id"]] = (
                set(tokenize(row["product_name"])),
                set(tokenize(row["product_class"] + " " + row["category hierarchy"])),
            )
        numeric = products.set_index("product_id")[["review_count", "average_rating"]]
        self.numeric = numeric.apply(pd.to_numeric, errors="coerce").fillna(0).to_dict("index")

    def transform(self, query, candidates, rankings, depth):
        terms = set(tokenize(query))
        channels = {name: {c.item_id: c for c in rows} for name, rows in rankings.items()}
        rows = []
        for c in candidates:
            lexical, dense = [channels[name].get(c.item_id) for name in ["bm25", "dense"]]
            title, category = self.text[c.item_id]
            numeric = self.numeric[c.item_id]
            rows.append(
                [
                    lexical.score if lexical else 0,
                    dense.score if dense else 0,
                    int(lexical is not None),
                    int(dense is not None),
                    lexical.rank if lexical else depth + 1,
                    dense.rank if dense else depth + 1,
                    c.score,
                    len(terms & title) / max(len(terms), 1),
                    len(terms & category) / max(len(terms), 1),
                    len(terms),
                    len(title),
                    np.log1p(max(0, numeric["review_count"])),
                    numeric["average_rating"],
                ]
            )
        return pd.DataFrame(rows, columns=FEATURE_NAMES, dtype=np.float32)
