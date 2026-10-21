import re
import time
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse

from joysearch.common import Candidate, read_json, write_json


def tokenize(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).lower())


class BM25Index:
    version = "bm25-v1"

    def __init__(self, weights, vocabulary, products, manifest):
        self.weights = weights.tocsc()
        self.vocabulary = vocabulary
        self.products = products
        self.item_ids = products.product_id.to_numpy(dtype=str)
        self.manifest = manifest

    @classmethod
    def build(cls, products, data_manifest=None, k1=1.2, b=0.75):
        started = time.perf_counter()
        products = products.sort_values("product_id").reset_index(drop=True)
        fields = [
            "product_name",
            "product_class",
            "category hierarchy",
            "product_features",
            "product_description",
        ]
        texts = products[fields].fillna("").astype(str)
        vocabulary, rows, cols, counts, lengths = {}, [], [], [], []
        for row, values in enumerate(texts.itertuples(index=False, name=None)):
            # Repeat title once to give it more lexical weight.
            terms = Counter(tokenize(" ".join([values[0], *values])))
            lengths.append(sum(terms.values()))
            for term, count in terms.items():
                col = vocabulary.setdefault(term, len(vocabulary))
                rows.append(row)
                cols.append(col)
                counts.append(count)
        tf = sparse.csr_matrix(
            (np.asarray(counts, dtype=np.float32), (rows, cols)),
            shape=(len(products), len(vocabulary)),
        )
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log1p((len(products) - df + 0.5) / (df + 0.5))
        lengths = np.asarray(lengths)
        norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1))
        coo = tf.tocoo()
        weights = sparse.csc_matrix(
            (idf[coo.col] * coo.data * (k1 + 1) / (coo.data + norm[coo.row]), (coo.row, coo.col)),
            shape=tf.shape,
        )
        manifest = {
            "version": cls.version,
            "catalog_id": "wands_search",
            "catalog_size": len(products),
            "k1": k1,
            "b": b,
            "tokenizer": "NFKC lowercase Unicode words, no stopword removal",
            "text_fields": fields,
            "title_repetitions": 2,
            "data_manifest": data_manifest,
            "build_seconds": time.perf_counter() - started,
        }
        return cls(weights, vocabulary, products, manifest)

    def save(self, output):
        output.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(output / "weights.npz", self.weights)
        write_json(output / "vocabulary.json", self.vocabulary)
        self.products.to_parquet(output / "products.parquet", index=False)
        write_json(output / "manifest.json", self.manifest)

    @classmethod
    def load(cls, output):
        manifest = read_json(output / "manifest.json")
        if manifest["version"] != cls.version or manifest["catalog_id"] != "wands_search":
            raise ValueError(f"Incompatible BM25 artifact: {output}")
        weights = sparse.load_npz(output / "weights.npz")
        products = pd.read_parquet(output / "products.parquet")
        vocabulary = read_json(output / "vocabulary.json")
        if weights.shape != (len(products), len(vocabulary)):
            raise ValueError(f"BM25 artifact dimensions disagree: {output}")
        return cls(weights, vocabulary, products, manifest)

    def retrieve(self, query, k):
        columns = [self.vocabulary[t] for t in sorted(set(tokenize(query))) if t in self.vocabulary]
        if not columns:
            return []
        scores = np.asarray(self.weights[:, columns].sum(axis=1)).ravel()
        matches = np.flatnonzero(scores > 0)
        order = matches[np.lexsort((self.item_ids[matches], -scores[matches]))[:k]]
        return [
            Candidate("wands_search", self.item_ids[i], float(scores[i]), "bm25", rank)
            for rank, i in enumerate(order, 1)
        ]
