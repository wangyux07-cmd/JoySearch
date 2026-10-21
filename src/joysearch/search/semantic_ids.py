"""Unsupervised balanced catalog addresses and legal-prefix constraints."""

import numpy as np


def catalog_codes(embeddings, item_ids, depth):
    if len(item_ids) > 2**depth or len(set(item_ids)) != len(item_ids):
        raise ValueError("Semantic address capacity exceeded or duplicate product IDs")
    codes = np.zeros(len(item_ids), dtype=np.int64)
    item_ids = np.asarray(item_ids)

    def partition(indices, level, prefix):
        if len(indices) == 1:
            codes[indices[0]] = prefix << (depth - level)
            return
        axis = int(np.var(embeddings[indices], axis=0).argmax())
        order = np.lexsort((item_ids[indices], embeddings[indices, axis]))
        ordered = indices[order]
        middle = len(ordered) // 2
        partition(ordered[:middle], level + 1, prefix * 2)
        partition(ordered[middle:], level + 1, prefix * 2 + 1)

    if len(item_ids):
        partition(np.arange(len(item_ids)), 0, 0)
    return codes


def prefix_tree(codes, depth):
    leaves = np.full(2 ** (depth + 1) - 1, -1, dtype=np.int32)
    children = np.full((len(leaves), 2), -1, dtype=np.int32)
    if len(set(codes.tolist())) != len(codes) or np.any(codes < 0) or np.any(codes >= 2**depth):
        raise ValueError("Duplicate or out-of-range semantic codes")
    for item, code in enumerate(codes.tolist()):
        node = 0
        for shift in range(depth - 1, -1, -1):
            bit = (code >> shift) & 1
            child = node * 2 + 1 + bit
            children[node, bit] = child
            node = child
        leaves[node] = item
    return children, leaves


def code_bits(codes, depth):
    return ((np.asarray(codes)[:, None] >> np.arange(depth - 1, -1, -1)) & 1).astype(np.int64)
