import numpy as np
import pandas as pd
import pytest
import torch

from joysearch.common import ROOT, read_json
from joysearch.search.generative import BinaryGenerator, GenerativeRetriever
from joysearch.search.generative_training import training_pairs
from joysearch.search.semantic_ids import catalog_codes, code_bits, prefix_tree


def test_semantic_codes_complete_unique_deterministic_and_capacity():
    vectors = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=np.float32
    )
    ids = ["b", "a", "c", "d", "e"]
    codes = catalog_codes(vectors, ids, 3)
    assert len(set(codes.tolist())) == 5
    assert np.array_equal(codes, catalog_codes(vectors, ids, 3))
    children, leaves = prefix_tree(codes, 3)
    for item, bits in enumerate(code_bits(codes, 3)):
        node = 0
        for bit in bits:
            node = children[node, bit]
            assert node >= 0
        assert leaves[node] == item
    with pytest.raises(ValueError):
        catalog_codes(vectors, ids, 2)


def test_neural_beam_probabilities_constraints_and_mapping_failures():
    torch.manual_seed(42)
    network = BinaryGenerator(2, 4, 3)
    for param in network.parameters():
        param.data.zero_()
    model = GenerativeRetriever(network, ["a", "b", "c"], [0, 2, 3], 2, 4)
    constrained, diagnostic = model.decode(np.zeros(2, dtype=np.float32))
    raw, raw_diagnostic = model.decode(np.zeros(2, dtype=np.float32), False)
    assert [c.item_id for c in constrained] == ["a", "b", "c"]
    assert len(raw) == 3 and raw_diagnostic["failed_mapping_count"] == 1
    assert diagnostic["failed_mapping_count"] == 0
    assert constrained[0].score == pytest.approx(-2 * np.log(2))
    network.output.bias.data[1] = 2
    changed = model.decode(np.zeros(2, dtype=np.float32))[0]
    assert changed[0].item_id == "c" and changed[0].score != constrained[0].score


def test_training_sampling_excludes_heldout_and_unjudged_negatives():
    frame = pd.DataFrame(
        {
            "query_id": ["train"] * 8 + ["val", "test"],
            "product_id": list("abcdefghij"),
            "grade": [0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        }
    )
    pairs = training_pairs(frame, ["train"], 4, 42)
    assert len(pairs) == 4 and pairs.grade.tolist().count(2) == 2
    assert pairs.query_id.eq("train").all() and pairs.grade.ge(1).all()
    assert pairs.equals(training_pairs(frame, ["train"], 4, 42))


def test_real_generative_persistence_full_catalog_and_train_lineage():
    if not (ROOT / "artifacts/search_generative/manifest.json").exists():
        pytest.skip("Train the optional generative retriever first")
    model = GenerativeRetriever.load(ROOT)
    cached = np.load(ROOT / "artifacts/generative_inputs/queries.npz", allow_pickle=False)
    pool, diagnostic = model.decode(cached["vectors"][0])
    assert len(model.item_ids) == 42994
    assert len(pool) == len({c.item_id for c in pool}) == 100
    assert all(c.catalog_id == "wands_search" and c.item_id in model.item_ids for c in pool)
    assert diagnostic["mapped_count"] == 100
    reloaded = GenerativeRetriever.load(ROOT)
    assert pool == reloaded.decode(cached["vectors"][0])[0]
    run = ROOT / "reports" / model.manifest["training_run"]
    pairs = pd.read_parquet(run / "training_pairs.parquet")
    splits = read_json(ROOT / "data/processed/wands/splits.json")
    assert set(pairs.query_id) <= set(splits["train"])
