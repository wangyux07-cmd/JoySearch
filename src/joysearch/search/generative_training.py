"""Train-only teacher forcing with full-query validation and bounded CPU fitting."""

import random
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from joysearch.common import read_json, write_json
from joysearch.data.wands import file_hash
from joysearch.evaluation.metrics import ndcg
from joysearch.search.dense import DenseIndex
from joysearch.search.generative import BinaryGenerator, GenerativeRetriever
from joysearch.search.semantic_ids import catalog_codes, code_bits


def prepare_inputs(root):
    config = read_json(root / "configs/generative.json")
    folder = root / "artifacts/generative_inputs"
    paths = [
        "src/joysearch/search/semantic_ids.py",
        "src/joysearch/search/dense.py",
        "configs/dense.json",
        "data/processed/wands/queries.parquet",
        "artifacts/search_dense/manifest.json",
        "artifacts/search_dense/item_ids.json",
        "artifacts/search_dense/embeddings.npy",
    ]
    signature = {"depth": config["depth"], "sources": {p: file_hash(root / p) for p in paths}}
    if (folder / "manifest.json").exists():
        manifest = read_json(folder / "manifest.json")
        if manifest["input_signature"] != signature:
            raise ValueError(
                "Generative catalog/query inputs changed; preserve old inputs before rebuilding"
            )
        for path, digest in manifest["files"].items():
            if file_hash(folder / path) != digest:
                raise ValueError(f"Generative input cache checksum mismatch: {path}")
        return folder
    started = time.perf_counter()
    dense = DenseIndex.load(root / "artifacts/search_dense")
    queries = pd.read_parquet(root / "data/processed/wands/queries.parquet").sort_values("query_id")
    codes = catalog_codes(dense.embeddings, dense.item_ids, config["depth"])
    vectors = dense.encoder.encode(queries["query"].tolist())
    folder.mkdir(parents=True, exist_ok=True)
    write_json(
        folder / "semantic_ids.json",
        {
            "item_ids": dense.item_ids.tolist(),
            "codes": codes.tolist(),
            "depth": config["depth"],
            "collision_count": len(codes) - len(set(codes.tolist())),
            "method": config["coding"],
        },
    )
    np.savez(
        folder / "queries.npz", query_ids=queries.query_id.to_numpy(dtype=str), vectors=vectors
    )
    write_json(
        folder / "manifest.json",
        {
            "input_signature": signature,
            "files": {p: file_hash(folder / p) for p in ["semantic_ids.json", "queries.npz"]},
            "build_seconds": time.perf_counter() - started,
            "catalog_size": len(codes),
            "query_count": len(queries),
            "query_encoder": "Frozen MiniLM; independently encode query texts, no fitting or labels",
        },
    )
    return folder


def protocol_signature(root):
    paths = [
        "configs/generative.json",
        "src/joysearch/search/semantic_ids.py",
        "src/joysearch/search/generative.py",
        "src/joysearch/search/generative_training.py",
        "data/processed/wands/qrels.parquet",
        "data/processed/wands/splits.json",
        "artifacts/generative_inputs/manifest.json",
    ]
    return {path: file_hash(root / path) for path in paths}


def training_pairs(qrels, train_qids, cap, seed):
    positives = qrels[qrels.query_id.isin(train_qids) & qrels.grade.ge(1)]
    rng, selected = random.Random(seed), []
    for _, group in positives.sort_values(["query_id", "product_id"]).groupby(
        "query_id", sort=True
    ):
        chosen = []
        for grade in [2, 1]:
            indices = group[group.grade.eq(grade)].index.tolist()
            chosen.extend(rng.sample(indices, min(cap // 2, len(indices))))
        chosen_set = set(chosen)
        remaining = [i for i in group.index if i not in chosen_set]
        chosen.extend(rng.sample(remaining, min(cap - len(chosen), len(remaining))))
        selected.extend(chosen)
    return positives.loc[selected].reset_index(drop=True)


def fit(root, pilot=False):
    started = time.perf_counter()
    config = read_json(root / "configs/generative.json")
    inputs = prepare_inputs(root)
    mapping = read_json(inputs / "semantic_ids.json")
    cached = np.load(inputs / "queries.npz", allow_pickle=False)
    vectors = torch.from_numpy(cached["vectors"])
    query_map = {q: i for i, q in enumerate(cached["query_ids"].tolist())}
    item_map = {pid: i for i, pid in enumerate(mapping["item_ids"])}
    codes = np.asarray(mapping["codes"])
    bits = torch.from_numpy(code_bits(codes, config["depth"]))
    splits = read_json(root / "data/processed/wands/splits.json")
    if any(
        set(splits[a]) & set(splits[b])
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("Generative query splits overlap")
    qrels = pd.read_parquet(root / "data/processed/wands/qrels.parquet")
    pairs = training_pairs(qrels, splits["train"], config["max_pairs_per_query"], config["seed"])
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    network = BinaryGenerator(
        vectors.shape[1], config["hidden_size"], config["token_embedding_size"]
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=config["learning_rate"])
    qindex = torch.tensor([query_map[q] for q in pairs.query_id])
    targets = bits[torch.tensor([item_map[p] for p in pairs.product_id])]
    teacher = torch.cat([torch.full((len(pairs), 1), 2), targets[:, :-1]], dim=1)
    folder = root / "reports" / ("run_e6_pilot_" if pilot else "run_e6_train_")
    from datetime import UTC, datetime

    folder = folder.with_name(folder.name + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    folder.mkdir(parents=True)
    signature = protocol_signature(root)
    write_json(
        folder / "inputs.json",
        {
            "config": config,
            "signature": signature,
            "training_query_ids": sorted(pairs.query_id.unique().tolist()),
            "pairs": len(pairs),
            "train_positive_population": int(
                (qrels.query_id.isin(splits["train"]) & qrels.grade.ge(1)).sum()
            ),
            "parameter_count": sum(p.numel() for p in network.parameters()),
            "pilot": pilot,
        },
    )
    pairs.to_parquet(folder / "training_pairs.parquet", index=False)
    model = GenerativeRetriever(
        network, mapping["item_ids"], codes, config["depth"], config["beam_size"]
    )
    history, best, best_epoch, best_state = [], -1.0, None, None
    fitting = time.perf_counter()
    expired = False
    epochs = 1 if pilot else config["max_epochs"]
    for epoch in range(epochs):
        network.train()
        order = torch.randperm(len(pairs))
        losses = []
        for offset in range(0, len(pairs), config["batch_size"]):
            if (
                time.perf_counter() - started > config["pilot_budget_seconds"]
                if pilot
                else time.perf_counter() - fitting > config["train_budget_seconds"]
            ):
                expired = True
                break
            ix = order[offset : offset + config["batch_size"]]
            logits = network(vectors[qindex[ix]], teacher[ix])
            loss = F.cross_entropy(logits.reshape(-1, 2), targets[ix].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if pilot and len(losses) == config["pilot_steps"]:
                break
        network.eval()
        validation = None
        if not pilot:
            selected = qrels[qrels.query_id.isin(splits["validation"])]
            judged = {q: dict(zip(g.product_id, g.grade)) for q, g in selected.groupby("query_id")}
            validation = float(
                np.mean(
                    [
                        ndcg(
                            [c.item_id for c in model.decode(vectors[query_map[q]].numpy())[0]],
                            judged.get(q, {}),
                        )
                        for q in splits["validation"]
                    ]
                )
            )
            if validation > best:
                best, best_epoch = validation, epoch + 1
                best_state = {
                    name: tensor.detach().clone() for name, tensor in network.state_dict().items()
                }
        record = {
            "epoch": epoch + 1,
            "steps": len(losses),
            "mean_loss": float(np.mean(losses)) if losses else None,
            "validation_ndcg_at_10": validation,
            "losses": losses if pilot else None,
            "elapsed_fitting_seconds": time.perf_counter() - fitting,
        }
        history.append(record)
        write_json(folder / "history.json", history)
        print({k: v for k, v in record.items() if k != "losses"}, flush=True)
        if expired or (not pilot and epoch + 1 - best_epoch >= config["patience"]):
            break
    probe_qids = pairs.query_id.unique().tolist()[:4]
    tick = time.perf_counter()
    probe = [model.decode(vectors[query_map[q]].numpy())[1] for q in probe_qids]
    metrics = {
        "status": "completed" if not expired else "stopped_at_budget",
        "pilot": pilot,
        "wall_seconds": time.perf_counter() - started,
        "fitting_seconds": time.perf_counter() - fitting,
        "parameter_count": sum(p.numel() for p in network.parameters()),
        "pairs": len(pairs),
        "selected_epoch": best_epoch,
        "selected_validation_ndcg_at_10": best if best_epoch else None,
        "probe_query_ids": probe_qids,
        "probe_decode_seconds": time.perf_counter() - tick,
        "probe_mapped_counts": [p["mapped_count"] for p in probe],
        "budget_expired": expired,
    }
    write_json(folder / "metrics.json", metrics)
    if pilot:
        write_json(root / "reports/generative_pilot_latest.json", {"run": folder.name})
        return folder
    if best_state is None or protocol_signature(root) != signature:
        raise ValueError("No completed validation checkpoint or training protocol changed")
    output = root / "artifacts/search_generative"
    if (output / "manifest.json").exists():
        raise ValueError("Generative model exists; preserve it before fitting another version")
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output / "model.pt")
    write_json(output / "semantic_ids.json", mapping)
    write_json(
        output / "manifest.json",
        {
            "version": config["version"],
            "config": config,
            "query_dimension": int(vectors.shape[1]),
            "protocol_signature": signature,
            "files": {p: file_hash(output / p) for p in ["model.pt", "semantic_ids.json"]},
            "training_run": folder.name,
            "selected_epoch": best_epoch,
            "validation_ndcg_at_10": best,
        },
    )
    write_json(root / "reports/generative_training_latest.json", {"run": folder.name})
    return folder
