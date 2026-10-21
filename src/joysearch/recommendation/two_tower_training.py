import copy
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.nn import functional as F

from joysearch.common import read_json, write_json
from joysearch.data.wands import file_hash
from joysearch.recommendation.pipeline import RecPipeline
from joysearch.recommendation.two_tower import TwoTower, TwoTowerNetwork, padded_histories


def sample_negatives(rng, user_rows, known_positive, count):
    """Uniform train-catalog draws, rejecting all known training positives."""
    item_count = known_positive.shape[1]
    if np.any(known_positive.getnnz(axis=1)[np.unique(user_rows)] >= item_count):
        raise ValueError("Negative sampling requires an unobserved train item per user")
    negatives = rng.integers(item_count, size=(len(user_rows), count))
    rows = np.repeat(user_rows, count)
    rejected = np.asarray(known_positive[rows, negatives.ravel()], dtype=bool).reshape(
        negatives.shape
    )
    while rejected.any():
        negatives[rejected] = rng.integers(item_count, size=int(rejected.sum()))
        rejected = np.asarray(known_positive[rows, negatives.ravel()], dtype=bool).reshape(
            negatives.shape
        )
    return torch.from_numpy(negatives)


def validation_metrics(model, validation, batch_size):
    """Exact target ranks over the full train catalog with bounded batch memory."""
    hits, gains = 0, 0.0
    items = model.item_map
    users = validation.user_id.tolist()
    targets = validation.parent_asin.tolist()
    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        histories = [model.popularity.histories[u] for u in batch_users]
        scores = model.score_histories(histories).numpy()
        for row, history in enumerate(histories):
            scores[row, history] = -np.inf
            target = targets[start + row]
            if target not in items:
                continue
            index = items[target]
            target_score = scores[row, index]
            rank = 1 + int((scores[row] > target_score).sum())
            rank += int((scores[row, :index] == target_score).sum())
            if rank <= 10:
                hits += 1
                gains += 1 / np.log2(rank + 1)
    return {
        "hr_at_10": hits / len(users),
        "ndcg_at_10": gains / len(users),
        "users": len(users),
        "unavailable_targets": sum(t not in items for t in targets),
    }


def train(root):
    started = time.perf_counter()
    output = root / "artifacts/recommendation_two_tower"
    source = root / "data/processed/amazon"
    config = read_json(root / "configs/two_tower.json")
    data_manifest = read_json(source / "manifest.json")
    if file_hash(source / "events.parquet") != data_manifest["split_sha256"]:
        raise ValueError("Amazon split changed since preparation")
    signature = {
        "config": config,
        "data_split_sha256": data_manifest["split_sha256"],
        "base_model_sha256": file_hash(root / "artifacts/recommendation/model.json"),
        "data_manifest_sha256": file_hash(source / "manifest.json"),
        "inference_code_sha256": file_hash(root / "src/joysearch/recommendation/two_tower.py"),
        "training_code_sha256": file_hash(Path(__file__)),
        "torch": importlib.metadata.version("torch"),
    }
    base = RecPipeline.load(root / "artifacts/recommendation", enable_two_tower=False)
    final = output / "manifest.json"
    if final.exists() and read_json(final)["training_signature"] == signature:
        TwoTower.load(output, base.popularity, base.manifest)
        print("Matching trained Two-Tower artifact reused", flush=True)
        return read_json(final)
    events = pd.read_parquet(source / "events.parquet")
    train_events = events[events.split.eq("train")].sort_values(
        ["user_id", "timestamp", "parent_asin"]
    )
    validation = events[events.split.eq("validation")].sort_values("user_id")
    pop = base.popularity
    items = {item: i for i, item in enumerate(pop.items)}
    histories = {u: [items[i] for i in g.parent_asin] for u, g in train_events.groupby("user_id")}
    if histories != pop.histories or set(validation.user_id) != set(histories):
        raise ValueError("Two-Tower histories/cohort differ from frozen baseline")
    users = sorted(histories)
    user_map = {u: i for i, u in enumerate(users)}
    rows = train_events.user_id.map(user_map).to_numpy()
    columns = train_events.parent_asin.map(items).to_numpy()
    known = sparse.csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, columns)), shape=(len(users), len(items))
    )
    eligible = [u for u in users if len(histories[u]) > 1]
    examples = train_events[train_events.user_id.isin(eligible)]
    example_users = examples.user_id.map(user_map).to_numpy()
    positives = torch.tensor(examples.parent_asin.map(items).to_numpy(), dtype=torch.long)
    contexts = padded_histories(
        [histories[u] for u in examples.user_id],
        config["max_history"],
        len(items),
        positives.tolist(),
    )
    if len(examples) == 0 or contexts.eq(len(items)).all(dim=1).any():
        raise ValueError("Two-Tower requires nonempty target-excluded training histories")
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config["seed"])
    network = TwoTowerNetwork(len(items), config["dimension"])
    optimizer = torch.optim.Adam(
        network.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    output.mkdir(parents=True, exist_ok=True)
    history, best_score, best_epoch, best_state = [], -1.0, 0, None
    for epoch in range(1, config["epochs"] + 1):
        tick = time.perf_counter()
        network.train()
        total_loss = 0.0
        permutation = rng.permutation(len(examples))
        for start in range(0, len(examples), config["batch_size"]):
            indices = permutation[start : start + config["batch_size"]]
            negatives = sample_negatives(rng, example_users[indices], known, config["negatives"])
            user_vectors = network.encode_users(contexts[indices])
            positive_scores = (user_vectors * network.encode_items(positives[indices])).sum(dim=-1)
            negative_scores = (user_vectors[:, None] * network.encode_items(negatives)).sum(dim=-1)
            loss = F.softplus(
                (negative_scores - positive_scores[:, None]) / config["temperature"]
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        model = TwoTower(network, pop, {"config": config})
        metrics = validation_metrics(model, validation, config["validation_batch_size"])
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(examples),
            "validation": metrics,
            "seconds": time.perf_counter() - tick,
        }
        history.append(row)
        if metrics["ndcg_at_10"] > best_score:
            best_score, best_epoch = metrics["ndcg_at_10"], epoch
            best_state = copy.deepcopy(network.state_dict())
            torch.save(best_state, output / "best_state.pt")
        write_json(
            output / "training_progress.json",
            {
                "training_signature": signature,
                "epochs": history,
                "best_epoch": best_epoch,
                "resume_policy": "Interrupted run restarts deterministic training; complete model is reused",
            },
        )
        print(
            f"epoch={epoch} loss={row['train_loss']:.6f} validation={metrics} seconds={row['seconds']:.2f}",
            flush=True,
        )
        if epoch - best_epoch >= config["patience"]:
            break
    torch.save(best_state, output / "model.pt")
    manifest = {
        "version": TwoTower.version,
        "catalog_id": "amazon_baby_rec",
        "items": pop.items.tolist(),
        "config": config,
        "training_signature": signature,
        "data_split_sha256": data_manifest["split_sha256"],
        "base_model_sha256": signature["base_model_sha256"],
        "inference_code_sha256": signature["inference_code_sha256"],
        "model_sha256": file_hash(output / "model.pt"),
        "best_epoch": best_epoch,
        "best_validation": history[best_epoch - 1]["validation"],
        "epochs": history,
        "train_events": len(train_events),
        "training_examples": len(examples),
        "train_users": len(users),
        "training_users": len(eligible),
        "singleton_users_excluded_from_fitting": len(users) - len(eligible),
        "negative_policy": "Uniform full train catalog, reject all user train positives; no held-out exclusion",
        "context_policy": "Mean of last 50 train items; remove positive target before capping in fitting",
        "validation_policy": "All frozen users, full train catalog, exclude all train positives; unavailable targets miss",
        "build_seconds": time.perf_counter() - started,
    }
    write_json(final, manifest)
    print(
        f"Saved Two-Tower: best_epoch={best_epoch}, validation={manifest['best_validation']}",
        flush=True,
    )
    return manifest
