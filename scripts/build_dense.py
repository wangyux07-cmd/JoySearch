import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joysearch.common import ROOT, read_json, write_json
from joysearch.data.wands import file_hash
from joysearch.search.dense import TEXT_FIELDS, DenseIndex, SentenceEncoder, product_texts


def checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, state)
    temporary.replace(path)


def build(source, output, config):
    # Public fixed-revision weights only; serving never connects to Hugging Face.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    data_manifest = read_json(source / "manifest.json")
    signature = {
        "config": config,
        "products_sha256": file_hash(source / "products.parquet"),
        "version": DenseIndex.version,
        "text_fields": TEXT_FIELDS,
    }
    if (output / "manifest.json").exists():
        if read_json(output / "manifest.json")["signature"] != signature:
            raise ValueError("Dense build settings changed; use a new --output directory")
        print(f"Dense index already complete: {output}; skipping encoding", flush=True)
        return
    state_path = output / "build_state.json"
    state = read_json(state_path) if state_path.exists() else None
    if state is not None and state["signature"] != signature:
        raise ValueError("Cannot resume dense build with changed input/config; use new --output")
    model_path = ROOT / "artifacts/models/all-MiniLM-L6-v2" / config["revision"]
    required = [
        "config.json",
        "modules.json",
        "model.safetensors",
        "sentence_bert_config.json",
        "config_sentence_transformers.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "1_Pooling/config.json",
    ]
    if not all((model_path / name).exists() for name in required):
        snapshot_download(
            config["model_id"],
            revision=config["revision"],
            local_dir=model_path,
            token=False,
            max_workers=4,
            allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/config.json"],
        )
    encoder = SentenceEncoder(model_path, config["max_seq_length"], config["torch_threads"])
    products = (
        pd.read_parquet(source / "products.parquet")
        .sort_values("product_id")
        .reset_index(drop=True)
    )
    output.mkdir(parents=True, exist_ok=True)
    if state is not None:
        embeddings = np.lib.format.open_memmap(output / "embeddings.npy", mode="r+")
    else:
        embeddings = np.lib.format.open_memmap(
            output / "embeddings.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(products), encoder.dimension),
        )
        state = {
            "signature": signature,
            "completed_rows": 0,
            "total_rows": len(products),
            "encoding_seconds": 0.0,
        }
        write_json(output / "item_ids.json", products.product_id.tolist())
        checkpoint(state_path, state)
    if embeddings.shape != (len(products), encoder.dimension):
        raise ValueError(f"Dense resume shape mismatch: {output}")
    for start in range(state["completed_rows"], len(products), config["chunk_size"]):
        tick = time.perf_counter()
        end = min(start + config["chunk_size"], len(products))
        embeddings[start:end] = encoder.encode(
            product_texts(products.iloc[start:end]), config["batch_size"]
        )
        embeddings.flush()
        state["completed_rows"] = end
        state["encoding_seconds"] += time.perf_counter() - tick
        checkpoint(state_path, state)
        print(
            f"Encoded {end}/{len(products)} products; encoding_seconds={state['encoding_seconds']:.1f}",
            flush=True,
        )
    manifest = {
        "version": DenseIndex.version,
        "catalog_id": "wands_search",
        "status": "completed",
        "catalog_size": len(products),
        "dimension": encoder.dimension,
        "normalization": "l2",
        "model_id": config["model_id"],
        "revision": config["revision"],
        "model_path": model_path.relative_to(ROOT).as_posix(),
        "model_files_sha256": {
            p.relative_to(model_path).as_posix(): file_hash(p)
            for p in sorted(model_path.rglob("*"))
            if p.is_file() and ".cache" not in p.parts
        },
        "max_seq_length": config["max_seq_length"],
        "torch_threads": config["torch_threads"],
        "text_fields": TEXT_FIELDS,
        "data_manifest": data_manifest,
        "signature": signature,
        "encoding_seconds": state["encoding_seconds"],
        "build_seconds_this_invocation": time.perf_counter() - started,
    }
    checkpoint(output / "manifest.json", manifest)
    print(f"Completed dense index: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/wands")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/search_dense")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dense.json")
    args = parser.parse_args()
    build(args.source, args.output, read_json(args.config))
