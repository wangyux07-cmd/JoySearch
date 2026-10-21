"""Local CPU generation with content-addressed, resumable raw-response caching."""

import hashlib
import importlib.metadata
import json
import time
from datetime import UTC, datetime

from joysearch.common import read_json, write_json
from joysearch.data.wands import file_hash


class LocalGenerator:
    def __init__(self, root):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        self.root = root
        self.config = read_json(root / "configs/llm.json")
        path = root / self.config["model_path"]
        manifest = read_json(path.parent / "manifest.json")
        if (
            manifest["revision"] != self.config["revision"]
            or manifest["repo_id"] != self.config["repo_id"]
        ):
            raise ValueError("Local LLM revision differs; run scripts/prepare_llm.py")
        for name, checksum in manifest["files"].items():
            if file_hash(path / name) != checksum:
                raise ValueError(f"Local LLM asset checksum mismatch: {name}")
        torch.set_num_threads(self.config["threads"])
        torch.manual_seed(self.config["seed"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="sdpa",
        ).eval()
        self.load_seconds = time.perf_counter() - started
        self.identity = {
            "assets": manifest["files"],
            "revision": self.config["revision"],
            "dtype": "float32",
            "threads": self.config["threads"],
            "engine_sha256": file_hash(root / "src/joysearch/search/local_model.py"),
            "packages": {
                p: importlib.metadata.version(p) for p in ["torch", "transformers", "safetensors"]
            },
        }

    def generate(self, messages, max_new_tokens):
        import torch

        request = {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "max_input_tokens": self.config["max_input_tokens"],
            "generation_seconds": self.config["generation_seconds"],
            "identity": self.identity,
        }
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        cache = self.root / "artifacts/llm_cache" / f"{digest}.json"
        if cache.exists():
            saved = read_json(cache)
            if saved["request"] != request:
                raise ValueError(f"LLM response cache request mismatch: {cache}")
            return {**saved["response"], "cache_hit": True, "cache_key": digest}
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        tokens = encoded["input_ids"].shape[-1]
        if tokens > self.config["max_input_tokens"]:
            result = {
                "raw": "",
                "input_tokens": tokens,
                "output_tokens": 0,
                "wall_ms": 0.0,
                "cpu_seconds": 0.0,
                "stop_reason": "prompt_too_long",
            }
        else:
            started, cpu = time.perf_counter(), time.process_time()
            with torch.inference_mode():
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    max_time=self.config["generation_seconds"],
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )[0, tokens:]
            elapsed = time.perf_counter() - started
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            reason = "eos" if int(output[-1]) in eos else "token_limit"
            if elapsed >= self.config["generation_seconds"]:
                reason = "deadline_exceeded"
            result = {
                "raw": self.tokenizer.decode(output, skip_special_tokens=True),
                "input_tokens": tokens,
                "output_tokens": len(output),
                "wall_ms": elapsed * 1000,
                "cpu_seconds": time.process_time() - cpu,
                "stop_reason": reason,
            }
        result["created_at"] = datetime.now(UTC).isoformat()
        write_json(cache, {"request": request, "response": result})
        return {**result, "cache_hit": False, "cache_key": digest}
