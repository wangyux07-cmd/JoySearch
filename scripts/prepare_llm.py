import time

from huggingface_hub import snapshot_download

from joysearch.common import ROOT, read_json, write_json
from joysearch.data.wands import file_hash

if __name__ == "__main__":
    started = time.perf_counter()
    config = read_json(ROOT / "configs/llm.json")
    output = ROOT / config["model_path"]
    snapshot_download(
        repo_id=config["repo_id"],
        revision=config["revision"],
        local_dir=output,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE", "README.md"],
    )
    files = {
        p.relative_to(output).as_posix(): file_hash(p)
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    write_json(
        output.parent / "manifest.json",
        {
            "repo_id": config["repo_id"],
            "revision": config["revision"],
            "model_path": config["model_path"],
            "files": files,
            "download_seconds": time.perf_counter() - started,
            "license": "Apache-2.0; official model card saved with the snapshot",
        },
    )
    print(
        {
            "model_path": str(output),
            "files": list(files),
            "download_seconds": time.perf_counter() - started,
        }
    )
