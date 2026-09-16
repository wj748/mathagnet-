#!/usr/bin/env python
"""从 ModelScope 下载中文向量模型 BAAI/bge-small-zh-v1.5。

下载后 embeddings.py 的 ``bge`` 后端会自动启用（真实语义向量）；
未下载时系统自动回退到零依赖的 hashing 向量，功能不受影响。

用法::
    python scripts/download_embedding_model.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_agent.config import cfg, resolve_path  # noqa: E402

API = "https://www.modelscope.cn/api/v1/models/{model}/repo/files?Revision=master&Recursive=True"
FILE_API = "https://www.modelscope.cn/api/v1/models/{model}/repo?Revision=master&FilePath={path}"
NEED = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
        "special_tokens_map.json", "modules.json", "sentence_bert_config.json",
        "1_Pooling/config.json"]
WEIGHTS = ["model.safetensors", "pytorch_model.bin"]


def _get(url: str, stream: bool = False):
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=120)


def main() -> int:
    model = cfg.get("embedding.model_id", "BAAI/bge-small-zh-v1.5")
    dest = resolve_path(cfg.get("embedding.model_dir", "models/bge-small-zh-v1.5"))
    dest.mkdir(parents=True, exist_ok=True)

    try:
        with _get(API.format(model=model)) as r:
            data = json.load(r)
        files = {f["Path"]: int(f["Size"]) for f in data["Data"]["Files"] if f["Type"] == "blob"}
    except Exception as e:
        print(f"✘ 无法访问 ModelScope（{type(e).__name__}: {e}），可继续使用 hashing 向量")
        return 1

    want = [p for p in NEED if p in files]
    w = next((p for p in WEIGHTS if p in files and (dest / p).exists() is False), None)
    if w:
        want.append(w)

    for path in want:
        out = dest / path
        if out.exists() and out.stat().st_size > 0:
            print(f"[skip] {path}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        size = files[path]
        print(f"[get ] {path} （{size / 1e6:.1f} MB）")
        t0 = time.time()
        with _get(FILE_API.format(model=model, path=path)) as r, open(out, "wb") as f:
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if size:
                    print(f"\r  {done * 100 // size}%", end="", flush=True)
        print(f"\r  ✔ {path} 用时 {time.time() - t0:.0f}s")

    print(f"✔ 模型已保存到 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
