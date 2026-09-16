#!/usr/bin/env python
"""Phase 1 —— 题库数据下载。

数据源：GitHub 开源仓库 ``math-eval/TAL-SCQ5K``（初中数学选择题，含知识点/难度/解析）。
国内网络下 GitHub 直连不稳定，脚本内置多个镜像自动降级：

    gh-proxy  →  jsdelivr  →  hf-mirror(HuggingFace 同源镜像)

产出：data/raw/TAL-SCQ5K-CN_{train,test}.jsonl
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_agent.config import RAW_DIR  # noqa: E402

REPO = "math-eval/TAL-SCQ5K"
FILES = ["TAL-SCQ5K-CN/train.jsonl", "TAL-SCQ5K-CN/test.jsonl"]

MIRRORS = [
    lambda p: f"https://gh-proxy.com/https://raw.githubusercontent.com/{REPO}/master/{p}",
    lambda p: f"https://cdn.jsdelivr.net/gh/{REPO}@master/{p}",
    lambda p: f"https://hf-mirror.com/datasets/{REPO}/resolve/main/{p}",
    lambda p: f"https://raw.githubusercontent.com/{REPO}/master/{p}",
]


def http_download(url: str, dest: Path, timeout: int = 300) -> bool:
    import urllib.request

    headers = {"User-Agent": "Mozilla/5.0 (math-agent)"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r  {dest.name} {pct}%", end="", flush=True)
        print()
        return dest.exists() and dest.stat().st_size > 1024
    except Exception as e:
        print(f"\n  失败 {type(e).__name__}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for rel in FILES:
        dest = RAW_DIR / (rel.replace("/", "_"))
        if dest.exists() and dest.stat().st_size > 1024:
            print(f"[skip] {dest.name} 已存在（{dest.stat().st_size / 1024:.0f} KB）")
            continue
        print(f"[get ] {rel}")
        for i, mk in enumerate(MIRRORS, 1):
            url = mk(rel)
            print(f"  镜像 {i}/{len(MIRRORS)}: {url[:70]}...")
            if http_download(url, dest):
                print(f"  ✔ 保存 {dest}（{dest.stat().st_size / 1024:.0f} KB）")
                break
            time.sleep(1)
        else:
            print(f"  ✘ 所有镜像均失败：{rel}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
