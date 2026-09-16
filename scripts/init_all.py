#!/usr/bin/env python
"""一键初始化：下载题库 → ETL 入库 → 构建向量库 → 跑验收测试。

    python scripts/init_all.py            # 全量
    python scripts/init_all.py --skip-eval
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(name: str, args: list[str]) -> bool:
    print(f"\n{'=' * 60}\n▶ {name}\n{'=' * 60}")
    t0 = time.time()
    r = subprocess.run([PY, *args], cwd=ROOT)
    ok = r.returncode == 0
    print(f"{'✔' if ok else '✘'} {name} 用时 {time.time() - t0:.1f}s")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    steps = [
        ("下载题库（GitHub / 镜像）", ["scripts/download_dataset.py"]),
        ("ETL 入库 + 构建向量知识库", ["scripts/build_question_bank.py", "--with-vector"]),
    ]
    if not args.skip_eval:
        steps.append(("四项优化验收测试", ["scripts/eval_optimizations.py"]))

    for name, argv in steps:
        if not run(name, [str(ROOT / a) if not a.startswith("-") else a for a in argv]):
            print(f"\n✘ 步骤失败：{name}")
            return 1
    print("\n🎉 全部完成！运行 `python scripts/run_cli.py --demo` 体验完整流程，"
          "或 `python -m math_agent.api.server` 启动 Web 界面。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
