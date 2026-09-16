#!/usr/bin/env python
"""命令行入口：交互式对话 or 一键演示完整闭环。

    python scripts/run_cli.py                 # 交互式
    python scripts/run_cli.py --demo          # 自动演示：刷题→作答→交卷→报告→习惯
    python scripts/run_cli.py --user u_001    # 指定用户（记忆按 user_id 隔离）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_agent.db.repository import topic_stats  # noqa: E402
from math_agent.graph.builder import MathAgent  # noqa: E402

DEMO_SCRIPT = [
    "我要刷函数的选择题，来3道",
    "A",
    "B",
    "解析",
    "C",
    "交卷",
    "我的学习习惯怎么样",
    "一次函数的斜率是什么",
]

BANNER = """
╔══════════════════════════════════════════════════════╗
║   初中数学辅导 Agent  (LangGraph)                     ║
║   刷题 / 错题分析 / 作业批改 / 习惯追踪               ║
║   输入 quit 退出                                      ║
╚══════════════════════════════════════════════════════╝
"""


def run_demo(user_id: str) -> None:
    agent = MathAgent(user_id=user_id)
    for line in DEMO_SCRIPT:
        print(f"\n\033[36m你 ▸\033[0m {line}")
        out = agent.invoke(line)
        print(f"\033[32m助手 ▸\033[0m {out.get('response')}")
        print(f"\033[90m   [intent={out.get('intent')} | trace={' → '.join(out.get('trace', []))}]\033[0m")


def run_interactive(user_id: str) -> None:
    print(BANNER)
    try:
        stats = topic_stats()
        total = sum(sum(v.values()) for v in stats.values())
        print(f"题库：{total} 题  " +
              "  ".join(f"{k}{sum(v.values())}" for k, v in stats.items()))
    except Exception as e:
        print(f"（题库未就绪：{e}；请先运行 scripts/build_question_bank.py）")
    agent = MathAgent(user_id=user_id)
    print(f"用户：{user_id}\n")
    while True:
        try:
            text = input("\033[36m你 ▸\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 👋")
            break
        if text.lower() in ("quit", "exit", "q", "退出"):
            print("再见 👋")
            break
        if not text:
            continue
        out = agent.invoke(text)
        print(f"\033[32m助手 ▸\033[0m {out.get('response')}")
        trace = out.get("trace") or []
        if trace:
            print(f"\033[90m   [{' → '.join(trace)}]\033[0m")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="自动演示完整闭环")
    ap.add_argument("--user", default="u_cli", help="用户 ID")
    args = ap.parse_args()
    if args.demo:
        run_demo(args.user)
    else:
        run_interactive(args.user)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
