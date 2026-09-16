"""知识点讲解 Agent 评测（学习模式）。

覆盖：大纲完整性 / 离线讲解可用 / 学习流程端到端（讲解→继续→换例题→小测→退出）/
      学习中不产出错题报告 / 讲解工具可调用。

    python scripts/eval_teaching.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'✔' if ok else '✘'} {name}" + (f"  — {detail}" if detail else ""))
    if not ok and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(f"{name} — {detail}")


def main() -> int:
    print("=" * 64)
    print("知识点讲解 Agent（学习模式）评测")
    print("=" * 64)

    from math_agent.teaching import (CURRICULUM, build_lecture, get_point,
                                     list_curriculum, match_topic, rule_lecture)

    # 1) 大纲完整性
    print("\n── 1. 知识大纲 ──")
    check("覆盖 6 个板块", len(CURRICULUM) == 6, str(list(CURRICULUM)))
    bad = []
    for topic, kps in CURRICULUM.items():
        if len(kps) < 4:
            bad.append(f"{topic} 仅 {len(kps)} 个知识点")
        for kp in kps:
            for f in ("name", "summary", "points", "pitfalls", "tip"):
                if not kp.get(f):
                    bad.append(f"{topic}/{kp.get('name')} 缺 {f}")
    check("每板块 ≥4 个知识点且字段完整", not bad, "; ".join(bad[:3]))
    check("板块关键词可识别",
          match_topic("我想学一次函数") == "函数"
          and match_topic("方程组怎么解") == "方程与不等式"
          and match_topic("全等三角形") == "几何"
          and match_topic("学二次根式") == "数与式"
          and match_topic("统计里的方差") == "统计与概率"
          and match_topic("正弦怎么算") == "锐角三角函数"
          and match_topic("解直角三角形") == "锐角三角函数"
          and match_topic("一元二次方程") == "方程与不等式"
          and match_topic("分式方程") == "方程与不等式")

    # 2) 离线讲解（无 API Key 也必须可用）
    print("\n── 2. 离线讲解 ──")
    kp = get_point("函数", 2)
    text = rule_lecture(kp, "函数")
    check("规则讲解含四要素",
          all(k in text for k in ("◆ 定义", "◆ 要点", "◆ 易错点", "◆ 口诀")))
    lec = build_lecture("函数", kp)
    check("build_lecture 有产出且标注来源", bool(lec["text"]) and lec["source"] in ("llm", "rule"),
          f"source={lec['source']}")
    check("目录可输出", "可学习的知识点" in list_curriculum("函数"))

    # 3) 学习流程端到端
    print("\n── 3. 学习流程（端到端）──")
    from math_agent.graph.builder import MathAgent

    agent = MathAgent(user_id="u_teach_eval")
    r1 = agent.chat("我要学函数", session_mode="learn")
    s1 = agent.last_state.get("learning_session") or {}
    check("进入学习：返回讲解", "◆ 定义" in r1 and "要点" in r1)
    check("进入学习：建立学习会话", s1.get("topic") == "函数", str(s1)[:80])
    check("首讲带配套小测", bool(s1.get("pending_quiz")), "pending_quiz 为空")

    r2 = agent.chat("继续", session_mode="learn")
    s2 = agent.last_state.get("learning_session") or {}
    check("「继续」推进到下一个知识点",
          int(s2.get("index", -1)) == int(s1.get("index", -2)) + 1,
          f"index {s1.get('index')} → {s2.get('index')}")

    r3 = agent.chat("换个例题", session_mode="learn")
    s3 = agent.last_state.get("learning_session") or {}
    check("「换个例题」换出新题", bool(s3.get("pending_quiz")) and "【" in r3)

    r4 = agent.chat("A", session_mode="learn")
    s4 = agent.last_state.get("learning_session") or {}
    check("小测作答被判分", ("✅" in r4 or "❌" in r4), r4[:60].replace("\n", " "))
    check("小测计数累加", int(s4.get("total", 0)) >= 1, f"total={s4.get('total')}")

    r5 = agent.chat("没听懂", session_mode="learn")
    check("「没听懂」重新讲解", "◆ 定义" in r5 or "要点" in r5)

    r6 = agent.chat("退出学习", session_mode="learn")
    check("退出后学习会话清空",
          not (agent.last_state.get("learning_session") or {}).get("topic"))

    # 4) 硬约束：学习模式不产出错题报告
    print("\n── 4. 硬约束 ──")
    check("学习全程无错题报告",
          not any("错题分析报告" in r or "【薄弱板块】" in r
                  for r in (r1, r2, r3, r4, r5, r6)))

    # 5) 讲解工具
    print("\n── 5. 讲解工具 ──")
    from math_agent.tools.registry import run_tool

    res = run_tool("explain_knowledge_point", {"topic": "几何图像", "point": "勾股定理",
                                               "with_examples": 1})
    check("explain_knowledge_point 可调用", res.ok and res.output.get("ok"),
          res.error[:60])
    check("工具返回要点/易错点/口诀",
          bool(res.output.get("points")) and bool(res.output.get("pitfalls"))
          and bool(res.output.get("tip")) if res.ok else False)
    check("工具可带例题", bool((res.output or {}).get("examples")) if res.ok else False)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 64)
    print(f"结果：{passed}/{len(RESULTS)} 项通过")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  ✘ {name}  {detail}")
    print("=" * 64)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
