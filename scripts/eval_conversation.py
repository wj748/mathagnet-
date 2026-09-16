"""对话评测集（方案 Phase 4）：意图路由准确率 + 关键行为约束回归。

方案要求"建立基础评测集（20-30 个典型对话脚本），每次改动回归测试"，
且 Phase 4 的验收标准是**通过率 > 85%**。

覆盖两类：
  1) 意图路由：30 条真实用户输入（含短输入、歧义、错别字等易错场景）
  2) 行为约束：多轮会话脚本，断言"不逐题出报告""交卷才出报告""工具防重"等硬约束

    python scripts/eval_conversation.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ==========================================================================
# 1) 意图路由评测集
#    (输入, 是否处于刷题会话中, 期望意图)
#    易错点：短输入作答、歧义词（"作业"既可能是批改也可能是刷题）、错别字
# ==========================================================================
ROUTING_CASES = [
    # ---- quiz：开始/继续刷题 ----
    ("我要刷函数的选择题", False, "quiz"),
    ("来5道二元一次方程的题", False, "quiz"),
    ("我想练习几何图像", False, "quiz"),
    ("给我出几道应用题", False, "quiz"),
    ("开始做题", False, "quiz"),
    ("来一套综合测验", False, "quiz"),
    ("再刷3道", True, "quiz"),
    ("刷方程", False, "quiz"),
    # ---- quiz：新增教学板块（2026-09-10 修复：原路由完全漏掉这三个板块）----
    ("刷概率题", False, "quiz"),
    ("来5道统计题", False, "quiz"),
    ("刷三角函数", False, "quiz"),
    ("做几道锐角三角函数的题", False, "quiz"),
    # ---- quiz：刷题中的短输入作答（最容易漂移成 chat 的场景）----
    ("A", True, "quiz"),
    ("（B）", True, "quiz"),
    ("c.", True, "quiz"),
    ("12", True, "quiz"),
    ("-3", True, "quiz"),
    ("3/4", True, "quiz"),
    # ---- submit ----
    ("交卷", True, "submit"),
    ("我不做了，出报告吧", True, "submit"),
    ("结束刷题", True, "submit"),
    ("总结一下我这组", True, "submit"),
    ("看看结果", True, "submit"),
    # ---- grade ----
    ("帮我批改作业", False, "grade"),
    ("我上传了试卷图片", False, "grade"),
    ("这道题判一下分", False, "grade"),
    ("成绩单分析", False, "grade"),
    # ---- habit ----
    ("我的学习习惯怎么样", False, "habit"),
    ("我哪些地方比较薄弱", False, "habit"),
    ("最近有进步吗", False, "habit"),
    ("分析一下我的学情", False, "habit"),
    # ---- chat：不应被误判成业务意图 ----
    ("什么是一次函数", False, "chat"),
    ("你好", False, "chat"),
    ("这道题怎么解：2x+3=7", False, "chat"),
    # ---- chat：刷题会话中问概念，不能被当成作答判错 ----
    ("什么是斜率", True, "chat"),
    ("这道题怎么求", True, "chat"),
    # ---- quiz：单题反馈指令仍留在刷题会话内 ----
    ("解析", True, "quiz"),
    ("提示", True, "quiz"),
]

# 学习模式（知识点讲解 Agent）：(输入, 是否处于刷题会话, 是否在学习会话, 会话模式, 期望意图)
LEARN_CASES = [
    ("我想学函数的知识点", False, False, "", "learn"),
    ("教我一次函数", False, False, "", "learn"),
    ("帮我系统学习几何图像", False, False, "", "learn"),
    ("学习 函数 的 一次函数 y=kx+b", False, False, "learn", "learn"),
    ("继续", False, True, "learn", "learn"),          # 学习中：下一个知识点
    ("没听懂", False, True, "learn", "learn"),        # 学习中：换讲法
    ("A", False, True, "learn", "learn"),            # 学习中：小测作答（不落刷题）
    ("退出学习", False, True, "learn", "learn"),      # 学习中：退出仍交给学习节点清理
    ("我要刷函数的选择题", False, True, "learn", "quiz"),   # 学习中可切回刷题
    # 学习中切刷题后的作答：必须路由回 quiz 判分，不能被学习会话吞掉（P0 修复回归）
    ("A", True, True, "learn", "quiz"),
    ("12", True, True, "learn", "quiz"),
    # 学习中（未切刷题）的短输入仍是学习小测作答
    ("B", False, True, "learn", "learn"),
    ("什么是一次函数", False, False, "", "chat"),     # 概念咨询不受影响
]


# ==========================================================================
# 2) 行为约束评测（多轮会话脚本）
# ==========================================================================
def _behavior_cases(agent_cls):
    """返回 [(名称, 是否通过, 详情)]；每个用例独立建 agent 避免状态污染。"""
    out = []

    def run(name, script, assertions):
        """script: [(输入,)] 顺序执行；assertions: [(说明, 判定函数)]"""
        try:
            agent = agent_cls(user_id=f"u_eval_{abs(hash(name)) % 9999}")
            replies = []
            for item in script:
                # 支持 (文本,) 或 (文本, 会话模式) 两种写法
                msg, mode = (list(item) + [""])[:2] if isinstance(item, tuple) else (item, "")
                r = agent.chat(msg, **({"session_mode": mode} if mode else {}))
                replies.append(r if isinstance(r, str) else str(r))
            for desc, fn in assertions:
                ok, detail = fn(replies)
                out.append((f"{name} — {desc}", ok, "" if ok else detail))
        except Exception as e:
            out.append((name, False, f"{type(e).__name__}: {e}"))

    # --- 约束 1：单题答错只提示对错，绝不输出报告 ---
    def no_report(replies):
        bad = [r for r in replies if "错题分析报告" in r or "【薄弱板块】" in r]
        return not bad, f"混入了 {len(bad)} 份报告"

    run("刷题中答错", [
        "我要刷函数的选择题，来3道",
        "zzz_故意答错",
    ], [
        ("答错不给报告", no_report),
        ("给出对错反馈", lambda r: (any("❌" in x or "✅" in x for x in r),
                                    "无对错提示")),
    ])

    # --- 约束 2：交卷后才出报告，且报告结构完整 ---
    run("交卷出报告", [
        "我要刷二元一次方程的选择题，来2道",
        "A", "B", "交卷",
    ], [
        ("交卷后有报告", lambda r: (any("错题分析报告" in x for x in r), "未生成报告")),
        ("报告含正确率", lambda r: (any("正确率" in x for x in r), "缺正确率")),
    ])

    # --- 约束 3：刷题中不逐题出报告（连续多题）---
    run("连续作答不出报告", [
        "来3道几何图像的选择题",
        "A", "B", "C",
    ], [
        ("全程无报告", no_report),
    ])

    # --- 约束 5：学习模式（知识点讲解 Agent） ---
    def has_lecture(replies):
        good = [r for r in replies if "◆ 定义" in r or "知识点" in r or "📘" in r]
        return len(good) >= 2, f"只找到 {len(good)} 段讲解"

    run("学习模式讲解", [
        ("我要学函数", "learn"),
        ("继续", "learn"),
        ("退出学习", "learn"),
    ], [
        ("连续讲解两个知识点", has_lecture),
        ("学习模式不给错题报告", no_report),
        ("退出后清除学习会话", lambda r: ("本次学习了" in r[-1] or "学完了" in r[-1],
                                        f"末轮={r[-1][:40]}")),
    ])

    # --- 约束 4：习惯查询返回习惯报告 ---
    run("习惯查询", [
        "我的学习习惯怎么样",
    ], [
        ("返回习惯报告", lambda r: (any("习惯" in x for x in r), "无习惯内容")),
    ])

    return out


# ==========================================================================
RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"{'✔' if ok else '✘'} {name}" + (f"  — {detail}" if detail else ""))
    if not ok and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(f"{name} — {detail}")
    return ok


def main() -> int:
    print("=" * 64)
    print("对话评测集（Phase 4）—— 意图路由 + 行为约束")
    print("=" * 64)

    from math_agent.graph.router import route_intent

    print("\n── 1. 意图路由（{} 条）──".format(len(ROUTING_CASES)))
    hit = 0
    for text, in_quiz, expect in ROUTING_CASES:
        got, conf = route_intent(text, in_quiz=in_quiz, use_llm=False)
        ok = got == expect
        hit += ok
        if not ok:
            print(f"✘ {text!r} (in_quiz={in_quiz}) 期望={expect} 实际={got} conf={conf}")
    rate = hit / len(ROUTING_CASES) * 100
    check(f"意图路由准确率 {hit}/{len(ROUTING_CASES)} = {rate:.1f}%",
          rate > 85.0, f"阈值 85%")

    print("\n── 1b. 学习模式路由（{} 条）──".format(len(LEARN_CASES)))
    hit2 = 0
    for text, in_quiz, in_learn, mode, expect in LEARN_CASES:
        got, conf = route_intent(text, in_quiz=in_quiz, use_llm=False,
                                 mode=mode, in_learn=in_learn)
        ok = got == expect
        hit2 += ok
        if not ok:
            print(f"✘ {text!r} (mode={mode!r} in_learn={in_learn}) "
                  f"期望={expect} 实际={got} conf={conf}")
    check(f"学习模式路由准确率 {hit2}/{len(LEARN_CASES)}", hit2 == len(LEARN_CASES),
          f"学习意图必须 100%")

    print("\n── 2. 行为约束 ──")
    from math_agent.graph.builder import MathAgent

    for name, ok, detail in _behavior_cases(MathAgent):
        check(name, ok, detail)

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
