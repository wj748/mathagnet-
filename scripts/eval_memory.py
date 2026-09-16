"""记忆优化评测：多轮对话的存储 / 压缩 / 释放（memory/compressor.py）。

验证目标（对应"多轮对话后模型思考变弱、token 变大"的优化）：
  1) **state 有界**：多轮之后 tool_call_history / observations 不随轮数线性增长
     （tool_call_history 旧记录压缩为 name+args_hash+ok，observations 只留窗口）
  2) **digest 有界**：注入 prompt 的压缩上下文 ≤ 预算，不随轮数增长
  3) **实体保留**：早期提到的板块/概念，多轮之后仍出现在压缩上下文中
  4) **无关释放**：无实体的闲聊轮次滑出窗口后被整轮释放（不进要点、不占预算）
  5) **防重语义不变**：tool_call_history 压缩后，同参数调用依旧被拦截
  6) **已交卷会话释放**：questions 整包释放，records/结论保留

    python scripts/eval_memory.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from math_agent.config import cfg  # noqa: E402
from math_agent.graph.builder import MathAgent  # noqa: E402
from math_agent.memory.compressor import (ConversationCompressor,  # noqa: E402
                                          prune_state)
from math_agent.tools.registry import run_tool  # noqa: E402

_PASS = 0
_FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    tag = "✔" if ok else "✘"
    print(f"{tag} {name}" + (f"  — {detail}" if detail else ""))
    _PASS += ok
    _FAIL += (not ok)
    if not ok and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(f"{name} — {detail}")


# ==========================================================================
# 1) 单元级：实体抽取 / 释放判定 / prune_state
# ==========================================================================
def test_unit() -> None:
    from math_agent.memory.compressor import _extract_entities, est_tokens

    ents = _extract_entities("我想刷函数的一次函数题目，TAL_abc12345 这道不会")
    check("实体抽取：板块+概念+题目ID", "函数" in ents and "一次函数" in ents
          and any(e.startswith("TAL") for e in ents), f"{ents}")

    c = ConversationCompressor("eval_unit")
    c.add_turn("user", "今天天气不错哦", intent="chat")
    t = c.turns[-1]
    check("无实体闲聊 → 判定为可释放", not t.relevant)

    check("est_tokens 非负且有量级", 0 < est_tokens("一二三四五") <= 10)

    # prune_state：旧历史压缩（单条瘦身，条数保留以维持防重语义）
    import json as _json
    hist = [{"name": "search_knowledge", "args": {"query": f"q{i}"},
             "args_hash": f"hash{i}", "ok": True, "error": ""} for i in range(30)]
    before = len(_json.dumps(hist, ensure_ascii=False))
    payload = prune_state({"tool_call_history": hist})
    new_hist = payload["tool_call_history"]
    after = len(_json.dumps(new_hist, ensure_ascii=False))
    full_keep = int(cfg.get("memory.hist_full", 8))
    beyond = int(cfg.get("memory.hist_compact_beyond", 24))
    check("prune_state：历史超阈值触发压缩（体积显著缩小）",
          after < before * 0.7 and len(new_hist) <= beyond,
          f"{before} → {after} 字符，30 → {len(new_hist)} 条（阈值 {beyond}）")
    check("压缩记录保留 name+args_hash+ok（防重/依赖链语义不变）",
          all(("args_hash" in h and "name" in h and "ok" in h)
              for h in new_hist[:-full_keep])
          and all("args" not in h for h in new_hist[:-full_keep]))

    # prune_state：observations 只留窗口
    obs = [{"tool": "search_knowledge", "observation": "x" * 500} for _ in range(20)]
    payload = prune_state({"short_term_memory": {"observations": obs}})
    window = int(cfg.get("memory.obs_window", 8))
    check("prune_state：observations 释放到窗口", len(
        payload["short_term_memory"]["observations"]) == window,
        f"20 → {window}")

    # prune_state：已交卷会话释放 questions、保留 records
    sess = {"finished": True, "questions": [{"stem": "长题干"}] * 5,
            "records": [{"correct": True}], "topic": "函数"}
    payload = prune_state({"quiz_session": sess})
    check("prune_state：已交卷释放 questions、保留 records",
          "questions" not in payload["quiz_session"]
          and payload["quiz_session"]["records"] == [{"correct": True}]
          and payload["quiz_session"]["topic"] == "函数")

    # 进行中的会话不释放
    sess_live = {"questions": [{"stem": "s"}], "index": 0, "records": []}
    payload = prune_state({"quiz_session": sess_live})
    check("prune_state：进行中会话的 questions 不释放",
          "questions" in payload["quiz_session"])


# ==========================================================================
# 2) 压缩器行为：预算、实体保留、无关释放
# ==========================================================================
def test_compressor() -> None:
    c = ConversationCompressor("eval_comp")
    budget = int(cfg.get("memory.context_budget", 1600))
    recent_n = int(cfg.get("memory.recent_turns", 6))

    # 第 1 轮：实体（函数）+ 第 2 轮：概念（斜率）
    c.add_turn("user", "我要刷函数的选择题", intent="quiz")
    c.add_turn("assistant", "好的，函数·选择题开始，第一题：一次函数 y=kx+b 的图像经过…", intent="quiz")
    c.add_turn("user", "什么是斜率", intent="chat")
    c.add_turn("assistant", "斜率表示直线的倾斜程度…", intent="chat")
    # 中间塞 30 轮无关闲聊 + 8 轮相关内容，把早期轮次挤出窗口
    for i in range(30):
        c.add_turn("user", f"哈哈{i}哈哈哈", intent="chat")
        c.add_turn("assistant", f"嗯嗯{i}，我们继续吧～", intent="chat")
    for i in range(4):
        c.add_turn("user", "再看看几何图像的全等三角形", intent="quiz")
        c.add_turn("assistant", f"全等第{i}题：△ABC≌△DEF 的判定条件…", intent="quiz")

    digest = c.build_context()
    check("digest 在预算内（不随轮数增长）", len(digest) <= budget + 200,
          f"{len(digest)} ch ≤ {budget}+200")
    check("早期实体'函数'仍保留（实体锚永不释放）", "函数" in digest)
    check("早期概念'斜率'仍保留", "斜率" in digest)
    check("近期实体'全等'保留", "全等" in digest)
    check("无关闲聊已释放（'哈哈3'不在上下文）", "哈哈3" not in digest)
    check("近期原文保留在窗口内", f"全等第{3}题" in digest)
    check("窗口轮数符合配置", digest.count("学生:") <= recent_n // 1 + 2,
          f"{digest.count('学生:')} 条学生消息 ≤ {recent_n}+2")
    check("已释放/压缩计数生效", c.stats()["released_turns"] > 0
          and c.stats()["compressed_turns"] > 0, str(c.stats()))


# ==========================================================================
# 3) 端到端：30 轮混合对话后 state 有界 + 防重语义不变
# ==========================================================================
def test_e2e() -> None:
    agent = MathAgent("eval_mem_user")
    turns = []
    for i in range(10):
        turns += [f"什么是正比例函数{i}", f"斜率是什么{i}", "随便聊聊今天的心情"]
    for t in turns:
        agent.invoke(t)

    st = agent.last_state
    hist = st.get("tool_call_history") or []
    obs = (st.get("short_term_memory") or {}).get("observations") or []
    beyond = int(cfg.get("memory.hist_compact_beyond", 24))
    window = int(cfg.get("memory.obs_window", 8))
    check("30 轮后 tool_call_history 有界", len(hist) <= beyond + 8, f"{len(hist)} ≤ {beyond}+8")
    max_loops = int(cfg.get("graph.max_react_loops", 8))
    check("30 轮后 observations 有界（窗口+末轮增量）", len(obs) <= window + max_loops,
          f"{len(obs)} ≤ {window}+{max_loops}")

    digest = agent.compressor.build_context()
    check("30 轮后 digest 仍含早期实体'正比例函数'", "正比例函数" in digest)
    check("30 轮后 digest 不含被释放的闲聊", "随便聊聊今天的心情" not in digest
          or digest.count("随便聊聊") <= 2)   # 窗口内允许保留极少几条

    # 防重语义：压缩后的历史仍拦截同参数调用
    args = {"query": "正比例函数", "top_k": 5}
    r1 = run_tool("search_knowledge", dict(args), history=hist, check_duplicate=False)
    r2 = run_tool("search_knowledge", dict(args),
                  history=[{"name": "search_knowledge", "args_hash": r1.args_hash, "ok": True}],
                  check_duplicate=True)
    check("压缩后的历史仍触发防重拦截", r2.blocked, r2.error[:40])


if __name__ == "__main__":
    print("=" * 64)
    print("记忆优化评测：存储 / 压缩 / 释放")
    print("=" * 64)
    test_unit()
    print("-" * 64)
    test_compressor()
    print("-" * 64)
    test_e2e()
    print("=" * 64)
    print(f"结果：{_PASS}/{_PASS + _FAIL} 项通过")
    print("=" * 64)
    sys.exit(0 if _FAIL == 0 else 1)
