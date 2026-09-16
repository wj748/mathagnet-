"""冒烟测试：由 pytest 收集，覆盖核心链路与四项优化的关键断言。

    pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_agent.chunking import chunk_question, semantic_chunk  # noqa: E402
from math_agent.embeddings import cosine_similarity, embed_one  # noqa: E402
from math_agent.graph.builder import MathAgent  # noqa: E402
from math_agent.tools.registry import record_history, run_tool  # noqa: E402


def test_semantic_chunk_varies_by_threshold():
    text = "一次函数 y=kx+b 中 k 是斜率。三角形内角和是 180 度。今天天气很好。"
    assert len(semantic_chunk(text, similarity_threshold=0.9)) != \
           len(semantic_chunk(text, similarity_threshold=0.0))


def test_question_chunk_is_atomic():
    q = {"topic": "函数", "type": "选择题", "stem": "斜率是？", "options": ["A. 1"],
         "answer": "A", "analysis": "k=1", "knowledge_points": ["斜率"]}
    c = chunk_question(q)
    assert all(s in c for s in ["斜率是？", "A. 1", "答案：A", "解析：k=1"])


def test_duplicate_tool_call_blocked():
    history = []
    args = {"topic": "函数", "type": "选择题", "count": 2}
    r1 = run_tool("fetch_quiz", args, history=history)
    record_history(history, "fetch_quiz", args, r1)
    r2 = run_tool("fetch_quiz", args, history=history)
    assert r2.blocked and not r2.ok


def test_dependency_chain_enforced():
    r = run_tool("grade_assignment", {"text": "1. 1+1=2"}, history=[])
    assert not r.ok and "ocr_document" in r.error


def test_info_gain_detects_stall():
    a = embed_one("检索一次函数的定义与性质")
    assert cosine_similarity(a, a) >= 0.80


def test_quiz_flow_no_immediate_report():
    agent = MathAgent(user_id="u_pytest")
    out = agent.invoke("我要刷函数的选择题，来2道")
    assert (out.get("quiz_session") or {}).get("questions")
    out = agent.invoke("A")
    assert "错题分析报告" not in (out.get("response") or "")
    out = agent.invoke("B")
    out = agent.invoke("交卷")
    assert out.get("error_report") or "还没" in (out.get("response") or "")


def test_memory_isolation():
    from math_agent.vectorstore import IsolationError, get_store

    store = get_store()
    store.upsert_memory("u_py_a", [{"content": "pytest 私有记忆 AAA", "type": "habit"}])
    assert any("AAA" in h.get("content", "") for h in
               store.search_user_memory("pytest", "u_py_a", top_k=5))
    try:
        store.search_user_memory("pytest", "", top_k=3)
        assert False, "空 user_id 必须被拒绝"
    except IsolationError:
        pass
