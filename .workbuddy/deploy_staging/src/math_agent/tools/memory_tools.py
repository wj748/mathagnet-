"""记忆 / 检索工具（方案 4.1、4.2）。

两个检索工具**严格分离**，对应双 Collection：
* ``search_knowledge``              → knowledge_base（全局共享）
* ``search_user_memory``            → user_long_term_memory（强带 user_id 过滤）
* ``update_long_term_memory``       → 写入长期记忆（仅阶段报告后由 Memory Manager 调用）
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ..config import cfg
from ..vectorstore import IsolationError, get_store
from .registry import register_tool

MEMORY_TYPES = ["habit", "weak_point", "strength", "report"]


class SearchKnowledgeArgs(BaseModel):
    query: str = Field(..., description="检索内容，如概念、定理、题目关键词")
    top_k: int = Field(5, ge=1, le=20)
    topic: str | None = Field(None, description="限定板块：函数 / 二元一次方程 / 几何图像")


class SearchMemoryArgs(BaseModel):
    query: str = Field(..., description="检索内容，如'我的计算习惯'")
    user_id: str = Field(..., description="用户 ID（必填，用于记忆隔离）")
    top_k: int = Field(5, ge=1, le=20)


class UpdateMemoryArgs(BaseModel):
    user_id: str = Field(..., description="用户 ID（必填）")
    content: str = Field(..., description="记忆内容，一条具体的习惯或薄弱点描述")
    type: str = Field("habit", description=f"记忆类型：{MEMORY_TYPES}")
    source: str = Field("", description="来源，如 quiz_session_20260907")


@register_tool(
    "search_knowledge",
    "在数学知识库（题库/概念/定理）中检索相关内容。全局共享，与用户无关。",
    SearchKnowledgeArgs,
)
def search_knowledge(query: str, top_k: int = 5, topic: str | None = None) -> Dict[str, Any]:
    hits = get_store().search_knowledge(query, top_k=top_k, topic=topic)
    thr = float(cfg.get("retrieval.score_threshold", 0.10))
    hits = [h for h in hits if float(h.get("score", 0)) >= thr]

    def _text(h: Dict[str, Any]) -> str:
        if h.get("text"):
            return h["text"]
        # 兜底：旧库可能没有 text 字段，用结构化字段拼回可读内容
        parts = [h.get("stem", ""), " ".join(h.get("options") or [])]
        if h.get("answer"):
            parts.append(f"答案：{h['answer']}")
        if h.get("analysis"):
            parts.append(f"解析：{h['analysis']}")
        return " ".join(p for p in parts if p).strip()

    return {"query": query, "count": len(hits),
            "results": [{"text": _text(h), "score": round(float(h.get("score", 0)), 4),
                         "topic": h.get("topic", ""), "knowledge_points": h.get("knowledge_points", [])}
                        for h in hits]}


@register_tool(
    "search_user_memory",
    "检索某个用户的长期记忆（错题习惯、薄弱点、历史报告）。"
    "必须传 user_id，系统会在代码层强制过滤，绝不会返回其他用户的记忆。",
    SearchMemoryArgs,
)
def search_user_memory(query: str, user_id: str, top_k: int = 5) -> Dict[str, Any]:
    if not user_id or not str(user_id).strip():
        raise IsolationError("search_user_memory 必须提供非空 user_id")
    hits = get_store().search_user_memory(query, user_id=user_id, top_k=top_k)
    return {"query": query, "user_id": user_id, "count": len(hits),
            "results": [{"content": h.get("content", ""), "type": h.get("type", ""),
                         "score": round(float(h.get("score", 0)), 4),
                         "timestamp": h.get("timestamp", ""), "source": h.get("source", "")}
                        for h in hits]}


@register_tool(
    "update_long_term_memory",
    "把用户的一条学习习惯/薄弱点写入长期记忆。"
    "仅在完成一组完整刷题或批改后调用，不要对单题做记忆写入。",
    UpdateMemoryArgs,
    retryable=False,   # 写入类工具：重试可能产生重复记忆条目，失败交回 Agent 处理
)
def update_long_term_memory(user_id: str, content: str, type: str = "habit",
                            source: str = "") -> Dict[str, Any]:
    if type not in MEMORY_TYPES:
        raise ValueError(f"type 必须是 {MEMORY_TYPES} 之一，收到：{type}")
    n = get_store().upsert_memory(user_id, [{
        "content": content.strip(), "type": type, "source": source,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }])
    return {"written": n, "user_id": user_id, "type": type}


__all__ = ["search_knowledge", "search_user_memory", "update_long_term_memory",
           "MEMORY_TYPES", "IsolationError"]
