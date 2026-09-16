"""知识点讲解工具（供 ReAct / 学习模式共用）。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..teaching import (CURRICULUM, build_lecture, find_point_index, get_point,
                        match_topic, pick_examples)
from ..topics import TEACHING_TOPICS
from .registry import register_tool


class ExplainArgs(BaseModel):
    topic: str = Field("", description=f"教学板块，枚举值之一：{TEACHING_TOPICS}")
    point: str = Field("", description="知识点名称或序号，留空则从第一个开始")
    with_examples: int = Field(1, ge=0, le=3, description="附带几道配套例题")


@register_tool(
    "explain_knowledge_point",
    "讲解一个初中数学知识点：返回定义、要点、易错点、口诀，并可附带配套例题。"
    "用于学习知识点/答疑场景，不是刷题。",
    ExplainArgs,
)
def explain_knowledge_point(topic: str = "", point: str = "",
                            with_examples: int = 1) -> Dict[str, Any]:
    topic = match_topic(topic) or (topic if topic in CURRICULUM else "")
    if not topic:
        topic = (point and match_topic(point)) or "函数"
    idx = find_point_index(topic, point)
    kp = get_point(topic, idx)
    if kp is None:
        return {"ok": False, "message": f"未找到知识点：{point}", "topic": topic}
    lecture = build_lecture(topic, kp)
    out: Dict[str, Any] = {
        "ok": True, "topic": topic, "point": kp["name"], "index": idx,
        "total": len(CURRICULUM[topic]), "text": lecture["text"],
        "source": lecture["source"], "summary": kp["summary"],
        "points": kp["points"], "pitfalls": kp["pitfalls"], "tip": kp["tip"],
    }
    if with_examples:
        out["examples"] = pick_examples(topic, max(1, min(with_examples, 3)))
    return out


__all__ = ["explain_knowledge_point"]
