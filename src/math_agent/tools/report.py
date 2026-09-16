"""错题报告工具（对 LLM 暴露，与 Error Analyzer 节点共用同一份分析逻辑）。"""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ..analysis import analyze_errors
from .registry import register_tool


class ReportArgs(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ..., description="答题记录列表，每项含 stem/user_answer/answer/correct/knowledge_points")
    with_narrative: bool = Field(True, description="是否生成可直接展示的文本报告")


@register_tool(
    "generate_error_report",
    "汇总本次所有错题，输出结构化错题报告（错因分类占比、逐题分析、修改建议、练习推荐）。"
    "仅在用户交卷或批改完成后调用，不要对单题调用。",
    ReportArgs,
)
def generate_error_report(records: List[Dict[str, Any]],
                          with_narrative: bool = True) -> Dict[str, Any]:
    real = [r for r in records if not r.get("skipped")]
    if not real:
        raise ValueError("没有可分析的答题记录")
    rep = analyze_errors(real)
    if not with_narrative:
        rep.pop("narrative", None)
    return rep


__all__ = ["generate_error_report"]
