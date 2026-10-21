"""作业批改工具（方案 3.3）：OCR → 结构化 → 判分 → 进入错题分析。

判分策略（三档，避免"没答案就瞎判"）
------------------------------------
1. **向量匹配**：用题干去知识库检索相似题，命中则取标准答案做**代码判分**（最准）。
2. **LLM 判定**：未命中且 LLM 可用时，交给模型做过程评分。
3. **待确认**：两者都不可用时标记 ``unknown``，绝不臆造对错。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..config import cfg
from ..llm import get_llm
from .quiz import answers_equal
from .registry import register_tool

_ITEM_SPLIT = re.compile(r"(?:^|\n)\s*(?:第?\s*)(\d{1,2})\s*[\.、．\)）:：]\s*")
_ANS_MARK = re.compile(r"(答案|答|解)\s*[:：]?\s*(.+)")


class GradeAssignmentArgs(BaseModel):
    text: str = Field(..., description="OCR 解析后的作业文本（含题目与学生作答）")
    use_llm: bool = Field(True, description="向量未命中时是否使用 LLM 判定")


class GradeApplicationArgs(BaseModel):
    question: str = Field(..., description="应用题题干")
    user_answer: str = Field(..., description="学生作答（可含过程）")


def parse_items(text: str) -> List[Dict[str, str]]:
    """把 OCR 文本切成 {stem, user_answer} 列表。"""
    text = (text or "").strip()
    if not text:
        return []
    matches = list(_ITEM_SPLIT.finditer(text))
    items: List[Dict[str, str]] = []
    if not matches:
        m = _ANS_MARK.search(text)
        if m:
            items.append({"stem": text[:m.start()].strip(), "user_answer": m.group(2).strip()})
        else:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            for i in range(0, len(lines) - 1, 2):
                items.append({"stem": lines[i], "user_answer": lines[i + 1]})
        return items
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.end():end].strip()
        am = _ANS_MARK.search(block)
        if am:
            items.append({"stem": block[:am.start()].strip(),
                          "user_answer": am.group(2).strip()})
        else:
            items.append({"stem": block, "user_answer": ""})
    return [it for it in items if it["stem"]]


def _retrieve_standard(stem: str) -> Optional[Dict[str, Any]]:
    """用题干检索知识库，取最相似题目的标准答案。"""
    try:
        from ..vectorstore import get_store
        hits = get_store().search_knowledge(stem, top_k=1)
    except Exception:
        return None
    if not hits:
        return None
    top = hits[0]
    if float(top.get("score", 0)) < float(cfg.get("retrieval.score_threshold", 0.10)):
        return None
    if not top.get("answer"):
        return None
    return top


def _llm_judge(stem: str, user_answer: str, reference: str = "") -> Dict[str, Any]:
    llm = get_llm()
    if not llm.available:
        return {"correct": None, "reason": "LLM 不可用", "method": "unknown"}
    prompt = (
        "你是初中数学阅卷老师。判断学生作答是否正确。\n"
        f"题目：{stem}\n"
        f"{('参考答案：' + reference) if reference else ''}\n"
        f"学生作答：{user_answer}\n"
        '只输出 JSON：{"correct": true/false, "reason": "一句话理由", "score": 0-10}'
    )
    data = llm.chat_json(prompt, fallback=None)
    if not isinstance(data, dict) or "correct" not in data:
        return {"correct": None, "reason": "LLM 输出解析失败", "method": "unknown"}
    return {"correct": bool(data["correct"]), "reason": str(data.get("reason", "")),
            "score": data.get("score"), "method": "llm"}


@register_tool(
    "grade_assignment",
    "批改作业/试卷：输入 OCR 文本，逐题切分并判分。"
    "必须先调用 ocr_document 获取文本。返回每题对错与整体统计。",
    GradeAssignmentArgs,
    requires=["ocr_document"],
)
def grade_assignment(text: str, use_llm: bool = True) -> Dict[str, Any]:
    items = parse_items(text)
    if not items:
        raise ValueError("未能从文本中解析出题目，请检查 OCR 结果格式")
    llm = get_llm()
    results: List[Dict[str, Any]] = []
    for idx, it in enumerate(items, 1):
        stem, ua = it["stem"], it["user_answer"]
        ref = _retrieve_standard(stem)
        if ref and ua:
            correct = answers_equal(ref["answer"], ua)
            results.append({"no": idx, "stem": stem, "user_answer": ua,
                            "correct": correct, "standard_answer": ref.get("answer", ""),
                            "analysis": ref.get("analysis", ""),
                            "topic": ref.get("topic", ""), "method": "vector+code"})
            continue
        if use_llm and llm.available:
            j = _llm_judge(stem, ua, ref.get("answer", "") if ref else "")
            results.append({"no": idx, "stem": stem, "user_answer": ua,
                            "correct": j["correct"],
                            "standard_answer": ref.get("answer", "") if ref else "",
                            "analysis": j.get("reason", ""),
                            "topic": ref.get("topic", "") if ref else "",
                            "method": j["method"]})
        else:
            results.append({"no": idx, "stem": stem, "user_answer": ua,
                            "correct": None, "standard_answer": "",
                            "analysis": "无法自动判定，请人工确认或配置 LLM",
                            "topic": "", "method": "unknown"})
    total = len(results)
    ok = sum(1 for r in results if r["correct"] is True)
    wrong = sum(1 for r in results if r["correct"] is False)
    unknown = sum(1 for r in results if r["correct"] is None)
    return {"items": results, "summary": {"total": total, "correct": ok,
                                          "wrong": wrong, "unknown": unknown}}


@register_tool(
    "grade_application_question",
    "应用题/解答题判定：结合过程给分。优先用知识库标准答案做代码比对，"
    "未命中时交由 LLM 做过程评分。",
    GradeApplicationArgs,
)
def grade_application_question(question: str, user_answer: str) -> Dict[str, Any]:
    ref = _retrieve_standard(question)
    if ref:
        correct = answers_equal(ref["answer"], user_answer)
        return {"correct": correct, "standard_answer": ref.get("answer", ""),
                "analysis": ref.get("analysis", ""), "method": "vector+code"}
    return {"question": question, "user_answer": user_answer, **_llm_judge(question, user_answer)}


__all__ = ["grade_assignment", "grade_application_question", "parse_items"]
