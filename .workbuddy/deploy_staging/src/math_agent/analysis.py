"""错题分析器（方案 3.2）：归类错因 → 结构化报告 → 习惯抽取。

错因四类：概念不清 / 计算错误 / 审题偏差 / 方法缺失。
判定为**规则优先 + LLM 润色**：规则保证任何环境下都有稳定输出，
LLM 可用时再补充"修改建议"的自然语言表述。
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List

from .config import cfg
from .llm import get_llm

ERROR_TYPES = ["概念不清", "计算错误", "审题偏差", "方法缺失"]

# ---------------------------------------------------------------- 规则特征
_CALC_KP = ("计算", "运算", "求解", "估算", "简便", "代数式", "整式", "分式", "有理数", "实数")
_CONCEPT_KP = ("概念", "定义", "性质", "判定", "图像", "图象", "函数", "坐标", "象限", "意义")
_METHOD_KP = ("方程", "应用", "证明", "辅助线", "综合", "探究", "建模")
_READING_KP = ("审题", "单位", "取值范围", "条件", "隐含")


def _to_number(s: str) -> float | None:
    s = (s or "").strip().replace("$", "").replace("\\", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def classify_error(q: Dict[str, Any], user_answer: str) -> str:
    """按题目特征 + 作答偏差模式归类错因。"""
    kp_text = "".join(q.get("knowledge_points") or []) + q.get("topic", "")
    qtype = q.get("type", "选择题")
    difficulty = int(q.get("difficulty", 3) or 3)
    std = str(q.get("answer", ""))

    ua, sa = _to_number(user_answer), _to_number(std)
    if ua is not None and sa is not None and ua != sa:
        # 典型"算错"特征：符号反 / 小数点错位 / 差一个整倍数 / 倒数
        if abs(abs(ua) - abs(sa)) < 1e-6:
            return "计算错误"                      # 只错符号
        if sa != 0 and abs(abs(ua / sa) - 1) < 0.15:
            return "计算错误"                      # 数值接近
        if sa != 0 and abs(abs(ua / sa) - 10 ** round(
                __import__("math").log10(abs(ua / sa)))) < 1e-6:
            return "计算错误"                      # 小数点错位

    if any(k in kp_text for k in _CALC_KP):
        return "计算错误"
    if any(k in kp_text for k in _METHOD_KP) or qtype in ("应用题", "解答题"):
        return "方法缺失"
    if any(k in kp_text for k in _CONCEPT_KP):
        return "概念不清"
    if any(k in kp_text for k in _READING_KP):
        return "审题偏差"
    if difficulty >= 4:
        return "方法缺失"
    return "概念不清"


# ---------------------------------------------------------------- 报告
def analyze_errors(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """输入答题记录，输出结构化错题报告。"""
    wrong = [r for r in records if r.get("correct") is False]
    total = len(records)
    correct = sum(1 for r in records if r.get("correct") is True)

    per_item: List[Dict[str, Any]] = []
    for i, r in enumerate(wrong, 1):
        etype = classify_error(r, str(r.get("user_answer", "")))
        per_item.append({
            "no": i,
            "question_id": r.get("question_id", ""),
            "stem": (r.get("stem", "") or "")[:120],
            "topic": r.get("topic", ""),
            "qtype": r.get("type", r.get("qtype", "")),
            "user_answer": r.get("user_answer", ""),
            "standard_answer": r.get("answer", ""),
            "error_type": etype,
            "analysis": r.get("analysis", ""),
            "knowledge_points": r.get("knowledge_points", []),
        })

    dist = Counter(x["error_type"] for x in per_item)
    kp_dist = Counter(kp for x in per_item for kp in (x["knowledge_points"] or []))
    topic_dist = Counter(x["topic"] for x in per_item)

    report: Dict[str, Any] = {
        "summary": {
            "total": total,
            "correct": correct,
            "wrong": len(wrong),
            "accuracy": round(correct / total * 100, 1) if total else 0.0,
        },
        "error_distribution": {k: dist.get(k, 0) for k in ERROR_TYPES},
        "error_ratio": {k: (round(dist.get(k, 0) / len(wrong) * 100, 1) if wrong else 0.0)
                        for k in ERROR_TYPES},
        "weak_knowledge_points": kp_dist.most_common(8),
        "weak_topics": topic_dist.most_common(5),
        "items": per_item,
        "suggestions": _build_suggestions(dist, kp_dist, topic_dist, len(wrong)),
        "recommendations": _build_recommendations(kp_dist, topic_dist),
    }
    report["narrative"] = _render_report(report)
    return report


def _build_suggestions(dist: Counter, kp_dist: Counter, topic_dist: Counter,
                       n_wrong: int) -> List[str]:
    if n_wrong == 0:
        return ["本次全部答对，保持！建议提升难度或换一个板块继续挑战。"]
    out: List[str] = []
    top = dist.most_common(1)[0][0] if dist else "概念不清"
    playbook = {
        "计算错误": "建议每天做 10 分钟限时计算训练，做题时在草稿纸上写出每一步，"
                    "并检查符号与小数点；完成后用 calculator 工具验算。",
        "概念不清": "回到教材把相关定义、性质抄写一遍，用自己的话复述；"
                    "配合 3-5 道基础概念题巩固。",
        "审题偏差": "读题时圈出关键条件（取值范围、单位、'不正确的是'等），"
                    "读完先复述题意再动笔。",
        "方法缺失": "先做 2 道同类型例题，总结解题模板（设元 → 列式 → 求解 → 检验），"
                    "再回到错题重做。",
    }
    out.append(f"本次错因以【{top}】为主。" + playbook[top])
    if kp_dist:
        kps = "、".join(k for k, _ in kp_dist.most_common(3))
        out.append(f"薄弱知识点集中在：{kps}，建议优先复习这几处。")
    if topic_dist:
        out.append("板块表现：" + "；".join(
            f"{t} 错 {c} 题" for t, c in topic_dist.most_common(3)))
    out.append("错题重做计划：当天订正 → 隔天重做 → 一周后再测一次。")
    return out


def _build_recommendations(kp_dist: Counter, topic_dist: Counter) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for topic, _ in topic_dist.most_common(2):
        recs.append({"topic": topic, "type": "选择题", "count": 5,
                     "reason": "该板块错题较多，先用基础题重建信心"})
    for kp, _ in kp_dist.most_common(3):
        recs.append({"knowledge_point": kp, "count": 3,
                     "reason": "针对薄弱知识点的专项练习"})
    if not recs:
        recs.append({"topic": "综合", "type": "混合", "count": 10,
                     "reason": "综合练习"})
    return recs


def _render_report(rep: Dict[str, Any]) -> str:
    s = rep["summary"]
    lines = [
        "📊 错题分析报告",
        "=" * 40,
        f"总题数 {s['total']}　答对 {s['correct']}　答错 {s['wrong']}　正确率 {s['accuracy']}%",
        "",
        "【错因分类】",
    ]
    ratio = rep["error_ratio"]
    for k in ERROR_TYPES:
        cnt = rep["error_distribution"].get(k, 0)
        if cnt:
            bar = "█" * max(1, int(ratio[k] / 10))
            lines.append(f"  {k}：{cnt} 题（{ratio[k]}%） {bar}")
    if rep["weak_topics"]:
        lines.append("")
        lines.append("【薄弱板块】" + "、".join(f"{t}({c})" for t, c in rep["weak_topics"]))
    if rep["weak_knowledge_points"]:
        lines.append("")
        lines.append("【薄弱知识点】" + "、".join(
            f"{k}({c})" for k, c in rep["weak_knowledge_points"]))
    if rep["items"]:                       # 全对时不打印空标题
        lines.append("")
        lines.append("【逐题分析】")
        for it in rep["items"]:
            lines.append(f"  {it['no']}. [{it['topic']}·{it['qtype']}] {it['stem']}")
            lines.append(f"     你的答案：{it['user_answer']}　正确答案：{it['standard_answer']}"
                         f"　错因：{it['error_type']}")
            if it["analysis"]:
                lines.append(f"     解析：{str(it['analysis'])[:160]}")
    if rep["suggestions"]:
        lines.append("")
        lines.append("【修改建议】")
        for sg in rep["suggestions"]:
            lines.append(f"  · {sg}")
    if rep["recommendations"]:
        lines.append("")
        lines.append("【针对性练习】")
        for r in rep["recommendations"]:
            if "topic" in r:
                lines.append(f"  · {r['topic']} {r.get('type', '')} × {r['count']} —— {r['reason']}")
            else:
                lines.append(f"  · 知识点 {r['knowledge_point']} × {r['count']} —— {r['reason']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 习惯抽取
def extract_habits(report: Dict[str, Any]) -> List[Dict[str, str]]:
    """从错题报告抽取写入长期记忆的结构化习惯（方案 3.4）。"""
    habits: List[Dict[str, str]] = []
    n = report["summary"]["wrong"]
    if n == 0:
        acc = report["summary"]["accuracy"]
        habits.append({"content": f"本次练习全部答对（正确率 {acc}%），当前难度偏低，可提升难度",
                       "type": "strength", "source": "error_report"})
        return habits

    for k, cnt in report["error_distribution"].items():
        if cnt and cnt / max(n, 1) >= 0.34:
            habits.append({
                "content": f"高频错因：{k}（本次 {cnt}/{n} 题），需要针对性训练",
                "type": "weak_point", "source": "error_report"})
    for kp, cnt in report["weak_knowledge_points"][:3]:
        if cnt >= 1:
            habits.append({
                "content": f"薄弱知识点：{kp}（错 {cnt} 题）",
                "type": "weak_point", "source": "error_report"})
    for topic, cnt in report["weak_topics"][:2]:
        habits.append({
            "content": f"板块「{topic}」表现较弱，本次错 {cnt} 题",
            "type": "habit", "source": "error_report"})
    acc = report["summary"]["accuracy"]
    if acc >= 85:
        habits.append({"content": f"本次正确率 {acc}%，整体掌握良好",
                       "type": "strength", "source": "error_report"})
    elif acc <= 50:
        habits.append({"content": f"本次正确率仅 {acc}%，建议降低难度先打基础",
                       "type": "habit", "source": "error_report"})
    return habits


def llm_polish_report(report: Dict[str, Any]) -> str:
    """LLM 可用时在结构化报告后追加一段老师建议；**结构化正文永不交由 LLM 覆写**。

    硬约束：报告标题、正确率、错因分布等由代码层产出的数据必须原样保留，
    LLM 只负责补充"怎么练"的建议——避免模型改写数字造成幻觉。
    """
    body = report["narrative"]
    llm = get_llm()
    if not llm.available:
        return body
    prompt = (
        "你是初中数学辅导老师。下面是某位同学的错题报告：\n"
        f"{body}\n\n"
        "请只针对报告中的薄弱点，补充 2-3 条具体、可执行的练习建议"
        "（120 字以内，不要重复报告里已有的数字，不要改写报告内容）。"
    )
    text = llm.chat(prompt)
    if not text:
        return body
    return f"{body}\n\n💡 老师建议\n{text}"


__all__ = ["analyze_errors", "classify_error", "extract_habits", "llm_polish_report",
           "ERROR_TYPES"]
