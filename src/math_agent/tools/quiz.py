"""刷题工具：fetch_quiz / check_answer（方案 3.1）。

判分为**纯代码判定**，不走 LLM——保证客观题 100% 准确、零 token 消耗。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..db.repository import fetch_questions
from ..topics import BANK_TOPICS
from .registry import register_tool

TOPIC_ENUM = BANK_TOPICS          # 单一真源：math_agent.topics
TYPE_ENUM = ["选择题", "填空题", "应用题", "混合"]
MAX_COUNT = 20                    # 单次出题上限（nodes.parse_quiz_request 同步用这个值）


class FetchQuizArgs(BaseModel):
    topic: str = Field(..., description=f"题目板块，枚举值之一：{TOPIC_ENUM}")
    type: str = Field("选择题", description=f"题型，枚举值之一：{TYPE_ENUM}")
    count: int = Field(5, ge=1, le=MAX_COUNT, description="题目数量")
    difficulty: Optional[int] = Field(None, ge=1, le=5, description="难度 1-5，不填则不限")
    exclude_ids: List[str] = Field(default_factory=list, description="需要排除的题目 ID")


class CheckAnswerArgs(BaseModel):
    question_id: str = Field(..., description="题目 ID")
    user_answer: str = Field(..., description="用户答案")


# ------------------------------------------------------------------ 归一化
_LATEX = re.compile(r"[\$\\\s　]+")
# 严格版：标号后必须跟分隔符，避免把 "a+b" 这类以变量开头的表达式误判成选项行
_OPTION_LEAD = re.compile(r"^\s*\(?([A-Da-d])\s*[\.、:：。．）\)]\s*")
# 宽松版：识别纯字母答案 "B" / "(c)" / "A "
_BARE_LETTER = re.compile(r"^\s*\(?([A-Da-d])\s*\)?\s*$")
_NUM = re.compile(r"^-?\d+(\.\d+)?$")
# 常见单位后缀（长度/面积/体积/质量/时间/角度/速度/人民币），判分前从数值答案两端剥掉
_UNIT = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?(?:/\d+)?)\s*"
    r"(?:(?:平方|立方)?(?:千米|米|分米|厘米|毫米)|km²|m²|dm²|cm²|mm²|km|m|dm|cm|mm|"
    r"毫升|升|L|mL|吨|千克|公斤|克|t|kg|g|"
    r"千米/时|米/秒|km/h|m/s|小时|时|分钟|分|秒|h|min|s|"
    r"元|角|分|度|°|％|%|个|人|只|本|棵|万元|万吨|万人|万)\s*$",
    re.IGNORECASE)
# 多答案分隔："1或-1"（任一即可） vs "1、2"（需答全，顺序无关）
_ALT_ANY = re.compile(r"\s*(?:或|或者|和|与)\s*")
_ALT_SET = re.compile(r"\s*(?:、|，|;|；|,)\s*")


def normalize(text: str) -> str:
    t = (text or "").strip()
    t = _LATEX.sub("", t)
    t = t.replace("．", ".").replace("，", ",").replace("（", "(").replace("）", ")")
    t = t.replace("＋", "+").replace("－", "-").replace("＝", "=")
    return t.lower()


def _sympy_equal(a: str, b: str) -> bool:
    try:
        import sympy
        ea = sympy.sympify(a)
        eb = sympy.sympify(b)
        return bool(sympy.simplify(ea - eb) == 0)
    except Exception:
        return False


def strip_option_lead(text: str) -> str:
    """去掉选项前缀："B." / "(C)" / "A、4+..." → "4+..."；非选项前缀则原样返回。"""
    t = normalize(text)
    m = _OPTION_LEAD.match(t)
    if m and _OPTION_LEAD.sub("", t):        # 去掉前缀后仍有内容 ⇒ 确实是选项行
        return _OPTION_LEAD.sub("", t)
    mc = _BARE_LETTER.match(t)
    return "" if mc else t                   # 纯字母标号本身没有内容


def _strip_unit(t: str) -> str:
    """"5cm" → "5"；"1/2 千米" → "1/2"；纯数字原样返回。仅对可解析出数值的串生效。"""
    m = _UNIT.match(t)
    return m.group(1) if m else t


def text_equal(a: str, b: str) -> bool:
    """纯文本/数值/代数等价判定（不含选项语义）。"""
    s, u = normalize(a), normalize(b)
    if not s or not u:
        return False
    if s == u:
        return True
    # 单位剥离：带单位的数值答案（5cm / 30°）与裸数值（5 / 30）互相等价
    ss, uu = _strip_unit(s), _strip_unit(u)
    if ss != s or uu != u:                   # 至少一侧剥到了单位
        if ss == uu:
            return True
    if _NUM.match(ss) and _NUM.match(uu):
        try:
            return abs(float(ss) - float(uu)) < 1e-6
        except ValueError:
            pass
    return _sympy_equal(ss, uu) or _sympy_equal(s, u)


def resolve_option(text: str, options: List[str]) -> Optional[int]:
    """把答案文本解析成选项下标（0-based）；无法识别返回 None。

    同时支持三种写法：字母 "B"、带标号 "B." / "(C)"、直接写选项内容。
    """
    if not options:
        return None
    t = normalize(text)
    if not t:
        return None
    mc = _BARE_LETTER.match(t)               # 纯字母：B / (c) / "A "
    if mc:
        idx = "abcd".find(mc.group(1).lower())
        return idx if 0 <= idx < len(options) else None
    body = strip_option_lead(t)              # 带标号：去掉 "A." 后按内容匹配
    for i, opt in enumerate(options):
        if text_equal(strip_option_lead(opt), body):
            return i
    return None


# 用户答案侧统一按任意分隔符拆（"1和2"、"2,1"、"1或-1" 都拆成集合）
_ALT_USER = re.compile(r"\s*(?:或|或者|和|与|、|，|;|；|,)\s*")


def _multi_equal(std: str, user: str) -> Optional[bool]:
    """多答案匹配（仅当标准答案含分隔符时生效）。

    * "或/和/与" 连接：任选其一即可，答全（顺序无关）也算对 → "1或-1" 接受 "1"/"-1"/"-1和1"
    * "、/，/,/;" 连接：需要答全且数量一致 → "1、2" 只答 "1" 判不完整（错误）
    标准答案不含分隔符时返回 None（走单值比较）。
    """
    if _ALT_ANY.search(std):
        alts = [p for p in _ALT_ANY.split(std) if p]
        got = [p for p in _ALT_USER.split(user) if p]
        if len(got) == len(alts):
            return all(any(text_equal(a, g) for a in alts) for g in got)
        return any(text_equal(a, user) for a in alts)
    if _ALT_SET.search(std):
        want = [p for p in _ALT_SET.split(std) if p]
        got = [p for p in _ALT_USER.split(user) if p]
        return len(got) == len(want) and all(
            any(text_equal(w, g) for w in want) for g in got)
    return None


def answers_equal(std: str, user: str, options: Optional[List[str]] = None) -> bool:
    """标准答案 vs 用户答案。

    选择题（传 options）时**先统一映射到选项下标再比较**——
    避免 "答案是A / 用户答B" 被字符串包含关系误判为正确。
    """
    if not (user or "").strip():
        return False
    if options:
        si = resolve_option(std, options)
        ui = resolve_option(user, options)
        if si is not None and ui is not None:
            return si == ui
        if si is not None:                   # 标准答案是选项，用户写的是内容
            return text_equal(strip_option_lead(options[si]), user)
        if ui is not None:                   # 用户答选项字母，标准答案是内容
            return text_equal(strip_option_lead(options[ui]), std)
    multi = _multi_equal(normalize(std), normalize(user))
    if multi is not None:
        return multi
    return text_equal(std, user)


# ------------------------------------------------------------------ 工具
@register_tool(
    "fetch_quiz",
    "按板块与题型从题库拉取题目。返回题目列表（含 question_id / 题干 / 选项）。"
    "严禁使用相同参数重复调用；如需更多题目请更改 count 或 difficulty。",
    FetchQuizArgs,
)
def fetch_quiz(topic: str, type: str = "选择题", count: int = 5,
               difficulty: Optional[int] = None,
               exclude_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    if topic not in TOPIC_ENUM:
        raise ValueError(f"topic 必须是 {TOPIC_ENUM} 之一，收到：{topic}")
    if type not in TYPE_ENUM:
        raise ValueError(f"type 必须是 {TYPE_ENUM} 之一，收到：{type}")
    qs = fetch_questions(topic=topic, qtype=type, count=count,
                         difficulty=difficulty, exclude_ids=exclude_ids or [])
    if not qs:
        return {"ok": False, "message": f"题库中没有符合条件的题目（topic={topic}, type={type}, "
                                        f"difficulty={difficulty}）。请放宽条件重试。",
                "questions": []}
    return {"ok": True, "count": len(qs), "questions": qs}


@register_tool(
    "check_answer",
    "客观题判分：传入 question_id 与用户答案，返回 correct / 标准答案 / 解析。"
    "本工具只判对错，不做错因分析（分析报告由交卷后统一生成）。",
    CheckAnswerArgs,
)
def check_answer(question_id: str, user_answer: str) -> Dict[str, Any]:
    from ..db.models import Question, get_session
    from sqlalchemy import select

    with get_session() as s:
        q = s.execute(select(Question).where(
            Question.question_id == question_id)).scalar_one_or_none()
        if q is None:
            raise ValueError(f"题目不存在：{question_id}")
        data = q.to_dict()

    correct = answers_equal(data["answer"], user_answer,
                            options=data.get("options") or None)
    return {
        "question_id": question_id,
        "correct": correct,
        "user_answer": user_answer,
        "standard_answer": data["answer"],
        "analysis": data["analysis"],
        "topic": data["topic"],
        "qtype": data["type"],
        "knowledge_points": data["knowledge_points"],
    }


__all__ = ["fetch_quiz", "check_answer", "answers_equal", "text_equal",
           "resolve_option", "strip_option_lead", "normalize",
           "TOPIC_ENUM", "TYPE_ENUM"]
