"""Intent Router（方案 2.2）：关键词规则优先 + LLM 兜底。

规则优先的原因：意图路由误判是方案风险表中的中风险项，
纯 LLM 路由在"选B""3"这类短输入上极易漂移，因此把确定性最高的部分
（交卷、作答、批改、习惯）交给规则，开放域才交给 LLM。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from .. import metrics
from ..config import cfg
from ..llm import get_llm

INTENTS = ["quiz", "submit", "grade", "habit", "learn", "chat"]

# ------------------------------------------------------------------ 规则
_P_SUBMIT = re.compile(r"(交卷|结束刷题|结束练习|不做了|生成报告|出报告|看看结果|总结一下|提交)")
# 学习知识点（学习模式 Agent）：与"答题时的概念追问"区分开——
# 这里是"系统性学习/开课"，所以关键词用 学/教/课程/知识点 等，不含"讲一下/什么是"
_P_LEARN = re.compile(r"(学知识点|学一下|学习一下|我要学|我想学|教教我|教我|开课|上课|"
                      r"知识点讲解|讲解知识点|系统学习|课程|学习模式|学学)")
_P_QUIZ = re.compile(r"(刷题|练习|出题|来\s*[几\d]+\s*道|做\s*[几\d]+\s*道|开始做|我要做|"
                     r"来一套|专项|测验|我要刷|我想刷|刷函数|刷方程|刷几何|刷概率|刷统计|"
                     r"刷三角|统计题|概率题|三角函数题|再刷|继续刷|"
                     r"[几\d]+\s*道)")
_P_GRADE = re.compile(r"(批改|改作业|作业|试卷|成绩单|上传|阅卷|判\s*(?:一\s*下|下)?\s*分)")
_P_HABIT = re.compile(r"(习惯|薄弱|弱点|我的问题|进步|表现|分析我|学情|优缺点)")
_P_OPTION = re.compile(r"^\s*\(?([A-Da-d])\)?[\.、:：\s]*$")
_P_PURE_NUM = re.compile(r"^\s*-?\d+(\.\d+)?(/\d+)?\s*$")
# 概念咨询：即便处于刷题会话也应走答疑，否则学生问"什么是斜率"会被判成答错。
# 注意不要包含"解析"/"提示"——那是 Quiz Manager 的单题反馈指令。
_P_CONCEPT = re.compile(r"(什么是|什么叫|什么叫作|怎么求|如何求|怎么算|如何算|"
                        r"为什么|讲解|讲一下|解释一下|我不懂|看不懂|举个例子)")


def _rule_route(text: str, in_quiz: bool, mode: str = "",
                in_learn: bool = False) -> Tuple[str, float]:
    t = (text or "").strip()
    if not t:
        return "chat", 0.3
    # 学习模式（Web 端切换 / 已在学习会话内）：除明确切换业务外一律走讲解 Agent。
    # 例外：in_quiz（刷题会话进行中）时不能吞掉短输入作答——
    # 学习页签下发"我要刷…"切到刷题后，答案 A/12 仍须路由回 quiz 判分。
    if (mode == "learn" or in_learn) and not in_quiz:
        if _P_CONCEPT.search(t) and not _P_LEARN.search(t):
            return "chat", 0.9                   # 学习中问"什么是XX"仍走答疑
        if not (_P_QUIZ.search(t) or _P_GRADE.search(t) or _P_HABIT.search(t)
                or _P_SUBMIT.search(t)):
            return "learn", 0.9
    if _P_CONCEPT.search(t):
        return "chat", 0.9                       # 概念咨询优先于作答判定
    if in_quiz and (_P_OPTION.match(t) or _P_PURE_NUM.match(t)):
        return "quiz", 0.95                      # 刷题进行中 → 短输入即作答
    if _P_SUBMIT.search(t):
        return "submit", 0.95
    if _P_GRADE.search(t):
        return "grade", 0.85
    if _P_HABIT.search(t):
        return "habit", 0.85
    if _P_LEARN.search(t):
        return "learn", 0.9                      # 学习知识点（系统学习）
    if _P_QUIZ.search(t):
        return "quiz", 0.9
    if in_quiz:
        return "quiz", 0.7                       # 会话未结束，默认继续作答
    return "", 0.0                               # 交给 LLM / 默认闲聊


_LLM_ROUTER_PROMPT = """你是意图分类器。按规则判断用户输入属于哪一类，只输出 JSON。
类别：
- quiz：开始或继续刷题（选择题/填空题/应用题），或正在作答
- submit：交卷、结束刷题、要求生成错题报告
- grade：上传/批改作业、试卷、成绩单
- habit：查询自己的学习习惯、薄弱点、进步情况
- learn：系统学习/讲解某个知识点（开课、教我XX、学习知识点）
- chat：其它（概念咨询、闲聊、解题答疑）

用户输入：{text}
当前是否处于刷题会话中：{in_quiz}

只输出：{{"intent": "...", "confidence": 0.xx}}"""


def route_intent(text: str, in_quiz: bool = False, use_llm: bool = True,
                 mode: str = "", in_learn: bool = False) -> Tuple[str, float]:
    intent, conf = _rule_route(text, in_quiz, mode=mode, in_learn=in_learn)
    if intent:
        metrics.record_intent(intent)
        return intent, conf
    if use_llm:
        llm = get_llm()
        if llm.available:
            data = llm.chat_json(_LLM_ROUTER_PROMPT.format(text=text, in_quiz=in_quiz),
                                 fallback=None)
            if isinstance(data, dict) and data.get("intent") in INTENTS:
                try:
                    return data["intent"], float(data.get("confidence", 0.6))
                except (TypeError, ValueError):
                    return data["intent"], 0.6
    metrics.record_intent("chat")
    return "chat", 0.4


__all__ = ["route_intent", "INTENTS"]
