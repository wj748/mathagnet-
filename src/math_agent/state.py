"""LangGraph 全局状态定义（对应方案 2.1 节）。

补充说明（实现时的必要扩展）
----------------------------
* `pending_answer`  —— 用户本轮作答，由 Quiz Manager 消费。
* `submit_event`    —— 代码层硬约束的"交卷"事件标记，Error Analyzer 只认它，
                       不依赖 LLM 自觉（方案 7.1 风险表最后一条）。
* `react_thoughts`  —— 每轮 Thought 文本，供信息增益判定回溯。
* `final_report` / `awaiting_user` —— 输出与多轮挂起控制。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage


class QuizQuestion(TypedDict, total=False):
    question_id: str
    topic: str
    type: str                # 选择题 / 填空题 / 应用题
    difficulty: int          # 1-5
    stem: str
    options: List[str]
    answer: str
    analysis: str
    knowledge_points: List[str]


class QuizRecord(TypedDict, total=False):
    question_id: str
    topic: str
    type: str
    stem: str
    options: List[str]
    user_answer: str
    correct: bool
    answer: str
    analysis: str
    knowledge_points: List[str]


class MathAgentState(TypedDict, total=False):
    # ---------------- 对话 ----------------
    messages: List[BaseMessage]
    user_input: str
    user_id: str

    # ---------------- 意图 ----------------
    intent: str                     # quiz | grade | habit | chat | submit
    intent_confidence: float

    # ---------------- 刷题会话 ----------------
    quiz_session: Dict[str, Any]    # {questions, index, records, topic, type, ...}
    pending_answer: Optional[str]   # 用户本轮作答
    submit_event: bool              # 交卷事件（代码层硬约束）
    awaiting_user: bool             # 是否等待用户下一轮输入

    # ---------------- 批改 ----------------
    ocr_payload: Optional[Dict[str, Any]]
    grade_result: Optional[Dict[str, Any]]

    # ---------------- 学习（知识点讲解 Agent） ----------------
    session_mode: str                   # ""（默认） | "learn"（Web 端切换为学习模式）
    learning_session: Dict[str, Any]    # {topic, index, stage, pending_quiz, correct, total}

    # ---------------- 记忆 ----------------
    short_term_memory: Dict[str, Any]
    long_term_memory_hits: List[Dict[str, Any]]
    memory_written: bool

    # ---------------- 报告 ----------------
    error_report: Optional[Dict[str, Any]]
    final_report: Optional[str]
    _records: Optional[List[Dict[str, Any]]]   # 批改流程产出的待分析记录

    # ---------------- ReAct 控制 ----------------
    react_loop_count: int
    react_thoughts: List[str]
    last_result_vector: Optional[List[float]]
    loop_stop_reason: Optional[str]  # max_loop | information_gain | finished | fatal_error | degraded
    task_id: str                          # 本次任务链标识（toolguard 台账/追踪）
    tool_replan_count: int                # 已重规划轮数（toolguard：上限 max_replans）
    react_forced_action: Optional[List]   # 重规划强制动作 [备选工具名, args]（用后即清）

    # ---------------- 工具 ----------------
    tool_call_history: List[Dict[str, Any]]
    tool_blocked_notice: Optional[str]

    # ---------------- 其它 ----------------
    response: Optional[str]
    trace: List[str]


def new_state(user_id: str = "u_default", user_input: str = "") -> MathAgentState:
    """创建一个干净状态，避免节点里到处 .get(...) 判空。"""
    return MathAgentState(
        messages=[],
        user_input=user_input,
        user_id=user_id,
        intent="",
        intent_confidence=0.0,
        quiz_session={},
        pending_answer=None,
        submit_event=False,
        awaiting_user=False,
        ocr_payload=None,
        grade_result=None,
        session_mode="",
        learning_session={},
        short_term_memory={},
        long_term_memory_hits=[],
        memory_written=False,
        error_report=None,
        final_report=None,
        react_loop_count=0,
        react_thoughts=[],
        last_result_vector=None,
        loop_stop_reason=None,
        tool_call_history=[],
        tool_blocked_notice=None,
        response=None,
        trace=[],
    )


__all__ = ["MathAgentState", "QuizQuestion", "QuizRecord", "new_state"]
