"""LangGraph 图构建 + Agent 运行时（方案 2.2 流转图）。"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from .. import safety, trace as trace_mod

from ..state import MathAgentState
from .knowledge_tutor import knowledge_tutor_node
from .nodes import (error_analyzer_node, grading_node, habit_node,
                    intent_router_node, memory_manager_node, quiz_manager_node)
from .react import react_step_node, should_continue_react

log = logging.getLogger(__name__)


def route_by_intent(state: MathAgentState) -> str:
    intent = state.get("intent", "chat")
    return {"quiz": "quiz_manager", "submit": "error_analyzer",
            "grade": "grading", "habit": "habit",
            "learn": "knowledge_tutor"}.get(intent, "react")


def after_grading(state: MathAgentState) -> str:
    if state.get("grade_result"):
        return "error_analyzer"
    return "finalize"


def after_analyzer(state: MathAgentState) -> str:
    if state.get("error_report"):
        return "memory_manager"
    return "finalize"


def build_graph(checkpointer=None):
    sg = StateGraph(MathAgentState)

    sg.add_node("intent_router", intent_router_node)
    sg.add_node("quiz_manager", quiz_manager_node)
    sg.add_node("grading", grading_node)
    sg.add_node("error_analyzer", error_analyzer_node)
    sg.add_node("memory_manager", memory_manager_node)
    sg.add_node("habit", habit_node)
    sg.add_node("knowledge_tutor", knowledge_tutor_node)
    sg.add_node("react", react_step_node)
    sg.add_node("finalize", lambda s: {})

    sg.set_entry_point("intent_router")
    sg.add_conditional_edges("intent_router", route_by_intent,
                             {"quiz_manager": "quiz_manager",
                              "error_analyzer": "error_analyzer",
                              "grading": "grading",
                              "habit": "habit",
                              "knowledge_tutor": "knowledge_tutor",
                              "react": "react"})
    sg.add_edge("quiz_manager", "finalize")
    sg.add_conditional_edges("grading", after_grading,
                             {"error_analyzer": "error_analyzer", "finalize": "finalize"})
    sg.add_conditional_edges("error_analyzer", after_analyzer,
                             {"memory_manager": "memory_manager", "finalize": "finalize"})
    sg.add_edge("memory_manager", "finalize")
    sg.add_edge("habit", "finalize")
    sg.add_edge("knowledge_tutor", "finalize")
    sg.add_conditional_edges("react", should_continue_react,
                             {"react": "react", "finalize": "finalize"})
    sg.add_edge("finalize", END)

    return sg.compile(checkpointer=checkpointer) if checkpointer else sg.compile()


# --------------------------------------------------------------------------
def _make_checkpointer():
    try:  # langgraph >= 0.2.20
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver()
    except Exception:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()
        except Exception:
            return None


class MathAgent:
    """面向调用方的门面：一次实例化 = 一个用户的长期会话。

    内置分层对话记忆（memory/compressor.py）：
    每轮 invoke 前 → ``prune_state`` 释放无关 state 内容；
    压缩上下文 ``conversation_digest``（实体锚 + 历史要点 + 近期原文）
    注入 state 供 ReAct prompt 使用；轮次结束后回写压缩存储。
    """

    def __init__(self, user_id: str = "u_default", use_checkpointer: bool = True):
        self.user_id = user_id
        self.graph = build_graph(_make_checkpointer() if use_checkpointer else None)
        self.config: Dict[str, Any] = {"configurable": {"thread_id": f"math_{user_id}"}}
        self.last_state: Dict[str, Any] = {}
        self._mode = ""                  # '' 默认（刷题/答疑） | 'learn' 学习知识点
        self._lock = threading.RLock()   # 同用户并发串行化：checkpointer/last_state 不可重入
        from ..memory.compressor import ConversationCompressor
        self.compressor = ConversationCompressor(user_id)
        # ② 会话状态持久化：进程重启后恢复进行中的刷题/学习会话（不再"重启即失忆"）
        self.last_state = self._restore_state() if use_checkpointer else {}

    _RESET: Dict[str, Any] = {
        "pending_answer": None, "submit_event": False, "awaiting_user": False,
        "react_loop_count": 0, "react_thoughts": [], "last_result_vector": None,
        "loop_stop_reason": None, "trace": [], "response": None,
        "error_report": None, "final_report": None, "tool_blocked_notice": None,
        "_records": None, "ocr_payload": None, "grade_result": None,
        "task_id": "", "tool_replan_count": 0, "react_forced_action": None,
    }

    # 允许持久化的键（大对象/临时产物不入库，避免无限膨胀）
    _PERSIST_KEYS = ("quiz_session", "learning_session", "session_mode",
                     "short_term_memory", "grade_result", "ocr_payload")

    def _restore_state(self) -> Dict[str, Any]:
        """启动时从 SQLite 恢复跨轮状态；失败降级为空（不影响可用性）。"""
        try:
            from ..db.repository import load_agent_state

            data = load_agent_state(self.user_id)
        except Exception as e:
            log.warning("恢复会话状态失败 user=%s: %s", self.user_id, e)
            return {}
        if not data:
            return {}
        self._mode = data.get("session_mode", "") or ""
        log.info("已恢复会话状态 user=%s keys=%s", self.user_id, list(data)[:6])
        return data

    def _persist_state(self, out: Dict[str, Any]) -> None:
        """每轮结束后落盘跨轮状态（失败只记日志，不阻断主流程）。"""
        snap: Dict[str, Any] = {k: out[k] for k in self._PERSIST_KEYS if k in out}
        snap["session_mode"] = getattr(self, "_mode", "") or out.get("session_mode", "")
        if not snap:
            return
        try:
            from ..db.repository import save_agent_state

            if not save_agent_state(self.user_id, snap):
                log.warning("会话状态落盘失败 user=%s", self.user_id)
        except Exception as e:
            log.warning("会话状态落盘异常 user=%s: %s", self.user_id, e)

    def invoke(self, text: str, **extra: Any) -> Dict[str, Any]:
        with self._lock:                 # P1：同用户并发下丢失更新防护
            return self._invoke_locked(text, **extra)

    def _invoke_locked(self, text: str, **extra: Any) -> Dict[str, Any]:
        from ..memory.compressor import active_task_of, prune_state

        t0 = time.time()
        # ⑤ 内容审核（硬约束）：违规输入不进入主流程，直接返回引导话术
        verdict = safety.check_input(text)
        if not verdict.get("allowed"):
            blocked_out: Dict[str, Any] = {
                "response": verdict.get("reply", ""), "intent": "blocked",
                "intent_confidence": 1.0, "blocked_category": verdict.get("category", ""),
                "session_mode": getattr(self, "_mode", ""), "trace": ["safety:blocked"],
                "tool_call_history": [], "user_id": self.user_id,
            }
            self.last_state = blocked_out
            trace_mod.record_turn(self.user_id, text, blocked_out,
                                  int((time.time() - t0) * 1000))
            return blocked_out

        payload: Dict[str, Any] = {"user_input": text, "user_id": self.user_id}
        payload.update(self._RESET)
        # 跨轮保留的键：从上一轮持久化状态取出 → 裁剪（释放无关内容）→ 回注。
        # 不回注的话 checkpointer 里的旧值会原样存活，裁剪等于没做。
        prev = self.last_state or {}
        carried = {k: prev[k] for k in ("tool_call_history", "short_term_memory",
                                        "quiz_session", "learning_session",
                                        "ocr_payload", "grade_result") if k in prev}
        payload.update(prune_state(carried))
        payload.update(extra)
        # 会话模式（'' / 'learn'）跨轮保持：本轮显式传入 > 门面设置 > 上一轮状态
        payload.setdefault("session_mode",
                           getattr(self, "_mode", "") or prev.get("session_mode", ""))
        payload["conversation_digest"] = self.compressor.build_context(
            active_task=active_task_of(payload))             # 压缩上下文注入
        try:
            out = self.graph.invoke(payload, config=self.config)
        except Exception as e:                                # ① 异常也留痕
            trace_mod.record_turn(self.user_id, text, {"trace": ["graph:error"]},
                                  int((time.time() - t0) * 1000), error=str(e))
            raise
        # ⑤ 输出审核：脱敏 + 移除索取未成年人隐私的话术
        resp = safety.sanitize_output(out.get("response") or "")
        out["response"] = resp
        # 轮次结束 → 回写分层记忆（user + assistant 各一轮，均已脱敏）
        self.compressor.add_turn("user", safety.mask_pii(text), intent=out.get("intent", ""))
        if resp:
            self.compressor.add_turn("assistant", resp, intent=out.get("intent", ""))
        trace = list(out.get("trace") or [])
        trace.append(f"memory:hist={len(out.get('tool_call_history') or [])}"
                     f" digest={len(payload['conversation_digest'])}ch")
        out["trace"] = trace
        self.last_state = out
        self._persist_state(out)          # ② 落盘：进程重启后可恢复
        # ① 调用链追踪
        trace_mod.record_turn(self.user_id, text, out, int((time.time() - t0) * 1000))
        return out

    def chat(self, text: str, **extra: Any) -> str:
        out = self.invoke(text, **extra)
        return out.get("response") or "(无回复)"

    def set_mode(self, mode: str) -> None:
        """切换会话模式：''（默认/刷题答疑）或 'learn'（学习知识点）。"""
        with self._lock:
            self._mode = mode or ""
            self.last_state = dict(self.last_state or {})
            self.last_state["session_mode"] = self._mode

    @property
    def mode(self) -> str:
        return getattr(self, "_mode", "")

    def learn(self, topic: str = "") -> str:
        """进入学习模式：不给板块则先出知识点目录。"""
        self.set_mode("learn")
        return self.chat(f"学知识点 {topic}".strip())

    def reset(self) -> None:
        """真重置：换新 checkpointer 丢弃跨轮状态（旧 quiz_session 不再存活）+ 清压缩任务。"""
        with self._lock:
            self.graph = build_graph(_make_checkpointer())
            self.last_state = {}
            # 持久化快照也要清掉，否则重启后又把旧会话读回来
            try:
                from ..db.repository import clear_agent_state

                clear_agent_state(self.user_id)
            except Exception as e:
                log.warning("清理持久化状态失败 user=%s: %s", self.user_id, e)
            reset_task = getattr(self.compressor, "reset_task", None)
            if callable(reset_task):
                reset_task()


__all__ = ["build_graph", "MathAgent", "route_by_intent"]
