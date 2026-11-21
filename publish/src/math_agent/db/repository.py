"""题库 / 答题记录仓储层。"""
from __future__ import annotations

import datetime as dt
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select

from ..topics import BANK_CORE_TOPICS          # 板块命名单一真源
from .models import (AgentState, Attempt, Question, QuizSession, get_session,
                     init_db)

log = logging.getLogger(__name__)

TOPICS = list(BANK_CORE_TOPICS)               # 独立出题板块（不含兜底"综合"）
Q_TYPES = ["选择题", "填空题", "应用题"]


def reset_questions() -> int:
    """清空题库（重建 ETL 前调用，避免脏数据残留）。"""
    from sqlalchemy import delete

    init_db()
    with get_session() as s:
        n = s.execute(delete(Question)).rowcount or 0
        s.commit()
    return int(n)


def upsert_questions(questions: List[Dict[str, Any]]) -> int:
    """批量写入题库（按 question_id 去重更新）。"""
    init_db()
    n = 0
    with get_session() as s:
        for q in questions:
            obj = s.execute(select(Question).where(
                Question.question_id == q["question_id"])).scalar_one_or_none()
            if obj is None:
                obj = Question(question_id=q["question_id"])
                s.add(obj)
            obj.topic = q["topic"]
            obj.type = q["type"]
            obj.difficulty = int(q.get("difficulty", 3))
            obj.stem = q["stem"]
            obj.options = q.get("options") or []
            obj.answer = q["answer"]
            obj.analysis = q.get("analysis", "")
            obj.knowledge_points = q.get("knowledge_points") or []
            obj.source = q.get("source", "")
            n += 1
        s.commit()
    return n


def fetch_questions(topic: str, qtype: str, count: int = 5,
                    difficulty: Optional[int] = None,
                    exclude_ids: Optional[List[str]] = None,
                    seed: Optional[int] = None) -> List[Dict[str, Any]]:
    init_db()
    exclude_ids = set(exclude_ids or [])
    with get_session() as s:
        stmt = select(Question)
        if topic and topic != "综合":
            stmt = stmt.where(Question.topic == topic)
        if qtype and qtype != "混合":
            stmt = stmt.where(Question.type == qtype)
        if difficulty:
            stmt = stmt.where(Question.difficulty == int(difficulty))
        rows = [q.to_dict() for q in s.scalars(stmt).all()
                if q.question_id not in exclude_ids]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:max(1, count)]


def topic_stats() -> Dict[str, Dict[str, int]]:
    init_db()
    with get_session() as s:
        rows = s.execute(select(Question.topic, Question.type,
                                func.count(Question.id)).group_by(
            Question.topic, Question.type)).all()
    out: Dict[str, Dict[str, int]] = {}
    for topic, qtype, cnt in rows:
        out.setdefault(topic, {})[qtype] = cnt
    return out


# ------------------------------------------------------------------ 会话状态持久化
# InMemorySaver 只在进程内有效，重启后正在进行的刷题/学习会话全部丢失。
# 这里把跨轮状态的白名单键落盘，进程启动时恢复（轻量持久化，不引入新依赖）。
_STATE_MAX_CHARS = 200_000          # 单用户状态上限，防止无限膨胀


def save_agent_state(user_id: str, payload: Dict[str, Any]) -> bool:
    """落盘跨轮会话状态。返回是否成功（失败不抛，由调用方记日志）。"""
    try:
        blob = json.dumps(payload or {}, ensure_ascii=False, default=str)
    except Exception as e:
        log.warning("save_agent_state 序列化失败 user=%s: %s", user_id, e)
        return False
    if len(blob) > _STATE_MAX_CHARS:
        log.warning("save_agent_state 状态过大跳过 user=%s chars=%d",
                    user_id, len(blob))
        return False
    init_db()
    try:
        with get_session() as s:
            obj = s.get(AgentState, user_id)
            if obj is None:
                s.add(AgentState(user_id=user_id, payload=blob))
            else:
                obj.payload = blob
                obj.updated_at = dt.datetime.now()
            s.commit()
        return True
    except Exception as e:
        log.error("save_agent_state 写入失败 user=%s: %s", user_id, e)
        return False


def load_agent_state(user_id: str) -> Dict[str, Any]:
    """读取跨轮会话状态；无记录或解析失败返回 {}。"""
    try:
        init_db()
        with get_session() as s:
            obj = s.get(AgentState, user_id)
            blob = obj.payload if obj else ""
        if not blob:
            return {}
        data = json.loads(blob)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log.warning("load_agent_state 读取失败 user=%s: %s", user_id, e)
        return {}


def clear_agent_state(user_id: str) -> None:
    """清空某用户的跨轮状态（如用户主动重置）。"""
    try:
        init_db()
        with get_session() as s:
            obj = s.get(AgentState, user_id)
            if obj is not None:
                s.delete(obj)
                s.commit()
    except Exception as e:
        log.warning("clear_agent_state 失败 user=%s: %s", user_id, e)


# ------------------------------------------------------------------ 记录
def create_session(session_id: str, user_id: str, topic: str, qtype: str) -> None:
    init_db()
    with get_session() as s:
        if s.execute(select(QuizSession).where(
                QuizSession.session_id == session_id)).scalar_one_or_none() is None:
            s.add(QuizSession(session_id=session_id, user_id=user_id,
                              topic=topic, qtype=qtype))
            s.commit()


def record_attempt(session_id: str, user_id: str, q: Dict[str, Any],
                   user_answer: str, correct: bool, error_type: str = "") -> None:
    init_db()
    last_err: Exception | None = None
    for attempt in range(3):                     # P1：写锁冲突重试，失败必须留痕
        try:
            with get_session() as s:
                s.add(Attempt(session_id=session_id, user_id=user_id,
                              question_id=q.get("question_id", ""),
                              topic=q.get("topic", ""), qtype=q.get("type", ""),
                              user_answer=user_answer, correct=bool(correct),
                              error_type=error_type))
                sess = s.execute(select(QuizSession).where(
                    QuizSession.session_id == session_id)).scalar_one_or_none()
                if sess:
                    sess.total = (sess.total or 0) + 1
                    sess.correct = (sess.correct or 0) + (1 if correct else 0)
                s.commit()
                return
        except Exception as e:                   # OperationalError: locked 等
            last_err = e
            time.sleep(0.2 * (attempt + 1))
    log.error("record_attempt 写入失败（session=%s user=%s question=%s）: %s",
              session_id, user_id, q.get("question_id", ""), last_err)
    raise RuntimeError(f"答题记录写入失败：{last_err}") from last_err


def finish_session(session_id: str) -> None:
    init_db()
    with get_session() as s:
        sess = s.execute(select(QuizSession).where(
            QuizSession.session_id == session_id)).scalar_one_or_none()
        if sess:
            sess.finished_at = dt.datetime.now()
            s.commit()


def reset_user_data(user_id: str) -> Dict[str, int]:
    """清空某用户的全部学习数据：答题记录 / 刷题会话 / 落盘状态。
    供前端「重置我的学习数据」调用（隐私说明中承诺的能力）。"""
    from sqlalchemy import delete

    init_db()
    counts: Dict[str, int] = {}
    with get_session() as s:
        counts["attempts"] = int(s.execute(
            delete(Attempt).where(Attempt.user_id == user_id)).rowcount or 0)
        counts["sessions"] = int(s.execute(
            delete(QuizSession).where(QuizSession.user_id == user_id)).rowcount or 0)
        counts["states"] = int(s.execute(
            delete(AgentState).where(AgentState.user_id == user_id)).rowcount or 0)
        s.commit()
    log.info("reset_user_data user=%s %s", user_id, counts)
    return counts


def user_history(user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    init_db()
    with get_session() as s:
        rows = s.execute(select(Attempt).where(
            Attempt.user_id == user_id).order_by(
            Attempt.created_at.desc()).limit(limit)).scalars().all()
    return [{"question_id": r.question_id, "topic": r.topic, "qtype": r.qtype,
             "user_answer": r.user_answer, "correct": r.correct,
             "error_type": r.error_type,
             "created_at": r.created_at.isoformat() if r.created_at else ""}
            for r in rows]


__all__ = ["upsert_questions", "reset_questions", "fetch_questions", "topic_stats", "create_session",
           "record_attempt", "finish_session", "user_history", "reset_user_data",
           "TOPICS", "Q_TYPES"]
