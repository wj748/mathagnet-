"""业务库表结构（SQLAlchemy 2.0）。默认 SQLite，生产可一键切 MySQL。"""
from __future__ import annotations

import datetime as dt
from typing import Any, List

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, Integer,
                        String, Text, create_engine)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..config import cfg, resolve_path


class Base(DeclarativeBase):
    pass


class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question_id = Column(String(64), unique=True, index=True, nullable=False)
    topic = Column(String(32), index=True, nullable=False)          # 函数/二元一次方程/几何图像
    type = Column(String(16), index=True, nullable=False)           # 选择题/填空题/应用题
    difficulty = Column(Integer, default=3, index=True)             # 1-5
    stem = Column(Text, nullable=False)
    options = Column(JSON, default=list)
    answer = Column(Text, nullable=False)
    analysis = Column(Text, default="")
    knowledge_points = Column(JSON, default=list)
    source = Column(String(64), default="")

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "topic": self.topic,
            "type": self.type,
            "difficulty": self.difficulty,
            "stem": self.stem,
            "options": self.options or [],
            "answer": self.answer,
            "analysis": self.analysis or "",
            "knowledge_points": self.knowledge_points or [],
            "source": self.source or "",
        }


class QuizSession(Base):
    __tablename__ = "quiz_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    user_id = Column(String(64), index=True, nullable=False)
    topic = Column(String(32), default="")
    qtype = Column(String(16), default="")
    started_at = Column(DateTime, default=dt.datetime.now)
    finished_at = Column(DateTime, nullable=True)
    total = Column(Integer, default=0)
    correct = Column(Integer, default=0)


class TraceTurn(Base):
    """调用链记录持久化。

    内存环形缓冲（trace.py）进程重启即归零，无法追溯历史问题；
    这里补一份落盘，启动时把最近的记录读回内存供 /api/traces 展示。
    """

    __tablename__ = "trace_turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(Float, index=True)
    user_id = Column(String(64), index=True)
    intent = Column(String(32), default="", index=True)
    elapsed_ms = Column(Integer, default=0)
    error = Column(Text, default="")
    payload = Column(Text, default="")            # JSON：完整 TurnTrace


class AgentState(Base):
    """跨轮 Agent 会话状态快照（修复「进程重启即失忆」）。

    langgraph 的 InMemorySaver 只在进程内有效，重启后正在进行的刷题/学习会话
    全部丢失。这里把**白名单键**序列化落盘，启动时恢复，实现轻量持久化。
    只存会话必需字段，不存大对象（tool_call_history 会被裁剪）。
    """

    __tablename__ = "agent_states"

    user_id = Column(String(64), primary_key=True)
    payload = Column(Text, default="")            # JSON：跨轮状态白名单
    updated_at = Column(DateTime, default=dt.datetime.now,
                        onupdate=dt.datetime.now)


class Attempt(Base):
    __tablename__ = "attempts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    user_id = Column(String(64), index=True)
    question_id = Column(String(64), index=True)
    topic = Column(String(32), index=True)
    qtype = Column(String(16))
    user_answer = Column(Text, default="")
    correct = Column(Boolean, default=False)
    error_type = Column(String(32), default="")       # 概念不清/计算错误/审题偏差/方法缺失
    created_at = Column(DateTime, default=dt.datetime.now, index=True)


# ---------------------------------------------------------------- engine
_engine = None
_SessionLocal = None


def _db_url() -> str:
    url = cfg.get("database.url", "sqlite:///data/math_agent.db")
    if url.startswith("sqlite:///"):
        path = resolve_path(url.replace("sqlite:///", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path.as_posix()}"
    return url


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        url = _db_url()
        _engine = create_engine(url, future=True,
                                connect_args={"check_same_thread": False,
                                              "timeout": 30}
                                if url.startswith("sqlite") else {})
        if url.startswith("sqlite"):
            # P1：WAL 允许读写并发；busy_timeout 让写锁冲突等待而非立刻报错丢数据
            from sqlalchemy import event

            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, _record):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.close()

        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session() -> Session:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


def init_db() -> None:
    get_engine()
    Base.metadata.create_all(get_engine())


__all__ = ["Base", "Question", "QuizSession", "Attempt", "AgentState",
           "TraceTurn", "init_db", "get_engine", "get_session"]
