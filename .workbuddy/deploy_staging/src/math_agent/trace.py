"""自研调用链追踪（Phase 5 可观测性，替代 LangSmith 外部依赖）。

每轮对话落一条 :class:`TurnTrace`，存进程内环形缓冲（默认 500 条，可配置）。
经 ``GET /api/traces`` 输出明细与聚合统计，前端 ``/trace`` 页面直接消费。

设计原则：
  * 零外部依赖、零阻塞：写缓冲只加一把小锁，不落盘、不联网；
  * **绝不因追踪失败影响主流程**——``record_turn`` 内部全量 try/except；
  * 敏感内容（用户输入、回复）只截断保存，配合 ``safety.mask_pii`` 脱敏。
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .config import cfg

log = logging.getLogger(__name__)
_PREVIEW = 200          # 输入/回复预览长度
_DB_MAX_ROWS = 2000     # 落盘上限，超出清理最旧的（避免无限增长）
_PERSIST_ON = True      # 落盘失败后自动关闭，避免每轮刷日志
_PRUNE_TICK = 0         # 清理计数（每 50 轮检查一次总量）


def _clip(s: Any, n: int = _PREVIEW) -> str:
    t = str(s or "")
    return t if len(t) <= n else t[:n] + "…"


@dataclass
class TurnTrace:
    """一轮对话的完整调用链。"""

    ts: float
    turn_id: str
    user_id: str
    message: str
    intent: str = ""
    confidence: float = 0.0
    mode: str = ""
    nodes: List[str] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    loop_stop: str = ""
    elapsed_ms: int = 0
    response: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["time"] = time.strftime("%H:%M:%S", time.localtime(self.ts))
        return d


class TraceRecorder:
    def __init__(self, maxlen: int = 500) -> None:
        self._buf: deque = deque(maxlen=max(10, maxlen))
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 写入
    def add(self, trace: TurnTrace) -> None:
        with self._lock:
            self._buf.append(trace)
        self._persist(trace)

    def _persist(self, trace: TurnTrace) -> None:
        """落盘一份（内存缓冲重启归零，这里保证历史可追溯）。

        失败自动关闭落盘并只警告一次——追踪绝不能影响主流程。
        """
        global _PERSIST_ON
        if not _PERSIST_ON:
            return
        try:
            from .db.models import TraceTurn, get_session, init_db

            init_db()
            blob = json.dumps(trace.to_dict(), ensure_ascii=False, default=str)
            with get_session() as s:
                s.add(TraceTurn(ts=float(trace.ts), user_id=trace.user_id or "",
                                intent=trace.intent or "",
                                elapsed_ms=int(trace.elapsed_ms or 0),
                                error=(trace.error or "")[:500],
                                payload=blob[:16000]))
                s.commit()
            self._prune_db()
        except Exception as e:
            _PERSIST_ON = False
            log.warning("trace 落盘失败，后续仅内存记录：%s", e)

    def _prune_db(self) -> None:
        """控制落盘总量：超过上限删掉最旧的。每 50 轮才查一次，省 IO。"""
        global _PRUNE_TICK
        _PRUNE_TICK += 1
        if _PRUNE_TICK % 50 != 0:
            return
        try:
            from sqlalchemy import delete, func, select

            from .db.models import TraceTurn, get_session

            with get_session() as s:
                n = s.execute(select(func.count()).select_from(TraceTurn)).scalar() or 0
                if n <= _DB_MAX_ROWS:
                    return
                old = list(s.execute(
                    select(TraceTurn.id).order_by(TraceTurn.ts.asc())
                    .limit(n - _DB_MAX_ROWS)).scalars().all())
                if old:
                    s.execute(delete(TraceTurn).where(TraceTurn.id.in_(old)))
                    s.commit()
        except Exception:
            pass

    def load_recent_from_db(self, limit: int = 100) -> int:
        """进程启动时把最近的调用链读回内存（否则重启后面板一片空白）。"""
        try:
            from .db.models import TraceTurn, get_session, init_db
            from sqlalchemy import select

            init_db()
            with get_session() as s:
                rows = s.execute(select(TraceTurn.payload).order_by(
                    TraceTurn.ts.desc()).limit(max(1, limit))).scalars().all()
            got = 0
            for blob in reversed(rows):          # 升序回填，保持时间顺序
                try:
                    d = json.loads(blob or "{}")
                except Exception:
                    continue
                d.pop("time", None)
                try:
                    self._buf.append(TurnTrace(**d))
                    got += 1
                except Exception:
                    continue
            return got
        except Exception as e:
            log.warning("trace 历史加载失败（仅用内存记录）：%s", e)
            return 0

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    # ---------------------------------------------------------------- 读取
    def recent(self, limit: int = 50, user_id: Optional[str] = None,
               intent: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._buf)
        if user_id:
            items = [t for t in items if t.user_id == user_id]
        if intent:
            items = [t for t in items if t.intent == intent]
        items.sort(key=lambda t: t.ts, reverse=True)
        return [t.to_dict() for t in items[:max(1, limit)]]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            items = list(self._buf)
        if not items:
            return {"turns": 0, "error_turns": 0, "error_rate": 0.0,
                    "latency": {"p50": 0, "p95": 0, "avg": 0},
                    "intents": {}, "loop_stops": {}, "tools": {},
                    "tool_error_rate": 0.0, "blocked_tools": 0}

        lat = sorted(t.elapsed_ms for t in items)
        p50 = lat[len(lat) // 2]
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        intents = Counter(t.intent or "unknown" for t in items)
        stops = Counter(t.loop_stop for t in items if t.loop_stop)

        tool_total = 0
        tool_err = 0
        tool_blocked = 0
        by_tool: Counter = Counter()
        for t in items:
            for c in t.tools:
                tool_total += 1
                by_tool[c.get("name", "?")] += 1
                if c.get("blocked"):
                    tool_blocked += 1
                elif not c.get("ok", True):
                    tool_err += 1
        err_turns = sum(1 for t in items if t.error)
        return {
            "turns": len(items),
            "error_turns": err_turns,
            "error_rate": round(err_turns / len(items) * 100, 1),
            "latency": {"p50": p50, "p95": p95,
                        "avg": round(sum(lat) / len(lat))},
            "intents": dict(intents),
            "loop_stops": dict(stops),
            "tools": dict(by_tool),
            "tool_calls": tool_total,
            "tool_error_rate": round(tool_err / tool_total * 100, 1) if tool_total else 0.0,
            "blocked_tools": tool_blocked,
        }


_recorder: Optional[TraceRecorder] = None
_lock = threading.Lock()


def get_recorder() -> TraceRecorder:
    global _recorder
    if _recorder is None:
        with _lock:
            if _recorder is None:
                rec = TraceRecorder(int(cfg.get("trace.max_items", 500)))
                if cfg.get("trace.persist", True):
                    rec.load_recent_from_db(limit=min(100, int(
                        cfg.get("trace.max_items", 500))))
                _recorder = rec
    return _recorder


def record_turn(user_id: str, message: str, out: Dict[str, Any],
                elapsed_ms: int, error: str = "") -> Optional[TurnTrace]:
    """把一轮 invoke 的结果记进追踪缓冲；任何异常都吞掉，绝不影响主流程。"""
    try:
        from .safety import mask_pii

        msg = mask_pii(_clip(message))
        resp = mask_pii(_clip((out or {}).get("response")))
    except Exception:
        msg, resp = _clip(message), _clip((out or {}).get("response"))

    try:
        hist = list((out or {}).get("tool_call_history") or [])
        tools = [{"name": h.get("name", "?"), "ok": bool(h.get("ok")),
                  "blocked": bool(h.get("blocked")),
                  "error": _clip(h.get("error", ""), 80)}
                 for h in hist[-10:]]
        trace = TurnTrace(
            ts=time.time(), turn_id=uuid.uuid4().hex[:10], user_id=user_id,
            message=msg, intent=(out or {}).get("intent", "") or "",
            confidence=float((out or {}).get("intent_confidence", 0) or 0),
            mode=(out or {}).get("session_mode", "") or "",
            nodes=list((out or {}).get("trace") or [])[-12:],
            tools=tools, loop_stop=(out or {}).get("loop_stop_reason", "") or "",
            elapsed_ms=int(elapsed_ms), response=resp, error=_clip(error, 120),
        )
        get_recorder().add(trace)
        return trace
    except Exception:
        return None


__all__ = ["TurnTrace", "TraceRecorder", "get_recorder", "record_turn"]
