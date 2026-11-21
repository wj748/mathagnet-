"""轻量可观测性（方案 Phase 5）。

监控三项指标：
  1. **循环终止原因分布** —— max_loop / information_gain / finished
  2. **工具调用错误率**   —— 成功 / 被拦截（防重、参数校验、依赖链）/ 执行报错
  3. **意图分布**         —— quiz / submit / grade / habit / chat

进程内计数器，零外部依赖；接 Prometheus 时把 snapshot() 的结果暴露出去即可。
"""
from __future__ import annotations

import threading
from collections import Counter
from typing import Any, Dict

_lock = threading.Lock()

_LOOP_STOPS: Counter = Counter()
_TOOL_CALLS: Counter = Counter()          # 结果维度：ok / blocked / error
_TOOL_BY_NAME: Counter = Counter()        # 工具维度：name -> 次数
_INTENTS: Counter = Counter()
_CONV: Counter = Counter()                # 对话压缩：released / compressed / digest_chars(累计)
_PERSIST: Counter = Counter()             # 持久化失败：record_attempt / create_session / finish_session
_CONC: Counter = Counter()                # 并发治理：admitted / busy / qps / overload / queue_timeout / replay
_TG: Counter = Counter()                  # 工具失败治理：retry / timeout / circuit_open / circuit_reject / replan
_TG_FAIL: Counter = Counter()             # 工具维度最终失败次数（重试耗尽后）


def _bump(counter: Counter, key: str, n: int = 1) -> None:
    with _lock:
        counter[key] += n


# ------------------------------------------------------------------ 记录
def record_loop_stop(reason: str) -> None:
    """reason: max_loop / information_gain / finished"""
    _bump(_LOOP_STOPS, reason or "unknown")


def record_tool_call(name: str, ok: bool, blocked: bool) -> None:
    _bump(_TOOL_BY_NAME, name)
    _bump(_TOOL_CALLS, "blocked" if blocked else ("ok" if ok else "error"))


def record_intent(intent: str) -> None:
    _bump(_INTENTS, intent or "unknown")


def record_conversation(released: int = 0, compressed: int = 0,
                        digest_chars: int = 0) -> None:
    """对话压缩/释放累计量（compressor 调用）。

    ⚠️ 本函数曾缺失，调用方 `compressor._record_conv` 用 try/except 兜底，
    导致 conversation 段指标长期静默为 0——改动此处务必核对函数名。
    """
    if released:
        _bump(_CONV, "released", released)
    if compressed:
        _bump(_CONV, "compressed", compressed)
    if digest_chars:
        _bump(_CONV, "digest_chars", digest_chars)
    _bump(_CONV, "turns")


def record_persist_failure(kind: str) -> None:
    """持久化失败计数（答题记录/会话落库）。

    这些失败此前被 `except Exception: pass` 静默吞掉，导致习惯报告与长期记忆
    失真且无告警。现在统一计数，可在 /api/metrics 的 persist_failures 段看到。
    """
    _bump(_PERSIST, kind or "unknown")


def record_toolguard(event: str, tool: str = "") -> None:
    """工具失败治理事件计数（toolguard.py）。

    event: retry（瞬时重试）/ timeout（执行超时）/ circuit_open（熔断打开）/
           circuit_reject（熔断拒绝）/ replan（Agent 重规划）/
           fallback（启用备选工具）/ fatal_stop（致命错误终止）
    """
    _bump(_TG, event or "unknown")
    if tool:
        _bump(_TG_FAIL, tool)


def record_concurrency(kind: str) -> None:
    """并发治理事件计数。

    kind: admitted（放行）/ busy（同用户任务互斥拒绝）/ qps（限流拒绝）/
          overload（在途超限熔断）/ queue_timeout（排队超时降级）/ replay（幂等回放）
    """
    _bump(_CONC, kind or "unknown")


# ------------------------------------------------------------------ 读取
def _rate(counter: Counter, key: str, total: int) -> float:
    return round(counter.get(key, 0) / total * 100, 1) if total else 0.0


def snapshot() -> Dict[str, Any]:
    with _lock:
        stops = dict(_LOOP_STOPS)
        calls = dict(_TOOL_CALLS)
        tools = dict(_TOOL_BY_NAME)
        intents = dict(_INTENTS)
        conv = dict(_CONV)
        persist = dict(_PERSIST)
        conc = dict(_CONC)
        tg = dict(_TG)
        tg_fail = dict(_TG_FAIL)

    total_calls = sum(calls.values())
    total_stops = sum(stops.values())
    return {
        "loop_stops": {
            "counts": stops,
            "total": total_stops,
            "max_loop_rate": _rate(_LOOP_STOPS, "max_loop", total_stops),
            "info_gain_rate": _rate(_LOOP_STOPS, "information_gain", total_stops),
        },
        "tool_calls": {
            "counts": calls,
            "total": total_calls,
            "error_rate": _rate(_TOOL_CALLS, "error", total_calls),
            "blocked_rate": _rate(_TOOL_CALLS, "blocked", total_calls),
            "by_name": tools,
        },
        "intents": {"counts": intents, "total": sum(intents.values())},
        "conversation": {"counts": conv, "turns": conv.get("turns", 0)},
        "persist_failures": {"counts": persist, "total": sum(persist.values())},
        "concurrency": {"counts": conc, "total": sum(conc.values())},
        "toolguard": {"counts": tg, "total": sum(tg.values()),
                      "failures_by_tool": tg_fail},
    }


def reset() -> None:
    with _lock:
        _LOOP_STOPS.clear()
        _TOOL_CALLS.clear()
        _TOOL_BY_NAME.clear()
        _INTENTS.clear()
        _CONV.clear()
        _PERSIST.clear()
        _CONC.clear()
        _TG.clear()
        _TG_FAIL.clear()


__all__ = ["record_loop_stop", "record_tool_call", "record_intent",
           "record_conversation", "record_persist_failure", "record_concurrency",
           "record_toolguard", "snapshot", "reset"]
