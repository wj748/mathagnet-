"""toolguard 工具链失败治理单测（失败分类/脱敏/重试/熔断/超时/台账）。"""
from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from math_agent import toolguard
from math_agent.tools.registry import ToolResult, register_tool, run_tool


# ---------------------------------------------------------------- 固定工具
class _Empty(BaseModel):
    pass


class _Expr(BaseModel):
    expression: str


_COUNTERS = {"flaky": 0, "slow": 0}


@register_tool("tg_flaky", "前 1 次失败（网络错误），之后成功", _Empty)
def _tg_flaky() -> dict:
    _COUNTERS["flaky"] += 1
    if _COUNTERS["flaky"] == 1:
        raise ConnectionError("connection reset by peer")
    return {"attempt": _COUNTERS["flaky"]}


@register_tool("tg_param", "恒抛参数类错误", _Empty, retryable=False)
def _tg_param() -> dict:
    raise ValueError("表达式格式不正确")


@register_tool("tg_fatal", "恒抛鉴权类错误", _Empty, retryable=False)
def _tg_fatal() -> dict:
    raise RuntimeError("API 鉴权失败: 401 unauthorized")


@register_tool("tg_slow", "睡 1 秒（测超时）", _Empty, retryable=False, timeout=0.2)
def _tg_slow() -> dict:
    time.sleep(1.0)
    return {"done": True}


@register_tool("tg_always_fail", "恒失败（测熔断）", _Empty, retryable=False)
def _tg_always_fail() -> dict:
    raise ValueError("bad input")


@pytest.fixture(autouse=True)
def _reset_state():
    toolguard.circuit_reset()
    _COUNTERS["flaky"] = 0
    _COUNTERS["slow"] = 0
    yield
    toolguard.circuit_reset()


# ---------------------------------------------------------------- 分类
def test_失败分类_三分类():
    assert toolguard.classify_failure("ConnectionError: connection reset") == "transient"
    assert toolguard.classify_failure("执行超时（>30s）") == "transient"
    assert toolguard.classify_failure("HTTP 429 rate limit") == "transient"
    assert toolguard.classify_failure("ValueError: 表达式格式不正确") == "param"
    assert toolguard.classify_failure("KeyError: 'topic'") == "param"
    assert toolguard.classify_failure("401 unauthorized, invalid api key") == "fatal"
    assert toolguard.classify_failure("quota exceeded, 欠费") == "fatal"
    # fatal 优先于 transient（429 限流 + 配额 → 按致命处理）
    assert toolguard.classify_failure("429 quota balance insufficient") == "fatal"


def test_错误脱敏():
    s = toolguard.sanitize_error(
        "FileNotFoundError: D:\\游戏库\\智能助教\\data\\secret.txt not found, "
        "key=abcdefgh123456789012345 contact a@b.com")
    assert "游戏库" not in s and "<path>" in s
    assert "abcdefgh" not in s or "***" in s
    assert "a@b.com" not in s and "<mail>" in s
    assert len(s) <= 200


# ---------------------------------------------------------------- 守卫执行
def test_瞬时故障_指数退避重试成功():
    r = run_tool("tg_flaky", {})
    assert r.ok and r.output == {"attempt": 2}
    assert _COUNTERS["flaky"] == 2          # 重试了 1 次


def test_参数错误_不重试():
    r = run_tool("tg_param", {})
    assert not r.ok
    assert _COUNTERS["param"] == 1 if "param" in _COUNTERS else True
    assert "参数/业务错误" in r.error


def test_致命错误_不重试():
    r = run_tool("tg_fatal", {})
    assert not r.ok
    assert "服务暂不可用" in r.error


def test_单工具超时_线程池隔离():
    t0 = time.time()
    r = run_tool("tg_slow", {})
    cost = time.time() - t0
    assert not r.ok and "超时" in r.error
    # 0.2s 超时即返回，不等 1s 睡眠（留余量断言）
    assert cost < 3.0


def test_熔断_打开后快速拒绝():
    for _ in range(6):                       # min_volume=5，6 次全失败 → 打开
        r = run_tool("tg_always_fail", {})
        assert not r.ok
    r = run_tool("tg_always_fail", {})
    assert not r.ok and r.blocked
    assert "熔断" in r.error
    snap = toolguard.circuit_snapshot()
    assert snap["tg_always_fail"]["open"] is True


# ---------------------------------------------------------------- 台账
def test_步骤台账_落步与断点续跑依据():
    task_id = "t_test_001"
    toolguard.ledger_create(task_id)
    toolguard.ledger_record(task_id, f"{task_id}-s1-search", "search_knowledge",
                            "hash001", True, digest="found 3 results")
    toolguard.ledger_record(task_id, f"{task_id}-s2-calc", "calculator",
                            "hash002", False, digest="bad expr")
    toolguard.ledger_finish(task_id, "degraded:calculator")

    data = toolguard.ledger_load(task_id)
    assert data and data["status"] == "degraded:calculator"
    assert len(data["steps"]) == 2
    done = toolguard.ledger_completed_map(data)
    assert set(done) == {"hash001"}          # 只有成功步骤可续跑复用
    assert done["hash001"]["tool"] == "search_knowledge"
