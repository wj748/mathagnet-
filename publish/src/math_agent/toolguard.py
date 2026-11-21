"""多工具链式调用的失败治理（toolguard，2026-09-15）。

对应「Agent 多工具链式调用，中途失败处理」规范，五件事全部代码层硬约束：

1. **失败分类**：``classify_failure`` → transient / param / fatal
   - transient（网络/超时/临时限流）：指数退避重试，仅重试当前步骤
   - param（参数/业务错误）：重试无效，把脱敏错误交回 Agent 重规划
   - fatal（鉴权/配额）：不重试，直接降级终止
2. **幂等防重复**：registry 的 args_hash 防重 + 写类工具 ``retryable=False``
   （重试不产生重复副作用）
3. **每工具熔断**：滑动窗口失败率超阈值即打开，冷却后半开探测，防雪崩
4. **单工具超时**：线程池隔离执行，单工具卡死不拖垮整条链
5. **步骤台账 StepLedger**：task_id/step_id 绑定，每步落存储（Redis/内存），
   支持断点续跑与失败诊断

错误信息统一 ``sanitize_error`` 脱敏后才进入 ToolResult / trace / LLM prompt。

边界：重试/熔断/超时为**进程内**实现，匹配单 uvicorn 进程部署；多实例需迁移
Redis 分布式锁与分布式熔断（同 concurrency.py 的边界声明）。
"""
from __future__ import annotations

import random
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional, Tuple

from .config import cfg
from . import metrics

# ---------------------------------------------------------------- 失败分类
# 命中顺序：fatal > transient > param（配额/鉴权优先于限流判定）
_FATAL_PAT = re.compile(
    r"401|403|authentication|unauthorized|invalid.{0,6}api.?key|"
    r"quota|balance|欠费|鉴权失败|配额|密钥无效", re.I)
_TRANSIENT_PAT = re.compile(
    r"timeout|timed? ?out|connection|connect|reset|refused|broken|"
    r"429|rate.?limit|too many requests|5\d{2}|server error|temporar|unavailable|"
    r"执行超时|网络|限流|暂时|稍后重试", re.I)


def classify_failure(error: str) -> str:
    """把错误文本分类为 transient / param / fatal。

    瞬时故障（网络抖动、超时、临时限流）→ 重试有意义；
    参数/业务错误（值非法、类型不对）→ 重试无意义，交回 Agent 修正；
    资源配额/鉴权类致命错误 → 直接终止，不重试。
    """
    e = error or ""
    if _FATAL_PAT.search(e):
        return "fatal"
    if _TRANSIENT_PAT.search(e):
        return "transient"
    return "param"


# ---------------------------------------------------------------- 错误脱敏
_PATH_PAT = re.compile(r"(?:[A-Za-z]:[\\/](?:[\w.\- \u4e00-\u9fff]+[\\/])+[\w.\- \u4e00-\u9fff]+)")
_LONG_TOKEN_PAT = re.compile(r"\b([A-Za-z0-9]{6})[A-Za-z0-9._]{14,}\b")
_MAIL_PAT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_WS_PAT = re.compile(r"\s+")


def sanitize_error(text: str, limit: int = 200) -> str:
    """错误信息脱敏：绝对路径、长 token/密钥、邮箱打码；截断到 limit 字符。

    工具异常文本（含文件路径、内部地址）不允许原样进入 ToolResult / trace /
    LLM prompt / 前端——所有执行类错误必须先过这里。
    """
    s = _PATH_PAT.sub("<path>", text or "")
    s = _LONG_TOKEN_PAT.sub(r"\1***", s)
    s = _MAIL_PAT.sub("<mail>", s)
    s = _WS_PAT.sub(" ", s).strip()
    return s[:limit]


# ---------------------------------------------------------------- 熔断器
class ToolCircuit:
    """单工具熔断器：滑动窗口失败率 + 冷却后半开探测。"""

    def __init__(self, window: int, min_volume: int, fail_rate: float,
                 cooldown_s: float):
        self.window = max(5, int(window))
        self.min_volume = max(1, int(min_volume))
        self.fail_rate = float(fail_rate)
        self.cooldown_s = float(cooldown_s)
        self._results: deque = deque(maxlen=self.window)
        self._opened_at: float = 0.0

    @property
    def is_open(self) -> bool:
        if len(self._results) < self.min_volume:
            return False
        fails = sum(1 for r in self._results if not r)
        if fails / len(self._results) < self.fail_rate:
            return False
        # 打开状态超过冷却期 → 半开（清窗重探）
        if self._opened_at and time.time() - self._opened_at >= self.cooldown_s:
            self._results.clear()
            self._opened_at = 0.0
            return False
        return True

    def record(self, ok: bool) -> None:
        self._results.append(bool(ok))
        if self.is_open and not self._opened_at:
            self._opened_at = time.time()


_CIRCUITS: Dict[str, ToolCircuit] = {}
_CK_LOCK = None  # 延迟初始化，避免 import 顺序问题


def _ck_lock():
    global _CK_LOCK
    import threading
    if _CK_LOCK is None:
        _CK_LOCK = threading.Lock()
    return _CK_LOCK


def _circuit(name: str) -> ToolCircuit:
    with _ck_lock():
        c = _CIRCUITS.get(name)
        if c is None:
            c = ToolCircuit(
                window=int(cfg.get("toolguard.circuit_window", 20)),
                min_volume=int(cfg.get("toolguard.circuit_min_volume", 5)),
                fail_rate=float(cfg.get("toolguard.circuit_fail_rate", 0.8)),
                cooldown_s=float(cfg.get("toolguard.circuit_cooldown_s", 30)),
            )
            _CIRCUITS[name] = c
        return c


def circuit_allow(name: str) -> bool:
    return not _circuit(name).is_open


def circuit_record(name: str, ok: bool) -> None:
    c = _circuit(name)
    was_open = c.is_open
    c.record(ok)
    if not was_open and c.is_open:
        metrics.record_toolguard("circuit_open", name)


def circuit_snapshot() -> Dict[str, Any]:
    with _ck_lock():
        return {n: {"open": c.is_open, "recent": len(c._results)}
                for n, c in _CIRCUITS.items()}


# ---------------------------------------------------------------- 守卫执行
_POOL: Optional[ThreadPoolExecutor] = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="toolguard")
    return _POOL


def _tool_timeout(spec_timeout: float) -> float:
    return float(spec_timeout or cfg.get("toolguard.default_timeout_s", 30))


def execute_with_guard(spec, args: Dict[str, Any]) -> Tuple[bool, Any, str]:
    """带失败治理的工具执行：熔断 → 超时 → 分类重试 → 脱敏。

    返回 ``(ok, output, error, rejected)``。error 已脱敏并附带失败类别提示，
    可安全回注给 Agent / LLM / 前端；``rejected=True`` 表示熔断快速拒绝
    （非工具本身报错），调用方应置 blocked 标记。
    """
    name = spec.name
    if not circuit_allow(name):
        metrics.record_toolguard("circuit_reject", name)
        return False, None, ("工具熔断中（近期连续失败过多），已按降级策略处理；"
                             "请稍后再试或换用其它方式。"), True

    timeout = _tool_timeout(getattr(spec, "timeout", None))
    max_retries = int(cfg.get("toolguard.max_retries", 2))
    base = float(cfg.get("toolguard.backoff_base_s", 0.4))
    attempts = (max_retries + 1) if getattr(spec, "retryable", True) else 1

    last_err = ""
    cls = "param"
    for i in range(attempts):
        try:
            fut = _pool().submit(spec.func, **args)
            out = fut.result(timeout=timeout)
            circuit_record(name, True)
            return True, out, "", False
        except FuturesTimeoutError:
            last_err = f"执行超时（>{timeout:.0f}s）"
            cls = "transient"
            metrics.record_toolguard("timeout", name)
        except Exception as e:                       # noqa: BLE001 统一治理入口
            last_err = f"{type(e).__name__}: {e}"
            cls = classify_failure(last_err)
        circuit_record(name, False)
        # 只对瞬时故障重试；param 重试无意义、fatal 不重试
        if cls != "transient" or i >= attempts - 1:
            break
        metrics.record_toolguard("retry", name)
        time.sleep(min(base * (2 ** i), 5.0) + random.uniform(0, 0.1))

    if cls == "transient":
        msg = f"已重试 {attempts - 1} 次仍失败：{sanitize_error(last_err)}"
    elif cls == "fatal":
        msg = f"服务暂不可用（配额/鉴权类错误）：{sanitize_error(last_err)}"
    else:
        msg = f"参数/业务错误：{sanitize_error(last_err)}（请修正参数后重试，或换用其它工具）"
    return False, None, msg, False


# ---------------------------------------------------------------- 步骤台账
_LEDGER_CACHE: Dict[str, Any] = {}


def _ledger_backend():
    """短期记忆存储（Redis 启用即分布式，否则进程内），与 compressor 同源。"""
    if "stm" in _LEDGER_CACHE:
        return _LEDGER_CACHE["stm"]
    from .memory.short_term import ShortTermMemory
    stm = ShortTermMemory()
    _LEDGER_CACHE["stm"] = stm
    return stm


def _ledger_key(task_id: str) -> str:
    return f"math:task:{task_id}"


def ledger_create(task_id: str) -> None:
    try:
        _ledger_backend().backend.set(
            _ledger_key(task_id),
            {"task_id": task_id, "status": "running", "steps": [],
             "started_at": time.time()},
            int(cfg.get("toolguard.ledger_ttl_s", 3600)))
    except Exception:
        pass  # 台账失败不阻断主链路（仅损失可观测/续跑能力）


def ledger_record(task_id: str, step_id: str, tool: str, args_hash: str,
                  ok: bool, digest: str = "") -> None:
    """记录一步：task_id/step_id/tool/args_hash/status + 结果摘要（防大对象）。"""
    try:
        stm = _ledger_backend()
        key = _ledger_key(task_id)
        data = stm.backend.get(key) or {"task_id": task_id, "status": "running",
                                        "steps": [], "started_at": time.time()}
        steps = list(data.get("steps") or [])
        steps.append({"step_id": step_id, "tool": tool, "args_hash": args_hash,
                      "status": "ok" if ok else "failed",
                      "digest": (digest or "")[:200],
                      "ts": time.time()})
        data["steps"] = steps[-32:]              # 有界：只留最近 32 步
        stm.backend.set(key, data, int(cfg.get("toolguard.ledger_ttl_s", 3600)))
    except Exception:
        pass


def ledger_finish(task_id: str, status: str) -> None:
    try:
        stm = _ledger_backend()
        key = _ledger_key(task_id)
        data = stm.backend.get(key)
        if data:
            data["status"] = status
            data["finished_at"] = time.time()
            stm.backend.set(key, data, int(cfg.get("toolguard.ledger_ttl_s", 3600)))
    except Exception:
        pass


def ledger_load(task_id: str) -> Optional[Dict[str, Any]]:
    try:
        return _ledger_backend().backend.get(_ledger_key(task_id))
    except Exception:
        return None


def ledger_completed_map(data: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """从台账数据提取「已成功步骤」映射：args_hash → step（断点续跑用）。"""
    if not data:
        return {}
    return {s["args_hash"]: s for s in (data.get("steps") or [])
            if s.get("status") == "ok" and s.get("args_hash")}


def circuit_reset() -> None:
    """测试辅助：清空全部熔断器状态。"""
    with _ck_lock():
        _CIRCUITS.clear()


__all__ = ["classify_failure", "sanitize_error", "ToolCircuit",
           "circuit_allow", "circuit_record", "circuit_snapshot", "circuit_reset",
           "execute_with_guard", "ledger_create", "ledger_record", "ledger_finish",
           "ledger_load", "ledger_completed_map"]
