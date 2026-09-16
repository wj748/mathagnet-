"""多用户并发治理（P0）——全部代码层硬约束，不依赖客户端自觉。

对应六项设计要求：
  1. 强会话隔离  —— 状态/记忆/记录均已以 user_id 为键（Agent 池、SQLite、
                    Qdrant 强制过滤、Redis 键前缀）；本模块新增的排队/限流
                    状态同样按 user_id 隔离，无任何跨用户共享可变态。
  2. 并发控制    —— UserGate：同一用户同一时刻只允许 1 条智能体任务
                    （非阻塞抢占，第二个请求立即 429 而非无限排队）+
                    滑动窗口 QPS 限流。
  3. 幂等设计    —— IdempotencyStore：请求 ID（X-Request-Id / request_id）
                    在 TTL 内重放直接返回首次结果，不重复执行智能体。
  4. 资源隔离    —— 进程内按用户隔离（现状为单 uvicorn 进程部署）；
                    GlobalBreaker 的全局信号量保护下游 LLM API 配额不被
                    少数用户打爆。GPU/显存池化不适用（CPU 推理 + 外部 API），
                    对应物即全局并发预算。
  5. 熔断降级    —— GlobalBreaker：在途任务达硬上限立即拒绝新请求；
                    并发满时排队 queue_wait_s 秒，超时降级 503，避免雪崩。
  6. 记忆层锁/版本 —— InMemoryStore 加 RLock、compressor 会话版本戳
                    （见 short_term.py / compressor.py；同用户任务已被
                     UserGate 串行化，锁是纵深防御的第二道）。

⚠️ 边界：以上锁均为进程内锁，适用于当前单 uvicorn 进程部署；若未来多进程/
多实例，需将 UserGate/幂等迁移到 Redis 分布式锁（SET NX + TTL）。
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Any, Dict, Optional, Tuple

from .config import cfg

__all__ = ["UserGate", "GlobalBreaker", "IdempotencyStore",
           "get_gate", "get_breaker", "get_idempotency"]


class UserGate:
    """同一用户串行 + QPS 限流。

    acquire() 返回 (ok, reason)；reason 为 "" / "busy"（已有任务在跑）/
    "qps"（触发每秒请求数上限）。acquire 成功后**必须**在 finally 里
    调 release(user_id)。
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: Dict[str, threading.Lock] = {}
        self._hits: Dict[str, deque] = {}

    def acquire(self, user_id: str) -> Tuple[bool, str]:
        qps = int(cfg.get("concurrency.per_user_qps", 3))
        now = time.monotonic()
        with self._guard:
            # QPS 滑动窗口（1 秒），对所有到达请求计数（含被拒的）
            hits = self._hits.setdefault(user_id, deque())
            while hits and now - hits[0] > 1.0:
                hits.popleft()
            if len(hits) >= max(1, qps):
                return False, "qps"
            hits.append(now)
            lock = self._locks.setdefault(user_id, threading.Lock())
        # 锁操作放在 guard 之外：互斥等待不能抱着全局字典锁
        if not lock.acquire(blocking=False):
            return False, "busy"
        return True, ""

    def release(self, user_id: str) -> None:
        with self._guard:
            lock = self._locks.get(user_id)
        if lock is not None:
            try:
                lock.release()
            except RuntimeError:
                pass    # 已被释放（防御：调用方重复 release 不应炸主流程）


class GlobalBreaker:
    """全局并发预算 + 熔断。

    - 在途任务 >= max_inflight：立即拒绝（"overload"），不排队——防雪崩；
    - 否则在 max_agent_tasks 信号量上排队，最多等 queue_wait_s 秒，
      超时降级拒绝（"queue_timeout"）。
    enter() 成功后必须在 finally 里调用 exit()。
    """

    def __init__(self, max_inflight: Optional[int] = None,
                 max_tasks: Optional[int] = None,
                 wait_s: Optional[float] = None) -> None:
        self._max_inflight = max_inflight
        self._max_tasks = max_tasks
        self._wait_s = wait_s
        self._guard = threading.Lock()
        self._cond = threading.Condition(self._guard)
        self._active = 0          # 正在执行智能体任务数（含排队中的配额占用）
        self._rejected = 0

    def _limits(self) -> Tuple[int, int, float]:
        mi = self._max_inflight if self._max_inflight is not None \
            else int(cfg.get("concurrency.max_inflight", 32))
        mt = self._max_tasks if self._max_tasks is not None \
            else int(cfg.get("concurrency.max_agent_tasks", 8))
        ws = self._wait_s if self._wait_s is not None \
            else float(cfg.get("concurrency.queue_wait_s", 15))
        return mi, mt, ws

    def enter(self) -> Tuple[bool, str]:
        max_inflight, max_tasks, wait_s = self._limits()
        deadline = time.monotonic() + wait_s
        with self._cond:
            if self._active >= max_inflight:
                self._rejected += 1
                return False, "overload"
            while self._active >= max_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._rejected += 1
                    return False, "queue_timeout"
                self._cond.wait(remaining)
            self._active += 1
            return True, ""

    def exit(self) -> None:
        with self._cond:
            self._active = max(0, self._active - 1)
            self._cond.notify()

    def stats(self) -> Dict[str, int]:
        max_inflight, max_tasks, _ = self._limits()
        with self._guard:
            return {"active": self._active, "max_agent_tasks": max_tasks,
                    "max_inflight": max_inflight, "rejected_total": self._rejected}


class IdempotencyStore:
    """请求 ID 幂等缓存：TTL + LRU 双重淘汰，进程内实现。

    key 约定为 f"{user_id}:{request_id}"（调用方拼接），天然按用户隔离。
    """

    def __init__(self, ttl_s: Optional[float] = None,
                 max_entries: Optional[int] = None) -> None:
        self._ttl = ttl_s
        self._max = max_entries
        self._d: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._guard = threading.Lock()

    def _limits(self) -> Tuple[float, int]:
        ttl = self._ttl if self._ttl is not None \
            else float(cfg.get("concurrency.idempotency_ttl_s", 300))
        mx = self._max if self._max is not None \
            else int(cfg.get("concurrency.idempotency_max", 500))
        return ttl, mx

    def get(self, key: str) -> Optional[Any]:
        ttl, _ = self._limits()
        now = time.monotonic()
        with self._guard:
            item = self._d.get(key)
            if item is None:
                return None
            ts, payload = item
            if now - ts > ttl:
                self._d.pop(key, None)
                return None
            self._d.move_to_end(key)        # LRU 触碰
            return payload

    def put(self, key: str, payload: Any) -> None:
        ttl, mx = self._limits()
        now = time.monotonic()
        with self._guard:
            # 先淘汰过期项，再按容量淘汰最旧项
            stale = [k for k, (ts, _) in self._d.items() if now - ts > ttl]
            for k in stale:
                self._d.pop(k, None)
            while len(self._d) >= mx:
                self._d.popitem(last=False)
            self._d[key] = (now, payload)


# ------------------------------------------------------------------ 单例
_gate: Optional[UserGate] = None
_breaker: Optional[GlobalBreaker] = None
_idem: Optional[IdempotencyStore] = None
_singletons_lock = threading.Lock()


def get_gate() -> UserGate:
    global _gate
    with _singletons_lock:
        if _gate is None:
            _gate = UserGate()
        return _gate


def get_breaker() -> GlobalBreaker:
    global _breaker
    with _singletons_lock:
        if _breaker is None:
            _breaker = GlobalBreaker()
        return _breaker


def get_idempotency() -> IdempotencyStore:
    global _idem
    with _singletons_lock:
        if _idem is None:
            _idem = IdempotencyStore()
        return _idem
