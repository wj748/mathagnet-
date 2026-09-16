"""多用户并发治理单测（concurrency.py）。

覆盖：同用户互斥 / 跨用户并行 / QPS 滑窗 / 熔断（硬上限+排队超时）/
幂等缓存（命中·过期·LRU 淘汰）/ InMemoryStore 并发读写安全。
全部纯进程内，不依赖 Qdrant / LLM / Redis，秒级跑完。
"""
from __future__ import annotations

import threading

import pytest

from math_agent.concurrency import GlobalBreaker, IdempotencyStore, UserGate
from math_agent.memory.short_term import InMemoryStore


# ---------------------------------------------------------------- UserGate
def test_gate_同用户互斥():
    g = UserGate()
    ok1, r1 = g.acquire("u1")
    assert ok1 and r1 == ""
    ok2, r2 = g.acquire("u1")          # 第二条任务必须被拒（429 busy）
    assert not ok2 and r2 == "busy"
    g.release("u1")
    ok3, _ = g.acquire("u1")           # 释放后可再次进入
    assert ok3
    g.release("u1")


def test_gate_跨用户互不干扰():
    g = UserGate()
    ok1, _ = g.acquire("u1")
    ok2, _ = g.acquire("u2")
    assert ok1 and ok2                 # 隔离：u1 在跑不影响 u2
    g.release("u1")
    g.release("u2")


def test_gate_qps_滑动窗口():
    g = UserGate()
    results = []
    for _ in range(4):                 # 默认 per_user_qps=3：逐次 acquire+release
        ok, r = g.acquire("u1")        # 到达即计数（含被拒），第 4 次触发 qps
        results.append((ok, r))
        if ok:
            g.release("u1")
    assert [ok for ok, _ in results[:3]] == [True, True, True]
    assert results[3] == (False, "qps")


# ------------------------------------------------------------ GlobalBreaker
def test_breaker_硬上限立即熔断():
    b = GlobalBreaker(max_inflight=2, max_tasks=2, wait_s=0.1)
    assert b.enter()[0]
    assert b.enter()[0]
    ok, reason = b.enter()             # 达到硬上限 → 不排队直接拒
    assert not ok and reason == "overload"
    b.exit()
    b.exit()
    assert b.stats()["active"] == 0


def test_breaker_排队超时降级():
    b = GlobalBreaker(max_inflight=8, max_tasks=1, wait_s=0.2)
    assert b.enter()[0]
    import time as _t
    t0 = _t.monotonic()
    ok, reason = b.enter()             # 并发满 → 最多等 0.2s → 降级拒绝
    assert not ok and reason == "queue_timeout"
    assert _t.monotonic() - t0 >= 0.15
    b.exit()
    assert b.enter()[0]                # 释放后可进入
    b.exit()


def test_breaker_统计口径():
    b = GlobalBreaker(max_inflight=1, max_tasks=1, wait_s=0.05)
    b.enter()
    b.enter()                          # rejected
    s = b.stats()
    assert s["active"] == 1 and s["rejected_total"] == 1
    b.exit()


# ---------------------------------------------------------- IdempotencyStore
def test_idempotency_命中与未命中():
    s = IdempotencyStore(ttl_s=10, max_entries=10)
    assert s.get("u1:rid1") is None
    s.put("u1:rid1", {"reply": "hello"})
    assert s.get("u1:rid1") == {"reply": "hello"}
    assert s.get("u2:rid1") is None    # 用户隔离：key 前缀不同不可见


def test_idempotency_ttl_过期():
    s = IdempotencyStore(ttl_s=0.05, max_entries=10)
    s.put("u1:rid", {"a": 1})
    import time as _t
    _t.sleep(0.08)
    assert s.get("u1:rid") is None


def test_idempotency_lru_容量淘汰():
    s = IdempotencyStore(ttl_s=60, max_entries=2)
    s.put("k1", 1)
    s.put("k2", 2)
    s.get("k1")                        # 触碰 k1 → k2 成为最旧
    s.put("k3", 3)
    assert s.get("k2") is None         # 最旧未触碰者被逐
    assert s.get("k1") == 1 and s.get("k3") == 3


# ---------------------------------------------------------- InMemoryStore
def test_inmemory_store_并发读写不抛错且最终一致():
    store = InMemoryStore()
    errs: list = []

    def _worker(tid: int) -> None:
        try:
            for i in range(200):
                store.set(f"k{i % 10}", {"tid": tid, "i": i}, ttl=60)
                store.get(f"k{(i + 1) % 10}")
                if i % 50 == 0:
                    store.delete(f"k{(i + 3) % 10}")
        except Exception as e:         # noqa: BLE001
            errs.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=_worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errs


def test_inmemory_store_ttl_过期():
    store = InMemoryStore()
    store.set("k", "v", ttl=1)
    assert store.get("k") == "v"
    store._d["k"] = ("'v'", 0.0)       # 手动构造已过期项
    assert store.get("k") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
