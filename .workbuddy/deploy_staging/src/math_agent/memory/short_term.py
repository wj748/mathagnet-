"""短期记忆：Redis（可选）+ 内存兜底（方案 4.2）。

短期记忆保存"当前会话/正在刷的题组/刚上传的文件解析结果"，TTL 默认 24h；
长期记忆则由 ``vectorstore`` 的 ``user_long_term_memory`` Collection 承载。
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Optional

from ..config import cfg


class InMemoryStore:
    """内存兜底后端。

    ⚠️ 并发安全：FastAPI sync 端点跑在线程池里，裸 dict 读写存在竞态
    （get 检查过期 → pop 与另一线程 set 交错）。加 RLock 串行化；
    RedisStore 无需加锁——redis-py 连接池自带协议级原子性（单命令粒度）。
    """

    def __init__(self) -> None:
        self._d: Dict[str, tuple[str, float]] = {}
        self._lock = threading.RLock()

    def set(self, key: str, value: Any, ttl: int = 0) -> None:
        raw = json.dumps(value, ensure_ascii=False, default=str)
        exp = time.time() + ttl if ttl else 0
        with self._lock:
            self._d[key] = (raw, exp)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._d.get(key)
            if not item:
                return None
            raw, exp = item
            if exp and time.time() > exp:
                self._d.pop(key, None)
                return None
        # json 解析放锁外：大 payload 不拖慢其它线程
        try:
            return json.loads(raw)
        except Exception:
            return None

    def delete(self, key: str) -> None:
        with self._lock:
            self._d.pop(key, None)


class RedisStore:
    def __init__(self, url: str) -> None:
        import redis  # type: ignore

        # 懒连接客户端：构造不报错 ≠ 服务可达，必须显式探活；
        # 否则坏地址会静默假装 Redis 正常，首次写数据时才炸且无降级
        self.r = redis.Redis.from_url(url, decode_responses=True,
                                      socket_connect_timeout=2.0,
                                      socket_timeout=2.0)
        self.r.ping()

    def set(self, key: str, value: Any, ttl: int = 0) -> None:
        raw = json.dumps(value, ensure_ascii=False, default=str)
        if ttl:
            self.r.setex(key, ttl, raw)
        else:
            self.r.set(key, raw)

    def get(self, key: str) -> Optional[Any]:
        raw = self.r.get(key)
        return json.loads(raw) if raw else None

    def delete(self, key: str) -> None:
        self.r.delete(key)


class ShortTermMemory:
    """统一入口；Redis 不可用时自动回退内存，绝不因缓存故障中断主流程。"""

    def __init__(self) -> None:
        self.ttl = int(cfg.get("redis.session_ttl", 86400))
        self.backend: Any
        if cfg.get("redis.enabled", False):
            try:
                self.backend = RedisStore(cfg.get("redis.url", "redis://localhost:6379/0"))
                self.name = "redis"
                return
            except Exception as e:
                print(f"[short_term] Redis 不可用（{e}），回退内存存储")
        self.backend = InMemoryStore()
        self.name = "memory"

    def save_session(self, user_id: str, payload: Dict[str, Any]) -> None:
        self.backend.set(f"math:session:{user_id}", payload, self.ttl)

    def load_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.backend.get(f"math:session:{user_id}")

    def clear_session(self, user_id: str) -> None:
        self.backend.delete(f"math:session:{user_id}")


__all__ = ["ShortTermMemory", "InMemoryStore", "RedisStore"]
