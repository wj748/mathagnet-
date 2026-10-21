"""向量库：双 Collection 隔离（方案 4.1）。

* ``knowledge_base``        —— 题库 / 概念知识，全局共享，只允许检索。
* ``user_long_term_memory`` —— 用户错题习惯 / 历史报告。

**隔离硬约束**：``search_user_memory`` 在代码层强制注入 ``user_id`` 过滤条件，
即便调用方忘记传（或传空），也会抛出错误而不是退化为"全库检索"。
这是方案"风险表 - 长期记忆越权串扰（高）"对应的代码级防线。
"""
from __future__ import annotations

import atexit
import time
import uuid
from typing import Any, Dict, List, Optional

from .config import cfg, resolve_path
from .embeddings import embed_one, embed_texts, current_dim


class IsolationError(RuntimeError):
    """长期记忆检索缺少 user_id 时抛出，拒绝降级为全库检索。"""


_CLEANUP_REGISTRY: List[Any] = []


def _register_cleanup(store: "VectorStore") -> None:
    """登记实例，进程退出前统一 close，规避 qdrant __del__ 的 ImportError。"""
    _CLEANUP_REGISTRY.append(store)


def _cleanup_all() -> None:
    while _CLEANUP_REGISTRY:
        try:
            _CLEANUP_REGISTRY.pop().close()
        except Exception:
            pass
    global _store
    _store = None


atexit.register(_cleanup_all)


class VectorStore:
    def __init__(self, client=None, dim: int | None = None):
        self.dim = dim or current_dim()
        self.kb = cfg.get("vectorstore.collection_knowledge", "knowledge_base")
        self.mem = cfg.get("vectorstore.collection_memory", "user_long_term_memory")
        self.client = client or self._make_client()
        self._ensure()
        _register_cleanup(self)

    # ------------------------------------------------------------ 生命周期
    def close(self) -> None:
        """显式释放本地嵌入式 Qdrant，避免解释器退出时的 __del__ 噪声。"""
        try:
            cl = getattr(self, "client", None)
            if cl is not None:
                cl.close()
                self.client = None
        except Exception:
            pass

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):                      # 兜底：解释器退出时不再抛异常
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------ 连接
    @staticmethod
    def _search(client, collection_name: str, query_vector, limit: int,
                query_filter=None, with_payload: bool = True):
        """兼容新旧 qdrant-client：>=1.12 用 query_points，旧版用 search。"""
        if hasattr(client, "query_points"):
            resp = client.query_points(collection_name=collection_name,
                                       query=query_vector, limit=limit,
                                       query_filter=query_filter,
                                       with_payload=with_payload)
            return getattr(resp, "points", resp)
        return client.search(collection_name=collection_name,
                             query_vector=query_vector, limit=limit,
                             query_filter=query_filter, with_payload=with_payload)

    def _make_client(self):
        from qdrant_client import QdrantClient

        backend = cfg.get("vectorstore.backend", "qdrant_local")
        if backend == "qdrant_server":
            return QdrantClient(host=cfg.get("vectorstore.host", "localhost"),
                                port=int(cfg.get("vectorstore.port", 6333)))
        path = resolve_path(cfg.get("vectorstore.path", "data/qdrant"))
        path.mkdir(parents=True, exist_ok=True)
        # 本地嵌入式模式：无需 Docker，数据落盘
        return QdrantClient(path=str(path))

    def _ensure(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self.client.get_collections().collections}
        for name in (self.kb, self.mem):
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                )

    # ------------------------------------------------------------ 写入
    @staticmethod
    def _pid() -> str:
        return str(uuid.uuid4())

    def upsert_knowledge(self, chunks: List[Dict[str, Any]], batch: int = 128) -> int:
        """chunks: [{text, metadata...}]"""
        from qdrant_client.models import PointStruct

        if not chunks:
            return 0
        vectors = embed_texts([c["text"] for c in chunks])
        points = []
        for c, v in zip(chunks, vectors):
            # text 必须入库：检索结果要直接用于 RAG 作答与展示
            payload = dict(c)
            payload.setdefault("created_at", time.time())
            points.append(PointStruct(id=self._pid(), vector=v, payload=payload))
        for i in range(0, len(points), batch):
            self.client.upsert(collection_name=self.kb, points=points[i:i + batch])
        return len(points)

    def upsert_memory(self, user_id: str, records: List[Dict[str, Any]]) -> int:
        """写入长期记忆；user_id 是强制字段，写入时统一打标。"""
        from qdrant_client.models import PointStruct

        if not records:
            return 0
        if not user_id:
            raise IsolationError("写入长期记忆必须携带 user_id")
        vectors = embed_texts([r["content"] for r in records])
        points = []
        for r, v in zip(records, vectors):
            payload = dict(r)
            payload["user_id"] = user_id
            payload.setdefault("type", "habit")
            payload.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
            points.append(PointStruct(id=self._pid(), vector=v, payload=payload))
        self.client.upsert(collection_name=self.mem, points=points)
        return len(points)

    # ------------------------------------------------------------ 检索
    def search_knowledge(self, query: str, top_k: int = 5,
                         topic: str | None = None,
                         score_threshold: float | None = None) -> List[Dict[str, Any]]:
        """检索知识库。score_threshold 过滤低相关片段（默认读 config
        ``retrieval.score_threshold``，与 memory_tools/grading 共用同一配置）。"""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        flt = None
        if topic:
            flt = Filter(must=[FieldCondition(key="topic", match=MatchValue(value=topic))])
        hits = self._search(self.client, self.kb, embed_one(query), top_k,
                            query_filter=flt)
        thr = float(score_threshold if score_threshold is not None
                    else cfg.get("retrieval.score_threshold", 0.10))
        return [{"score": h.score, **(h.payload or {})} for h in hits
                if float(h.score) >= thr]

    def search_user_memory(self, query: str, user_id: str,
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """强制 user_id 过滤——这是记忆隔离的唯一入口。"""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not user_id or not str(user_id).strip():
            raise IsolationError("search_user_memory 必须提供非空 user_id（记忆隔离硬约束）")
        hits = self._search(
            self.client, self.mem, embed_one(query), top_k,
            query_filter=Filter(must=[FieldCondition(key="user_id",
                                                     match=MatchValue(value=user_id))]),
        )
        return [{"score": h.score, **(h.payload or {})} for h in hits]

    # ------------------------------------------------------------ 运维
    def counts(self) -> Dict[str, int]:
        out = {}
        for name in (self.kb, self.mem):
            try:
                out[name] = self.client.count(collection_name=name, exact=True).count
            except Exception:
                out[name] = 0
        return out

    def clear(self, collection: str | None = None) -> None:
        if collection:
            self.client.delete_collection(collection)
        else:
            for name in (self.kb, self.mem):
                try:
                    self.client.delete_collection(name)
                except Exception:
                    pass
        self._ensure()

    def delete_memory(self, user_id: str, content: str) -> int:
        """按 user_id + content 精确删除记忆（用于清理审计探针）。"""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not user_id:
            raise IsolationError("删除长期记忆必须携带 user_id")
        self.client.delete(
            collection_name=self.mem,
            points_selector=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="content", match=MatchValue(value=content)),
            ]),
        )
        return 1

    def audit_isolation(self, user_a: str = "u_audit_a", user_b: str = "u_audit_b") -> Dict[str, Any]:
        """隔离审计：写入 A 的私有记忆，B 检索不得命中；结束后自清理探针。"""
        probe = f"___isolation_probe___专有记忆内容_{uuid.uuid4().hex[:8]}"
        try:
            self.upsert_memory(user_a, [{"content": probe, "type": "habit",
                                         "source": "audit"}])
            hits_a = self.search_user_memory(probe, user_a, top_k=5)
            hits_b = self.search_user_memory(probe, user_b, top_k=5)
            leaked = [h for h in hits_b if h.get("content") == probe]
            return {
                "passed": bool(hits_a) and not leaked,
                "a_hits": len(hits_a),
                "b_leaked": len(leaked),
            }
        except Exception as e:
            return {"passed": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            try:                              # 探针不得污染真实记忆
                self.delete_memory(user_a, probe)
            except Exception:
                pass


_store: VectorStore | None = None


def get_store(rebuild: bool = False) -> VectorStore:
    global _store
    if _store is None or rebuild:
        _store = VectorStore()
    return _store


__all__ = ["VectorStore", "get_store", "IsolationError"]
