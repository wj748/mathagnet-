# -*- coding: utf-8 -*-
"""为公网发布包构建「哈希向量」索引。

为什么不用本机现有的 BGE 索引：
  发布沙箱无 torch，BGEEmbedder 会抛 ImportError → get_embedder() 回退到
  HashingEmbedder。两者 dim 都是 512，Qdrant 不会报维度错，但向量空间完全不同，
  用哈希 query 去检索 BGE 索引 → 余弦相似度近乎随机 → 检索静默失效（不报错、召回归零）。
  因此发布包必须用「与沙箱一致的 embedder」重建索引。

数据源：data/math_agent.db 的 questions 表（= ETL 产物 to_records 的等价集）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 用环境变量把「embedder 类型」和「向量库路径」指到发布包
os.environ["MATH_AGENT_EMBEDDING__BACKEND"] = "hashing"
os.environ["MATH_AGENT_VECTORSTORE__PATH"] = "publish_qdrant_hashing"

from math_agent.chunking import chunk_question          # noqa: E402
from math_agent.db.models import get_session             # noqa: E402
from math_agent.db.models import Question                # noqa: E402
from math_agent.embeddings import get_embedder           # noqa: E402
from math_agent.vectorstore import get_store             # noqa: E402


def main() -> int:
    emb = get_embedder()
    print(f"embedder = {emb.name} (dim={emb.dim})")
    assert emb.name == "hashing", "发布包索引必须用 hashing 构建"

    sess = get_session()
    rows = sess.query(Question).all()
    print(f"题库 {len(rows)} 题")

    store = get_store()
    store.clear(store.kb)

    chunks = []
    for r in rows:
        d = r.to_dict()
        chunks.append({
            "text": chunk_question(d),
            "question_id": d["question_id"],
            "topic": d["topic"],
            "type": d["type"],
            "difficulty": d["difficulty"],
            "knowledge_points": d.get("knowledge_points") or [],
            "answer": d.get("answer") or "",
        })

    n = 0
    B = 128
    for i in range(0, len(chunks), B):
        n += store.upsert_knowledge(chunks[i:i + B])
        print(f"  upsert {min(i + B, len(chunks))}/{len(chunks)}", end="\r")
    print(f"\n✔ 写入向量 {n} 条 → {store.client.get_collections()}")

    # 立刻自测召回（用同一 embedder，验证索引与查询同空间）
    for q in ["一元二次方程怎么解", "一次函数图像怎么画", "三角形面积怎么算"]:
        hits = store.search_knowledge(q, top_k=3)
        print(f"\nquery: {q}")
        for h in hits:
            print(f"  score={h.get('score'):.4f}  {h.get('text','')[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
