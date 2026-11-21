"""动态切片（方案 4.1）：语义相似度分块，替代固定 chunk_size。

三种策略
--------
1. ``chunk_question``  —— 题目：以"题干-选项-答案-解析"为**完整 Chunk**，绝不切断。
2. ``chunk_by_structure`` —— 教材：按"定义 / 定理 / 例题"等标题结构分块。
3. ``semantic_chunk``  —— 自由文本：计算相邻句 Embedding 相似度，低于阈值处切分。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from .config import cfg
from .embeddings import cosine_similarity, embed_texts

_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])")
_HEADING = re.compile(r"^\s*(定义|定理|性质|公式|例题|例\s*\d+|推论|法则|注意|小结)\s*[:：、.．]?",
                      re.M)


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [s for s in _SENT_SPLIT.split(text) if s and s.strip()]
    merged: List[str] = []
    for s in parts:
        if merged and len(merged[-1]) < 12:   # 过短句并入前一句
            merged[-1] += s
        else:
            merged.append(s)
    return merged


def semantic_chunk(text: str,
                   similarity_threshold: float | None = None,
                   max_chars: int | None = None,
                   min_chars: int | None = None) -> List[str]:
    """相邻句语义相似度 < 阈值 → 切分；同时受最大长度约束。"""
    thr = float(similarity_threshold if similarity_threshold is not None
                else cfg.get("chunking.similarity_threshold", 0.55))
    max_chars = int(max_chars if max_chars is not None
                    else cfg.get("chunking.max_chunk_chars", 900))
    min_chars = int(min_chars if min_chars is not None
                    else cfg.get("chunking.min_chunk_chars", 40))

    sents = split_sentences(text)
    if len(sents) <= 1:
        return [text] if text.strip() else []

    vecs = embed_texts(sents)
    chunks: List[str] = []
    buf = sents[0]
    for i in range(1, len(sents)):
        sim = cosine_similarity(vecs[i - 1], vecs[i])
        too_long = len(buf) + len(sents[i]) > max_chars
        if sim < thr or too_long:
            chunks.append(buf)
            buf = sents[i]
        else:
            buf += sents[i]
    chunks.append(buf)

    # 合并过短碎片
    final: List[str] = []
    for c in chunks:
        if final and len(c) < min_chars:
            final[-1] += c
        else:
            final.append(c)
    return [c.strip() for c in final if c.strip()]


def chunk_by_structure(text: str, max_chars: int = 900) -> List[str]:
    """教材：按"定义/定理/例题"等结构标记切分。"""
    text = (text or "").strip()
    if not text:
        return []
    positions = [m.start() for m in _HEADING.finditer(text)]
    if len(positions) <= 1:
        return semantic_chunk(text, max_chars=max_chars)
    blocks: List[str] = []
    for i, p in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        blocks.append(text[p:end].strip())
    out: List[str] = []
    for b in blocks:
        if len(b) > max_chars:
            out.extend(semantic_chunk(b, max_chars=max_chars))
        else:
            out.append(b)
    return [b for b in out if b]


def chunk_question(q: Dict[str, Any]) -> str:
    """题目 → 单条完整 Chunk（题干 + 选项 + 答案 + 解析）。"""
    lines = [f"【{q.get('topic', '数学')}·{q.get('type', '题目')}】{q.get('stem', '')}".strip()]
    for opt in q.get("options") or []:
        lines.append(f"  {opt}")
    if q.get("answer"):
        lines.append(f"答案：{q['answer']}")
    if q.get("analysis"):
        lines.append(f"解析：{q['analysis']}")
    if q.get("knowledge_points"):
        lines.append(f"知识点：{'、'.join(q['knowledge_points'])}")
    return "\n".join(lines)


__all__ = ["semantic_chunk", "chunk_by_structure", "chunk_question", "split_sentences"]
