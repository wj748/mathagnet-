"""Embedding 层：真实中文语义向量 + 零依赖兜底。

后端优先级（backend=auto）
-------------------------
1. ``bge``     —— BAAI/bge-small-zh-v1.5（从 ModelScope 下载，见 scripts/download_embedding_model.py）
                  需要 transformers + torch(CPU)。
2. ``hashing`` —— 中文字符 uni/bi-gram 加权哈希向量，零依赖、确定性。
                  语义表达弱于 BGE，但足以支撑"信息增益判定"与关键词检索，
                  保证无网 / 无 torch 环境下整条链路依然可运行。
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List, Sequence

from .config import cfg, resolve_path

_DIM_FALLBACK = 512


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def l2_normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


_TOKEN = re.compile(r"[a-zA-Z]+|\d+\.?\d*|[\u4e00-\u9fff]")

# 中文高频虚词/常用字：参与向量的话会让任意两句的余弦相似度虚高到 0.9+，
# 直接抹掉语义分块与信息增益判定的区分度，因此做停用处理。
_COMMON_CHARS = set(
    "的一是不了在人有我他这个们来上下大小多少和与对为以会要就都也很再又及其中于之则而且被把从向"
    "吗呢吧啊哦嗯于将如若但或所很更最还只又再没无能让使得被由到过给等于求解设"
)


def _is_common(g: str) -> bool:
    return len(g) > 0 and all(c in _COMMON_CHARS for c in g)


def _ngrams(text: str) -> List[str]:
    """中文按字、英文数字按词，取 uni/bi-gram，并过滤纯高频字组合。"""
    text = (text or "").lower()
    toks = _TOKEN.findall(text)
    grams: List[str] = []
    for t in toks:
        if not (_CJK_ONLY(t) and _is_common(t)):
            grams.append(t)
    for i in range(len(toks) - 1):
        g = toks[i] + toks[i + 1]
        if not _is_common(g):
            grams.append(g)
    return grams or [text]


def _CJK_ONLY(t: str) -> bool:
    return bool(t) and all("\u4e00" <= c <= "\u9fff" for c in t)


# --------------------------------------------------------------------------
# Backend: hashing
# --------------------------------------------------------------------------
class HashingEmbedder:
    """确定性哈希向量（subword TF → 哈希 → 加权 → L2 归一）。"""

    name = "hashing"
    dim = _DIM_FALLBACK

    def __init__(self, dim: int = _DIM_FALLBACK):
        self.dim = dim

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            grams = _ngrams(t or "")
            for g in grams:
                h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                vec[idx] += sign * (1.0 / (1 + len(g) * 0.3))
            out.append(l2_normalize(vec))
        return out

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0]


# --------------------------------------------------------------------------
# Backend: BGE (transformers)
# --------------------------------------------------------------------------
class BGEEmbedder:
    name = "bge"
    dim = 512

    def __init__(self, model_dir: str, max_length: int = 512, batch_size: int = 32,
                 device: str = "cpu"):
        import torch  # noqa: F401  (延迟导入，缺 torch 时不影响其它后端)
        from transformers import AutoModel, AutoTokenizer

        path = resolve_path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"BGE 模型目录不存在：{path}，请先运行 scripts/download_embedding_model.py")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModel.from_pretrained(str(path))
        self.model.eval()
        self.device = device
        self.model.to(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.dim = int(getattr(self.model.config, "hidden_size", 512))

    @staticmethod
    def _pool(last_hidden, attention_mask):
        import torch

        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = (last_hidden * mask).sum(dim=1)
        counted = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counted

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        import torch

        texts = [t or "" for t in texts]
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out_t = self.model(**enc).last_hidden_state
            pooled = self._pool(out_t, enc["attention_mask"])
            for row in pooled.cpu().numpy():
                out.append(l2_normalize([float(x) for x in row]))
        return out

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0]


# --------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------
_embedder = None


def get_embedder(force: str | None = None):
    """按配置返回全局单例 embedder。"""
    global _embedder
    if _embedder is not None and force is None:
        return _embedder

    backend = (force or cfg.get("embedding.backend", "auto")).lower()
    model_dir = cfg.get("embedding.model_dir", "models/bge-small-zh-v1.5")

    if backend in ("auto", "bge"):
        try:
            _embedder = BGEEmbedder(
                model_dir=model_dir,
                max_length=int(cfg.get("embedding.max_length", 512)),
                batch_size=int(cfg.get("embedding.batch_size", 32)),
                device=cfg.get("embedding.device", "cpu"),
            )
            return _embedder
        except Exception as e:  # 缺 torch / 模型未下载
            if backend == "bge":
                raise
            print(f"[embeddings] BGE 不可用（{type(e).__name__}: {e}），回退到 hashing 向量")

    _embedder = HashingEmbedder()
    return _embedder


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    return get_embedder().encode(texts)


def embed_one(text: str) -> List[float]:
    return get_embedder().encode_one(text)


def current_dim() -> int:
    return int(getattr(get_embedder(), "dim", _DIM_FALLBACK))


def peek_embedder():
    """返回已加载的 embedder 单例；未加载则返回 None（不触发模型加载）。

    供 /api/health 这类轻量接口使用，避免健康检查被 BGE 首加载阻塞数十秒。
    """
    return _embedder


__all__ = ["get_embedder", "peek_embedder", "embed_texts", "embed_one",
           "cosine_similarity", "l2_normalize", "current_dim",
           "HashingEmbedder", "BGEEmbedder"]
