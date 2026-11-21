"""多轮对话的分层记忆：存储 → 压缩 → 释放（记忆优化模块）。

解决的问题
----------
多路/多轮对话后：无界的工具观察与调用历史随 checkpointer 无限累积
（模型思考质量变弱、token 消耗持续变大），而真正有价值的对话上下文
反而没有结构化沉淀。

核心策略（代码层硬约束，离线可用；LLM 仅可选润色）
--------------------------------------------------
1. **实体锚（永不释放）**：板块、题型、题目 ID、数学概念、当前刷题进度。
   只要"还相关"，就从原始轮次提升到实体池，与具体轮次解耦存储。
2. **近期原文窗口**：最近 N 轮对话逐字保留（当前话题的完整语境）。
3. **历史要点压缩**：滑出窗口但仍相关的旧轮次 → 压缩为一行要点
   （意图 + 实体 + 结论摘要），超过摘要容量后按新旧淘汰。
4. **无关内容释放**：既无实体又非任务性意图（quiz/submit/grade/habit）
   的闲聊轮次，滑出窗口后**整轮释放**，不进摘要、不占预算。
5. **state 瘦身**：``prune_state`` 在每轮开始前裁剪 LangGraph 状态——
   旧 ``tool_call_history`` 压缩为防重/依赖链判定所需的最小字段
   （name + args_hash + ok，语义不变），``observations`` 只留滑动窗口，
   已交卷会话释放整包 questions（保留 records 等实体内容）。

防重语义保持：压缩后的 history 记录仍带 ``args_hash`` 与 ``ok``，
工具防重（Phase 3 硬约束）与依赖链检查行为完全不变。
"""
from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import cfg

# ---------------------------------------------------------------- 实体抽取
_TOPIC_PAT = re.compile(
    r"函数|方程|不等式|几何|三角形|四边形|圆|全等|相似|勾股|平行|垂直|"
    r"对称|平移|旋转|面积|周长|扇形|整式|分式|因式分解|幂|实数|有理数|"
    r"绝对值|坐标系|坐标|正比例|反比例|一次函数|二次函数|抛物线|斜率|"
    r"象限|统计|概率|平均数|中位数|众数|方差|应用题")
_QID_PAT = re.compile(r"\b(?:TAL|GEO|ALG|hw)[_\-]?[A-Za-z0-9]{3,}")
_INTENT_LABEL = {"quiz": "刷题", "submit": "交卷", "grade": "批改",
                 "habit": "习惯查询", "chat": "答疑", "": "对话"}
# 这些意图天然与"任务/实体"相关，即便没抽到实体也不得整轮释放
_TASK_INTENTS = {"quiz", "submit", "grade", "habit"}


def est_tokens(text: str) -> int:
    """粗估 token 数（中英混合 ≈ 0.7 字符/token），仅用于预算控制。"""
    return max(1, int(len(text or "") * 0.7))


@dataclass
class Turn:
    role: str                 # user | assistant
    content: str
    intent: str = ""
    entities: List[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    @property
    def relevant(self) -> bool:
        """是否为"实体相关"内容：决定滑出窗口后是压缩保留还是整轮释放。"""
        return bool(self.entities) or self.intent in _TASK_INTENTS


def _extract_entities(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    out.extend(_QID_PAT.findall(text))
    for m in _TOPIC_PAT.finditer(text):
        w = m.group(0)
        if w not in out:
            out.append(w)
    return out[:8]


# ---------------------------------------------------------------- 压缩器
class ConversationCompressor:
    """每个用户一份；原始轮次有界存储，上下文按预算动态压缩。"""

    def __init__(self, user_id: str = "u_default"):
        self.user_id = user_id
        self.turns: List[Turn] = []
        # 实体池：entity -> 最近出现时间；容量有界（LRU 淘汰）
        self.entity_pool: "OrderedDict[str, float]" = OrderedDict()
        self.released_turns = 0        # 已整轮释放的无关轮次
        self.compressed_turns = 0      # 已压缩为要点的轮次
        self.version = 0               # 会话版本戳：每次 _save 自增（并发写冲突可观测）
        self._load()

    # ---------------- 配置 ----------------
    @property
    def _budget(self) -> int:
        return int(cfg.get("memory.context_budget", 1600))

    @property
    def _recent_n(self) -> int:
        return max(2, int(cfg.get("memory.recent_turns", 6)))

    @property
    def _max_summary(self) -> int:
        return max(0, int(cfg.get("memory.max_summary_lines", 12)))

    @property
    def _max_entities(self) -> int:
        return max(0, int(cfg.get("memory.max_entities", 10)))

    @property
    def _max_turns(self) -> int:
        return max(self._recent_n * 2, int(cfg.get("memory.max_turns_kept", 60)))

    # ---------------- 写入 ----------------
    def add_turn(self, role: str, content: str, intent: str = "") -> None:
        content = (content or "").strip()
        if not content:
            return
        ents = _extract_entities(content)
        self.turns.append(Turn(role=role, content=content,
                               intent=intent or "", entities=ents))
        for e in ents:
            self.entity_pool[e] = time.time()
            self.entity_pool.move_to_end(e)
        while len(self.entity_pool) > max(self._max_entities * 4, 24):
            self.entity_pool.popitem(last=False)
        # 原始轮次有界：最老的先淘汰（其价值已沉淀进实体池/摘要）
        while len(self.turns) > self._max_turns:
            self.turns.pop(0)
        self._save()

    # ---------------- 读取（压缩上下文） ----------------
    def build_context(self, active_task: str = "") -> str:
        budget = self._budget
        lines: List[str] = []

        # 1) 实体锚（与轮次解耦，永不释放的"骨架"）
        ents = list(self.entity_pool.keys())[-self._max_entities:]
        if ents:
            lines.append("【已确认实体】" + "、".join(ents))
        if active_task:
            lines.append(f"【进行中】{active_task}")

        # 2) 近期原文窗口（从最新往回收集，受预算约束）
        recent: List[Turn] = []
        spent = est_tokens("\n".join(lines))
        for t in reversed(self.turns):
            if len(recent) >= self._recent_n:
                break
            cost = est_tokens(t.content)
            if spent + cost > budget and recent:
                break
            recent.append(t)
            spent += cost
        recent.reverse()

        # 3) 窗口外的旧轮次：相关 → 压缩要点；无关 → 释放
        window_ids = {id(t) for t in recent}
        summary: List[str] = []
        for t in self.turns:
            if id(t) in window_ids:
                continue
            if t.relevant:
                if len(summary) < self._max_summary:
                    ent = "·".join(t.entities[:3]) if t.entities else _INTENT_LABEL.get(t.intent, "")
                    snippet = t.content[:60].replace("\n", " ")
                    summary.append(f"- [{_INTENT_LABEL.get(t.intent, '对话')}]"
                                   f"{ent + '：' if ent else ''}{snippet}")
                self.compressed_turns += 1
            else:
                self.released_turns += 1   # 无实体的闲聊：整轮释放

        if summary:
            lines.append("【此前要点】")
            lines.extend(summary)
        if recent:
            lines.append("【最近对话】")
            for t in recent:
                tag = "学生" if t.role == "user" else "老师"
                body = t.content if len(t.content) <= 300 else t.content[:300] + "…"
                lines.append(f"{tag}: {body}")

        _record_conv(turns=len(self.turns), released=self.released_turns,
                     compressed=self.compressed_turns,
                     digest_chars=len("\n".join(lines)))
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        return {"turns": len(self.turns), "entities": len(self.entity_pool),
                "released_turns": self.released_turns,
                "compressed_turns": self.compressed_turns}

    # ---------------- 持久化（经短期记忆后端，TTL 随会话） ----------------
    def _key(self) -> str:
        return f"math:conv:{self.user_id}"

    def _save(self) -> None:
        try:
            from .short_term import ShortTermMemory
            stm = _stm()
            self.version += 1          # 版本戳：同用户任务已被 UserGate 串行化，
            stm.backend.set(self._key(), self.to_dict(),   # 版本单调递增可校验乱序写
                            int(cfg.get("redis.session_ttl", 86400)))
        except Exception:
            pass

    def _load(self) -> None:
        try:
            from .short_term import ShortTermMemory
            data = _stm().backend.get(self._key())
            if data:
                self.from_dict(data)
        except Exception:
            pass

    def to_dict(self) -> Dict[str, Any]:
        return {"turns": [{"role": t.role, "content": t.content, "intent": t.intent,
                           "entities": t.entities, "ts": t.ts} for t in self.turns],
                "entity_pool": list(self.entity_pool.items())[-48:],
                "released": self.released_turns, "compressed": self.compressed_turns,
                "version": self.version}

    def from_dict(self, data: Dict[str, Any]) -> None:
        self.turns = [Turn(role=d.get("role", "user"), content=d.get("content", ""),
                           intent=d.get("intent", ""), entities=list(d.get("entities") or []),
                           ts=float(d.get("ts") or 0)) for d in (data.get("turns") or [])]
        self.entity_pool = OrderedDict((k, float(v)) for k, v in (data.get("entity_pool") or []))
        self.released_turns = int(data.get("released", 0))
        self.compressed_turns = int(data.get("compressed", 0))
        self.version = int(data.get("version", 0))


_stm_obj: Optional[Any] = None


def _stm() -> Any:
    global _stm_obj
    if _stm_obj is None:
        from .short_term import ShortTermMemory
        _stm_obj = ShortTermMemory()
    return _stm_obj


# ---------------------------------------------------------------- state 瘦身
def prune_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    """每轮 invoke 前裁剪 LangGraph 持久化状态，释放无关内容（硬约束）。

    * tool_call_history：超过阈值后，旧记录压缩为 {name, args_hash, ok}——
      工具防重（args_hash）与依赖链（name+ok）判定所需字段完整保留，语义不变。
    * short_term_memory.observations：只保留滑动窗口内的最新观察，
      旧观察的结论已沉淀进对话摘要。
    * quiz_session：已交卷（finished）的会话释放整包 questions，
      保留 records / topic / 正确率等实体内容。
    """
    hist = payload.get("tool_call_history") or []
    full_keep = max(4, int(cfg.get("memory.hist_full", 8)))
    compact_beyond = max(full_keep, int(cfg.get("memory.hist_compact_beyond", 24)))
    if len(hist) > compact_beyond:
        head = [{"name": h.get("name"), "args_hash": h.get("args_hash", ""),
                 "ok": bool(h.get("ok"))} for h in hist[:-full_keep]]
        # 条数硬上限：压缩后整体仍超过阈值时，最老的直接释放
        # （跨轮防重只保留近期窗口；更早的同参数调用允许重新执行）
        head = head[-(compact_beyond - full_keep):]
        payload["tool_call_history"] = head + list(hist[-full_keep:])
        _record_conv(hist_compacted=len(hist) - full_keep - len(head),
                     hist_compacted_kept=len(head))

    stm = payload.get("short_term_memory") or {}
    obs = stm.get("observations") or []
    window = max(2, int(cfg.get("memory.obs_window", 8)))
    if len(obs) > window:
        stm = {**stm, "observations": obs[-window:]}
        payload["short_term_memory"] = stm
        _record_conv(obs_released=len(obs) - window)

    sess = payload.get("quiz_session") or {}
    if sess.get("finished") and sess.get("questions"):
        payload["quiz_session"] = {k: v for k, v in sess.items() if k != "questions"} | {
            "questions_released": True}
        _record_conv(quiz_released=len(sess["questions"]))
    return payload


def active_task_of(payload: Dict[str, Any]) -> str:
    """从 payload 提取"进行中任务"锚点（永不释放的实体内容）。"""
    sess = payload.get("quiz_session") or {}
    if sess.get("questions") and not sess.get("finished"):
        idx = int(sess.get("index", 0))
        total = len(sess["questions"])
        return (f"{sess.get('topic', '')}·{sess.get('type', '')}"
                f" 第{min(idx + 1, total)}/{total}题进行中")
    return ""


# ---------------------------------------------------------------- 可观测性
def _record_conv(**kw: Any) -> None:
    try:
        from .. import metrics
        metrics.record_conversation(**kw)
    except Exception:
        pass


__all__ = ["ConversationCompressor", "Turn", "prune_state", "active_task_of",
           "est_tokens", "_extract_entities"]
