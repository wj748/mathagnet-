"""ReAct Executor + 双重循环终止控制（方案 4.3）。

终止条件（代码层硬约束，不依赖 LLM 自觉）
----------------------------------------
1. **最大循环次数**：``react_loop_count >= max_react_loops``(默认 8) → 强制输出。
2. **信息增益判定**：本轮 Thought 向量与上一轮余弦相似度 ≥ 0.80（思维停滞）
   → 立即停止并输出当前结论。

``loop_stop_reason`` 会被写入 trace 与响应，便于线上监控"循环终止原因分布"。
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .. import metrics
from ..config import cfg
from ..embeddings import cosine_similarity, embed_one
from ..llm import get_llm
from ..state import MathAgentState
from ..toolguard import (classify_failure, ledger_create, ledger_finish,
                         ledger_record)
from ..tools.registry import args_hash, record_history, run_tool

_EXPR = re.compile(r"[\d\)\]]\s*[\+\-\*/x×÷]\s*[\d\(]|solve\s*\(|=\s*\?")


def _ctx_block(digest: str) -> str:
    """把压缩对话上下文拼进 prompt；空 digest 直接跳过。"""
    digest = (digest or "").strip()
    return f"【此前对话要点（已压缩）】\n{digest}\n" if digest else ""


# ------------------------------------------------------------------ Thought
def _generate_thought(user_input: str, obs: List[Dict[str, Any]], n: int,
                      digest: str = "") -> str:
    llm = get_llm()
    if llm.available:
        prompt = (
            "你是一个使用 ReAct 框架的初中数学辅导 Agent。\n"
            f"{_ctx_block(digest)}"
            f"用户问题：{user_input}\n"
            f"已获得的观察（Observation）：\n{json.dumps(obs[-3:], ensure_ascii=False)}\n"
            f"当前是第 {n + 1} 轮。请只输出本轮的 Thought（不超过 60 字），"
            "说明你接下来要做什么或是否已经可以回答。不要输出 Action。"
        )
        t = llm.chat(prompt, temperature=0.2)
        if t:
            return t.strip()
    # 规则式 Thought（LLM 不可用时的确定性轨迹）
    if n == 0:
        return f"Thought {n + 1}：用户问「{user_input[:40]}」，先检索知识库获取相关题目与概念。"
    if n == 1:
        return f"Thought {n + 1}：已检索到资料，检查是否涉及数值计算，必要时用计算器验算。"
    if n == 2:
        return f"Thought {n + 1}：结合检索结果与计算过程，组织最终答案。"
    return f"Thought {n + 1}：信息已足够，可以给出结论。"


# ------------------------------------------------------------------ Action
def _called(history: List[Dict[str, Any]], name: str, args: Dict[str, Any]) -> bool:
    return any(h.get("args_hash") == args_hash(name, args) for h in history)


def _decide_action(user_input: str, obs: List[Dict[str, Any]], n: int,
                   history: List[Dict[str, Any]],
                   digest: str = "") -> Optional[Tuple[str, Dict[str, Any]]]:
    """决定本轮 Action；返回 None 表示可以结束。"""
    llm = get_llm()
    if llm.available:
        from ..tools.registry import list_tools
        tools_desc = "\n".join(f"- {t['name']}: {t['description']}" for t in list_tools())
        prompt = (
            "你是 ReAct Agent，只能从下列工具中选择，或输出 finish。\n"
            f"工具：\n{tools_desc}\n\n"
            f"{_ctx_block(digest)}"
            f"用户问题：{user_input}\n"
            f"已获得观察：{json.dumps(obs[-4:], ensure_ascii=False)}\n"
            '只输出 JSON：{"action": "finish"} 或 {"action": "tool", "name": "...", "args": {...}}'
        )
        data = llm.chat_json(prompt, fallback=None)
        if isinstance(data, dict):
            if data.get("action") == "finish":
                return None
            if data.get("action") == "tool" and data.get("name"):
                return data["name"], dict(data.get("args") or {})

    # 规则式计划
    if n == 0:
        args = {"query": user_input[:200], "top_k": 5}
        return None if _called(history, "search_knowledge", args) else ("search_knowledge", args)
    if n == 1 and _EXPR.search(user_input or ""):
        expr = (user_input or "").strip()
        args = {"expression": expr[:120]}
        return None if _called(history, "calculator", args) else ("calculator", args)
    if n == 2:
        kp = ""
        for o in obs:
            if o.get("tool") == "search_knowledge":
                try:
                    payload = json.loads(o["observation"])
                    kp = (payload.get("results") or [{}])[0].get("topic", "")
                except Exception:
                    pass
        if kp:
            args = {"query": f"{user_input[:80]} {kp}", "top_k": 5, "topic": kp}
            return None if _called(history, "search_knowledge", args) else ("search_knowledge", args)
    return None


# ------------------------------------------------------------------ Answer
def _compose_answer(user_input: str, obs: List[Dict[str, Any]],
                    stop_reason: str, digest: str = "") -> str:
    facts: List[str] = []
    for o in obs:
        if o.get("tool") == "search_knowledge":
            try:
                payload = json.loads(o["observation"])
                for r in (payload.get("results") or [])[:3]:
                    if r.get("text"):
                        facts.append(r["text"][:400])
            except Exception:
                pass
        elif o.get("tool") == "calculator":
            try:
                payload = json.loads(o["observation"])
                facts.append(f"计算结果：{payload.get('result')}")
            except Exception:
                pass

    llm = get_llm()
    if llm.available:
        prompt = (
            "你是初中数学辅导老师，用简洁、鼓励的口吻回答（300 字以内）。\n"
            f"{_ctx_block(digest)}"
            f"学生问题：{user_input}\n"
            f"参考资料：\n" + "\n---\n".join(facts[:4]) + "\n请作答："
        )
        t = llm.chat(prompt)
        if t:
            return _suffix(t, stop_reason)

    if facts:
        body = "📖 我在知识库里找到这些内容：\n\n" + "\n\n".join(
            f"· {f}" for f in facts[:3])
        body += "\n\n（想看更详细的讲解，可以告诉我具体哪个步骤不懂～）"
    else:
        body = ("🤔 知识库里暂时没有找到直接相关的资料。\n"
                "你可以换个说法，或者直接开始刷题（例如：「我要刷函数的选择题」）。")
    return _suffix(body, stop_reason)


def _suffix(text: str, stop_reason: str) -> str:
    if stop_reason == "information_gain":
        return text + "\n\n（注：检测到思考已收敛，直接给出当前最优结论）"
    if stop_reason == "max_loop":
        return text + "\n\n（注：已达到最大思考轮次，基于已有信息给出回答）"
    return text


# ------------------------------------------------------------------ 节点
def react_step_node(state: MathAgentState) -> Dict[str, Any]:
    n = int(state.get("react_loop_count", 0))
    max_loops = int(cfg.get("graph.max_react_loops", 8))
    thr = float(cfg.get("graph.info_gain_threshold", 0.80))
    max_replans = int(cfg.get("toolguard.max_replans", 2))

    user_input = state.get("user_input", "") or ""
    digest = state.get("conversation_digest") or ""
    thoughts: List[str] = list(state.get("react_thoughts") or [])
    history: List[Dict[str, Any]] = list(state.get("tool_call_history") or [])
    stm: Dict[str, Any] = dict(state.get("short_term_memory") or {})
    obs: List[Dict[str, Any]] = list(stm.get("observations") or [])
    trace = list(state.get("trace", []))

    # 0) 任务标识 + 步骤台账（toolguard：task_id/step_id 绑定，断点续跑依据）
    task_id = state.get("task_id") or uuid.uuid4().hex[:12]
    if n == 0:
        ledger_create(task_id)

    # 1) Thought
    thought = _generate_thought(user_input, obs, n, digest)
    vec = embed_one(thought)
    last = state.get("last_result_vector")

    # 2) 双重终止判定
    stop: Optional[str] = None
    if last is not None and cosine_similarity(vec, last) >= thr:
        stop = "information_gain"
    elif n + 1 >= max_loops:
        stop = "max_loop"

    thoughts.append(thought)
    trace.append(f"react[{n + 1}]{':stop=' + stop if stop else ''}")
    patch: Dict[str, Any] = {
        "task_id": task_id,
        "react_loop_count": n + 1,
        "react_thoughts": thoughts,
        "last_result_vector": vec,
        "loop_stop_reason": stop,
        "tool_call_history": history,
        "short_term_memory": {**stm, "observations": obs},
        "trace": trace,
    }
    if stop:
        metrics.record_loop_stop(stop)
        ledger_finish(task_id, f"stopped:{stop}")
        patch["response"] = _compose_answer(user_input, obs, stop, digest)
        return patch

    # 3) Action：优先执行上轮失败后的「备选工具重规划」，否则正常决策
    action: Optional[Tuple[str, Dict[str, Any]]] = None
    forced = state.get("react_forced_action")
    if forced and len(forced) == 2 and not _called(history, forced[0], forced[1]):
        fb_args = dict(forced[1])
        fb_args.setdefault("user_id", state.get("user_id", "u_default"))
        action = (forced[0], fb_args)
        metrics.record_toolguard("fallback", forced[0])
    if action is None:
        action = _decide_action(user_input, obs, n, history, digest)
    patch["react_forced_action"] = None            # 用后即清（无论来源）
    if action is None:
        metrics.record_loop_stop("finished")
        ledger_finish(task_id, "finished")
        patch["loop_stop_reason"] = "finished"
        patch["response"] = _compose_answer(user_input, obs, "finished", digest)
        return patch

    name, args = action
    res = run_tool(name, args, history=history)
    record_history(history, name, args, res)
    obs.append({"tool": name, "args": args, "observation": res.to_observation(),
                "ok": res.ok, "blocked": res.blocked})
    trace.append(f"react[{n + 1}]tool={name}:"
                 + ("ok" if res.ok else ("blocked" if res.blocked else "fail")))
    step_id = f"{task_id}-s{n + 1}-{name}"
    ledger_record(task_id, step_id, name,
                  getattr(res, "args_hash", "") or args_hash(name, args),
                  res.ok, digest=(res.to_observation()[:200] if res.ok else res.error[:200]))
    patch["short_term_memory"] = {**stm, "observations": obs}
    patch["tool_call_history"] = history
    patch["tool_blocked_notice"] = res.error if (not res.ok) else None

    # 4) 失败治理：分类 → 重规划 / 降级（toolguard 规范 §3/§4）
    if not res.ok:
        # blocked（同参重复/依赖链拦截）与参数错误同类：重试无意义，
        # 计入重规划上限，避免 LLM 反复同参空转浪费轮次
        cls = "param" if res.blocked else classify_failure(res.error)
        replans = int(state.get("tool_replan_count", 0) or 0)
        if cls == "fatal":
            # 致命错误（鉴权/配额）：不重试不重规划，降级输出已获取的部分结果
            metrics.record_toolguard("fatal_stop", name)
            metrics.record_loop_stop("fatal_error")
            ledger_finish(task_id, f"fatal:{name}")
            patch["loop_stop_reason"] = "fatal_error"
            patch["response"] = _compose_answer(user_input, obs, "fatal_error", digest)
            return patch
        if replans >= max_replans:
            # 重规划轮次用尽：返回部分结果（防死循环，代码层硬约束）
            metrics.record_loop_stop("degraded")
            ledger_finish(task_id, f"degraded:{name}")
            patch["loop_stop_reason"] = "degraded"
            patch["response"] = _compose_answer(user_input, obs, "degraded", digest)
            return patch
        # 重规划：错误信息已在 observation 里回注给 LLM（param 类可修正参数/换方案）；
        # 规则层兜底：配置了同能力备选工具则下一轮强制换用
        metrics.record_toolguard("replan", name)
        patch["tool_replan_count"] = replans + 1
        fb_map = cfg.get("toolguard.fallback_tools") or {}
        fb = fb_map.get(name)
        if fb:
            patch["react_forced_action"] = [fb, dict(args)]
    return patch


def should_continue_react(state: MathAgentState) -> str:
    if state.get("loop_stop_reason"):
        return "finalize"
    return "react"


__all__ = ["react_step_node", "should_continue_react"]
