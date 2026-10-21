"""知识点讲解 Agent 节点（学习模式）。

交互流程（讲练结合，不产出报告、不逐题点评）
------------------------------------------
   进入学习 → 选板块/知识点 → 讲解 → 例题 → 小测 → 判分讲评 → 下一个知识点
   指令：下一个/继续 · 没听懂/再讲一遍 · 换个例题 · A-D 作答 · 退出学习

会话状态存在 state["learning_session"]（跨轮由 compressor 的 carried 键保留）。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from ..state import MathAgentState
from ..teaching import (CURRICULUM, build_lecture, find_point_index, format_example,
                        get_point, is_again, is_example, is_exit, is_next,
                        is_option_answer, list_curriculum, match_topic,
                        pick_examples)
from ..tools.registry import record_history, run_tool


def _first_hint(analysis: str, kp: Dict[str, Any]) -> str:
    """取解析首句作为方向提示；无解析时退到知识点要点/口诀。"""
    a = (analysis or "").strip()
    if a:
        parts = re.split(r"(?<=[。；;!?])", a)
        return (parts[0] or a[:80]).strip()
    pts = kp.get("points") or []
    if pts:
        return str(pts[0])
    return str(kp.get("tip") or "")


def _new_session(topic: str, index: int) -> Dict[str, Any]:
    return {"topic": topic, "index": index, "stage": "teach",
            "asked_quiz": False, "pending_quiz": None, "correct": 0, "total": 0}


def _lecture_text(topic: str, kp: Dict[str, Any], style: str = "") -> str:
    return build_lecture(topic, kp, style)["text"]


def _start(topic: str, index: int, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    kp = get_point(topic, index)
    if kp is None:
        return {"learning_session": None, "awaiting_user": True,
                "response": list_curriculum(), "trace": ["learn:catalog"]}
    text = _lecture_text(topic, kp)
    examples = pick_examples(topic, 1)
    sess = _new_session(topic, index)
    body = [text, "", "👉 学会了吗？回复「继续」进入下一个知识点，"
                      "「没听懂」我换个讲法，「换个例题」再练一题，或提出你的疑问。"]
    if examples:
        q = examples[0]
        sess["pending_quiz"] = q
        sess["asked_quiz"] = True
        sess["stage"] = "quiz"
        body += ["", "小试一下：", format_example(q)]
    return {"learning_session": sess, "awaiting_user": True,
            "response": "\n".join(body), "tool_call_history": history,
            "trace": [f"learn:teach:{kp['name']}"]}


def knowledge_tutor_node(state: MathAgentState) -> Dict[str, Any]:
    text = (state.get("user_input") or "").strip()
    user_id = state.get("user_id", "u_default")
    history: List[Dict[str, Any]] = list(state.get("tool_call_history") or [])
    trace: List[str] = list(state.get("trace") or [])
    sess: Dict[str, Any] = dict(state.get("learning_session") or {})
    mode = state.get("session_mode", "")

    # 1) 退出学习
    if sess and is_exit(text):
        done = f"📕 本次学习了【{sess.get('topic', '')}】，完成小测 {sess.get('total', 0)} 题" \
               f"（答对 {sess.get('correct', 0)} 题）。随时说「学知识点」可以继续。"
        return {"learning_session": None, "awaiting_user": False,
                "response": done, "tool_call_history": history,
                "trace": trace + ["learn:exit"]}

    # 2) 首次进入 / 学习模式未指定知识点 → 给目录
    explicit_topic = match_topic(text)
    if not sess or not sess.get("topic"):
        if not explicit_topic:
            return {"learning_session": None, "awaiting_user": True,
                    "response": list_curriculum(), "tool_call_history": history,
                    "trace": trace + ["learn:catalog"]}
        idx = find_point_index(explicit_topic, text)
        out = _start(explicit_topic, idx, history)
        out["trace"] = trace + (out.get("trace") or [])
        return out

    topic = sess["topic"]
    idx = int(sess.get("index", 0))
    kp = get_point(topic, idx)
    if kp is None:
        out = _start(topic, 0, history)
        out["trace"] = trace + (out.get("trace") or [])
        return out

    # 3) 学习会话中的指令
    if is_again(text):
        style = "换一种讲法，用一个生活中的例子或更慢的步骤重新讲一遍，不要照抄上一版。"
        return await_next(state, history, trace, _lecture_text(topic, kp, style))

    if is_example(text):
        qs = pick_examples(topic, 1)
        if not qs:
            return {"learning_session": sess, "awaiting_user": True,
                    "response": "题库里暂时没有这个板块的例题，我们继续讲要点吧～"}
        sess["pending_quiz"] = qs[0]
        sess["asked_quiz"] = True
        sess["stage"] = "quiz"
        return {"learning_session": sess, "awaiting_user": True,
                "tool_call_history": history,
                "response": format_example(qs[0]) + "\n\n（直接回 A/B/C/D 作答）",
                "trace": trace + ["learn:example"]}

    if is_next(text):
        nxt = idx + 1
        if nxt >= len(CURRICULUM.get(topic, [])):
            total = sess.get("total", 0)
            correct = sess.get("correct", 0)
            return {"learning_session": None, "awaiting_user": False,
                    "tool_call_history": history,
                    "response": f"🎉 【{topic}】的知识点全部学完了！"
                                f"小测 {total} 题答对 {correct} 题。"
                                f"想换板块就说「学几何」「学二次函数」，或说「刷题」趁热打铁练一练。",
                    "trace": trace + ["learn:finished"]}
        out = _start(topic, nxt, history)
        out["learning_session"]["correct"] = sess.get("correct", 0)
        out["learning_session"]["total"] = sess.get("total", 0)
        out["trace"] = trace + (out.get("trace") or [])
        return out

    # 4) 小测作答（有 pending_quiz 且输入像答案）
    pq = sess.get("pending_quiz")
    if pq and (is_option_answer(text) or len(text) <= 12):
        args = {"question_id": pq.get("question_id", ""), "user_answer": text}
        res = run_tool("check_answer", args, history=history)
        record_history(history, "check_answer", args, res)
        out = res.output or {}
        correct = bool(out.get("correct"))
        sess["total"] = int(sess.get("total", 0)) + 1
        sess["correct"] = int(sess.get("correct", 0)) + (1 if correct else 0)
        sess["pending_quiz"] = None
        sess["stage"] = "taught"
        std = out.get("standard_answer", pq.get("answer", ""))
        analysis = (pq.get("analysis") or "").strip()
        if correct:
            head = f"✅ 答对了！标准答案是 {std}。"
            sess.pop("last_wrong", None)
        else:
            # 不直接揭晓答案（避免"看一眼就抄"）：先给方向提示，
            # 想要答案再说「看答案」。计数行为保持不变（total/correct 照常累加）。
            sess["last_wrong"] = {"answer": std, "analysis": analysis,
                                  "tip": kp.get("tip", ""), "kp": kp.get("name", "")}
            hint = _first_hint(analysis, kp)
            head = (f"❌ 差一点！先不告诉你答案，按这个方向再想想：\n💡 {hint}"
                    if hint else f"❌ 差一点！再检查一下～")
            head += "\n（想不出来就说「看答案」，我把思路和答案一起讲给你）"
        body = [head]
        if analysis and correct:
            body += ["", f"解析：{analysis[:220]}"]
        body += ["", f"📘 回到【{kp['name']}】：{kp['tip']}",
                 "回复「继续」学下一个知识点，「没听懂」我再讲一遍，或继续提问。"]
        return {"learning_session": sess, "awaiting_user": True,
                "tool_call_history": history, "response": "\n".join(body),
                "trace": trace + ["learn:quiz_graded"]}

    # 4') 答错后追问「看答案」：揭晓上一题（配合上面"先提示、不直接给答案"）
    if re.search(r"看答案|给答案|答案是什么|告诉我答案|公布答案", text) \
            and sess.get("last_wrong"):
        lw = sess.pop("last_wrong") or {}
        body = [f"💡 上一题参考答案：{lw.get('answer', '')}"]
        if lw.get("analysis"):
            body += ["", f"解析：{lw['analysis'][:220]}"]
        if lw.get("tip"):
            body += ["", f"📘 记住口诀：{lw['tip']}"]
        body += ["", "回复「继续」学下一个知识点，「没听懂」我再讲一遍。"]
        return {"learning_session": sess, "awaiting_user": True,
                "tool_call_history": history, "response": "\n".join(body),
                "trace": trace + ["learn:reveal_answer"]}

    # 5) 学习中的自由提问：知识库检索 + 当前知识点要点
    args = {"query": f"{kp['name']} {text}", "top_k": 3}
    res = run_tool("search_knowledge", args, history=history)
    record_history(history, "search_knowledge", args, res)
    facts = [f.get("text", "") for f in ((res.output or {}).get("results") or [])][:2]
    tail = ("\n\n参考：" + " / ".join(f[:80] for f in facts if f)) if facts else ""
    return {"learning_session": sess, "awaiting_user": True,
            "tool_call_history": history,
            "response": f"关于【{kp['name']}】你的疑问我记下了：{text}\n"
                        f"核心要点是——{kp['summary']}\n"
                        f"记住口诀：{kp['tip']}" + tail +
                        "\n（想听更细致的讲解可说「没听懂」）",
            "trace": trace + ["learn:question"]}


def await_next(state: MathAgentState, history: List[Dict[str, Any]],
               trace: List[str], text: str) -> Dict[str, Any]:
    """包装：把一段讲解作为本轮回复发出，保持会话。"""
    sess = dict(state.get("learning_session") or {})
    return {"learning_session": sess, "awaiting_user": True,
            "tool_call_history": history,
            "response": text + "\n\n👉 回复「继续」学下一个，「没听懂」我再换讲法。",
            "trace": trace + ["learn:again"]}


__all__ = ["knowledge_tutor_node"]
