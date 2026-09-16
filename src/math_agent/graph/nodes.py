"""LangGraph 业务节点（方案 2.2 / 3.1-3.4）。"""
from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any, Dict, List

from .. import metrics
from ..analysis import analyze_errors, extract_habits, llm_polish_report
from ..config import cfg
from ..db.repository import (create_session, fetch_questions, finish_session,
                             record_attempt, user_history)
from ..llm import get_llm
from ..state import MathAgentState
from ..topics import to_bank_topic
from ..tools.quiz import MAX_COUNT as _QUIZ_MAX_COUNT
from ..tools.registry import record_history, run_tool

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ 解析
# 注意顺序：锐角三角函数/统计与概率必须在"函数/几何"之前（含"函数""三角形"字样会被劫持）
_P_TOPIC = [
    (re.compile(r"三角函数|正弦|余弦|正切|解直角三角"), "锐角三角函数"),
    (re.compile(r"概率|统计|平均数|中位数|众数|方差|频数"), "统计与概率"),
    (re.compile(r"函数|一次函数|反比例|二次函数|图像性质"), "函数"),
    (re.compile(r"二元一次|方程组|方程|不等式"), "二元一次方程"),
    (re.compile(r"几何|三角形|四边形|圆|相似|全等|勾股|平行|角度|面积"), "几何图像"),
]
_P_TYPE = [
    (re.compile(r"应用题|解答题|大题"), "应用题"),
    (re.compile(r"填空"), "填空题"),
    (re.compile(r"选择"), "选择题"),
]
_P_COUNT = re.compile(r"(\d+)\s*(?:道|题|个)")
_P_DIFF = re.compile(r"(?:难度|等级)\s*(\d)")
# 刷题中的情绪表达/闲聊不当答案判分（"这道题好难"会被判错）；概念提问已由路由层拦截
_NON_ANSWER = re.compile(r"好难|太难|不会啊|求助|帮帮我|什么意思|说人话|放弃|无聊|换个思路")


def parse_quiz_request(text: str) -> Dict[str, Any]:
    topic = ""
    for p, v in _P_TOPIC:
        if p.search(text):
            topic = v
            break
    qtype = "选择题"
    for p, v in _P_TYPE:
        if p.search(text):
            qtype = v
            break
    m = _P_COUNT.search(text or "")
    count = int(m.group(1)) if m else 5
    d = _P_DIFF.search(text or "")
    difficulty = int(d.group(1)) if d else None
    return {"topic": topic or "综合", "type": qtype, "count": max(1, min(count, _QUIZ_MAX_COUNT)),
            "difficulty": difficulty}


def _format_question(q: Dict[str, Any], idx: int, total: int) -> str:
    lines = [f"第 {idx}/{total} 题【{q.get('topic', '')}·{q.get('type', '')}·难度 {q.get('difficulty', 3)}】",
             q.get("stem", "")]
    for opt in q.get("options") or []:
        lines.append(f"  {opt}")
    return "\n".join(lines)


# ------------------------------------------------------------------ 分层提示
# 答错就只回一句"回答错误"是教学上的浪费：学生既不知道下一步怎么想，
# 也不知道可以主动要解析（多数孩子不会主动打"解析"两个字）。
# 这里做三级递进：方向 → 关键步骤 → 揭晓；题库无解析时用通用步骤兜底，
# 绝不回"暂无解析"就结束。
_GENERAL_STEPS = ("① 读题圈已知：把题目给的数字和条件都标出来；\n"
                  "② 明确求什么：用一句话说清目标；\n"
                  "③ 选方法：看它属于哪一类（列方程 / 套公式 / 找图形关系）；\n"
                  "④ 算完检验：把结果代回原题看是否说得通。")
_TOPIC_STEPS = {
    "函数": "先设出解析式的一般形式（比如 y=kx+b），再把已知点的坐标代入求系数。",
    "二元一次方程": "设两个未知数，找两个等量关系列出方程组，再用代入消元或加减消元求解。",
    "几何图像": "先把已知的边、角标到图上，再找全等 / 相似 / 勾股 / 圆的性质这些关系。",
    "综合": "把大问题拆成几个小问题，逐个找等量关系，一步步化归。",
}


def _hint_for(q: Dict[str, Any], level: int) -> str:
    """分层提示：1=方向，2=关键步骤，3=完整解析（揭晓答案）。"""
    ana = (q.get("analysis") or "").strip()
    if ana:
        sents = [s.strip() for s in re.split(r"(?<=[。；;!?])", ana) if s.strip()]
        if level <= 1:
            first = sents[0] if sents else ana[:80]
            return f"{first}\n（先顺着这个方向想一步，想不出来可以再答一次～）"
        if level == 2:
            pre = re.sub(r"。{2,}", "。", "".join(sents[:2]).strip())
            if pre and not pre.endswith("。"):
                pre += "。"
            return f"{pre}\n（关键步骤已给出，再算一次试试？）"
        return ana
    # 题库无解析 → 通用兜底，不能只回"暂无解析"
    step = _TOPIC_STEPS.get(q.get("topic", ""), _TOPIC_STEPS["综合"])
    if level <= 1:
        return f"先判断这道题在考什么：{step}"
    if level == 2:
        return f"解题方向：{step}\n通用步骤：\n{_GENERAL_STEPS}"
    return (f"这类题的做法：{step}\n{_GENERAL_STEPS}\n"
            f"参考答案是 {q.get('answer', '')}。"
            f"（本题暂无详细解析，按上面的步骤自己推一遍，印象更深～）")


# ------------------------------------------------------------------ 节点
def intent_router_node(state: MathAgentState) -> Dict[str, Any]:
    from .router import route_intent

    text = state.get("user_input", "") or ""
    sess = state.get("quiz_session") or {}
    in_quiz = bool(sess.get("questions")) and not sess.get("finished")
    learn = state.get("learning_session") or {}
    in_learn = bool(learn.get("topic"))
    mode = state.get("session_mode", "") or ""
    intent, conf = route_intent(text, in_quiz=in_quiz, mode=mode, in_learn=in_learn)

    submit_event = intent == "submit"
    trace = list(state.get("trace", []))
    trace.append(f"intent={intent}({conf:.2f})")
    return {"intent": intent, "intent_confidence": conf,
            "submit_event": submit_event, "trace": trace}


def quiz_manager_node(state: MathAgentState) -> Dict[str, Any]:
    text = (state.get("user_input", "") or "").strip()
    sess: Dict[str, Any] = dict(state.get("quiz_session") or {})
    history: List[Dict[str, Any]] = list(state.get("tool_call_history", []))
    trace = list(state.get("trace", []))
    user_id = state.get("user_id", "u_default")

    # ---------- 单题解析（不算"即时分析报告"） ----------
    # 「看答案 / 不会」也能拿到思路——多数孩子不知道要主动打"解析"两个字。
    # 用 不会(?!做) 排除跳题指令「不会做」（见下方跳题分支）
    if sess.get("questions") and re.search(
            r"解析|讲解|为什么|怎么(做|解)|提示|答案|算了|不会(?!做)", text):
        idx = int(sess.get("index", 0))
        qs = sess["questions"]
        q = qs[min(idx, len(qs) - 1)]
        # 题库无解析时用通用步骤 + 参考答案兜底，绝不只回"暂无解析"
        body = (q.get("analysis") or "").strip() or _hint_for(q, 3)
        return {"response": f"💡 本题解析：\n{body}",
                "awaiting_user": True, "trace": trace + ["quiz:show_analysis"]}

    # ---------- 跳题 ----------
    if sess.get("questions") and re.search(r"跳过|下一题|不会做|换一题", text):
        sess["index"] = int(sess.get("index", 0))
        qs = sess["questions"]
        if sess["index"] < len(qs):
            q = qs[sess["index"]]
            sess["records"].append({"question_id": q["question_id"], "topic": q["topic"],
                                    "type": q["type"], "stem": q["stem"], "options": q["options"],
                                    "user_answer": "", "correct": False, "answer": q["answer"],
                                    "analysis": q.get("analysis", ""),
                                    "knowledge_points": q.get("knowledge_points", []),
                                    "skipped": True})
        sess["index"] += 1
        if sess["index"] >= len(qs):
            return {"quiz_session": sess, "awaiting_user": True,
                    "response": "已跳到最后一道题啦，回复「交卷」查看错题报告 📝",
                    "trace": trace + ["quiz:skip"]}
        q = qs[sess["index"]]
        return {"quiz_session": sess, "awaiting_user": True,
                "response": "⏭️ 已跳过。\n\n" + _format_question(q, sess["index"] + 1, len(qs)),
                "trace": trace + ["quiz:skip"]}

    # ---------- 收答 ----------
    answer = state.get("pending_answer")
    if answer is None and sess.get("questions") and not sess.get("finished"):
        if text and _NON_ANSWER.search(text):
            return {"quiz_session": sess, "awaiting_user": True,
                    "response": "💪 别急～ 直接回复选项字母（A/B/C/D）或答案即可；"
                                "看解析说「解析」，跳过说「下一题」，交卷说「交卷」。",
                    "trace": trace + ["quiz:chitchat"]}
        answer = text if text else None
    if answer and sess.get("questions") and int(sess.get("index", 0)) < len(sess["questions"]):
        qs = sess["questions"]
        idx = int(sess["index"])
        q = qs[idx]
        res = run_tool("check_answer", {"question_id": q["question_id"], "user_answer": answer},
                       history=history)
        record_history(history, "check_answer",
                       {"question_id": q["question_id"], "user_answer": answer}, res)
        if not res.ok:
            return {"response": f"判分失败：{res.error}", "awaiting_user": True,
                    "tool_call_history": history, "trace": trace + ["quiz:check_failed"]}
        out = res.output
        sess.setdefault("records", []).append({
            "question_id": q["question_id"], "topic": q.get("topic", ""),
            "type": q.get("type", ""), "stem": q.get("stem", ""),
            "options": q.get("options", []), "user_answer": answer,
            "correct": out["correct"], "answer": out["standard_answer"],
            "analysis": out.get("analysis", ""),
            "knowledge_points": out.get("knowledge_points", []),
        })
        # 落库（答题记录 → 习惯追踪数据源）
        # ⚠ 禁止静默吞：这是习惯追踪的唯一数据源，写失败会让错题报告长期失真且无告警
        try:
            record_attempt(sess.get("session_id", ""), user_id, q, answer, out["correct"])
        except Exception:
            log.exception("record_attempt 失败 user=%s qid=%s",
                          user_id, q.get("question_id"))
            metrics.record_persist_failure("record_attempt")
        sess["index"] = idx + 1

        # 代码层硬约束：单题只给对错 + 分层引导，绝不输出"错题分析报告"。
        # ⚠ "回答正确/回答错误"字样与"不含错题分析报告"都是评测断言，改动前先看
        #   tests/test_smoke.py::test_quiz_flow_no_immediate_report
        if out["correct"]:
            head = "✅ 回答正确！"
            (sess.get("hints") or {}).pop(q.get("question_id", ""), None)
        else:
            hints = sess.setdefault("hints", {})
            qid = q.get("question_id", "")
            lvl = hints.get(qid, 0) + 1
            hints[qid] = lvl
            if lvl < 3:
                head = f"❌ 回答错误。\n💡 提示{lvl}：{_hint_for(q, lvl)}"
            else:
                head = (f"❌ 回答错误。\n正确答案：{out.get('standard_answer', '')}\n"
                        f"💡 {_hint_for(q, 3)}")
        if sess["index"] >= len(qs):
            return {"quiz_session": sess, "awaiting_user": True,
                    "tool_call_history": history,
                    "response": f"{head}\n\n已经是最后一道题了，回复「交卷」获取本次错题报告 📝",
                    "trace": trace + ["quiz:answered(last)"]}
        nq = qs[sess["index"]]
        return {"quiz_session": sess, "awaiting_user": True, "tool_call_history": history,
                "response": f"{head} 继续下一题：\n\n" + _format_question(nq, sess["index"] + 1, len(qs)),
                "trace": trace + ["quiz:answered"]}

    # ---------- 新会话 ----------
    req = parse_quiz_request(text)
    # 板块回退：题库未收录的板块 → 映射/综合题库 + 明确提示（而非静默拉无关题）
    bank_topic = to_bank_topic(req["topic"])
    topic_notice = (f"📌 「{req['topic']}」板块题目正在补充中，先从{bank_topic}题库为你挑选～\n"
                    if bank_topic != req["topic"] else "")
    # 出题去重：排除上一轮会话已出过的题，避免"再刷3道"重复出题
    prev_ids = [q.get("question_id") for q in (sess.get("questions") or [])
                if q.get("question_id")]
    res = run_tool("fetch_quiz", {"topic": bank_topic, "type": req["type"],
                                  "count": req["count"], "difficulty": req["difficulty"],
                                  "exclude_ids": prev_ids},
                   history=history)
    if (not res.ok or not res.output.get("ok")) and prev_ids:
        # 排除旧题后拉不到（如剩余题不足）→ 放开去重重试一次
        res = run_tool("fetch_quiz", {"topic": bank_topic, "type": req["type"],
                                      "count": req["count"], "difficulty": req["difficulty"],
                                      "exclude_ids": []},
                       history=history)
    record_history(history, "fetch_quiz",
                   {"topic": bank_topic, "type": req["type"], "count": req["count"],
                    "difficulty": req["difficulty"]}, res)
    if not res.ok or not res.output.get("ok"):
        msg = res.error or res.output.get("message", "拉题失败")
        return {"response": f"😢 {msg}\n请换个板块或题型试试～", "awaiting_user": True,
                "tool_call_history": history, "trace": trace + ["quiz:fetch_failed"]}

    questions = res.output["questions"]
    session_id = f"s_{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    try:
        create_session(session_id, user_id, bank_topic, req["type"])
    except Exception:
        log.exception("create_session 失败 user=%s sid=%s", user_id, session_id)
        metrics.record_persist_failure("create_session")
    sess = {"session_id": session_id, "topic": bank_topic, "type": req["type"],
            "questions": questions, "index": 0, "records": [], "finished": False}
    head = (f"{topic_notice}📚 开始【{req['topic']}·{req['type']}】练习，共 {len(questions)} 题。\n"
            f"直接回复答案即可（选择题回 A/B/C/D）。回复「交卷」查看错题报告。\n")
    return {"quiz_session": sess, "awaiting_user": True, "tool_call_history": history,
            "response": head + "\n" + _format_question(questions[0], 1, len(questions)),
            "trace": trace + ["quiz:started"]}


def grading_node(state: MathAgentState) -> Dict[str, Any]:
    text = (state.get("user_input", "") or "").strip()
    history = list(state.get("tool_call_history", []))
    trace = list(state.get("trace", []))

    payload = state.get("ocr_payload") or {}
    if not payload:
        import re as _re
        m = _re.search(r"([A-Za-z]:\\[^\s]+|\./[^\s]+|/[^\s]+\.(?:png|jpg|jpeg|pdf|txt))", text)
        if not m:
            return {"response": ("📄 请把作业图片/PDF 的路径发给我（或先上传文件），"
                                 "例如：批改 D:\\homework\\week3.pdf"),
                    "awaiting_user": True, "trace": trace + ["grade:need_file"]}
        path = m.group(1)
        ocr = run_tool("ocr_document", {"file_path": path}, history=history)
        record_history(history, "ocr_document", {"file_path": path}, ocr)
        if not ocr.ok:
            return {"response": f"OCR 失败：{ocr.error}", "awaiting_user": True,
                    "tool_call_history": history, "trace": trace + ["grade:ocr_failed"]}
        payload = ocr.output

    if float(payload.get("confidence", 0)) < 0.6:
        return {"ocr_payload": payload, "awaiting_user": True,
                "response": (f"⚠️ 识别置信度仅 {payload.get('confidence', 0):.2f}，"
                             f"结果可能不准。识别内容如下，请确认是否正确：\n"
                             f"{payload.get('text', '')[:500]}\n确认请回复「确认批改」，重新上传请发新路径。"),
                "tool_call_history": history, "trace": trace + ["grade:low_confidence"]}

    res = run_tool("grade_assignment", {"text": payload.get("text", "")}, history=history)
    record_history(history, "grade_assignment", {"text": payload.get("text", "")[:80]}, res)
    if not res.ok:
        return {"response": f"批改失败：{res.error}", "awaiting_user": True,
                "tool_call_history": history, "trace": trace + ["grade:failed"]}

    out = res.output
    records = [{"question_id": f"hw_{it['no']}", "topic": it.get("topic", ""),
                "type": "解答题", "stem": it["stem"], "options": [],
                "user_answer": it["user_answer"], "correct": it["correct"],
                "answer": it.get("standard_answer", ""),
                "analysis": it.get("analysis", ""), "knowledge_points": []}
               for it in out["items"]]
    summary = out["summary"]
    resp = (f"✅ 批改完成：共 {summary['total']} 题，对 {summary['correct']}，"
            f"错 {summary['wrong']}，待确认 {summary['unknown']}。\n正在生成错题报告…")
    return {"grade_result": out, "tool_call_history": history, "ocr_payload": payload,
            "_records": records, "response": resp,
            "trace": trace + ["grade:done"]}


def error_analyzer_node(state: MathAgentState) -> Dict[str, Any]:
    trace = list(state.get("trace", []))
    sess = state.get("quiz_session") or {}
    records: List[Dict[str, Any]] = list(sess.get("records", []))
    if not records:
        records = list(state.get("_records") or (state.get("grade_result") or {}).get("_records", []))
    if not records:
        return {"response": "本次还没有答题记录，先做几道题再交卷吧～",
                "awaiting_user": True, "trace": trace + ["analyzer:empty"]}

    real = [r for r in records if not r.get("skipped")]
    report = analyze_errors(real)
    report["narrative"] = llm_polish_report(report)
    report["_records"] = real

    try:
        finish_session(sess.get("session_id", ""))
    except Exception:
        log.exception("finish_session 失败 sid=%s", sess.get("session_id", ""))
        metrics.record_persist_failure("finish_session")
    return {"error_report": report, "quiz_session": {**sess, "finished": True},
            "submit_event": True, "final_report": report["narrative"],
            "response": report["narrative"], "awaiting_user": False,
            "trace": trace + ["analyzer:done"]}


def memory_manager_node(state: MathAgentState) -> Dict[str, Any]:
    """错题报告 → 抽取习惯 → 写入长期记忆（方案 4.2）。"""
    trace = list(state.get("trace", []))
    report = state.get("error_report")
    if not report:
        return {"trace": trace + ["memory:skip"]}
    user_id = state.get("user_id", "u_default")
    habits = extract_habits(report)
    history = list(state.get("tool_call_history", []))
    written = 0
    for h in habits:
        args = {"user_id": user_id, "content": h["content"],
                "type": h["type"], "source": f"report_{time.strftime('%Y%m%d%H%M%S')}"}
        res = run_tool("update_long_term_memory", args, history=history, check_duplicate=False)
        record_history(history, "update_long_term_memory", args, res)
        written += res.ok
    return {"memory_written": written > 0, "tool_call_history": history,
            "trace": trace + [f"memory:written={written}"]}


def habit_node(state: MathAgentState) -> Dict[str, Any]:
    """习惯查询：长期记忆 + 答题历史 → 优缺点报告（方案 3.4）。"""
    text = (state.get("user_input", "") or "我的学习习惯和薄弱点")
    user_id = state.get("user_id", "u_default")
    trace = list(state.get("trace", []))
    history = list(state.get("tool_call_history", []))

    res = run_tool("search_user_memory", {"query": text, "user_id": user_id, "top_k": 8},
                   history=history)
    record_history(history, "search_user_memory",
                   {"query": text, "user_id": user_id, "top_k": 8}, res)
    mem = res.output or {} if res.ok else {}
    hits = mem.get("results", []) if isinstance(mem, dict) else []

    hist = user_history(user_id, limit=200)
    total = len(hist)
    correct = sum(1 for h in hist if h["correct"])
    acc = round(correct / total * 100, 1) if total else 0.0
    by_topic: Dict[str, List[int]] = {}
    for h in hist:
        by_topic.setdefault(h["topic"] or "未分类", [0, 0])
        by_topic[h["topic"] or "未分类"][0] += 1
        by_topic[h["topic"] or "未分类"][1] += 1 if h["correct"] else 0

    lines = ["🧭 学习习惯报告", "=" * 36,
             f"累计做题 {total} 道，正确率 {acc}%", ""]
    if by_topic:
        lines.append("【板块表现】")
        for t, (n, c) in sorted(by_topic.items(), key=lambda x: x[1][1] / max(x[1][0], 1)):
            lines.append(f"  · {t}：{n} 题，正确率 {round(c / n * 100, 1)}%")
        lines.append("")
    if hits:
        lines.append("【长期记忆中的关键点】")
        for h in hits[:8]:
            lines.append(f"  · [{h.get('type', 'habit')}] {h.get('content', '')}"
                         f"（{str(h.get('timestamp', ''))[:10]}）")
        lines.append("")
    else:
        lines.append("（暂无长期记忆，完成一次完整刷题并交卷后会自动积累）\n")

    strengths = [h["content"] for h in hits if h.get("type") == "strength"]
    weak = [h["content"] for h in hits if h.get("type") in ("weak_point", "habit")]
    lines.append("【优点】")
    lines.extend(f"  · {s}" for s in strengths[:5]) if strengths else lines.append("  · 继续练习，优点会逐渐显现")
    lines.append("")
    lines.append("【待改进】")
    if weak:
        lines.extend(f"  · {w}" for w in weak[:6])
    else:
        lines.append("  · 暂无明显薄弱点")

    body = "\n".join(lines)
    # 硬约束：结构化正文（标题 + 正确率 + 板块数据）由代码层产出，永不交给 LLM 覆写；
    # LLM 只追加一段建议，避免改写数字造成幻觉。
    llm = get_llm()
    if llm.available:
        tip = llm.chat(
            "你是初中数学辅导老师。下面是某位同学的学习习惯报告：\n"
            f"{body}\n\n"
            "请只给出 2-3 条具体、可执行的下一步练习建议"
            "（120 字以内，不要重复报告里已有的数字，不要改写报告内容）。"
        )
        if tip:
            body = f"{body}\n\n💡 老师建议\n{tip}"
    return {"long_term_memory_hits": hits, "tool_call_history": history,
            "final_report": body, "response": body, "awaiting_user": False,
            "trace": trace + ["habit:done"]}


def finalize_node(state: MathAgentState) -> Dict[str, Any]:
    return {"awaiting_user": state.get("awaiting_user", False)}


__all__ = ["intent_router_node", "quiz_manager_node", "grading_node", "error_analyzer_node",
           "memory_manager_node", "habit_node", "finalize_node", "parse_quiz_request"]
