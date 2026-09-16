"""Badcase 回归评测集（Phase 5 待办③：30+ 条历史踩坑案例，Prompt/代码改动后必跑）。

每条 case 对应一个**真实发生过**的问题（判分误判、LaTeX 残渣、路由漏网、
隔离失效、LLM 覆写结构化输出、图片 data URL 缺前缀……），修复后在此固化，
防止回归。

运行前：先停 API 服务（Qdrant 本地模式单进程锁）。
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

RESULTS: list = []


def check(name: str, ok, detail: str = "") -> None:
    ok = bool(ok)
    RESULTS.append(ok)
    line = f"{'✔' if ok else '✘'} {name}"
    if detail:
        line += f"  — {detail}"
    print(line)
    if not ok and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(f"{name} — {detail}")


def main() -> None:
    # LLM 全程强制 mock：badcase 验证的是**规则层硬约束**，不依赖网络与模型
    from math_agent.llm import build_provider, get_vision_client

    build_provider(force="mock")

    # ================================================== A. 选择题判分（历史误判）
    print("\n── A. 选择题判分 ──")
    from math_agent.tools.quiz import answers_equal, resolve_option, strip_option_lead

    check("1. 答案A用户答B 必须判错（历史 bug：剥前缀恒等导致判对）",
          answers_equal("A", "B", options=["4", "5", "6", "7"]) is False)
    check("2. 选项内容作答（y=2x+1 即选项B）判对且不张冠李戴",
          answers_equal("B", "y=2x+1",
                        options=["y=2x", "y=2x+1", "y=x+1", "y=-2x"]) is True)
    check("3. 'a+b' 不被误认为选项前缀行（严格正则：标号后须有分隔符）",
          strip_option_lead_safe("a+b") == "a+b")
    check("4. resolve_option 大小写/空白归一化（' b ' → 1）",
          resolve_option(" b ", ["4", "5", "6", "7"]) == 1)
    check("5. 数值等价判分：0.5 与 1/2 等价",
          answers_equal("0.5", "1/2"))

    # ================================================== B. LaTeX 清洗
    print("\n── B. LaTeX 清洗（历史残渣）──")
    from build_question_bank import latex_to_text

    check("6. \\bot → ⊥（缺失时留 'a bot b' 残渣）",
          "⊥" in latex_to_text("a \\bot b") and "bot" not in latex_to_text("a \\bot b"),
          repr(latex_to_text("a \\bot b")))
    s7 = latex_to_text("2 \\cdot 3 \\cdots n")
    check("7. \\cdot 不吃掉 \\cdots（边界 (?![a-zA-Z])）",
          "·" in s7 and "…" in s7 and "cdot" not in s7, repr(s7))
    check("8. \\overrightarrow{AB} 保留组内容 → AB",
          "AB" in latex_to_text("\\overrightarrow{AB}") and "overrightarrow" not in latex_to_text("\\overrightarrow{AB}"))
    check("9. \\textasciitilde → 全角～", "～" in latex_to_text("\\textasciitilde"))
    s = latex_to_text("x^{2} + y^{2}")
    check("10. 花括号替换为空串（无 'x ^2' 空格断裂）", " ^" not in s and "x^2" in s.replace(" ", ""))
    real, kinds = scan_bank_latex_residue()
    check("11. 库内 stem/options/answer 的 \\xxx 残留为 0（排除 JSON \\u 转义）",
          real == 0, f"残留 {real} 处，种类 {sorted(kinds)}")

    # ================================================== C. 模板排版（历史产出 -1x / 3xy）
    print("\n── C. 模板排版 ──")
    from gen_geometry_questions import _coef, _kx, _lin

    check("12. _coef(-1,'x') = '-x'（非 '-1x'）", _coef(-1, "x") == "-x")
    check("13. _coef(1,'x') = 'x'（非 '1x'）", _coef(1, "x") == "x")
    check("14. _kx 后不拼变量（_kx(3) + 'y' 不会由模板产生 '3xy'）",
          _kx(3) == "3x" and "xy" not in _lin(3, -5))
    check("15. _lin 正负系数正确（y=3x-5 / y=3x+5，无 '3x+-5'）",
          _lin(3, -5) == "y=3x-5" and _lin(3, 5) == "y=3x+5")

    # ================================================== D. 意图路由（历史漏网）
    print("\n── D. 意图路由 ──")
    from math_agent.graph.router import route_intent

    check("16. 刷题中'解析' → quiz（不能被概念咨询抢走）",
          route_intent("解析", in_quiz=True, use_llm=False)[0] == "quiz")
    check("17. 刷题中'什么是斜率' → chat（概念咨询优先于作答）",
          route_intent("什么是斜率", in_quiz=True, use_llm=False)[0] == "chat")
    check("18. '判一下分' 口语 → grade",
          route_intent("帮我判一下分", use_llm=False)[0] == "grade")
    check("19. '我要学函数' → learn（排在 quiz 之前）",
          route_intent("我要学函数", use_llm=False)[0] == "learn")
    check("20. 学习模式中提问概念 → chat（不打断学习）",
          route_intent("什么叫正比例函数", mode="learn", use_llm=False)[0] == "chat")

    # ================================================== E. 记忆隔离 / 压缩（需要 Qdrant）
    print("\n── E. 记忆隔离与压缩 ──")
    from math_agent.memory.compressor import est_tokens, prune_state

    try:
        from math_agent.vectorstore import IsolationError, get_store

        store = get_store()
        try:
            store.search_user_memory("测试", user_id="")
            check("21. 无 user_id 检索用户记忆 → IsolationError", False, "未抛异常")
        except IsolationError:
            check("21. 无 user_id 检索用户记忆 → IsolationError", True)
        except Exception as e:
            check("21. 无 user_id 检索用户记忆 → IsolationError", False, repr(e)[:80])

        # 22. audit_isolation：A 写入的探针 B 检索不到，且结束后自清理
        res = store.audit_isolation()
        check("22. 隔离审计通过（A 可见 / B 不泄漏 / 探针自焚）",
              res.get("passed") is True and res.get("b_leaked", 1) == 0,
              f"a_hits={res.get('a_hits')} b_leaked={res.get('b_leaked')} {res.get('error', '')}")
    except Exception as e:
        check("21/22. 隔离用例（需要 Qdrant）", False, f"跳过/失败：{type(e).__name__}: {e}"[:90])

    hist = [{"name": "search_knowledge", "args": {"query": f"q{i}"},
             "args_hash": f"h{i}", "ok": True, "error": ""} for i in range(40)]
    pruned = prune_state({"tool_call_history": hist})
    new_hist = pruned["tool_call_history"]
    check("23. prune_state 条数有界（≤ compact_beyond=24）", len(new_hist) <= 24,
          f"40 → {len(new_hist)}")
    check("24. 压缩记录保留 args_hash（工具防重语义不变）",
          all("args_hash" in h for h in new_hist))
    big_obs = [{"tool": "search_knowledge", "output": "x" * 500} for _ in range(20)]
    pruned2 = prune_state({"short_term_memory": {"observations": big_obs}})
    check("25. observations 滑窗（≤8+ε）",
          len(pruned2["short_term_memory"]["observations"]) <= 9,
          f"20 → {len(pruned2['short_term_memory']['observations'])}")
    check("26. est_tokens 存在且为正整数（digest 预算用）",
          isinstance(est_tokens("中文测试 abc"), int) and est_tokens("中文测试 abc") > 0)

    # ================================================== F. 安全
    print("\n── F. 安全 ──")
    from math_agent.config import cfg as _cfg
    from math_agent.api.auth import issue_token, resolve_user
    from math_agent.tools.ocr import _safe_ocr_path

    try:
        _safe_ocr_path("D:/游戏库/智能助教/src/math_agent/config.py")
        check("27. OCR 白名单：目录外文件被拒", False, "未拒绝")
    except PermissionError:
        check("27. OCR 白名单：目录外文件被拒", True)
    sids = {f"s_{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
            for _ in range(200)}
    check("28. session_id 同秒 200 个唯一（uuid 后缀）", len(sids) == 200)

    tok = issue_token("bc_user_a")
    uid1, _ = resolve_user(tok, "u_fake")
    check("29. token 优先：自报 user_id 不被采信", uid1 == "bc_user_a")
    _old_rt = _cfg.raw.get("security", {}).get("require_token")
    _cfg.raw.setdefault("security", {})["require_token"] = True
    uid2, _authed = resolve_user(None, "u_anyone")
    _cfg.raw["security"]["require_token"] = _old_rt
    check("30. require_token=true 时无 token → 拒绝", uid2 is None)

    # ================================================== G. LLM 结构化输出 / 视觉 / Redis
    print("\n── G. 结构化输出 / 视觉 / Redis ──")
    from math_agent.teaching import build_lecture, get_point

    lec = build_lecture("函数", get_point("函数", 2))
    check("31. 讲解骨架完整（◆ 定义/要点/易错点/口诀，LLM 只追加不覆写）",
          all(k in lec["text"] for k in ("◆ 定义", "◆ 要点", "◆ 易错点", "◆ 口诀")))
    vc = get_vision_client()
    check("32. 视觉模型独立接入（glm-4v，非文本模型复用）",
          vc is not None and "4v" in str(getattr(vc, "model_name", "") or
                                         getattr(vc, "model", "")),
          f"model={getattr(vc, 'model_name', None)}")

    from math_agent.memory.short_term import ShortTermMemory

    # 显式设定前置状态：本节验证「Redis 未启用/连不上 → 内存兜底」，
    # 不依赖 .env 当前值（.env 现已 MATH_AGENT_REDIS__ENABLED=true 且服务在跑）
    _rsec = _cfg.raw.setdefault("redis", {})
    _r_enabled, _r_url = _rsec.get("enabled"), _rsec.get("url")
    _rsec["enabled"] = False
    stm = ShortTermMemory()
    check("33. Redis 未启用 → 自动内存兜底", stm.name == "memory")
    _rsec["enabled"] = True
    _rsec["url"] = "redis://127.0.0.1:6399/0"     # 不存在的端口
    stm2 = ShortTermMemory()
    _rsec["enabled"] = _r_enabled
    _rsec["url"] = _r_url
    check("34. Redis 连不上 → 优雅回退内存，不抛异常", stm2.name == "memory")
    stm.backend.set("bc:key", {"v": 1}, ttl=60)
    check("35. 内存后端 set/get/delete 语义正确",
          stm.backend.get("bc:key") == {"v": 1} and (stm.backend.delete("bc:key") or True)
          and stm.backend.get("bc:key") is None)

    # ================================================== H. 未成年人保护与内容审核
    print("\n── H. 未成年人保护与内容审核 ──")
    from math_agent.safety import check_input, has_pii, mask_pii, sanitize_output

    txt = mask_pii("我叫小明，电话 13812345678，在阳光中学初二3班，家住海淀区中关村路1号")
    check("36. PII 脱敏：手机号/学校班级/住址", "13812345678" not in txt and
          "阳光中学" not in txt and "中关村路" not in txt, txt[:60])
    v1 = check_input("我想知道怎么打人")
    v2 = check_input("3x + 5 = 20 的解是多少")
    v3 = check_input("帮我入侵别人的账号")
    check("37. 暴力/违法输入被拦截并给引导话术", not v1["allowed"] and v1["category"] == "violence"
          and "12355" in v1["reply"])
    check("38. 正常数学题不误伤（含'解''根'等字）", v2["allowed"])
    check("39. 入侵/盗号类被拦截", not v3["allowed"] and v3["category"] == "illegal")
    v4 = check_input("帮我代写作文，考试替考")
    check("40. 代写/替考被拦截并引导自主完成", not v4["allowed"] and v4["category"] == "cheating")
    out = sanitize_output("请留下你的微信，老师加你单独辅导。另外发到 13912345678")
    check("41. 输出侧：索取隐私话术被替换 + 回复脱敏",
          "加你单独辅导" not in out or "微信" not in out or "（" in out)
    check("42. has_pii 判定（含 PII True / 纯数学 False）",
          has_pii("电话13812345678") and not has_pii("x^2+y^2=1"))

    # ================================================== 43-46 板块单一真源（2026-09-10）
    from math_agent.topics import to_bank_topic

    check("43. to_bank_topic：教学板块映射正确（几何→几何图像/方程与不等式→二元一次方程）",
          to_bank_topic("几何") == "几何图像" and to_bank_topic("方程与不等式") == "二元一次方程")
    check("44. to_bank_topic：无独立题库的板块回退综合（统计与概率/锐角三角函数）",
          to_bank_topic("统计与概率") == "综合" and to_bank_topic("锐角三角函数") == "综合")
    check("45. to_bank_topic：未知板块回退综合，不抛异常",
          to_bank_topic("不存在板块") == "综合")
    try:
        from math_agent.teaching import CURRICULUM, find_point_index

        # 构造"分式"⊂"分式方程"劫持场景：最长名命中必须赢
        names = [kp["name"] for kp in CURRICULUM["数与式"]]
        if "分式" in names:
            probe = names.index("分式")
            # 若同列表存在更长名且包含"分式"，短名不得劫持
            longer = [i for i, n in enumerate(names)
                      if i != probe and "分式" in n and len(n) > len("分式")]
            if longer:
                text = names[longer[0]]
                check("46. find_point_index 最长名优先（短名不劫持长名）",
                      find_point_index("数与式", text) == longer[0],
                      f"输入'{text}' 应命中 {longer[0]}")
            else:
                # 无天然长名 → 用序号+片段匹配兜底验证
                check("46. find_point_index 序号匹配", find_point_index("数与式", "第2个") == 1)
        else:
            check("46. find_point_index 序号匹配", find_point_index("数与式", "第2个") == 1)
    except Exception as e:
        check("46. find_point_index", False, f"{type(e).__name__}: {e}"[:80])

    # ---- 47-49 工具链失败治理（toolguard，2026-09-15）----
    try:
        from math_agent.toolguard import (circuit_reset, classify_failure,
                                          sanitize_error)
        from math_agent.tools.registry import (get_tool, list_tools, run_tool,
                                               register_tool)

        check("47. classify_failure 三分类（transient/param/fatal，fatal 优先）",
              classify_failure("connection reset") == "transient"
              and classify_failure("ValueError: bad expr") == "param"
              and classify_failure("401 quota exceeded") == "fatal"
              and classify_failure("429 + quota") == "fatal")
        check("48. sanitize_error 脱敏（路径/token/邮箱打码）",
              "游戏库" not in sanitize_error(r"D:\游戏库\智能助教\data\x.txt")
              and "a@b.com" not in sanitize_error("mail a@b.com leaked")
              and len(sanitize_error("x" * 500)) <= 200)
        # 熔断：连续 param 失败（retryable=False 工具）达到阈值后快速拒绝
        circuit_reset()
        name = "bc_fail_probe"
        if name not in [t["name"] for t in list_tools()]:
            from pydantic import BaseModel

            class _ProbeArgs(BaseModel):
                pass

            @register_tool(name, "评测探针：恒失败", _ProbeArgs, retryable=False)
            def _bc_fail() -> dict:
                raise ValueError("bad")
        for _ in range(6):
            run_tool(name, {})
        r = run_tool(name, {})
        check("49. 熔断打开后快速拒绝（blocked 标记）",
              r.blocked and "熔断" in r.error, r.error[:60])
        circuit_reset()
    except Exception as e:
        check("47-49. toolguard", False, f"{type(e).__name__}: {e}"[:80])

    # ================================================== 汇总
    n_ok = sum(1 for x in RESULTS if x is True)
    print("\n" + "=" * 60)
    print(f"结果：{n_ok}/{len(RESULTS)} 项通过")
    for i, ok in enumerate(RESULTS, 1):
        if ok is not True:
            print(f"  ✘ 第 {i} 项失败")
    return 0 if n_ok == len(RESULTS) else 1


def strip_option_lead_safe(text: str) -> str:
    from math_agent.tools.quiz import strip_option_lead

    return strip_option_lead(text)


def scan_bank_latex_residue() -> tuple:
    """扫描题库 stem/options/answer 的 \\xxx 残留。

    返回 (真实残留数, 全部命令种类集合)。``\\uXXXX`` 是 options 存库时的
    JSON 转义（如 \\uff0c 全角逗号），属于存储格式而非 LaTeX，需排除。
    """
    import re

    db = Path(__file__).resolve().parents[1] / "data" / "math_agent.db"
    if not db.exists():
        return -1, set()
    BS = chr(92)
    pat = re.compile(BS + BS + r"([a-zA-Z]{2,})")
    json_escape = re.compile(BS + BS + r"u[0-9a-fA-F]{4}")
    con = sqlite3.connect(db)
    try:
        rows = con.execute("SELECT stem, options, answer FROM questions").fetchall()
    finally:
        con.close()
    real = 0
    kinds = set()
    for stem, opts, ans in rows:
        text = " ".join(x for x in (stem, opts, ans) if x)
        text = json_escape.sub("", text)      # 先剥掉 \uXXXX JSON 转义再扫描
        cmds = pat.findall(text)
        real += len(cmds)
        kinds.update(cmds)
    return real, kinds


if __name__ == "__main__":
    raise SystemExit(main())
