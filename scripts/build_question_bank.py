#!/usr/bin/env python
"""Phase 1 —— 题库 ETL：TAL-SCQ5K → 统一 schema → SQLite + 向量知识库。

处理流程
--------
1. 读取 data/raw/TAL-SCQ5K-CN_*.jsonl
2. 清洗 LaTeX 噪音、抽取选项/答案/解析/知识点
3. 按关键词映射到三大板块：函数 / 二元一次方程 / 几何图像
4. 题型派生：
   * 选择题 —— 原始数据
   * 填空题 —— 选项内容较短的选择题，去掉选项、保留答案文本
   * 应用题 —— 命中"应用题/行程/工程/分百/列方程"等标签的选择题，去选项
5. 写入 SQLite（questions 表）并可选构建向量知识库

用法::
    python scripts/build_question_bank.py                 # 只写库
    python scripts/build_question_bank.py --with-vector   # 同时构建向量知识库
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_agent.config import RAW_DIR  # noqa: E402
from math_agent.db.repository import (reset_questions, topic_stats,  # noqa: E402
                                      upsert_questions)

# ------------------------------------------------------------------ 板块映射
TOPIC_RULES: List[tuple] = [
    ("函数", re.compile(r"函数|图象与性质|图像与性质|象限|自变量|斜率|坐标|正比例|反比例")),
    ("二元一次方程", re.compile(r"二元一次|一元一次|一元二次|方程组|解方程|不等式|分式方程|整式方程|方程")),
    ("几何图像", re.compile(r"几何|三角形|四边形|圆|相似|全等|勾股|平行|垂直|对称|平移|旋转|面积|周长|角度|扇形|多边形")),
]
APP_RULES = re.compile(r"应用题|行程|工程|分百|列方程解应用题|利润|浓度|平均|经济|植树|鸡兔")
# 初中内容判定（用于过滤小学奥数题）
MIDDLE_SCHOOL = re.compile(r"函数|方程|几何|三角形|四边形|圆|相似|全等|勾股|平行|坐标|"
                           r"不等式|整式|分式|根式|因式分解|幂|统计|概率|实数|有理数|绝对值|"
                           r"二元一次|一元二次|七年级|八年级|九年级|初中|中考")
PRIMARY_SCHOOL = re.compile(r"[一二三四五六]年级|小学|奥数|华杯|希望杯|走美杯")

_LATEX_NOISE = re.compile(
    r"\\(?:left|right|!|,|;|:|quad|qquad|displaystyle|limits|rm|bf|it|"
    r"mathsf|mathbf|mathop|not)\b")
_OPTION_LEAD = re.compile(r"^\s*\(?[A-Da-d]\)?[\.、:：]\s*")

# 通用知识点占位词，对"薄弱知识点"分析无信息量，剔除
_KP_BLACKLIST = {"课内体系", "知识体系", "数学思想", "拓展思维", "思想", "方法",
                 "能力", "素养", "知识点", "题型", "特色题型", "学科符号",
                 "七大能力", "学习能力", "知识模块", "小升初知识点", "课内知识点",
                 "课内题型", "Knowledge Point"}

# 注意：以下均为**正则**（用 re.sub 应用），反斜杠要写成 \\
# 每条规则自动追加 (?![a-zA-Z])，避免 \cdot 误吃 \cdots 这类长命令
_REPL_RAW = [
    (r"\\textless\s*\{\}", "<"), (r"\\textgreater\s*\{\}", ">"),
    (r"\\textbar\s*\{\}", "|"),
    (r"\\leqslant", "≤"), (r"\\geqslant", "≥"),
    (r"\\leq", "≤"), (r"\\geq", "≥"), (r"\\neq", "≠"), (r"\\approx", "≈"),
    (r"\\cdots", "…"), (r"\\ldots", "…"), (r"\\dots", "…"),
    (r"\\times", "×"), (r"\\div", "÷"), (r"\\cdot", "·"), (r"\\pm", "±"),
    (r"\\angle", "∠"), (r"\\triangle", "△"), (r"\\parallel", "∥"),
    (r"\\perp", "⊥"), (r"\\bot", "⊥"), (r"\\pi", "π"), (r"\\alpha", "α"), (r"\\beta", "β"),
    (r"\\gamma", "γ"), (r"\\theta", "θ"), (r"\\circ", "°"),
    (r"\\infty", "∞"), (r"\\log", "log"), (r"\\ln", "ln"), (r"\\sin", "sin"),
    (r"\\cos", "cos"), (r"\\tan", "tan"),
    (r"\\cup", "∪"), (r"\\cap", "∩"),
    (r"\\le", "≤"), (r"\\ge", "≥"),
    (r"\\(?:ne|neq)", "≠"), (r"\\in", "∈"),
    (r"\\(?:quad|qquad|hfill|break|hline|displaystyle|nonumber)", " "),
    # --- 补充：实测原始数据残留的高频 LaTeX 命令（2026-09-08 全量扫描） ---
    (r"\\textasciitilde", "～"),
    (r"\\Rightarrow", "⇒"), (r"\\Leftrightarrow", "⇔"),
    (r"\\rightarrow", "→"), (r"\\to", "→"), (r"\\colon", ":"),
    (r"\\sim", "∽"), (r"\\backsim", "∽"), (r"\\simeq", "≃"),
    (r"\\equiv", "≡"), (r"\\lt", "<"), (r"\\gt", ">"),
    (r"\\mid", "|"), (r"\\nmid", "∤"),
    (r"\\square", "□"), (r"\\blacksquare", "■"), (r"\\bigcirc", "○"),
    (r"\\bigstar", "★"), (r"\\blacktriangle", "▲"), (r"\\vartriangle", "△"),
    (r"\\heartsuit", "♥"), (r"\\diamondsuit", "♦"),
    (r"\\therefore", "∴"), (r"\\because", "∵"),
    (r"\\Delta", "Δ"), (r"\\Theta", "Θ"), (r"\\Omega", "Ω"), (r"\\Gamma", "Γ"),
    (r"\\omega", "ω"), (r"\\varphi", "φ"), (r"\\phi", "φ"), (r"\\lambda", "λ"),
    (r"\\xi", "ξ"), (r"\\rho", "ρ"), (r"\\mu", "μ"), (r"\\tau", "τ"),
    (r"\\delta", "δ"), (r"\\nabla", "∇"),
    (r"\\oplus", "⊕"), (r"\\otimes", "⊗"), (r"\\odot", "⊙"),
    (r"\\centerdot", "·"), (r"\\ast", "*"), (r"\\surd", "√"),
    (r"\\subseteq", "⊆"), (r"\\subsetneqq", "⊊"), (r"\\subset", "⊂"),
    (r"\\supset", "⊃"), (r"\\varnothing", "∅"), (r"\\complement", "∁"),
    (r"\\notin", "∉"), (r"\\forall", "∀"), (r"\\exists", "∃"),
    (r"\\vee", "∨"), (r"\\wedge", "∧"), (r"\\neg", "¬"),
    (r"\\lfloor", "⌊"), (r"\\rfloor", "⌋"), (r"\\langle", "⟨"), (r"\\rangle", "⟩"),
    (r"\\vdots", "…"), (r"\\ddots", "…"),
    (r"\\lg", "lg"), (r"\\cot", "cot"),
    (r"\\arcsin", "arcsin"), (r"\\arccos", "arccos"), (r"\\arctan", "arctan"),
    (r"\\min", "min"), (r"\\max", "max"),
    (r"\\(?:bmod|pmod)", "mod"),
    (r"\\prime", "′"),
]
_REPL = [(p if p.endswith(r"\s*\{\}") else p + r"(?![a-zA-Z])", v)
         for p, v in _REPL_RAW]


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = _LATEX_NOISE.sub("", s)
    s = s.replace("\\,", " ").replace("\\;", " ").replace("$$", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _take_group(s: str, i: int):
    """s[i] 必须是 '{'，返回 (花括号内内容, 结束后下标)；支持嵌套花括号。"""
    if i >= len(s) or s[i] != "{":
        return None, i
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    return s[i + 1:], len(s)


def _convert_frac_sqrt(s: str) -> str:
    r"""把 \frac{a}{b} / \sqrt{a} 递归转成 (a)/(b) / √(a)，支持嵌套。"""
    out: List[str] = []
    i = 0
    while i < len(s):
        m = re.match(r"\\(d?frac|sqrt)\s*(?:\[[^\]]*\])?", s[i:])
        if not m:
            out.append(s[i])
            i += 1
            continue
        cmd, j = m.group(1), i + m.end()
        if cmd.endswith("frac"):
            a, j2 = _take_group(s, j)
            b, j3 = _take_group(s, j2) if a is not None else (None, j2)
            if a is not None and b is not None:
                out.append(f"({_convert_frac_sqrt(a)})/({_convert_frac_sqrt(b)})")
                i = j3
                continue
        else:
            a, j2 = _take_group(s, j)
            if a is None:
                a, j2 = s[j:j + 1], j + 1
            out.append(f"√({_convert_frac_sqrt(a)})")
            i = j2
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def latex_to_text(s: str) -> str:
    """把 LaTeX 转成尽量接近教材排版的中文可读文本。"""
    if not s:
        return ""
    s = re.sub(r"</?[a-zA-Z][^>]{0,40}>", " ", s)       # HTML 标签（解析里常见 <p>）
    s = s.replace("``", "“").replace("''", "”")         # LaTeX 双引号
    # 环境名 + 可选参数（如 \begin{array}{ll}）整体移除
    s = re.sub(r"\\(?:begin|end)\s*\{[a-zA-Z*]+\}(?:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})?", " ", s)
    s = re.sub(r"\\(?:underline|uline|emph|textbf|textit|mathrm|mbox|"
               r"overrightarrow|overrghtarrow)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\underbrace\s*\{([^{}]*)\}(?:\s*_\s*\{[^{}]*\})?", r"\1", s)
    s = re.sub(r"\\(?:overline|bar|hat|vec|dot|tilde)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\xrightarrow\s*\{[^{}]*\}", " → ", s)
    s = re.sub(r"\\(?:boxed|operatorname|textsuperscript|textsubscript)\s*\{([^{}]*)\}",
               r"\1", s)
    s = _convert_frac_sqrt(s)
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    for pat, rep in _REPL:            # 注意：这些是正则，必须用 re.sub
        s = re.sub(pat, rep, s)
    s = re.sub(r"\\text[a-zA-Z]*", "", s)      # 兜底：去掉残余 \textxxx
    # 括号定界符与换行、数组环境
    s = re.sub(r"\\(?:left|right)\s*([()\[\]|.]|\{|\})?", r"\1", s)
    s = re.sub(r"\\(?:begin|end)\s*\{array\}\s*\{[^{}]*\}", " ", s)
    s = s.replace("\\\\", " ; ")
    s = s.replace("$", " ")
    s = re.sub(r"\\\|", "|", s)                # \| → 绝对值竖线
    s = s.replace("~", " ")                    # LaTeX 不断行空格
    s = re.sub(r"\\!", " ", s)
    s = re.sub(r"(?:\s*!\s*){2,}", " ", s)     # \qquad 等残留的 ! 序列
    s = re.sub(r"\^\s*\{([^{}]*)\}", r"^\1", s)
    s = re.sub(r"_\s*\{([^{}]*)\}", r"_\1", s)
    s = re.sub(r"[,\s]*&", "，", s)          # array 对齐符
    s = re.sub(r"\s*\^\s*", "^", s)
    s = re.sub(r"\s*_\s*", "_", s)
    # 花括号只是分组符，替换为空串（替换为空格会产生 "x ^2" 这类断裂）
    s = s.replace("\\", " ").replace("{", "").replace("}", "")
    s = re.sub(r"(?<!\w)\*\s*\d+\s*[lcrLCR](?![a-zA-Z])", " ", s)  # 列格式 *{35}{l} → *35l
    s = re.sub(r"\s+([,，。;；)）\]])", r"\1", s)
    s = re.sub(r"([(（\[])\s+", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_tex(s: str) -> str:
    """把 LaTeX 记号转成可读文本（用于填空题/应用题答案，尽量短）。"""
    return latex_to_text(clean_text(s))


# 英文自然语言高频词：命中多个说明整段是英文原题（TAL-SCQ5K 混有英文竞赛题）
_EN_STOPWORDS = {
    "the", "of", "is", "and", "let", "find", "what", "defined", "following",
    "such", "that", "for", "are", "was", "were", "be", "been", "this", "with",
    "as", "if", "then", "we", "have", "has", "given", "where", "which", "how",
    "many", "integer", "value", "compute", "determine", "show", "prove", "set",
    "function", "number", "numbers", "each", "from", "into", "when", "all",
    "there", "their", "them", "its", "not", "but", "can", "does", "do",
}
# 高中内容特征词：初中题库里不应出现
HIGH_SCHOOL = re.compile(
    r"双曲线|椭圆|正四面体|棱柱|棱锥|棱台|空间直角|弧度|\bcot\b|三角函数|导数|"
    r"\b向量\b|复数|排列组合|二项式|极限|映射|反三角|矩阵|行列式|外接球|内切球|"
    r"离心率|渐近线|准线|数学归纳法|恒成立问题|参数方程|极坐标|"
    # --- 以下为实测混入的高中/竞赛特征：函数板块曾出现反函数、等比数列、Σ 求和 ---
    r"反函数|互为反函数|等比数列|等差数列的前|求和公式|"
    r"\\sum|∑|连加|累乘|"
    r"\blog\s*_?\d|\bln\s*\(|指数函数y=a\^?|对数函数|"
    r"单调区间|定义域为R|值域为|复合函数|抽象函数|"
    r"递推公式|通项公式|数学期望|方差|标准差|概率分布|"
    r"不等式证明|\b均值不等式\b|柯西|"
    r"求导|积分|切线方程|法线|"
    # --- 高中必修概念：初中课标完全不涉及，可高置信过滤 ---
    r"奇函数|偶函数|周期函数|周期性|奇偶性|"
    r"\b数列\b|前n项和|"
    r"定义在实数集|实数集R|定义在R上|对于所有实数|对任意实数|任意实数x|"
    r"\bln\b|自然对数|\be\^|"
    r"∈\s*[NRZ]|N\^\*")


def is_english_original(text: str, threshold: int = 4) -> bool:
    """识别英文原题：英文自然词命中数 ≥ threshold 即判定为英文题。"""
    toks = [t.lower() for t in re.findall(r"[A-Za-z]{2,}", text or "")]
    if not toks:
        return False
    hits = sum(1 for t in toks if t in _EN_STOPWORDS)
    return hits >= threshold


def assign_topic(kp_text: str, stem: str) -> str:
    text = f"{kp_text} {stem}"
    for topic, pat in TOPIC_RULES:
        if pat.search(text):
            return topic
    return "综合"          # 未命中三大板块的初中题归入综合，避免数据浪费


def parse_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    stem = latex_to_text(clean_text(row.get("problem") or ""))
    if not stem or len(stem) < 8:
        return None
    if is_english_original(stem):
        return None

    options: List[str] = []
    answer_contents: Dict[str, str] = {}
    for grp in row.get("answer_option_list") or []:
        for o in grp:
            val = (o.get("aoVal") or "").strip().upper()
            content = strip_tex(clean_text(o.get("content") or ""))
            if not val or not content:
                continue
            options.append(f"{val}. {content}")
            answer_contents[val] = content

    ans_val = (row.get("answer_value") or "").strip().upper()
    if not ans_val or ans_val not in answer_contents:
        return None

    kps: List[str] = []
    for route in row.get("knowledge_point_routes") or []:
        for p in str(route).split("->"):
            p = p.strip()
            if p and p not in _KP_BLACKLIST and p not in kps:
                kps.append(p)
    kp_text = " ".join(kps)

    analysis = latex_to_text(" ".join(clean_text(a) for a in (row.get("answer_analysis") or [])))
    if is_english_original(analysis, threshold=8):
        analysis = ""
    difficulty = int(row.get("difficulty") or 0) + 1          # 0-4 → 1-5
    difficulty = max(1, min(5, difficulty))

    src = " ".join(row.get("competition_source_list") or [])
    topic = assign_topic(kp_text, stem)

    # 学段过滤：保留初中，剔除小学奥数与高中内容
    body = f"{kp_text} {stem} {src}"
    is_middle = bool(MIDDLE_SCHOOL.search(body))
    is_primary = bool(PRIMARY_SCHOOL.search(src)) and not re.search(r"[七八九]年级|初中|中考", src)
    if not is_middle or is_primary or HIGH_SCHOOL.search(body):
        return None

    return {
        "stem": stem,
        "options": options,
        "answer": ans_val,                       # 选择题标准答案（字母）
        "answer_content": answer_contents[ans_val],
        "analysis": analysis,
        "knowledge_points": kps[-4:],
        "difficulty": difficulty,
        "topic": topic,
        "is_app": bool(APP_RULES.search(kp_text)),
        "uid": row.get("queId") or row.get("qid") or "",
        "source": "TAL-SCQ5K",
    }


def to_records(parsed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """生成选择题 / 填空题 / 应用题三类记录。"""
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(parsed):
        if not p["topic"] or len(p["options"]) < 2:
            continue
        base = {
            "topic": p["topic"], "difficulty": p["difficulty"],
            "stem": p["stem"], "analysis": p["analysis"],
            "knowledge_points": p["knowledge_points"], "source": p["source"],
        }
        # question_id 必须与「原始数据」一一对应、与列表下标无关，
        # 否则重建 ETL 时旧数据会残留成脏数据。
        prefix = p.get("id_prefix", "TAL")
        qid = p["uid"]
        # 1) 选择题
        out.append({"question_id": f"{prefix}_C_{qid}", "type": "选择题",
                    "options": p["options"], "answer": p["answer"], **base})
        # 2) 填空题：正确答案内容较短（≤24 字符）时派生
        content = p["answer_content"]
        if 0 < len(content) <= 24:
            out.append({"question_id": f"{prefix}_F_{qid}", "type": "填空题",
                        "options": [], "answer": content, **base})
        # 3) 应用题：命中应用题标签，或题干较长（≥70 字，典型文字叙述题）
        if p["is_app"] or len(p["stem"]) >= 70:
            out.append({"question_id": f"{prefix}_A_{qid}", "type": "应用题",
                        "options": [], "answer": content, **base})
    return out


def load_raw() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for f in sorted(RAW_DIR.glob("TAL-SCQ5K-CN_*.jsonl")):
        with open(f, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
    return rows


SYNTH_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "synth_questions.jsonl"


def load_synthetic() -> List[Dict[str, Any]]:
    """载入模板生成的题目（scripts/gen_geometry_questions.py 产出）。

    TAL-SCQ5K 的初中子集里几何题不足 130 条、二元一次方程 175 条，
    达不到方案要求的「每板块 ≥200 题」，用参数化模板补齐。
    文件缺失时静默跳过，不影响主流程。
    """
    if not SYNTH_FILE.exists():
        print(f"  （未找到 {SYNTH_FILE}，跳过模板几何题；"
              f"可运行 scripts/gen_geometry_questions.py 生成）")
        return []
    out: List[Dict[str, Any]] = []
    with open(SYNTH_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
            except Exception:
                continue
            if q.get("stem") and len(q.get("options") or []) >= 2:
                q["id_prefix"] = "GEO"      # 与 TAL 区分，便于回溯来源
                out.append(q)
    return out


# ------------------------------------------------------------------ SXT001_CN
SXT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "sxt001"

_SXT_HTML_ENTITIES = {"&there4;": "∴", "&because;": "∵", "&nbsp;": " ",
                      "&le;": "≤", "&ge;": "≥", "&ne;": "≠", "&times;": "×",
                      "&div;": "÷", "&plusmn;": "±", "&pi;": "π", "&deg;": "°"}
# ${-7}^\circ C$ 这类「上标 °」记法先归一成 °，避免清洗后残留孤立 ^
_SXT_SUP_CIRC = re.compile(r"\{\}\s*\^\s*\\circ|\^\s*\\circ")


def _sxt_norm(s: str) -> str:
    """SXT001 原文字段进入通用清洗前的预处理：HTML 实体 + ^\\circ 归一。"""
    if not s:
        return s
    for ent, rep in _SXT_HTML_ENTITIES.items():
        s = s.replace(ent, rep)
    return _SXT_SUP_CIRC.sub("°", s)


def _sxt_unwrap(v: Any) -> List[str]:
    """SXT001 的 选择/分析 字段常是「字符串化的 Python list」，安全解出来。"""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        out: List[str] = []
        for x in v:
            out.extend(_sxt_unwrap(x))
        return out
    s = str(v).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return _sxt_unwrap(ast.literal_eval(s))
        except Exception:
            return [s]
    return [s] if s else []


def _sxt_options(sel: List[str]):
    """解析选择项：形如 "A : <p>$$3$$</p>." → [("A", "3")]。"""
    options: List[tuple] = []
    for it in sel:
        m = re.match(r"^\s*\(?([A-Da-d])\)?\s*[:：]\s*(.*)$", str(it))
        if not m:
            continue
        val = m.group(1).upper()
        content = strip_tex(_sxt_norm(m.group(2))).strip().rstrip("．.").strip()
        if content:
            options.append((val, content))
    return options


def load_sxt001() -> List[Dict[str, Any]]:
    """载入 ModelScope SXT001_CN 样例题（阿里云市场数学题库，仓库仅含公开样例）。

    原始字段：科目/答案/年级/一级知识点/题型/题干(HTML+$$LaTeX)/选择/分析/难易度/文件名称。
    完整版（初中约 1.2 万题）需在阿里云市场获取；下载更多样例后放入
    data/raw/sxt001/ 重跑本脚本即可自动并入。
    """
    if not SXT_DIR.exists():
        return []
    out: List[Dict[str, Any]] = []
    for f in sorted(SXT_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw_stem = _sxt_norm(str(d.get("题干") or ""))
        stem = latex_to_text(clean_text(raw_stem))
        if not stem or len(stem) < 8 or is_english_original(stem):
            continue

        options = _sxt_options(_sxt_unwrap(d.get("选择")))
        ans_flat = _sxt_unwrap(d.get("答案"))
        ans_val = ans_flat[0].strip().upper() if ans_flat else ""
        analysis = latex_to_text(" ".join(clean_text(_sxt_norm(a))
                                          for a in _sxt_unwrap(d.get("分析"))))
        kps = [k for k in (str(d.get("一级知识点") or "").replace("/", " ")
                           .replace("->", " ").split()) if k]
        kp_text = " ".join(kps)
        if not kp_text:
            kps, kp_text = [], ""
        try:
            difficulty = max(1, min(5, int(float(d.get("难易度") or 1))))
        except Exception:
            difficulty = 1

        body = f"{kp_text} {stem}"
        if HIGH_SCHOOL.search(body):
            continue
        if HIGH_SCHOOL.search(analysis or ""):
            analysis = ""               # 解析含高中内容时只丢解析，不丢题

        base = {"stem": stem, "analysis": analysis, "knowledge_points": kps[:4],
                "difficulty": difficulty,
                "topic": assign_topic(kp_text, stem),
                "is_app": bool(APP_RULES.search(kp_text)) or bool(APP_RULES.search(stem)),
                "uid": f.stem.removesuffix("_filtered").removesuffix(".json"),
                "source": "SXT001_CN", "id_prefix": "SXT"}

        if len(options) >= 2 and ans_val in dict(options):
            p = dict(base)
            p["options"] = [f"{v}. {c}" for v, c in options]
            p["answer"] = ans_val
            p["answer_content"] = dict(options)[ans_val]
            out.append(p)
        elif ans_val:
            # 无有效选项 → 按填空题入包（答案文本）
            ans_text = strip_tex(" ".join(ans_flat))
            if not ans_text:
                continue
            p = dict(base)
            p["options"] = []
            p["answer"] = ans_text
            p["answer_content"] = ans_text
            out.append(p)
    return out


def build_vector(records: List[Dict[str, Any]]) -> int:
    from math_agent.chunking import chunk_question
    from math_agent.vectorstore import get_store

    store = get_store()
    store.clear(store.kb)
    chunks = []
    for r in records:
        chunks.append({"text": chunk_question(r), "question_id": r["question_id"],
                       "topic": r["topic"], "type": r["type"],
                       "difficulty": r["difficulty"],
                       "knowledge_points": r["knowledge_points"],
                       "answer": r["answer"]})
    n = 0
    B = 256
    for i in range(0, len(chunks), B):
        n += store.upsert_knowledge(chunks[i:i + B])
        print(f"  向量化 {min(i + B, len(chunks))}/{len(chunks)}", end="\r")
    print()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-vector", action="store_true", help="同时构建向量知识库")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条原始数据（调试用）")
    args = ap.parse_args()

    raw = load_raw()
    if not raw:
        print("✘ data/raw 下没有原始数据，请先运行 scripts/download_dataset.py")
        return 1
    if args.limit:
        raw = raw[:args.limit]
    print(f"读取原始数据 {len(raw)} 条")

    parsed, seen_stem = [], set()
    dup = 0
    for p in (parse_row(r) for r in raw):
        if not p:
            continue
        # 原始数据集存在同题重复（不同 qid、相同题干），按归一化题干去重
        key = re.sub(r"[\s（）()、,，。．\.；;：:]+", "", p["stem"])[:120]
        if key in seen_stem:
            dup += 1
            continue
        seen_stem.add(key)
        parsed.append(p)
    n_tal = len(parsed)
    print(f"清洗+初中过滤后 {n_tal} 条（去重 {dup} 条）")

    # 模板生成题：补齐 TAL-SCQ5K 覆盖不足的「几何图像」「二元一次方程」板块
    synth = load_synthetic()
    added = 0
    for p in synth:
        key = re.sub(r"[\s（）()、,，。．\.；;：:]+", "", p["stem"])[:120]
        if key in seen_stem:
            continue
        seen_stem.add(key)
        parsed.append(p)
        added += 1
    if synth:
        n_geo = sum(1 for p in synth if p["topic"] == "几何图像")
        print(f"合并模板题 {added} 条（几何 {n_geo} / 代数 {len(synth) - n_geo}；"
              f"{len(synth) - added} 条与已有题重复被跳过）")

    # SXT001_CN 样例题（阿里云市场数学题库）
    sxt = load_sxt001()
    added = 0
    for p in sxt:
        key = re.sub(r"[\s（）()、,，。．\.；;：:]+", "", p["stem"])[:120]
        if key in seen_stem:
            continue
        seen_stem.add(key)
        parsed.append(p)
        added += 1
    if sxt:
        print(f"合并 SXT001_CN 样例 {added} 条（共 {len(sxt)} 条，"
              f"{len(sxt) - added} 条与已有题重复被跳过）")

    print("  板块分布：", Counter(p["topic"] for p in parsed).most_common())

    records = to_records(parsed)
    print(f"生成题目 {len(records)} 条")
    print("  题型分布：", Counter(r["type"] for r in records).most_common())

    if not args.limit:
        old = reset_questions()
        if old:
            print(f"  已清空旧题库 {old} 条")
    n = upsert_questions(records)
    print(f"✔ 写入题库 {n} 条；当前库存：{json.dumps(topic_stats(), ensure_ascii=False)}")

    if args.with_vector:
        print("构建向量知识库…")
        print(f"✔ 写入向量 {build_vector(records)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
