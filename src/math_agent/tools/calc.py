"""数学计算工具：SymPy 安全求值（不走 LLM，杜绝算术幻觉）。"""
from __future__ import annotations

import re
from typing import Any, Dict

from pydantic import BaseModel, Field

from .registry import register_tool


class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="数学表达式，如 '(2/3+0.25)*12' 或 'solve(2*x+3=7, x)' 或 'diff(x**2, x)'")


_ALLOWED_PREFIX = ("solve", "diff", "integrate", "simplify", "expand",
                   "factor", "limit", "Matrix", "sqrt", "Rational")

# SymPy 安全求值白名单：sympify 对不可信输入等同于 eval（可被诱导 RCE），
# LLM Action 参数同样不可信，计算前必须过此校验。
_EXPR_CHARS = re.compile(r"^[0-9a-zA-Z\s+\-*/^().,=<>!%:\[\]{}]+$")
_EXPR_BLOCK = re.compile(
    r"__|import|lambda|class|\bdef\b|mro|subclass|globals|locals|builtins|"
    r"getattr|setattr|\bopen\b|\beval\b|\bexec\b|compile|\bvars\b|\bos\.|\bsys\.|"
    r"\.pop|\.read|\.write|\.system", re.I)


def _safe_math_expr(s: str) -> bool:
    """仅允许数学表达式字符集，且不含属性访问 / 导入 / 内调等危险片段。"""
    s = s or ""
    if not s or len(s) > 500:
        return False
    return bool(_EXPR_CHARS.match(s)) and not _EXPR_BLOCK.search(s)


@register_tool(
    "calculator",
    "计算数学表达式 / 解方程 / 求导。基于 SymPy 精确计算，可用于验算。",
    CalculatorArgs,
)
def calculator(expression: str) -> Dict[str, Any]:
    import sympy

    expr = (expression or "").strip()
    if not expr:
        raise ValueError("expression 不能为空")
    if len(expr) > 300:
        raise ValueError("表达式过长")
    if not _safe_math_expr(expr):
        raise ValueError("表达式含不安全的字符或关键字，已拒绝计算")
    try:
        if any(expr.startswith(p) for p in _ALLOWED_PREFIX) and "(" in expr:
            val = sympy.sympify(expr, rational=True)
        else:
            val = sympy.sympify(expr.replace("^", "**"), rational=True)
        res = sympy.simplify(val)
        try:
            numeric = float(res.evalf())
        except Exception:
            numeric = None
        return {"expression": expr, "result": str(res), "numeric": numeric}
    except Exception as e:
        raise ValueError(f"无法计算表达式 {expr!r}：{type(e).__name__}: {e}")


__all__ = ["calculator"]
