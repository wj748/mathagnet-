"""工具注册中心 + 防重/防错拦截器（方案 4.4）。

三重防护
--------
1. **结构化参数**：Pydantic v2 严格校验，非法参数直接拒绝并回注错误信息。
2. **防重拦截**：``tool_name + arguments_hash`` 命中历史即拒绝，
   回注「该工具已用相同参数调用过…」要求 LLM 自我纠正。
3. **依赖链约束**：OCR → 批改 的调用顺序在代码层强制（``requires``）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError


@dataclass
class ToolResult:
    ok: bool
    name: str
    output: Any = None
    error: str = ""
    blocked: bool = False
    args_hash: str = ""          # 基于「校验后参数」的哈希，保证与防重判定一致

    def to_observation(self) -> str:
        if self.blocked or not self.ok:
            return f"[工具 {self.name} 调用失败] {self.error}"
        if isinstance(self.output, str):
            return self.output
        return json.dumps(self.output, ensure_ascii=False, default=str)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "name": self.name, "blocked": self.blocked,
                "output": self.output, "error": self.error}


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: Type[BaseModel]
    func: Callable[..., Any]
    requires: List[str] = field(default_factory=list)   # 前置工具名
    retryable: bool = True          # False=有副作用（写入类），失败不自动重试（幂等保护）
    timeout: Optional[float] = None  # 单工具超时秒数；None 用 toolguard.default_timeout_s


_REGISTRY: Dict[str, ToolSpec] = {}


def register_tool(name: str, description: str, args_model: Type[BaseModel],
                  requires: Optional[List[str]] = None,
                  retryable: bool = True, timeout: Optional[float] = None):
    def deco(fn):
        fn._tool_meta_ = (name, description, args_model, requires or [])
        _REGISTRY[name] = ToolSpec(name, description, args_model, fn, requires or [],
                                   retryable=retryable, timeout=timeout)
        return fn
    return deco


def get_tool(name: str) -> ToolSpec:
    if name not in _REGISTRY:
        raise KeyError(f"未注册的工具：{name}；已注册：{sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_tools() -> List[Dict[str, Any]]:
    return [{"name": t.name, "description": t.description,
             "parameters": t.args_model.model_json_schema(),
             "requires": t.requires} for t in _REGISTRY.values()]


def as_langchain_tools():
    """转成 LangChain Tool（真实 LLM 模式下用于 bind_tools）。"""
    from langchain_core.tools import StructuredTool

    tools = []
    for spec in _REGISTRY.values():
        def _run(_spec=spec, **kwargs):
            return run_tool(_spec.name, kwargs).to_observation()
        tools.append(StructuredTool.from_function(
            func=_run, name=spec.name, description=spec.description,
            args_schema=spec.args_model))
    return tools


# ------------------------------------------------------------------ 拦截
def args_hash(name: str, args: Dict[str, Any]) -> str:
    payload = json.dumps({"n": name, "a": args}, sort_keys=True,
                         ensure_ascii=False, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


class DuplicateToolCall(RuntimeError):
    pass


def run_tool(name: str,
             raw_args: Dict[str, Any],
             history: Optional[List[Dict[str, Any]]] = None,
             check_duplicate: bool = True,
             **context: Any) -> ToolResult:
    """统一工具执行入口：校验 → 依赖检查 → 防重 → 执行 → 错误回注。

    同时把 ok / blocked / error 结果记入 metrics（方案 Phase 5 可观测性）。
    """
    from .. import metrics

    result = _run_tool_inner(name, raw_args, history, check_duplicate, **context)
    metrics.record_tool_call(name, result.ok, result.blocked)
    return result


def _run_tool_inner(name: str,
                    raw_args: Dict[str, Any],
                    history: Optional[List[Dict[str, Any]]] = None,
                    check_duplicate: bool = True,
                    **context: Any) -> ToolResult:
    history = history if history is not None else []

    try:
        spec = get_tool(name)
    except KeyError as e:
        return ToolResult(ok=False, name=name, error=str(e))

    # 1) 参数校验
    try:
        args = spec.args_model(**(raw_args or {})).model_dump()
    except ValidationError as e:
        return ToolResult(ok=False, name=name,
                          error=f"参数校验失败：{e.errors()[0]['msg']}；"
                                f"正确 schema: {spec.args_model.model_json_schema()}")
    # 2) 依赖链
    called = {h.get("name") for h in history if h.get("ok")}
    missing = [r for r in spec.requires if r not in called]
    if missing:
        return ToolResult(ok=False, name=name,
                          error=f"依赖链约束：调用 {name} 之前必须先成功调用 {missing}。")

    # 3) 防重（哈希基于校验后的 args，与 record_history 保持一致）
    ah = args_hash(name, args)
    if check_duplicate and any(h.get("args_hash") == ah for h in history):
        return ToolResult(ok=False, name=name, blocked=True, args_hash=ah,
                          error="该工具已用相同参数调用过，请使用已有结果或更改参数。")

    # 4) 执行（toolguard 守卫：熔断 → 线程池超时 → 失败分类重试 → 错误脱敏）
    from ..toolguard import execute_with_guard
    call_args = {**args, **context} if context else args
    ok, out, err, rejected = execute_with_guard(spec, call_args)
    if ok:
        return ToolResult(ok=True, name=name, output=out, args_hash=ah)
    return ToolResult(ok=False, name=name, args_hash=ah, error=err, blocked=rejected)


def record_history(history: List[Dict[str, Any]], name: str,
                   raw_args: Dict[str, Any], result: ToolResult) -> None:
    history.append({"name": name, "args": raw_args,
                    "args_hash": getattr(result, "args_hash", "") or args_hash(name, raw_args),
                    "ok": result.ok, "blocked": result.blocked,
                    "error": result.error})


__all__ = ["register_tool", "get_tool", "list_tools", "run_tool", "record_history",
           "args_hash", "ToolResult", "ToolSpec", "DuplicateToolCall", "as_langchain_tools"]
