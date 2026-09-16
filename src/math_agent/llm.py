"""LLM 接入层：GLM / 任意 OpenAI 兼容端点，缺 Key 时优雅降级。

降级策略（关键设计）
--------------------
本项目所有关键行为（意图路由兜底、判分、错题报告、记忆隔离、循环终止）
均在 **代码层硬约束**，LLM 只负责润色与开放域答疑。
因此没有 API Key 时系统不是"报错"，而是自动进入 ``mock`` 模式：
全流程可跑通，只是文案由模板生成而非模型生成。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import cfg


class LLMProvider:
    def __init__(self, client=None, model: str = "", enabled: bool = True):
        self.client = client
        self.model = model
        self.enabled = enabled and client is not None

    # ------------------------------------------------------------------ 基础
    @property
    def available(self) -> bool:
        return self.enabled

    def chat(self,
             messages: Sequence[Tuple[str, str]] | str,
             system: str | None = None,
             temperature: float | None = None,
             max_tokens: int | None = None) -> Optional[str]:
        """返回模型文本；不可用时返回 None（调用方自行走兜底）。"""
        if not self.enabled:
            return None
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        except Exception:
            return None

        lc_msgs: List[Any] = []
        if system:
            lc_msgs.append(SystemMessage(content=system))
        if isinstance(messages, str):
            lc_msgs.append(HumanMessage(content=messages))
        else:
            for role, content in messages:
                if role in ("system", "sys"):
                    lc_msgs.append(SystemMessage(content=content))
                elif role in ("assistant", "ai"):
                    lc_msgs.append(AIMessage(content=content))
                else:
                    lc_msgs.append(HumanMessage(content=content))
        try:
            kwargs: Dict[str, Any] = {}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            resp = self.client.invoke(lc_msgs, **kwargs)
            return (resp.content or "").strip() or None
        except Exception as e:  # 网络/鉴权失败 → 本轮降级，不影响主流程
            print(f"[llm] 调用失败，本轮降级为规则模式：{type(e).__name__}: {e}")
            return None

    # ------------------------------------------------------------------ JSON
    _JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.S)

    def chat_json(self, prompt: str, system: str | None = None,
                  fallback: Any = None) -> Any:
        text = self.chat(prompt, system=system,
                         temperature=float(cfg.get("llm.temperature", 0.2)))
        if not text:
            return fallback
        # 去掉 ```json 代码块围栏
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        m = self._JSON_RE.search(text)
        if not m:
            return fallback
        try:
            return json.loads(m.group(0))
        except Exception:
            return fallback


# --------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------
_provider: LLMProvider | None = None


def build_provider(force: str | None = None) -> LLMProvider:
    global _provider
    if _provider is not None and force is None:
        return _provider

    mode = (force or cfg.get("llm.provider", "auto") or "auto").lower()
    if mode == "mock":
        _provider = LLMProvider(client=None, enabled=False)
        return _provider

    api_key = os.getenv(cfg.get("llm.api_key_env", "GLM_API_KEY"), "").strip()
    base_url = (os.getenv(cfg.get("llm.base_url_env", "GLM_BASE_URL"), "").strip()
                or cfg.get("llm.base_url_default", "https://open.bigmodel.cn/api/paas/v4/"))
    model = (os.getenv(cfg.get("llm.model_env", "GLM_MODEL"), "").strip()
             or cfg.get("llm.model_default", "glm-4-flash"))

    if mode == "openai_compatible" and not api_key:
        raise RuntimeError("llm.provider=openai_compatible 但未配置 API Key")

    if api_key:
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:
            print(f"[llm] langchain_openai 不可用（{e}），进入 mock 模式")
            _provider = LLMProvider(client=None, enabled=False)
            return _provider
        client = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=float(cfg.get("llm.temperature", 0.2)),
            max_tokens=int(cfg.get("llm.max_tokens", 2048)),
            timeout=int(cfg.get("llm.timeout", 60)),
        )
        _provider = LLMProvider(client=client, model=model, enabled=True)
        print(f"[llm] 已接入 {model} @ {base_url}")
    else:
        print("[llm] 未检测到 API Key，进入离线规则模式（全流程可用，文案为模板生成）")
        _provider = LLMProvider(client=None, enabled=False)
    return _provider


def get_llm(force: str | None = None) -> LLMProvider:
    return build_provider(force)


# --------------------------------------------------------------------------
# 视觉模型（Phase 5 待办②：GLM-4V 提升手写体识别）
# --------------------------------------------------------------------------
_vision_client = None


def get_vision_client():
    """返回视觉模型客户端（独立于文本模型，如 glm-4v-flash）；不可用返回 None。

    ⚠️ 文本模型（glm-4-flash）**不支持**图片输入——OCR 多模态兜底必须走本客户端，
    之前直接复用文本 client 会报"模型不支持图片"并静默降级，等于视觉能力缺失。
    """
    global _vision_client
    if _vision_client is not None:
        return _vision_client
    api_key = os.getenv(cfg.get("llm.api_key_env", "GLM_API_KEY"), "").strip()
    if not api_key:
        return None
    base_url = (os.getenv(cfg.get("llm.base_url_env", "GLM_BASE_URL"), "").strip()
                or cfg.get("llm.base_url_default", "https://open.bigmodel.cn/api/paas/v4/"))
    model = (os.getenv("GLM_VISION_MODEL", "").strip()
             or cfg.get("llm.vision_model_default", "glm-4v-flash"))
    try:
        from langchain_openai import ChatOpenAI
        _vision_client = ChatOpenAI(
            model=model, api_key=api_key, base_url=base_url,
            temperature=0.1, timeout=int(cfg.get("llm.timeout", 60)),
        )
        print(f"[llm] 视觉模型已接入 {model} @ {base_url}")
        return _vision_client
    except Exception as e:
        print(f"[llm] 视觉模型不可用（{e}），OCR 走本地/纯文本链")
        return None


__all__ = ["LLMProvider", "get_llm", "build_provider", "get_vision_client"]
