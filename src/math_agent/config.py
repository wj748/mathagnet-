"""统一配置加载：config.yaml + .env 覆盖。

设计要点
--------
* 所有路径以项目根目录为锚点，支持任意 cwd 调用。
* `LLM` 无 Key 时自动降级为离线规则模式（mock），保证全流程可跑通。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# ---------------------------------------------------------------- 路径常量
SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent          # src/
ROOT_DIR = PKG_DIR.parent         # 项目根
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROMPT_DIR = ROOT_DIR / "prompts"
WEB_DIR = ROOT_DIR / "web"

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, PROMPT_DIR, WEB_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    """极简 .env 解析（不强制依赖 python-dotenv）。"""
    env_file = ROOT_DIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """带层级访问的配置对象（cfg.get('llm.model')）。"""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> Dict[str, Any]:
        return dict(self._data.get(name, {}))

    @property
    def raw(self) -> Dict[str, Any]:
        return self._data

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Config {self._data.get('app', {})}>"


_DEFAULT_YAML = ROOT_DIR / "config.yaml"

_cached: Config | None = None


def load_config(path: str | Path | None = None, reload: bool = False) -> Config:
    global _cached
    if _cached is not None and not reload:
        return _cached
    _load_dotenv()
    cfg_path = Path(path) if path else _DEFAULT_YAML
    data: Dict[str, Any] = {}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    # 环境变量覆盖（MATH_AGENT_LLM_MODEL 之类）
    for k, v in os.environ.items():
        if k.startswith("MATH_AGENT_"):
            parts = [p.lower() for p in k[len("MATH_AGENT_"):].split("__")]
            node = data
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = v
    _cached = Config(data)
    return _cached


def resolve_path(p: str | Path) -> Path:
    """相对路径按项目根解析。"""
    path = Path(p)
    return path if path.is_absolute() else (ROOT_DIR / path)


cfg = load_config()

__all__ = ["cfg", "load_config", "resolve_path", "Config", "ROOT_DIR", "DATA_DIR",
           "RAW_DIR", "PROCESSED_DIR", "PROMPT_DIR", "WEB_DIR"]
