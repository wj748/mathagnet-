"""会话鉴权（P0：user_id 不可自报越权）。

机制
----
* ``POST /api/session`` 用 user_id 换取 HMAC 签名 token（``user_id:ts:sig``）。
* 受保护接口凭 token 识别身份：user_id 从 token 解出，**客户端传入的 user_id 一律忽略**。
* ``security.require_token: true`` 时，无有效 token 的请求一律 401；
  默认 false（本地开发兼容旧行为），公网部署务必开启。
* 密钥优先取环境变量 ``MATH_AGENT_SECRET``，否则自动生成并持久化到 ``data/.secret_key``。
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Optional, Tuple

from ..config import DATA_DIR, cfg

_SECRET: Optional[bytes] = None
TOKEN_TTL = 30 * 24 * 3600          # token 有效期 30 天


def _secret() -> bytes:
    global _SECRET
    if _SECRET is None:
        import os

        env = os.environ.get("MATH_AGENT_SECRET", "")
        if env:
            _SECRET = env.encode("utf-8")
        else:
            key_file = DATA_DIR / ".secret_key"
            if key_file.exists():
                _SECRET = key_file.read_text(encoding="utf-8").strip().encode("utf-8")
            else:
                _SECRET = secrets.token_hex(32).encode("utf-8")
                key_file.write_text(_SECRET.decode("ascii"), encoding="ascii")
    return _SECRET


def issue_token(user_id: str) -> str:
    ts = str(int(time.time()))
    msg = f"{user_id}:{ts}".encode("utf-8")
    sig = hmac.new(_secret(), msg, hashlib.sha256).hexdigest()[:32]
    return f"{user_id}:{ts}:{sig}"


def verify_token(token: str) -> Optional[str]:
    """校验通过返回 user_id，失败返回 None。"""
    try:
        user_id, ts, sig = token.split(":", 2)
    except ValueError:
        return None
    if not user_id or len(user_id) > 64:
        return None
    if abs(time.time() - int(ts)) > TOKEN_TTL:
        return None
    msg = f"{user_id}:{ts}".encode("utf-8")
    expect = hmac.new(_secret(), msg, hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(sig, expect):
        return None
    return user_id


def resolve_user(token: Optional[str], user_id: Optional[str]) -> Tuple[Optional[str], bool]:
    """返回 (生效的 user_id, 是否已鉴权)。require_token=true 且未鉴权时返回 (None, False)。"""
    if token:
        uid = verify_token(token)
        if uid:
            return uid, True
    require = bool(cfg.get("security.require_token", False))
    if require:
        return None, False
    return (user_id or "u_default"), False


__all__ = ["issue_token", "verify_token", "resolve_user"]
