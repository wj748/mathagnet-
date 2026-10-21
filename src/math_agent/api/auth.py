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
import os
import re
import secrets
import threading
import time
from typing import Optional, Tuple

from ..config import DATA_DIR, cfg

_SECRET: Optional[bytes] = None
_SECRET_LOCK = threading.Lock()
TOKEN_TTL = 30 * 24 * 3600          # token 有效期 30 天

# user_id 字符白名单：仅字母/数字/下划线/连字符/中文，长度 1-64。
# 这是上传文件名拼接与路径安全的前置约束——含 "/"、".."、控制字符的 user_id
# 一律拒绝，从源头堵死路径遍历与畸形 user_id 越权枚举。
_UID_RE = re.compile(r"^[0-9A-Za-z_\-\u4e00-\u9fa5]{1,64}$")


def valid_user_id(user_id: Optional[str]) -> bool:
    """user_id 合法性校验（签发 token / 校验 token 共用）。"""
    return bool(user_id) and bool(_UID_RE.match(user_id))


def _secret() -> bytes:
    global _SECRET
    if _SECRET is None:
        with _SECRET_LOCK:                      # 双检锁：防首次并发重复生成/重复写文件
            if _SECRET is None:
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
                        try:
                            os.chmod(key_file, 0o600)   # 密钥文件最小权限
                        except OSError:
                            pass
    return _SECRET


def issue_token(user_id: str) -> str:
    if not valid_user_id(user_id):
        raise ValueError("user_id 含非法字符（仅允许字母/数字/下划线/连字符/中文，1-64 位）")
    ts = str(int(time.time()))
    msg = f"{user_id}:{ts}".encode("utf-8")
    sig = hmac.new(_secret(), msg, hashlib.sha256).hexdigest()[:32]
    return f"{user_id}:{ts}:{sig}"


def verify_token(token: str) -> Optional[str]:
    """校验通过返回 user_id，失败返回 None（绝不因畸形 token 抛异常致 500）。"""
    try:
        user_id, ts, sig = token.split(":", 2)
    except ValueError:
        return None
    if not user_id or len(user_id) > 64:
        return None
    if not ts.isdigit():                     # 时间戳必须是数字，避免 int() 抛错
        return None
    if abs(time.time() - int(ts)) > TOKEN_TTL:
        return None
    if not valid_user_id(user_id):           # token 内的 user_id 也须过白名单
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


__all__ = ["issue_token", "verify_token", "resolve_user", "valid_user_id"]
