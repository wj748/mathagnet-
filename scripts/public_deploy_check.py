# -*- coding: utf-8 -*-
"""公网部署自检脚本（《公网部署清单.md》第五节的自动化实现）。

用法:
    python scripts/public_deploy_check.py [BASE_URL]
    # BASE_URL 默认 https://你的域名 ；本地自检传 http://127.0.0.1:8787

五项检查:
    1. /api/health 健康
    2. 无 token 调 /api/chat 必须 401（require_token 生效）
    3. 首页 HTML 可达（前端静态资源）
    4. HTTPS 证书有效期（仅 https:// 目标检查，自签会报 WARN）
    5. .env 未被 git 跟踪

退出码: 0=全部通过, 1=存在 FAIL
"""
import json
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = (sys.argv[1] if len(sys.argv) > 1 else "https://127.0.0.1:8787").rstrip("/")
ROOT = Path(__file__).resolve().parent.parent
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE  # 自签/验证由第 4 步单独做

results = []


def record(name, ok, detail="", warn=False):
    results.append(ok)
    tag = "WARN" if warn else ("PASS" if ok else "FAIL")
    print(f"{tag:4s} {name}" + (f" | {detail}" if detail else ""))


def fetch(path, method="GET", timeout=15):
    req = urllib.request.Request(BASE + path, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# 1) 健康
try:
    code, body = fetch("/api/health")
    h = json.loads(body) if code == 200 else {}
    record("1. /api/health 健康", code == 200 and h.get("ok") is True,
           f"llm={h.get('llm')} embedder_ready={h.get('embedder_ready')}")
except Exception as e:
    record("1. /api/health 健康", False, f"{type(e).__name__}: {e}")

# 2) 无 token 必须 401（body 必须合法，否则先触发 422 参数校验、走不到鉴权）
try:
    req = urllib.request.Request(BASE + "/api/chat", method="POST",
                                 data=json.dumps({"message": "ping"}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code
    record("2. 无 token 访问 /api/chat 被拒", code == 401, f"实际返回 {code}")
except Exception as e:
    record("2. 无 token 访问 /api/chat 被拒", False, f"{type(e).__name__}: {e}")

# 3) 首页可达
try:
    code, body = fetch("/")
    ok = code == 200 and b"katex" in body.lower()
    record("3. 前端首页可达", ok, f"HTTP {code}")
except Exception as e:
    record("3. 前端首页可达", False, f"{type(e).__name__}: {e}")

# 4) HTTPS 证书有效期（仅 https 目标）
if BASE.startswith("https://"):
    host = BASE.split("//", 1)[1].split("/")[0].split(":")[0]
    port = 443
    if ":" in BASE.split("//", 1)[1].split("/")[0]:
        port = int(BASE.split("//", 1)[1].split("/")[0].split(":")[1])
    try:
        cert = ssl.get_server_certificate((host, port))
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False) as f:
            f.write(cert)
            tmp = f.name
        x509 = ssl._ssl._test_decode_cert(tmp)  # noqa: SLF001 标准库私有 API，仅诊断用
        not_after = datetime.strptime(x509["notAfter"], "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        days = (not_after - datetime.now(timezone.utc)).days
        issuer = x509.get("issuer", ())
        self_signed = any("localhost" in str(v) or "self" in str(v).lower() for _, _, v in issuer)
        if self_signed:
            record("4. TLS 证书有效期", True, f"自签证书（上线请换 Let's Encrypt）", warn=True)
        elif days < 14:
            record("4. TLS 证书有效期", True, f"剩 {days} 天，即将到期", warn=True)
        else:
            record("4. TLS 证书有效期", True, f"剩 {days} 天")
        Path(tmp).unlink(missing_ok=True)
    except Exception as e:
        record("4. TLS 证书有效期", False, f"{type(e).__name__}: {e}")
else:
    record("4. TLS 证书有效期", True, "目标非 https，公网部署时必须启用 TLS", warn=True)

# 5) .env 未被 git 跟踪
try:
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", ".env"],
                             cwd=ROOT, capture_output=True, text=True)
    if tracked.returncode != 0:
        record("5. .env 未被 git 跟踪", True, "密钥不会进仓库")
    else:
        record("5. .env 未被 git 跟踪", False, "危险：.env 已被跟踪，立即 git rm --cached .env")
except FileNotFoundError:
    record("5. .env 未被 git 跟踪", True, "未安装 git，跳过", warn=True)

n_ok = sum(1 for r in results if r)
print(f"\n结果: {n_ok}/{len(results)} 项通过")
sys.exit(0 if all(results) else 1)
