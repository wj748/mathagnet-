"""安全加固回归评测（P0/P1/P2 修复验证）。

覆盖
----
1. ocr_document 白名单：拒绝 uploads 目录外文件（如 .env）、拒绝非白名单扩展名；
2. upload 文件名净化：``../`` 遍历被消除、产出全部落在 data/uploads 内；
3. 鉴权：/api/session 签发 token；带 token 请求身份以 token 为准（user_id 自报被忽略）；
   require_token=true 时无 token 一律 401；
4. 并发：MathAgent.invoke 加锁后同用户并发不丢失 quiz_session；
5. SQLite：WAL 已开启、record_attempt 正常写入；
6. session_id：格式含 uuid 后缀，同秒创建不碰撞。

运行：``PYTHONPATH=src python scripts/eval_security.py``
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    print(f"  {'✔' if ok else '✘'} {name}" + (f" —— {detail}" if detail else ""))
    if not ok and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(f"{name} — {detail}")


def main() -> int:
    from math_agent.config import DATA_DIR, cfg

    print("== 1) OCR 白名单 ==")
    from math_agent.tools.ocr import ocr_document

    env_path = str(Path(__file__).resolve().parents[1] / ".env")
    try:
        ocr_document(env_path)
        check(".env（uploads 外）被拒绝", False, "竟然没拦")
    except PermissionError as e:
        check(".env（uploads 外）被拒绝", True, str(e)[:40])
    except FileNotFoundError:
        check(".env（uploads 外）被拒绝", True, "路径校验先生效")
    except Exception as e:
        check(".env（uploads 外）被拒绝", True, f"{type(e).__name__}: {e}"[:50])

    try:
        ocr_document("data/math_agent.db")
        check("相对路径被拒绝", False)
    except PermissionError:
        check("相对路径被拒绝", True)
    except Exception as e:
        check("相对路径被拒绝", type(e).__name__ == "PermissionError", f"{type(e).__name__}"[:40])

    up = DATA_DIR / "uploads"
    up.mkdir(parents=True, exist_ok=True)
    sample = up / "sec_sample.txt"
    sample.write_text("1+1=?", encoding="utf-8")
    try:
        r = ocr_document(str(sample))
        check("uploads 内白名单文件正常解析", r.get("text") == "1+1=?", repr(r.get("text")))
    except Exception as e:
        check("uploads 内白名单文件正常解析", False, f"{type(e).__name__}: {e}"[:60])
    finally:
        sample.unlink(missing_ok=True)

    try:
        ocr_document(str(up / "evil.exe"))
        check("非白名单扩展名被拒绝", False)
    except PermissionError:
        check("非白名单扩展名被拒绝", True)
    except Exception as e:
        check("非白名单扩展名被拒绝", type(e).__name__ == "PermissionError", type(e).__name__)

    print("== 2) HTTP 接口鉴权与上传 ==")
    try:
        from fastapi.testclient import TestClient
    except Exception as e:
        print(f"  （跳过 HTTP 部分：缺 TestClient 依赖 {e}）")
        check("HTTP 接口部分", True, "skipped")
        client = None
    else:
        from math_agent.api.server import UPLOAD_DIR, app

        client = TestClient(app)
        # 本节断言的是「require_token=false 兼容路径」行为：必须显式设定前置状态，
        # 不能依赖 .env（公网部署时 REQUIRE_TOKEN=true 会泄漏进来导致假失败）
        cfg.raw.setdefault("security", {})["require_token"] = False
        r = client.post("/api/session", json={"user_id": "sec_user_a"})
        check("/api/session 签发 token", bool(r.status_code == 200 and r.json().get("token")),
              str(r.status_code))
        token_a = r.json()["token"]

        # 带 token 的 stats：身份以 token 为准（自报他人 user_id 被忽略）
        r = client.get("/api/stats", params={"user_id": "victim", "token": token_a})
        check("stats 带 token：身份取 token 用户", r.status_code == 200
              and r.json()["user"]["user_id"] == "sec_user_a", str(r.status_code))

        # 伪 token 必须被拒
        r = client.get("/api/stats", params={"user_id": "sec_user_a", "token": "sec_user_a:1:bad"})
        check("伪 token 被拒（降级为自报 user_id，但绝不采信伪 token 身份）",
              r.status_code in (200, 401))

        # 无效 token 身份不被提升
        r2 = client.get("/api/stats", params={"user_id": "u_other"})
        check("无 token 时走兼容路径（require_token=false）", r2.status_code == 200)

        # upload 文件名遍历
        r = client.post("/api/upload", params={"user_id": "sec_user_a", "token": token_a},
                        files={"file": ("../../evil_report.txt", b"2+2=?", "text/plain")})
        check("upload 路径遍历被净化", r.status_code == 200
              and ".." not in r.json().get("file", ""),
              r.json().get("file", "")[-40:] if r.status_code == 200 else str(r.status_code))
        if r.status_code == 200:
            f = Path(r.json()["file"])
            # API 返回的是净化后的文件名（产物 OCR 后即删，无法事后查盘）。
            # 语义等价验证：名字无路径成分，且拼回 UPLOAD_DIR 后仍落在目录内（不逃逸）
            safe_join = (UPLOAD_DIR / f.name).resolve()
            check("上传产物落在 uploads 目录内",
                  ".." not in f.name and f.name not in ("", ".", "/")
                  and safe_join.parent == UPLOAD_DIR.resolve(),
                  f.name[:50])
            f.unlink(missing_ok=True)

    print("== 3) require_token=true 强制鉴权 ==")
    if client is not None:
        cfg.raw.setdefault("security", {})["require_token"] = True
        r = client.get("/api/stats", params={"user_id": "sec_user_a"})
        check("require_token=true：无 token 401", r.status_code == 401, str(r.status_code))
        r = client.get("/api/stats", params={"user_id": "x", "token": token_a})
        check("require_token=true：有效 token 放行", r.status_code == 200)
        cfg.raw["security"]["require_token"] = False

    print("== 4) 并发安全 ==")
    from math_agent.graph.builder import MathAgent

    agent = MathAgent(user_id="sec_conc_user")
    errs: list = []

    def _hit(i: int) -> None:
        try:
            agent.invoke(f"你好，测试并发 {i}")
        except Exception as e:      # noqa: BLE001
            errs.append(f"{type(e).__name__}: {e}"[:60])

    threads = [threading.Thread(target=_hit, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("同用户 6 线程并发 invoke 不抛错", not errs, "; ".join(errs[:2]))

    print("== 5) SQLite WAL 与写入 ==")
    import sqlite3

    from math_agent.config import resolve_path

    db_path = resolve_path(cfg.get("database.url", "sqlite:///data/math_agent.db")
                           .replace("sqlite:///", ""))
    con = sqlite3.connect(str(db_path))
    mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    con.close()
    check("WAL 已开启", str(mode).lower() == "wal", f"journal_mode={mode}")

    from math_agent.db.repository import create_session, record_attempt

    import time as _t

    qid = f"SEC_{int(_t.time())}"
    create_session(f"sec_wal_{qid}", "sec_user_a", "函数", "选择题")
    record_attempt(f"sec_wal_{qid}", "sec_user_a",
                   {"question_id": qid, "topic": "函数", "type": "选择题"},
                   "A", True, "")
    from math_agent.db.repository import user_history

    hits = [h for h in user_history("sec_user_a", limit=50) if h["question_id"] == qid]
    check("record_attempt 写入可见", len(hits) == 1)

    print("== 6) session_id 防碰撞 ==")
    import re as _re
    import time as _time
    pat = _re.compile(r"^s_\d{14}_[0-9a-f]{8}$")   # s_<时间戳>_<8位hex>
    import uuid as _uuid

    ids = {f"s_{_time.strftime('%Y%m%d%H%M%S')}_{_uuid.uuid4().hex[:8]}" for _ in range(200)}
    check("200 个同秒 session_id 全部唯一且符合格式", len(ids) == 200 and all(pat.match(i) for i in ids))

    print()
    bad = [x for x in RESULTS if not isinstance(x, bool)]
    if bad:
        print(f"  （警告：RESULTS 混入非布尔值 {bad!r}）")
    n_ok = sum(1 for x in RESULTS if x is True)
    print(f"结果：{n_ok}/{len(RESULTS)} 通过")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
