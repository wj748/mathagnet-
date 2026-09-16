"""FastAPI 服务化（Phase 5）：REST + SSE 流式接口 + 文件上传。"""
from __future__ import annotations

import json
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from math_agent.config import DATA_DIR, WEB_DIR, cfg  # noqa: E402
from math_agent.concurrency import get_gate, get_breaker, get_idempotency  # noqa: E402
from math_agent.db.repository import topic_stats, user_history  # noqa: E402
from math_agent.graph.builder import MathAgent  # noqa: E402
from math_agent.llm import get_llm  # noqa: E402

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = float(cfg.get("security.max_upload_mb", 10))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

import logging  # noqa: E402

log = logging.getLogger("math_agent.api")


def _require_admin(token: Optional[str]) -> None:
    """监控接口（/api/metrics、/api/traces、/trace）鉴权门：
    require_token=true 时必须携带有效 token，防止运维数据（调用链/指标）对外裸奔。"""
    if not bool(cfg.get("security.require_token", False)):
        return
    from .auth import resolve_user

    _uid, authed = resolve_user(token, "")
    if not authed:
        raise HTTPException(status_code=401, detail="监控接口需要有效 token")

def _warmup() -> None:
    """后台预热向量模型，避免首个真实请求被 BGE 首加载阻塞。"""
    try:
        from math_agent.embeddings import get_embedder

        print(f"[warmup] embedder = {get_embedder().name}", flush=True)
    except Exception as e:
        print(f"[warmup] 跳过（{type(e).__name__}: {e}）", flush=True)


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(title="初中数学辅导 Agent", version="1.0.0", lifespan=lifespan)
_CORS_ORIGINS = cfg.get("security.cors_origins",
                        ["http://127.0.0.1:8787", "http://localhost:8787"])
app.add_middleware(CORSMiddleware, allow_origins=list(_CORS_ORIGINS),
                   allow_methods=["*"], allow_headers=["*"])

# 静态资源（KaTeX 等前端库本地化在 web/ 下，离线可用）
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# Agent 池：TTL 淘汰 + 数量上限（防随机 user_id 打爆内存；compressor 已持久化，可安全重建）
_AGENTS: Dict[str, tuple] = {}          # user_id -> (agent, last_used_ts)
_AGENT_TTL = float(cfg.get("security.agent_ttl", 1800))
_AGENT_MAX = int(cfg.get("security.agent_max", 100))
_AGENTS_LOCK = threading.Lock()


def _agent(user_id: str) -> MathAgent:
    now = time.time()
    with _AGENTS_LOCK:
        hit = _AGENTS.get(user_id)
        if hit is not None:
            _AGENTS[user_id] = (hit[0], now)
            return hit[0]
        # 顺手淘汰过期与超容（最久未用先出）
        stale = [k for k, (_, t) in _AGENTS.items() if now - t > _AGENT_TTL]
        for k in stale:
            _AGENTS.pop(k, None)
        while len(_AGENTS) >= _AGENT_MAX:
            oldest = min(_AGENTS, key=lambda k: _AGENTS[k][1])
            _AGENTS.pop(oldest, None)
        agent = MathAgent(user_id=user_id)
        _AGENTS[user_id] = (agent, now)
        return agent


class ChatRequest(BaseModel):
    message: str
    user_id: str = "u_default"
    token: Optional[str] = None
    mode: str = ""          # "" 默认（刷题/答疑） | "learn" 学习知识点
    request_id: str = ""    # 幂等键：同 user 同 request_id 在 TTL 内重放返回首次结果


class ChatResponse(BaseModel):
    reply: str
    intent: str = ""
    trace: list = []
    report: Optional[Dict[str, Any]] = None
    elapsed_ms: int = 0
    idempotent_replay: bool = False


# ---- 并发治理统一文案（面向初中生，直接进前端气泡） ----
_MSG_BUSY = "你的上一条消息还在处理中，等它完成再发哦～"
_MSG_QPS = "发送太频繁啦，休息一下再试～"
_MSG_OVERLOAD = "现在学习的小伙伴太多啦，请等几秒再试～"


def _admit(user_id: str) -> None:
    """并发准入：同用户互斥 + QPS + 全局熔断。不通过直接抛 429/503。

    调用方必须在 finally 里：get_breaker().exit() + get_gate().release(user_id)
    （仅当两个 enter 都成功时才需要 exit；本函数保证要么全成功要么已回滚）。
    返回值无意义，只为类型完备。
    """
    from math_agent import metrics as _m

    gate, breaker = get_gate(), get_breaker()
    ok, reason = gate.acquire(user_id)
    if not ok:
        _m.record_concurrency(reason)          # busy / qps
        raise HTTPException(status_code=429,
                            detail=_MSG_BUSY if reason == "busy" else _MSG_QPS)
    entered, breason = breaker.enter()
    if not entered:                            # overload / queue_timeout
        gate.release(user_id)
        _m.record_concurrency(breason)
        raise HTTPException(status_code=503, detail=_MSG_OVERLOAD)
    _m.record_concurrency("admitted")


class SessionRequest(BaseModel):
    user_id: str = "u_default"


@app.post("/api/session")
def create_session_token(req: SessionRequest):
    """用 user_id 换取签名 token；此后接口身份以 token 为准（客户端自报 user_id 不再被信任）。"""
    from .auth import issue_token

    user_id = (req.user_id or "u_default").strip()[:64]
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 不能为空")
    return {"user_id": user_id, "token": issue_token(user_id)}


@app.get("/", response_class=HTMLResponse)
def index():
    idx = WEB_DIR / "index.html"
    if not idx.exists():
        return HTMLResponse("<h1>Web UI 未找到</h1>")
    return FileResponse(idx)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    from .auth import resolve_user

    user_id, authed = resolve_user(req.token, req.user_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="缺少有效 token，请先调用 /api/session")
    _admit(user_id)                             # 同用户互斥 + QPS + 全局熔断
    t0 = time.time()
    # 幂等：同 user + request_id 在 TTL 内直接回放首次结果，不重复执行智能体
    rid = (req.request_id or "").strip()[:64]
    idem = get_idempotency()
    idem_key = f"{user_id}:{rid}" if rid else None
    if idem_key:
        cached = idem.get(idem_key)
        if cached is not None:
            from math_agent import metrics as _m

            _m.record_concurrency("replay")
            get_gate().release(user_id)
            get_breaker().exit()
            return ChatResponse(idempotent_replay=True, **cached)
    try:
        out = _agent(user_id).invoke(req.message,
                                     session_mode=(req.mode or "").strip())
    except Exception as e:
        # 内部异常细节只进日志，不外泄给客户端
        log.exception("chat 处理失败 user=%s", user_id)
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试") from e
    finally:
        get_breaker().exit()
        get_gate().release(user_id)
    payload = {"reply": out.get("response") or "(无回复)",
               "intent": out.get("intent", ""),
               "report": out.get("error_report"),
               "elapsed_ms": int((time.time() - t0) * 1000)}
    if idem_key:
        idem.put(idem_key, payload)             # trace 体积大且可再生，不入幂等缓存
    return ChatResponse(trace=out.get("trace", []), **payload)


@app.get("/api/chat/stream")
def chat_stream(message: str, user_id: str = "u_default", token: Optional[str] = None,
                mode: str = ""):
    """SSE 流式：先推 trace 事件，再逐字推回复。

    并发治理与 POST /api/chat 同源：同用户互斥/QPS/全局熔断在**响应头发出前**
    完成（保证 429/503 状态码可达客户端）；释放挂到生成器 finally（客户端断开
    也会触发 GeneratorExit 清理，锁不泄漏）。流式响应不做幂等回放（断线重连
    语义与请求级幂等冲突），由同用户互斥兜底防重复执行。
    """
    from fastapi.responses import StreamingResponse

    from .auth import resolve_user

    uid, authed = resolve_user(token, user_id)
    if not uid:
        raise HTTPException(status_code=401, detail="缺少有效 token，请先调用 /api/session")
    _admit(uid)

    def gen():
        t0 = time.time()
        try:
            out = _agent(uid).invoke(message, session_mode=(mode or "").strip())
        except Exception:
            log.exception("chat_stream 处理失败 user=%s", uid)
            yield f"data: {json.dumps({'type': 'error', 'data': '服务内部错误，请稍后重试'}, ensure_ascii=False)}\n\n"
            return
        finally:
            get_breaker().exit()
            get_gate().release(uid)
        yield f"data: {json.dumps({'type': 'trace', 'data': out.get('trace', [])}, ensure_ascii=False)}\n\n"
        for ch in (out.get("response") or ""):
            yield f"data: {json.dumps({'type': 'delta', 'data': ch}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'elapsed_ms': int((time.time() - t0) * 1000)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


import re as _re

_SAFE_NAME = _re.compile(r"[^0-9A-Za-z._\-\u4e00-\u9fa5]+")


@app.post("/api/upload")
async def upload(user_id: str = "u_default", token: Optional[str] = None,
                 file: UploadFile = File(...)):
    """上传作业图片/PDF，自动 OCR 并批改。"""
    from .auth import resolve_user

    uid, authed = resolve_user(token, user_id)
    if not uid:
        raise HTTPException(status_code=401, detail="缺少有效 token，请先调用 /api/session")
    # P0：文件名净化（防路径遍历）+ 大小上限（防内存 DoS）
    raw_name = Path(file.filename or "file").name
    safe_name = _SAFE_NAME.sub("_", raw_name)[:80] or "file"
    if not Path(safe_name).suffix:
        safe_name += Path(raw_name).suffix[:16]
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_UPLOAD_MB:g}MB 上限")
    dest = UPLOAD_DIR / f"{uid[:24]}_{int(time.time()*1000)}_{secrets.token_hex(4)}_{safe_name}"
    dest.write_bytes(content)
    # 并发准入（async 端点里可能排队/等待的操作都要进线程池，不能阻塞事件循环）
    import anyio

    _admit(uid)
    try:
        agent = _agent(uid)
        try:
            out = await anyio.to_thread.run_sync(
                lambda: agent.invoke(f"批改 {dest.as_posix()}"))
        except Exception as e:
            log.exception("upload 批改失败 user=%s", uid)
            raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试") from e
        # need_confirm 用 OCR 的结构化字段，不再依赖回复文案关键词
        need_confirm = bool((out.get("ocr_payload") or {}).get("need_confirm"))
        if not need_confirm and "识别置信度" in (out.get("response") or ""):
            need_confirm = True      # 兜底：旧链路文案匹配
    finally:
        get_breaker().exit()
        get_gate().release(uid)
        # 上传文件用完即删（OCR 文本已进会话状态），防止磁盘无限增长
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
    return {"file": safe_name, "reply": out.get("response"),
            "need_confirm": bool(need_confirm), "trace": out.get("trace", [])}


@app.get("/api/stats")
def stats(user_id: str = "u_default", token: Optional[str] = None):
    from .auth import resolve_user

    uid, authed = resolve_user(token, user_id)
    if not uid:
        raise HTTPException(status_code=401, detail="缺少有效 token，请先调用 /api/session")
    hist = user_history(uid, limit=500)
    total = len(hist)
    correct = sum(1 for h in hist if h["correct"])
    return {"question_bank": topic_stats(),
            "user": {"user_id": uid, "attempts": total,
                     "accuracy": round(correct / total * 100, 1) if total else 0.0}}


@app.post("/api/reset")
def reset_user(user_id: str = "u_default", token: Optional[str] = None):
    """清空该用户的全部学习数据（答题记录/刷题会话/落盘状态），
    同时丢弃内存中的 Agent 实例（下次请求按全新会话重建）。"""
    from .auth import resolve_user

    uid, authed = resolve_user(token, user_id)
    if not uid:
        raise HTTPException(status_code=401, detail="缺少有效 token，请先调用 /api/session")
    with _AGENTS_LOCK:
        agent = _AGENTS.pop(uid, None)
    if agent is not None:
        try:
            agent[0].reset()
        except Exception:
            log.exception("reset: Agent 重建失败 user=%s", uid)
    from math_agent.db.repository import reset_user_data

    counts = reset_user_data(uid)
    return {"ok": True, "cleared": counts}


@app.get("/api/curriculum")
def curriculum(topic: str = ""):
    """学习模式用的知识点目录（板块 → 知识点列表）。"""
    from math_agent.teaching import CURRICULUM

    topics = [topic] if topic and topic in CURRICULUM else list(CURRICULUM)
    return {"topics": [{"topic": t, "points": [
        {"index": i, "name": kp["name"], "summary": kp["summary"]}
        for i, kp in enumerate(CURRICULUM[t])]} for t in topics]}


@app.get("/api/metrics")
def metrics_snapshot(token: Optional[str] = None):
    """可观测性快照（方案 Phase 5）：循环终止原因分布 / 工具错误率 / 意图分布。"""
    _require_admin(token)
    from math_agent import metrics as _m

    return {**_m.snapshot(), "breaker": get_breaker().stats()}


@app.get("/api/traces")
def traces(limit: int = 50, user_id: str = "", intent: str = "", token: Optional[str] = None):
    """① 自研 trace 面板数据：最近 N 轮调用链 + 聚合统计。"""
    _require_admin(token)
    from math_agent import metrics as _m
    from math_agent.trace import get_recorder

    rec = get_recorder()
    return {"stats": rec.stats(), "metrics": _m.snapshot(),
            "items": rec.recent(limit=limit, user_id=user_id or None,
                                intent=intent or None)}


@app.get("/trace", response_class=HTMLResponse)
def trace_page(token: Optional[str] = None):
    """① trace 面板页面。"""
    _require_admin(token)
    page = WEB_DIR / "trace.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>trace.html 缺失</h3>", status_code=404)


@app.get("/api/health")
def health():
    """轻量健康检查：不触发 BGE 模型加载，避免首请求阻塞数十秒。"""
    from math_agent.embeddings import peek_embedder

    emb = peek_embedder()
    return {"ok": True, "llm": get_llm().available,
            "embedder": getattr(emb, "name", "not_loaded_yet"),
            "embedder_ready": emb is not None}


def main() -> None:
    import os

    import uvicorn

    # 容器/公网部署必须绑 0.0.0.0（Dockerfile 用）；本机默认 127.0.0.1
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8787"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
