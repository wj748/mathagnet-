# 智能助教 · API 调用指南

> 基准地址：`http://127.0.0.1:8787` ｜ 全部端点已于 2026-09-16 实测通过
> 服务启动：`start.bat` 或 `python -m math_agent.api.server`（需 `PYTHONPATH=src`）

---

## 0. 调用约定

1. **鉴权**：先 `POST /api/session` 用 user_id 换 HMAC token；之后所有业务接口都带 `token` 参数。token 服务端签名，客户端自报 user_id 不作为身份依据。
2. **中文参数必须 URL 编码**：`topic=函数` 这类 query 直接拼会报 `UnicodeEncodeError`/400，用 `urllib.parse.quote()` 或 curl 的 `--data-urlencode`。
3. **限流语义**（同用户）：上一条还在处理 → `429 busy`；发送过频 → `429 qps`；全局过载 → `503`。错误体为 `{"detail": "面向学生的话术"}`。
4. **LLM 在线时对话耗时 10~30s**，请把客户端超时设 ≥120s（前端用 SSE 流式缓解等待感）。

## 1. 端点总表

| 端点 | 方法 | 鉴权 | 说明 |
|---|---|---|---|
| `/api/health` | GET | 无 | 健康检查：`ok / llm / embedder / embedder_ready` |
| `/api/session` | POST | 无 | 入参 `{user_id}` → `{token}` |
| `/api/chat` | POST | token | 同步对话；入参 `{message, user_id, token, mode, request_id}` |
| `/api/chat/stream` | GET | token | SSE 流式对话，事件：`trace` / `delta` / `error` |
| `/api/upload` | POST | token | multipart 上传作业（png/jpg/bmp/webp/pdf/txt/md，≤10MB） |
| `/api/stats` | GET | token | 用户做题数/正确率 + 题库分布 |
| `/api/curriculum` | GET | 无 | 课程目录，`?topic=函数` 过滤板块 |
| `/api/reset` | POST | token | 清空该用户全部学习数据 |
| `/api/metrics` | GET | token* | 运行指标（工具成功率/意图分布/循环终止/toolguard 计数） |
| `/api/traces` | GET | token* | 调用链明细（脱敏），另有 `/trace` 可视化页 |

\* `security.require_token: true` 时监控接口也要求有效 token。

## 2. curl 示例

```bash
BASE=http://127.0.0.1:8787

# ① 换 token
TOKEN=$(curl -s -X POST $BASE/api/session -H "Content-Type: application/json" \
  -d '{"user_id":"u_demo"}' | python -c "import sys,json;print(json.load(sys.stdin)['token'])")

# ② 对话
curl -s -X POST $BASE/api/chat -H "Content-Type: application/json" \
  -d "{\"message\":\"什么是一次函数的斜率？\",\"user_id\":\"u_demo\",\"token\":\"$TOKEN\"}"

# ③ 学习统计
curl -s "$BASE/api/stats?user_id=u_demo&token=$TOKEN"

# ④ 课程目录（中文要编码）
curl -s "$BASE/api/curriculum?topic=%E5%87%BD%E6%95%B0"

# ⑤ SSE 流式对话（逐事件输出）
curl -N "$BASE/api/chat/stream?user_id=u_demo&token=$TOKEN&mode=&message=%E6%88%91%E8%A6%81%E5%88%B7%E5%87%BD%E6%95%B0%E7%9A%84%E9%80%89%E6%8B%A9%E9%A2%98"
```

## 3. Python 示例（完整闭环，已实测）

```python
import json, urllib.request, urllib.parse

BASE = "http://127.0.0.1:8787"

def post(path, payload, timeout=120):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def get(path, timeout=15):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())

# ① 鉴权
token = post("/api/session", {"user_id": "u_demo"})["token"]
q = urllib.parse.quote                      # 中文参数编码

# ② 对话（mode: ""=刷题/答疑, "learn"=学习知识点；request_id 传 UUID 可获幂等重放）
c = post("/api/chat", {"message": "来5道函数选择题", "user_id": "u_demo",
                       "token": token, "mode": "", "request_id": ""})
print(c["intent"], c["elapsed_ms"], "ms")
print(c["reply"])

# ③ 统计 / 目录 / 健康
st = get(f"/api/stats?user_id=u_demo&token={q(token)}")
cu = get("/api/curriculum?topic=" + q("函数"))
h  = get("/api/health")
```

## 4. SSE 流式对接要点（`/api/chat/stream`）

- 事件以空行分隔，每行 `data:{json}`；解析失败单事件跳过即可（前端即此策略）。
- `trace` 事件先到（可做阶段提示），`delta` 累积正文，结束后**整体做一次 KaTeX 渲染**（流式中途渲染公式会碎）。
- `mode` 参数：切学习模式传 `learn`；从学习页签切回刷题建议先发一条「退出学习」。

## 5. 实测记录（2026-09-16）

| 调用 | 结果 |
|---|---|
| `POST /api/session` | token 签发 ✓ |
| `POST /api/chat`（斜率是什么） | intent=chat，耗时 23440ms，LLM 正常讲解 ✓ |
| `GET /api/stats` | attempts=0, accuracy=0.0 ✓ |
| `GET /api/curriculum?topic=函数` | 6 个知识点，首个「平面直角坐标系」✓ |
| `GET /api/health` | ok=true, llm=true, embedder=bge ✓ |
