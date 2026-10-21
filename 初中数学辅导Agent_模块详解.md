# 初中数学辅导 Agent · 五大模块技术详解

> 围绕简历「项目经验」五条模块展开，所有细节均对应 `src/math_agent/` 下真实代码。

---

## 模块一：状态机与失败治理模块

### 1.1 9 节点状态机（`graph/builder.py`）
用 `StateGraph(MathAgentState)` 编译出一张图，9 个节点：

| 节点 | 职责 |
|---|---|
| `intent_router` | 意图路由（`route_intent`） |
| `quiz_manager` | 刷题管理（出题/收答/跳题/单题提示） |
| `grading` | 作业批改（OCR + 判分） |
| `error_analyzer` | 错题分析（交卷/批改后出报告） |
| `memory_manager` | 长期记忆写入（习惯沉淀） |
| `habit` | 习惯/薄弱点查询 |
| `knowledge_tutor` | 知识点学习模式 |
| `react` | ReAct 答疑循环 |
| `finalize` | 收尾 → `END` |

**入口与分流**：`intent_router` → `route_by_intent` 按意图分发：`quiz→quiz_manager`、`submit→error_analyzer`、`grade→grading`、`habit→habit`、`learn→knowledge_tutor`、其余→`react`。

**连线（关键分支）**：
- `quiz_manager → finalize`（单题只回对错，**不进分析**，这是防幻觉硬约束的落点）
- `grading →（有 grade_result）→ error_analyzer → memory_manager → finalize`（批改完自动出报告并沉淀习惯）
- `react ⇄ react` 自循环，直到 `should_continue_react` 判停 → `finalize`
- 所有路径最终汇入 `finalize → END`

**状态与持久化**：`MathAgent` 门面以 `thread_id = math_{user_id}` 绑定 checkpointer（InMemorySaver），进程重启可从 SQLite 恢复进行中的会话（`_restore_state`/`_persist_state`）；用 `threading.RLock` 做同用户并发串行化，防丢失更新。

### 1.2 ReAct 循环（`graph/react.py`）
`react_step_node` 实现 Thought → Action → Observation 循环，**双重硬终止（不依赖 LLM 自觉）**：
- **最大循环**：`react_loop_count >= max_react_loops`（默认 8）→ 强制输出；
- **信息增益判定**：本轮 Thought 向量与上一轮余弦相似度 ≥ `info_gain_threshold`（0.80，思维停滞）→ 立即停止。
`loop_stop_reason` 写入 trace 便于线上监控「循环终止原因分布」。LLM 不可用时回退到**规则式 Thought/Action**（确定性轨迹），保证离线可用。

### 1.3 toolguard 失败治理（`toolguard.py`）
五件事全部代码层硬约束：
1. **失败分类** `classify_failure`：`fatal`（401/403/配额/密钥）→ 不重试直接降级；`transient`（网络/超时/429）→ 指数退避重试；`param`（参数错误）→ 交回 Agent 重规划（重试无意义）。
2. **幂等防重**：registry 的 `args_hash` + 写类工具 `retryable=False`（重试无副作用）。
3. **每工具熔断** `ToolCircuit`：滑动窗口失败率 > `circuit_fail_rate`(0.8) 且样本 ≥ `circuit_min_volume`(5) 即打开，冷却 `circuit_cooldown_s`(30s) 后半开探测。
4. **单工具超时**：`ThreadPoolExecutor`(max_workers=16) 隔离执行，默认 `default_timeout_s`(30s)，单工具卡死不拖垮整链。
5. **步骤台账 StepLedger**：`ledger_create/record/finish` 以 `task_id/step_id` 绑定，落 Redis/内存，`ledger_completed_map` 提取已成功步骤支持断点续跑。

错误统一 `sanitize_error` **脱敏**（绝对路径/长 token/邮箱打码，截断 200 字）后才进 ToolResult / trace / LLM prompt。边界：熔断/重试/超时为进程内实现，匹配单 uvicorn 进程部署。

---

## 模块二：RAG 知识库模块

### 2.1 ETL 管线（`scripts/build_question_bank.py`）
- **数据来源**：好未来开源 `TAL-SCQ5K-CN_*.jsonl`，单题字段 `problem / answer_option_list（嵌套分组）/ answer_value / knowledge_point_routes（->分隔）/ answer_analysis（带 HTML 的 list）/ difficulty(0-4) / competition_source_list / queId`。
- **LaTeX→可读文本**：`latex_to_text` 递归把 `\frac{a}{b}`→`(a)/(b)`、`\sqrt`→`√(a)`，剥离 `\left\right`、array 环境、HTML 标签。
- **入料门槛**（四者齐全才保留）：`stem` 长度≥8、非英文原题、属初中（`MIDDLE_SCHOOL` 命中）、非高中（`HIGH_SCHOOL` 黑名单拦截双曲线/导数/复数等）、非小学奥数。
- **去重**：按归一化题干去重，剔除「同题异 qid」重复；**板块映射** 函数/二元一次方程/几何图像/综合；**题型派生** 选择题（原始）、填空题（答案≤24 字）、应用题（命中标签或题干≥70 字）。
- 另用 `gen_geometry_questions.py` 合成题补齐几何/代数，并入 SXT001_CN 公开样例。

### 2.2 整题原子切片（`chunking.py` → `chunk_question`）
把 `【topic·type】+ 题干 + 各选项 + 答案 + 解析 + 知识点` 拼成**一个 chunk**；一个 parsed 记录 = 恰好一个 chunk，**不按固定长度切**。边界锚定在 ETL：`parse_row` 仅当 stem/options/answer/analysis 四者齐全才返回记录，结构上不可能切出半个题。同类策略：教材→`chunk_by_structure` 按「定义/定理/例题」标题切；自由文本→`semantic_chunk` 按相邻句余弦<0.55 处切，受 `max_chunk_chars=900` 软约束。

### 2.3 向量化与双 Collection 隔离（`vectorstore.py`）
- **Qdrant 本地嵌入式**（`QdrantClient(path=...)`，零依赖落盘），`knowledge_base` 与 `user_long_term_memory` 两个 collection，均 `Distance.COSINE` + `size=current_dim`。
- **payload 差异（隔离在 schema 而非数据分桶）**：
  - `knowledge_base`：`text / question_id / topic / type / difficulty / knowledge_points / answer`，检索可 `topic` 过滤 + `score_threshold`(默认 0.10)；
  - `user_long_term_memory`：`content / user_id（强制）/ type（默认 habit）/ timestamp / source`，**检索不应用 score_threshold**（保召回）。
- **代码级隔离**：`search_user_memory` 入口 `if not user_id: raise IsolationError`，即使调用方忘传也报错，绝不退化全库检索；并有 `audit_isolation()` 探针自检（写 A 查 B 断言查不到，跑完自清理）。
- 一题一 upsert，批量 256；BGE 向量召回，`top_k=5`。

---

## 模块三：安全与防幻觉约束模块

### 3.1 单题遮罩硬约束（`nodes.py` → `quiz_manager_node`）
代码层锁死：**单题答错只回「❌ 回答错误」+ 分层提示**（1=方向 / 2=关键步骤 / 3=完整解析），绝不输出「错题分析报告」。该行为是测试断言 `tests/test_smoke.py::test_quiz_flow_no_immediate_report` 的锁点——改之前必须先看测试。只有在 `submit`（交卷）或 `grading`（批改整份）时才进入 `error_analyzer_node` 出阶段性错题报告（`analyze_errors` + `llm_polish_report`）。
**动机**：① 学生易「没思考就偷看解析」→ 遮罩逼其先想；② 单题上下文不足时错因分析不可靠 → 不析错因从源头压低幻觉。

### 3.2 多模态批改兜底（`grading_node`）
- 路径提取正则兼容**正/反斜杠与盘符**（修复过 `D:/xxx.png` 被第三个分支吞掉盘符变成 `/xxx.png`、相对路径被 OCR 白名单拒绝的 bug）；扩展名与 `ocr.ALLOWED_OCR_EXT` 对齐（补 bmp/webp/md）。
- `ocr_document → grade_assignment`；OCR 置信度 <0.6 让用户先确认；PaddleOCR 未装时 **GLM-4V 多模态兜底**自动接管读图。

### 3.3 三层安全（`builder.py` → `invoke`）
`safety.check_input`（输入审核，违规直接返回引导话术）→ 主流程 → `safety.sanitize_output`（输出脱敏 + 移除索取未成年人隐私话术）；用户轮次写入记忆前 `safety.mask_pii` 脱敏。

---

## 模块四：会话压缩与长程记忆模块

### 4.1 压缩器（`memory/compressor.py`）
每轮 `invoke` 前 `prune_state` 释放无关 state：`tool_call_history` 旧记录压成 `{name, args_hash, ok}`（防重/依赖链语义不变）、`observations` 只留滑动窗口、已交卷会话释放整包 `questions`；`build_context` 注入 `conversation_digest`。

**三层策略（实体锚 + 最近 N 轮 + 窗口外摘要）**：
1. **实体锚（永不释放）**：`add_turn` 时 `_extract_entities` 用正则把板块/题型/QID 提升进 `entity_pool`，与具体轮次解耦，永远置顶（`【已确认实体】…`）；
2. **近期原文窗口**：`recent_turns`(默认 6) 逐字保留，**受 `context_budget`(1600 字符) 硬约束**；
3. **窗口外旧轮次**：相关→压成一行要点（`【此前要点】`）；无关闲聊→**整轮释放**（不进摘要不占预算）。

**盲区（前面试聊已坦白）**：`_TOPIC_PAT` 是粗粒度主题词表，「只练一元一次方程」会退化成只存「方程」，细粒度约束特异性会在滑出窗口后丢失；修法为补细粒度词表 + 显式约束抽取 + 回归断言，已排入下一轮。

### 4.2 长程记忆
`memory_manager_node` 从错题报告 `extract_habits` → `update_long_term_memory` 写入 `user_long_term_memory`；`habit_node` 用 `search_user_memory`（强制 user_id 隔离）+ `user_history` 出习惯/薄弱点报告。**结构化数字由代码产出、LLM 只追加建议**，防数字幻觉。

---

## 模块五：后端服务与安全部署模块

### 5.1 API 与鉴权（`api/auth.py` + `api/server.py`）
- **FastAPI + uvicorn（端口 8787）**。`POST /api/session` 用 `user_id` 换 HMAC token：`user_id:ts:sig`（SHA256，sig 取前 32 位），`TOKEN_TTL = 30 天`。
- **身份不可自报**：受保护接口凭 token 解出 `user_id`，**客户端传入的 user_id 一律忽略**；`require_token=true` 时无有效 token 一律 401。
- **密钥管理**：优先环境变量 `MATH_AGENT_SECRET`，否则自动生成并持久化到 `data/.secret_key`。
- **CORS**：`cors_origins` 白名单，公网域名显式填入。

### 5.2 安全部署
- `Dockerfile`（`HOST=0.0.0.0`）、`docker-compose` 的 `caddy` profile(public)、`Caddyfile` 自动 TLS（HSTS/nosniff 安全头、上传体对齐 11MB）、uvicorn `SSL_CERTFILE/SSL_KEYFILE` 直连；`require_token` 强制开启。
- **单实例约束已文档化**：toolguard 熔断/台账/重规划与 Qdrant 文件锁均为进程内实现，不支持多 worker 横向扩展（文档 §9.6 声明，避免误用）。

---

### 一句话串起五模块
用户输入 → `intent_router` 路由 → 刷题走 `quiz_manager`（单题遮罩）→ 交卷/批改触发 `error_analyzer`+`memory_manager`（长程记忆）→ 答疑走 `react`（toolguard 兜底、compressor 控上下文）→ 全程基于 `knowledge_base`/`user_long_term_memory` 双 Collection 检索 → 经 FastAPI+HMAC+Caddy TLS 公网安全服务。
