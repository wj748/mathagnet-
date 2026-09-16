# 智能助教（初中数学辅导 Agent）项目长期记忆

## 项目定位
按 `实现方案.md` Phase 0~5 落地的 LangGraph 初中数学辅导 Agent。
核心原则：**关键行为在代码层硬约束，不依赖 LLM 自觉**；无 API Key 也能跑通全流程（离线规则模式）。

## 技术栈与关键约定
- Python venv：`C:/Users/lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe`
- 运行脚本**必须**带 `PYTHONPATH="D:/游戏库/智能助教/src"`，否则 `math_agent` 导入失败
- 向量库：Qdrant 本地嵌入式（`data/qdrant`），双 Collection 隔离
  - `knowledge_base`（全局题库，只读检索）
  - `user_long_term_memory`（用户私有，检索强制注入 user_id，缺失时抛 `IsolationError`）
- Embedding：BGE-small-zh-v1.5（ModelScope `BAAI/bge-small-zh-v1.5`），无 torch 时自动回退 hashing 向量
- 题库源：GitHub `math-eval/TAL-SCQ5K`（经 hf-mirror.com 镜像下载，直连 GitHub/ModelScope 不稳）
  - 该源**以小学奥数为主**：5000 条里仅 992 条初中、2905 条小学；初中桶里几何天花板 126 条
  - 备选源 `cn-k12` 经采样为 99.8% 英文且偏高中，**不可用**（不要再试）
  - 缺口由 `scripts/gen_geometry_questions.py` 的 39 个参数化模板补齐
- LLM：OpenAI 兼容端点，默认 GLM；无 Key 走规则模板

## 环境注意事项
- pip 装 torch 必须：`pip install --index-url https://download.pytorch.org/whl/cpu torch`
  （不能写进 requirements.txt 行内，会导致 `pip install -r` 失败）
- qdrant-client ≥1.12 用 `query_points()`，`search()` 已废弃；退出时 `__del__` 会抛
  `ImportError: sys.meta_path is None` — 已用 atexit + close() 兜底
- 中文哈希向量存在高频字导致相似度虚高问题，已改用 BGE 真实语义向量

## 已踩过的坑（务必避免复现）
1. **判分 Bug**：`answers_equal` 里 `_OPTION_LEAD.sub("", u) == ""` 对任意单字母都成立，
   导致"答案是A、用户答B"被判正确。修法：选择题**先映射到选项下标再比较**。
   选项前缀识别必须用严格正则（标号后要有分隔符），否则 `a+b` 会被误当成选项行。
2. **LaTeX 清洗**：`_REPL` 是正则不是字面量，必须用 `re.sub`；且要加 `(?![a-zA-Z])`
   避免 `\cdot` 吃掉 `\cdots`。花括号要替换成空串（替换成空格会产生 `x ^2` 断裂）。
   `\bot` 缺失会让未知命令在末尾 `replace("\\", " ")` 后留下单词残渣（`a bot b`）。
   2026-09-08 已对原始数据全量扫描 LaTeX 命令，把 `overrightarrow/sim/Rightarrow/
   textasciitilde/Delta/therefore/subseteq` 等 100+ 命令全部映射完毕；
   组内容保留型命令（boxed/operatorname/textsuperscript 等）单独用 `\{([^{}]*)\}` → `\1` 规则。
   验证方法：扫描库内 stem/options/answer 的 `\xxx` 残留，为 0 即干净
   （注意 Git Bash heredoc 会吃反斜杠，正则要用 `chr(92)+chr(92)+...` 构造）。
3. `audit_isolation` 探针必须在 finally 里删除，否则污染真实记忆（a_hits 会累积）。
4. **模板题排版**：f-string 拼 `y={k}x{b}` 会产出 `-1x`/`1x+5`。必须用 `_coef(k, sym)`、
   `_kx(k)`、`_lin(k, b)`；且 `_coef` 已自带变量名，**后面不能再拼 `x`/`y`**（会得到 `3xy`）。
5. **模板题干扰项必须手算验证**：曾出现干扰项恒等于正确答案（等腰三角形底角 `90-apex/2`）。
6. **LangGraph 跨轮状态**：真正持久的是 checkpointer 里的 state，每轮 invoke 的 payload 是
   新 dict——想释放/修改跨轮字段（tool_call_history 等），必须从上一轮 `last_state` 取出、
   裁剪后**显式回注 payload**（见 `memory/compressor.prune_state` + `MathAgent.invoke`）。
   只改 payload 上不存在的键等于没改。
7. **同一文件多处 Edit 不能并行批量提交**——会互相覆盖且部分静默丢失，必须逐个改+逐个验证。

## 题库构建流程（改动数据后按此顺序跑）
```
python scripts/gen_geometry_questions.py --per-template 10   # → data/processed/synth_questions.jsonl
python scripts/build_question_bank.py --with-vector          # 合并 TAL + 模板 → SQLite + Qdrant
```
`build_question_bank.py` 会 `reset_questions()` 清空旧库重建，避免脏数据残留
（question_id 必须来自数据本身而非列表下标，否则重建会残留）。

## 验收入口（七套都要跑，改动路由/判分/记忆/讲解/安全后必跑）
- `python scripts/eval_optimizations.py` —— 四项优化 + 判分准确性回归（34/34）
- `python scripts/eval_conversation.py` —— 34 条意图路由 + 10 条学习模式路由 + 行为约束（11/11）
- `python scripts/eval_memory.py` —— 记忆优化：多轮压缩/释放/state 有界（21/21）
- `python scripts/eval_security.py` —— 安全回归：OCR 白名单/鉴权/上传净化/并发/WAL（16/16）
- `python scripts/eval_teaching.py` —— 知识点讲解 Agent：大纲/讲解/学习闭环/硬约束（19/19）
- `python scripts/eval_badcase.py` —— 42 条历史踩坑回归（判分/LaTeX/模板/路由/隔离/审核等）
- `python -m pytest tests/ -q` —— 冒烟测试（7 passed）
- `python scripts/run_cli.py --demo` —— 端到端演示
- ⚠ **接了 LLM Key 后必须重跑全量**：离线规则模式不会暴露「LLM 覆写结构化输出」类问题
- ⚠ **跑评测前先停 API 服务**：Qdrant 本地模式单进程锁，否则报错伪装成代码回归
- 可观测性：`GET /api/metrics`（循环终止/工具错误率/意图分布/**conversation 压缩释放**）、
  `GET /api/traces` + `/trace` 页面（调用链明细 + P50/P95 耗时，输入/回复已脱敏）
- 安全约定：身份以 `/api/session` 签发的 HMAC token 为准（api/auth.py），客户端自报 user_id
  不可信；OCR 仅允许 `data/uploads/` 白名单扩展名；公网部署把 `security.require_token` 改 true
- 数据规模：原始 5000 条 → 清洗去重 542 题 + 模板 373 题 → 派生 **2009 条**
  （按板块×题型库存：二元一次方程 572 / 几何图像 553 / 函数 515 / 综合 369，均 ≥200 达标；
  板块分布日志里的数字是选择题口径）
- 备注：改 `HIGH_SCHOOL` 超纲过滤规则后，题库总量会下降，需确认板块仍达标

## LLM 使用红线（新增硬约束，2026-09-09 踩坑总结）
**结构化输出绝不能整段交给 LLM 改写**——报告标题、正确率、错因分布、讲解骨架
（◆ 定义/要点/易错点/口诀）一律由代码层渲染，LLM 只允许**追加**补充段
（习惯/错题报告加「💡 老师建议」，讲解插「◆ 老师讲讲」）。
反面案例：接 Key 后 habit 报告标题被改成「🌟 学习小记」、讲解骨架丢失，各掉 1 条验收。
判定 LLM 真的贡献了内容要用「是否拿到文本」而非「LLM 是否可用」（失败会返回 None）。

## 意图路由规则（改这里务必先看）
`graph/router.py` 的 `_rule_route(text, in_quiz, mode, in_learn)` 优先级：
空输入 → **学习模式内分支（概念咨询→chat，其余非任务指令→learn）** → 概念咨询 `_P_CONCEPT`
→ 刷题中短输入作答 → submit → grade → habit → **learn（在 quiz 之前）** → quiz → 兜底。
`_P_LEARN` 必须排在 `_P_QUIZ` 前面，否则"我想学函数"会被判成刷题。
`_P_CONCEPT` 里**不能**加"解析"/"提示"——那是 Quiz Manager 的单题反馈指令，
加进去会导致刷题中无法查看解析。
判分关键词用 `判\s*(?:一\s*下|下)?\s*分` 才能覆盖"判一下分"这类口语。

## 板块命名单一真源（2026-09-10 起）
- 板块/题量上限一律引用 `src/math_agent/topics.py`：BANK_TOPICS / TEACHING_TO_BANK /
  `to_bank_topic()`；禁止再散落维护映射 dict（旧 _EXAMPLE_TOPIC 与 _NO_BANK_TOPICS
  曾对"锐角三角函数"映射矛盾）。教学板块新增只改 CURRICULUM + topics.py 两处。
- 单次出题上限 MAX_COUNT=20 在 tools/quiz.py，nodes.py parse 引用同一常量。
- `search_knowledge` 已内置 retrieval.score_threshold（默认 0.10，勿在调用方再硬编码）。
- eval 断言必须语义可满足：勿用"绝对目录 in 纯文件名的 parents"这类永假式
  （eval_security 曾因此假绿，2026-09-10 修正）。
- 新增 POST /api/reset：清该用户 attempts/quiz_sessions/agent_states + 内存 Agent 弹池。

## 多用户并发治理（2026-09-15 起）
- `src/math_agent/concurrency.py` 单一入口：UserGate（同用户互斥+QPS 滑窗）、
  GlobalBreaker（max_agent_tasks=8 信号量 + max_inflight=32 熔断 + queue_wait_s=15）、
  IdempotencyStore（user:rid → TTL+LRU）。chat/chat_stream/upload 经 `_admit()` 准入，
  **finally 必须** breaker.exit()+gate.release()；SSE 准入在响应头前、释放在 gen finally。
- 配置全在 config.yaml `concurrency:` 段；拒绝=429/503+友好文案；metrics 有
  concurrency/breaker 段。锁是进程内的——多实例部署需换 Redis 分布式锁（未做）。
- redis 客户端必须 4.6.0（8.x RESP3 握手不兼容 Redis 5 服务端，勿升）。
- eval 断言要显式设定前置状态（如 require_token），不能依赖 .env 当前值——
  否则环境变化会造成假失败（eval_security 曾因此 15/16）。

## 工具链失败治理（2026-09-15 起）
- `src/math_agent/toolguard.py` 单一入口：classify_failure（fatal>transient>param）、
  execute_with_guard（线程池超时+指数退避重试+每工具熔断）、sanitize_error（所有执行类
  错误进 ToolResult/trace/prompt 前必经）、StepLedger（math:task:{task_id}，TTL 1h）。
- registry.run_tool 已接入守卫；写类工具 register_tool(retryable=False)（幂等保护），
  ocr_document timeout=90。react 节点失败治理：blocked 同参重复与 param 同类计入
  replan 上限（max_replans=2），超限/fatal 降级出部分结果（loop_stop_reason:
  degraded/fatal_error）；fallback_tools 强制换同能力备选。
- RedisStore 懒连接：构造不报错≠可达，已补 ping 探活（socket 超时 2s）。
- 同文件多处 Edit 严禁并行批量提交（本日三次静默丢编辑）；验证靠逐个 grep。
- 评测基线更新：badcase 49/49（47-49 toolguard）、pytest 26 passed+6 skipped。
