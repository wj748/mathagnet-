# 初中数学辅导 Agent（LangGraph 落地实现）

按 `实现方案.md` 的 Phase 0 → Phase 5 逐步落地，当前已完成 **Phase 0 ~ Phase 4 全量 + Phase 5 服务化与 Web 界面**。

> 关键设计原则：**凡是关键行为（不逐题出报告、记忆隔离、循环终止、工具防重）都在代码层硬约束，不依赖 LLM 自觉。**
>   
> 因此**没有 API Key 也能完整跑通**全流程（文案由规则模板生成）；配置 Key 后自动切换为模型润色。

---


## 1. 快速开始

```bash
# 0) 环境（Python 3.10+，本项目在 3.13 验证通过）
pip install -r requirements.txt
# 可选：真实中文语义向量（torch CPU 版）
pip install --index-url https://download.pytorch.org/whl/cpu torch

# 1) 下载题库（GitHub 开源仓库 math-eval/TAL-SCQ5K，内置多镜像自动降级）
python scripts/download_dataset.py

# 2) 生成模板题，补齐「几何图像」「二元一次方程」板块（可选，产物已入库）
python scripts/gen_geometry_questions.py --per-template 10

# 3) ETL 入库 + 构建向量知识库
python scripts/build_question_bank.py --with-vector

# 4) 跑一遍完整演示（刷题 → 作答 → 交卷 → 错题报告 → 习惯查询）
python scripts/run_cli.py --demo

# 5) 四项优化验收测试
python scripts/eval_optimizations.py

# 6) 对话评测集（意图路由 + 行为约束回归，Phase 4）
python scripts/eval_conversation.py

# 7) 记忆优化评测（多轮对话压缩/释放，state 有界 + 实体保留）
python scripts/eval_memory.py

# 8) 安全加固回归（OCR 白名单 / 鉴权 / 上传净化 / 并发 / WAL）
python scripts/eval_security.py

# 9) 知识点讲解 Agent 评测（课程大纲 / 讲解 / 讲练闭环）
python scripts/eval_teaching.py

# 10) Badcase 回归集（42 条历史踩坑案例，判分/LaTeX/路由/隔离/安全/内容审核）
python scripts/eval_badcase.py

# 11) 启动服务 + Web 界面
python -m math_agent.api.server      # 或 uvicorn math_agent.api.server:app
# 浏览器打开 http://127.0.0.1:8787   # 调用链追踪面板: http://127.0.0.1:8787/trace
```

想接入 GLM：复制 `.env.example` 为 `.env`，填入 `GLM_API_KEY` 即可（也支持 DeepSeek / 通义等任意 OpenAI 兼容端点）。

### 安全配置（config.yaml `security:` 段）

| 配置                        | 默认         | 说明                                                   |
| ------------------------- | ---------- | ---------------------------------------------------- |
| `require_token`           | false      | 公网部署**必须改 true**：无 `/api/session` 签发 token 的请求一律 401 |
| `cors_origins`            | 本机 8787    | CORS 白名单（默认不再全开）                                     |
| `max_upload_mb`           | 10         | 上传大小上限                                               |
| `agent_ttl` / `agent_max` | 1800 / 100 | 空闲 Agent 淘汰与池容量上限                                    |

OCR 只允许解析 `data/uploads/` 内白名单扩展名文件；身份以服务端 HMAC token 为准，客户端自报 `user_id` 不再被信任。

---


## 2. 目录结构

```
.
├── 实现方案.md                  # 原始方案
├── config.yaml                  # 全局配置（LLM / 向量库 / 图参数 / 分块阈值）
├── prompts/system_v1.md         # System Prompt（版本化管理）
├── src/math_agent/
│   ├── config.py                # 配置加载（config.yaml + .env 覆盖）
│   ├── state.py                 # MathAgentState（LangGraph 全局状态）
│   ├── llm.py                   # GLM / OpenAI 兼容接入 + 无 Key 降级
│   ├── embeddings.py            # BGE-small-zh（ModelScope） / hashing 兜底
│   ├── vectorstore.py           # Qdrant 本地模式，双 Collection 隔离
│   ├── chunking.py              # 语义分块 + 题目结构化 Chunk
│   ├── analysis.py              # 错因归类 / 错题报告 / 习惯抽取
│   ├── db/                      # SQLAlchemy：题库 / 会话 / 答题记录
│   ├── tools/
│   │   ├── registry.py          # ★ 工具注册 + 防重/依赖链/Fallback 拦截
│   │   ├── quiz.py              # fetch_quiz / check_answer（纯代码判分）
│   │   ├── grading.py           # grade_assignment / grade_application_question
│   │   ├── ocr.py               # ocr_document（PaddleOCR / PDF / 多模态兜底）
│   │   ├── calc.py              # calculator（SymPy 精确计算）
│   │   └── memory_tools.py      # search_knowledge / search_user_memory / update_long_term_memory
│   ├── graph/
│   │   ├── router.py            # Intent Router（规则优先 + LLM 兜底）
│   │   ├── nodes.py             # Quiz Manager / Grader / Error Analyzer / Memory Manager / Habit
│   │   ├── react.py             # ★ ReAct + 8 次上限 + 信息增益终止
│   │   └── builder.py           # 图组装 + MathAgent 运行时
│   └── api/server.py            # FastAPI：REST + SSE 流式 + 文件上传
├── scripts/                     # 数据下载 / ETL / 验收 / CLI
├── web/index.html               # Web 交互界面
└── data/                        # 原始数据 / SQLite / Qdrant 本地存储
```

---

## 3. LangGraph 流转图

```
用户输入 ─► [intent_router]
              ├─ quiz   ─► [quiz_manager] ──────────────► [finalize]
              ├─ submit ─► [error_analyzer] ─► [memory_manager] ─► [finalize]
              ├─ grade  ─► [grading] ─► (有批改结果) ─► [error_analyzer] ─► …
              ├─ habit  ─► [habit] ───────────────────► [finalize]
              ├─ learn  ─► [knowledge_tutor] ──────────► [finalize]
              └─ chat   ─► [react] ──(continue)──► [react]
                                └─(max_loop / information_gain / finished)─► [finalize]
```

**两种会话模式**（Web 端顶部切换，或 API 传 `mode`）：

| 模式   | `session_mode` | 主导节点              | 说明                                |
| ---- | -------------- | ----------------- | --------------------------------- |
| 刷题   | `quiz`（默认）     | `quiz_manager`    | 出题 → 逐题判分 → 交卷出报告                 |
| 学知识点 | `learn`        | `knowledge_tutor` | 选板块 → 逐知识点讲解 → 例题 → 小测 → 讲评 → 下一个 |

---

## 4. 四项优化的落地位置

| 优化项                          | 实现位置                                                                                                                                                                           | 验收方式                          |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| **① 动态切片 + 双 Collection 隔离** | `chunking.py`（语义相似度分块）、`vectorstore.py`（`knowledge_base` / `user_long_term_memory`）                                                                                            | `eval_optimizations.py` §1 §2 |
| **② 短期/长期记忆分层**              | `state.short_term_memory`（会话内）+ `graph/nodes.memory_manager_node`（报告后写入向量库）+ Redis 可选                                                                                          | §2 §5                         |
| **③ ReAct 循环控制**             | `react.py`：8 次上限 + Thought 向量余弦相似度 ≥ 0.80 终止，`loop_stop_reason` 落 trace                                                                                                        | §3                            |
| **④ 工具防重防错**                 | `tools/registry.py`：Pydantic 校验 + `args_hash` 防重 + `requires` 依赖链 + 错误回注                                                                                                       | §4                            |
| **⑤ 判分零误判**                  | `tools/quiz.py`：`resolve_option()` 把标准答案与用户答案统一映射到选项下标再比较                                                                                                                      | §7                            |
| **⑥ 多轮对话压缩/释放**              | `memory/compressor.py`：实体锚永不释放 + 近期原文窗口 + 历史要点压缩 + 无关闲聊整轮释放；`prune_state` 每轮瘦身 state（旧工具历史压缩为 `name+args_hash+ok`、observations 滑窗、已交卷会话释放 questions）；digest 注入 ReAct 三个 prompt | `eval_memory.py`              |

### 硬约束清单（不依赖 LLM）

1. **不逐题出报告** —— `Error Analyzer` 只由 `submit` 事件或批改完成触发；`quiz_manager` 单题只回 `✅/❌`。
2. **记忆隔离** —— `search_user_memory` 缺 `user_id` 直接抛 `IsolationError`，绝不降级为全库检索。
3. **循环终止** —— `react_loop_count >= 8` 或信息增益 ≥ 0.80 强制 `finalize`。
4. **工具防重** —— `tool_name + args_hash` 命中即拒绝并回注自我纠正提示；`prune_state` 压缩历史后保留 `args_hash`，防重语义不变。
5. **选择题判分** —— 先经 `resolve_option()` 归一化到选项下标再比对，杜绝"答错判对"。
     
   选项前缀识别用**严格正则**（标号后必须跟分隔符），否则 `a+b` 会被误认成选项行。
6. **state 有界** —— `prune_state` 每轮强制裁剪 `tool_call_history` / `observations` / 已交卷 `quiz_session`，
     
   多轮对话后 token 占用不随轮数线性增长（`memory:*` 配置段控制阈值）。
7. **结构化报告不被 LLM 覆写** —— 错题报告 / 习惯报告的标题、正确率、错因分布由代码层渲染，
     
   LLM 只能在正文后**追加**「💡 老师建议」，绝不允许改写已有数字（防幻觉）。
8. **上传与 OCR 白名单** —— `ocr_document` 只解析 `data/uploads/` 内白名单扩展名文件；
     
   上传文件名净化（防 `../` 遍历）+ 大小上限；身份以 `/api/session` 签发的 HMAC token 为准。

---


## 5. 题库数据

- 来源：GitHub 开源仓库 **`math-eval/TAL-SCQ5K`**（初中数学选择题，含题干/选项/答案/解析/知识点路径/难度）。
- 下载：脚本内置 `gh-proxy → jsdelivr → hf-mirror → GitHub 直连` 四级镜像自动降级。
- 追加来源：ModelScope **`market.aliyun/SXT001_CN`**（阿里云市场数学题库，仓库仅含公开样例；
  样例放入 `data/raw/sxt001/*.json` 后 ETL 自动并入，完整版初中约 1.2 万题需在阿里云市场获取）。
- ETL（`scripts/build_question_bank.py`）：
  - 清洗 LaTeX 噪音、HTML 标签/实体，抽取选项与解析、映射知识点路径；
  - 按关键词映射到三大板块 **函数 / 二元一次方程 / 几何图像**；
  - 过滤小学奥数题，保留初中内容；
  - 题型派生：**选择题**（原始）、**填空题**（答案文本 ≤ 24 字符的题去选项）、**应用题**（命中应用题/行程/工程/分百标签）。
- 每个题目作为**一个完整 Chunk**（题干-选项-答案-解析）写入向量库，检索时不会被切断。

**当前规模**：

| 板块     |     选择题 |     填空题 |     应用题 |       合计 |
| ------ | ------: | ------: | ------: | -------: |
| 函数     |     240 |     238 |      41 |  **519** |
| 二元一次方程 |     250 |     249 |      73 |  **572** |
| 几何图像   |     268 |     267 |      24 |  **559** |
| 综合     |     169 |     169 |      46 |  **384** |
| **合计** | **927** | **923** | **184** | **2034** |

> 各板块均已达方案要求的「每板块 ≥ 200 题」，向量库同步 2034 条。
> 数据构成：TAL-SCQ5K 清洗去重后 569 题（原始 5000 条）+ 模板生成 373 题
> + SXT001_CN 样例 12 题（派生 25 条，question_id 前缀 SXT_）。

**清洗要点**（`latex_to_text`）：LaTeX 符号转教材排版（`\frac`/`\sqrt` 支持嵌套花括号）、
  
滤除英文竞赛题与高中超纲内容、去 HTML 标签、修 `array`/`cases` 环境残留。

**超纲过滤**（`HIGH_SCHOOL`）：除双曲线/椭圆/导数/向量等显性高中概念外，
  
还针对实测混入的内容补充了反函数、等比数列、Σ 求和、对数式、复合函数/抽象函数、
  
通项公式、柯西与均值不等式、方差等特征词 —— 实测额外过滤 82 条高中/竞赛题，零误伤。

### 模板生成题（补齐板块缺口）

TAL-SCQ5K 以小学奥数竞赛题为主，其初中子集里**几何仅 126 条、二元一次方程仅 175 条**，
  
达不到每板块 200 题。备选源 `cn-k12` 经采样验证为 99.8% 英文且偏高中，不可用。

因此改用**参数化模板生成**（`scripts/gen_geometry_questions.py`）：

- **39 个模板**（几何 21 + 方程 10 + 函数 8），覆盖：
  - 几何：勾股定理、三角形角度、等腰三角形、面积周长、圆与扇形、相似与中位线、长方体体积
  - 方程：加减消元、代入消元、一般二元一次方程组、鸡兔同笼、票价问题、年龄问题、数字问题
  - 函数：函数值、待定系数法求解析式、图象与象限/坐标轴围成面积、正/反比例函数、两直线交点
- 答案由**程序计算**，不存在标注错误；干扰项来自**典型错法**
    
  （勾股定理忘记开方、消元后忘记除以 2、面积公式漏除 2），有教学价值。
- 解析是**真实分步推导**，不是空话。
- 参数随机前会校验「干扰项互不重复且不等于正确答案」，保证任何模板都不会静默失败。

产出 `data/processed/synth_questions.jsonl`，由 ETL 合并入库，
  
`source` 标记为 `GEO_TEMPLATE` / `ALG_TEMPLATE` 可追溯。

---

## 6. 配置说明

| 配置                              | 默认                             | 说明                                                                                        |
| ------------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------- |
| `llm.provider`                  | `auto`                         | 有 Key → GLM/OpenAI 兼容；无 Key → `mock` 离线规则模式                                               |
| `embedding.backend`             | `auto`                         | 优先 BGE-small-zh（需 `scripts/download_embedding_model.py` 从 ModelScope 下载），否则零依赖 hashing 向量 |
| `vectorstore.backend`           | `qdrant_local`                 | **本地嵌入式模式，无需 Docker**；切 `qdrant_server` 走独立服务                                             |
| `database.url`                  | `sqlite:///data/math_agent.db` | 生产可换 `mysql+pymysql://…`                                                                  |
| `graph.max_react_loops`         | `8`                            | ReAct 最大循环次数                                                                              |
| `graph.info_gain_threshold`     | `0.80`                         | 信息增益终止阈值                                                                                  |
| `chunking.similarity_threshold` | `0.55`                         | 语义分块切分阈值                                                                                  |
| `redis.enabled`                 | `false`                        | 开启后短期记忆落 Redis（TTL 24h）                                                                   |
| `security.require_token`        | `false`                        | 公网部署改 `true`：无 token 一律 401                                                               |

接入真实模型：复制 `.env.example` 为 `.env`，填 `GLM_API_KEY` 即可（默认 `glm-4-flash`，
  
也支持 DeepSeek / 通义等任意 OpenAI 兼容端点）。**注意**：接 Key 后结构化报告仍由代码层产出，
  
LLM 只追加建议段，验收断言不会因模型改写而失效。

---

## 7. 知识点讲解 Agent（学习模式）

与刷题并列的一条主线：学生选板块后按课程大纲**逐知识点讲解**，讲完配例题与小测形成闭环。

| 组成   | 位置                                           | 说明                                            |
| ---- | -------------------------------------------- | --------------------------------------------- |
| 课程大纲 | `teaching.py::CURRICULUM`                    | **6 大板块 38 个知识点**：数与式 6 / 方程与不等式 8 / 函数 6 / 几何 10 / 统计与概率 4 / 锐角三角函数 4，每个含定义、要点、易错点、口诀 |
| 讲解生成 | `teaching.py::build_lecture`                 | LLM 可用时润色，**无 Key 时用规则模板渲染**（离线全程可用）          |
| 工具   | `tools/teaching.py::explain_knowledge_point` | 注册进工具层，供 ReAct 与讲解节点共用                        |
| 图节点  | `graph/knowledge_tutor.py`                   | 阶段机：`catalog → teach → quiz → graded → next`  |
| 意图   | `router.py::_P_LEARN` + `learn`              | 学习模式内除刷题/批改/交卷外的输入默认走 `learn`                 |
| 状态   | `state.learning_session` / `session_mode`    | 跨轮持久化当前板块、知识点序号、答对数                           |

**交互闭环**：选板块 → 讲解知识点 → 出例题 → 学生作答 → `check_answer` 判分 + 解析 + 易错提示
  
→ 「继续」进入下一个 / 「没听懂」换讲法 / 「换个例题」换题 / 「退出学习」结束。
  
学习过程中**不会**触发错题报告（复用硬约束 1）。

**接入方式**：Web 端顶部「刷题 / 学知识点」切换；API 传 `mode=learn`（`/api/chat`、`/api/chat/stream`）；
  
课程目录由 `GET /api/curriculum` 提供。

---


## 8. 已完成 / 待办

**已完成**：环境骨架 · 题库 ETL · 工具层（含防重/依赖链/Fallback）· LangGraph 主干 · 四项优化 · 批改流程 · 习惯查询 · FastAPI + SSE + Web 界面 · 验收测试。

**验收结果**：

| 测试             | 命令                                     | 结果                                                          |
| -------------- | -------------------------------------- | ----------------------------------------------------------- |
| 四项优化 + 判分回归    | `python scripts/eval_optimizations.py` | **34/34 通过**                                                |
| 对话评测集（Phase 4） | `python scripts/eval_conversation.py`  | 行为 **11/11**（意图路由 **34/34** + 学习模式 **10/10** = 100%，阈值 85%） |
| 记忆压缩/释放        | `python scripts/eval_memory.py`        | **21/21 通过**                                                |
| 安全加固回归         | `python scripts/eval_security.py`      | **16/16 通过**                                                |
| 知识点讲解 Agent    | `python scripts/eval_teaching.py`      | **19/19 通过**                                                |
| Badcase 回归集      | `python scripts/eval_badcase.py`       | **42/42 通过**（8 组历史踩坑）                                |
| 冒烟测试           | `python -m pytest tests/ -q`           | **7 passed**                                                |
| 端到端演示          | `python scripts/run_cli.py --demo`     | 选板块 → 刷题 → 作答 → 解析 → 交卷 → 错题报告 → 习惯查询，全链路跑通                 |

**可观测性（Phase 5）**：进程内计数器 `src/math_agent/metrics.py`，经 `GET /api/metrics` 输出
  
循环终止原因分布（`max_loop` / `information_gain` / `finished`）、工具调用错误率与拦截率、意图分布。

**路由易错点已覆盖**：刷题会话中的短输入作答（`A` / `12` / `3/4`）、概念咨询不被当作答
  
（`什么是斜率` → chat）、单题反馈指令仍留在刷题内（`解析` / `提示` → quiz）。

> ⚠️ 已知限制：TAL-SCQ5K 的板块映射基于关键词，偶有难度偏高的竞赛题混入；
>   
> 已用 `HIGH_SCHOOL` 特征词做了一轮过滤（见上），但仍建议生产使用前再叠加人工抽检。
>   
> 模板生成题的题干为参数化套用，措辞重复度高于真实题目。

**待办（对应 Phase 5 运维侧）**：~~五项已于 2026-09-09 全部落地~~

- [x] **自研 trace 面板** —— `trace.py`（环形缓冲 500 轮）+ `GET /api/traces`（明细+聚合统计：
  意图/循环终止/工具错误率/P50·P95 耗时）+ 浏览器 `/trace` 页面（自动刷新、按意图/用户过滤）。
  过程中修复了 `metrics.record_conversation` 缺失导致的 conversation 指标静默为 0。
- [x] **GLM-4V 视觉识别** —— `llm.get_vision_client()`（`glm-4v-flash`，独立于文本模型）
  + OCR 多模态兜底 3 次重试；修复 data URL 缺 `image/` 前缀导致的 1210 解析错误。
  扫描件 PDF 无文本层时给出"转图片再上传"的明确提示。
- [x] **badcase 评测集** —— `scripts/eval_badcase.py` **42/42**：判分/LaTeX/模板排版/路由/
  隔离/压缩/安全/结构化输出/视觉/Redis/内容审核 8 组历史踩坑案例，改动后必跑。
- [x] **Redis 短期记忆** —— `docker-compose.yml`（redis:7-alpine + LRU + AOF）一键起；
  `redis.enabled=true` 后会话/对话压缩经 Redis 跨进程持久化（TTL 24h）；连不上自动回退内存。
- [x] **未成年人保护与内容审核** —— `safety.py` 规则层硬约束：输入审核（暴力/自伤/违法/
  代写作弊→温和引导话术，不进主流程）、PII 脱敏（手机号/身份证/邮箱/住址/学校班级，进
  日志/trace/记忆前打码）、输出审核（拦截索取隐私话术）。词表用词组，不误伤数学语境。
