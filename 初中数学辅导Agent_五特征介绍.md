# 初中数学辅导 Agent —— 以「Agent 五特征」为主线介绍

> 一句话定位：一个面向初中数学的**对话式、工具增强、受监督的智能辅导 Agent**。
> 技术底座：LangGraph（状态机编排）+ GLM 系列大模型（文本 `glm-4-flash` / 视觉 `glm-4v-flash`）+ Qdrant 本地向量库 + BGE 中文嵌入（`bge-small-zh-v1.5`）+ FastAPI / Redis / Caddy。
>
> 下面按业界公认的 Agent 五条特征，逐条说明本项目如何落地、以及哪些地方是刻意的「受限设计」。

---

## 特征一：自主决策与路由（Autonomy & Routing）

**落地：** 用 LangGraph 的 `StateGraph` 构建了 9 节点状态机（`graph/builder.py:44-52`）。入口 `intent_router` 节点调用 `route_intent`（`graph/router.py:86`），走**「关键词规则优先 + LLM 兜底」的混合路由**——确定性意图（交卷/作答/批改/习惯/概念咨询）由正则 `_rule_route` 秒级命中（confidence 0.7~0.95，不调 LLM），仅开放域才交 LLM 兜底（`_LLM_ROUTER_PROMPT`），LLM 不可用则降级为 `chat`。分类结果再经条件边 `route_by_intent`（`builder.py:22-26`）分流到对应节点：

| 意图 | 路由目标节点 |
|---|---|
| quiz（出题） | `quiz_manager` |
| submit（交卷） | `error_analyzer` |
| grade（批改） | `grading` |
| habit（习惯） | `habit` |
| learn（讲解） | `knowledge_tutor` |
| 其他 / 开放问答 | `react`（ReAct 循环） |

9 个节点：`intent_router → quiz_manager / grading / error_analyzer / memory_manager / habit / knowledge_tutor / react / finalize`（含统一出口 `finalize → END`）。

**受限设计（诚实点）：** 关键教学安全策略由代码硬锁死，而非 LLM 自主决定。例如「单题答错只回『回答错误』、不析错因，须交卷才出报告」这条硬约束，是 `quiz_manager_node` 用条件分支锁定的（并有测试断言兜底），LLM 在红线上被剥夺自主权——这是教育场景的刻意取舍，防幻觉也防偷改答案。此外，路由层还有安全前置闸：输入审核（safety 层）命中时，intent 被直接置为 `blocked`（`builder.py:165`），根本不进入任何业务节点。

---

## 特征二：规划能力（Planning）

**落地：** 核心辅导与开放问答走 `react_step_node`（`graph/react.py`）实现的 **ReAct 循环**，多步推进而非一步到位。每轮执行 `思考(Thought) → 行动决策(Action) → 观察(Observation)` 三步；`Thought`/`Action` 均有**规则式兜底**（`react.py:53-59,91-110`），无 LLM 时也能跑完确定性轨迹。

**双重循环终止（代码层硬约束，不依赖 LLM 自觉，`react.py:188-193`）：**
1. **最大轮数**：`react_loop_count >= max_react_loops`（默认 **8**）→ 强制输出。
2. **信息增益判定**：本轮 Thought 向量（BGE 编码）与上一轮余弦相似度 ≥ `info_gain_threshold`（默认 **0.80**）→ 判定思维停滞，立即停止，避免车轱辘话空转。

`loop_stop_reason`（`information_gain/max_loop/finished/fatal_error/degraded`）写入 trace 与响应，并 `metrics.record_loop_stop`——线上可监控「循环终止原因分布」，把「规划是否健康」做成可观测指标。规划过程还经 toolguard 台账持久化（`task_id`+`step_id`，断点续跑依据），并用 `args_hash` 防同参重复调用。

---

## 特征三：工具使用（Tool Use）

**落地：** 工具经统一 registry 注册，带 Pydantic 参数校验、依赖链声明与 `args_hash` 防重（同参不重复调用）。代表性工具：
- **`grading`**：先调 RAG 检索标准题（`_retrieve_standard`），再调 GLM 判分；
- **多模态批改**：上传作业走 GLM-4V 视觉模型兜底 OCR（paddleocr 缺失时自动接管）；
- **记忆读写**：长期记忆 / 习惯分析的读写工具；
- **讲解 / 出题**：知识检索 + 模板生成。

**受控调用（toolguard，`toolguard.py`）：** 工具不是「想调就调」，而是有治理层——失败分类（参数错误 / 网络 / 配额 / 致命）、熔断、超时、以及 Redis 持久化的 StepLedger 台账。调用稳定性由代码保障，而非依赖 LLM 自觉。

---

## 特征四：记忆（Memory）

**两层记忆，均已工程化：**

1. **短期会话记忆 —— `memory/compressor.py`**
   每轮 `invoke` 前执行 `prune_state / build_context`，采用**三层策略**：
   - **实体锚（永不释放）**：`_extract_entities` 把板块/概念/题目 ID 提升进 `entity_pool`，与具体轮次解耦，永远置顶；
   - **最近 N 轮原样**：`recent_turns=6`，且受 `context_budget=1600` 字符硬约束；
   - **窗口外旧轮次**：相关→压成一行要点；闲聊→整轮释放。

2. **长期记忆 —— Qdrant 双 Collection（`vectorstore.py`）**
   - `knowledge_base`（全局共享，可 topic 过滤 + `score_threshold=0.10` 控精度）；
   - `user_long_term_memory`（按 `user_id` **强制隔离**，检索入口忘传 `user_id` 直接抛 `IsolationError`，绝不退化为全库检索）；
   - 附 `audit_isolation()` 探针自检：写 A 的记忆用 B 查，断言查不到，防越权串扰。

**受限设计（诚实盲区）：** 实体锚用的是固定粗粒度词表（函数/方程/几何…），「只练一元二次方程」这类细粒度约束会退化为泛化主题「方程」，特异性在滑出窗口后丢失——这是已识别、待补细粒度约束抽取的潜伏弱点。

---

## 特征五：反思与自纠（Reflection）

**落地：** 反思贯穿两个层面。

1. **推理层反思 —— ReAct 信息增益**：`react.py:167` 算相邻 Thought 向量余弦，≥0.80 判思维停滞，自动收口，避免 LLM 在一个点上打转。
2. **执行层反思 —— 工具失败自治理**：`react.py:246-270` 在工具出错时走 toolguard 流程：
   - 分类错误类型 → 致命错误（鉴权/配额）**不重试不重规划**，降级输出已获取的部分结果；
   - 可修正错误 → 把错误信息回注 observation，触发**备选工具重规划**（最多 `max_replans=2` 次，`react.py:168`）；
   - 重规划用尽 → 返回部分结果（防死循环，代码层硬约束）。

即「观察到失败 → 反思原因 → 换个法子再来 → 不行就体面收尾」，是完整的自纠闭环。

---

## 收口：这算一个「真 Agent」吗

**算，但定性要准：它是一个「受监督、受约束、工具增强的对话式 Agent」。**

- **五特征全中且已工程化**：自主路由、ReAct 规划、受控工具调用、双层记忆、双层面反思——不是带 if 的 chatbot，是有感知-决策-行动-记忆-反思闭环并落地部署的生产级系统。
- **「受限」是刻意的**：核心安全与教学策略由状态机与代码约束框定，LLM 在框内自主、不越红线。教育场景下「可控的聪明」优于「自由的聪明」，这反而是它作为生产级 Agent 的优点。

> 一句话复述数据流：用户发言 → `intent_router` 路由 → 命中专项节点或进入 `react` 循环（调工具/RAG/多模态）→ 短期压缩 + 长期记忆读写 → 失败则 toolguard 自纠 → `finalize` 输出。
