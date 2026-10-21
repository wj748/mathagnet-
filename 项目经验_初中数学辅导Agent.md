# 项目经验

## 初中数学智能辅导 Agent
**技术栈：** Python、LangGraph、FastAPI、Qdrant、Redis、BGE Embedding、GLM 大模型、PaddleOCR / GLM-4V、Docker、Caddy

**项目背景：** 传统课后辅导依赖家长陪读或课外报班，教师与家长精力有限，批改反馈滞后，学生错题难以及时针对性巩固；通用大模型答疑常直接给出完整解析，学生缺少独立思考空间。需构建一套集题目引导讲解、作业批改、错题归因与学习习惯分析于一体的初中数学智能辅导系统。

**项目描述：** 开发集多轮辅导对话、RAG 知识检索、工具调用、失败治理、会话压缩于一体的初中数学辅导 Agent，完成 9 节点状态机设计、TAL-SCQ5K 知识库构建、安全约束落地、长程记忆隔离与公网安全部署全链路开发。

1. **状态机与失败治理模块：** 基于 LangGraph StateGraph 构建 9 节点状态机（router / grading / react 等），以 TypedDict 定义状态、ReAct 循环驱动多轮辅导；自研 toolguard 做失败分类、熔断、台账（Redis 持久化）与自动重规划，保障工具调用稳定性。
2. **RAG 知识库模块：** 基于 TAL-SCQ5K 数据集构建 ETL 管线，按「完整题目」原子切片（题干+选项+答案+解析合并为一个 chunk），避免固定长度切分造成的语义割裂；采用 Qdrant 本地嵌入式向量库做双 Collection 隔离（knowledge_base / user_long_term_memory），BGE 向量召回并以 score_threshold 控精度。
3. **安全与防幻觉约束模块：** 实现「单题答错仅提示『回答错误』、不析错因，须交卷或批改整份作业方出阶段性错题报告」的硬约束，从源头抑制偷看答案与逐题幻觉；前端上传作业走多模态（GLM-4V）兜底 OCR 批改。
4. **会话压缩与长程记忆模块：** 自研 compressor 在每轮 invoke 前执行 prune_state / build_context，采用「实体锚（永不释放）+ 最近 N 轮原样 + 窗口外摘要」三层策略保留用户关键约束；用户长期记忆按 user_id 强制隔离检索，并附隔离自检探针防越权串扰。
5. **后端服务与安全部署模块：** FastAPI + uvicorn 提供 API（端口 8787），HMAC token 鉴权（user_id:ts:sig，30 天 TTL）、CORS 白名单；Docker + Caddy 自动 TLS 公网发布，require_token 强制开启，单实例容量上限已文档化声明。

**项目成果：** 搭建「题目输入 - 意图路由 - 知识检索 - 智能讲解 / 批改」一体化辅导链路，支持单题遮罩、整卷错题报告与学习习惯分析；支持公网安全访问，满足初中数学实时辅导需求。
