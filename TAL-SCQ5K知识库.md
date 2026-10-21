# TAL-SCQ5K 知识库

> 初中数学辅导 Agent 的 RAG 检索底座：从开源题库到可召回向量库的全链路。

## 一、数据来源
- **好未来开源初中数学题库 TAL-SCQ5K**，原始文件为 `data/raw/TAL-SCQ5K-CN_*.jsonl`（每行一条 JSON 记录）。
- 单题核心字段：
  - `problem`：题干，混大量 LaTeX（如 `\frac{a}{b}`、`\sqrt{x}`）；
  - `answer_option_list`：**选项，是「分组套分组」的嵌套结构**，每项 `{aoVal:"A", content:"$$3$$"}`；
  - `answer_value`：标准答案字母；
  - `knowledge_point_routes`：知识点路径，用 `->` 分隔（如 `初中数学->函数与图象->...`）；
  - `answer_analysis`：解析，是 **list 且常带 `<p>` HTML 标签与 LaTeX**；
  - `difficulty`（0–4）、`competition_source_list`、`queId`。
- 另用**模板合成题**（`gen_geometry_questions.py`）补齐 TAL 初中子集里偏少的「几何图像」「二元一次方程」板块，并并入 SXT001_CN 公开样例题。

## 二、ETL 清洗（`scripts/build_question_bank.py`）
1. **LaTeX → 可读文本**：`latex_to_text` 递归把 `\frac{a}{b}` 转成 `(a)/(b)`、`\sqrt` 转 `√(a)`，剥离 `\left\right`、array 环境、HTML 标签，再做 `clean_text` 去噪音。
2. **入料门槛（四者齐全才保留）**：`stem` 长度 ≥ 8、非英文原题、属初中内容（`MIDDLE_SCHOOL` 命中）、非高中内容（`HIGH_SCHOOL` 黑名单拦截双曲线/导数/复数等）、非小学奥数。
3. **去重**：按归一化题干去重，剔除数据集中「同题异 qid」的重复记录。
4. **板块映射**：`函数 / 二元一次方程 / 几何图像 / 综合`（正则匹配 `knowledge_point_routes` + `stem`）。
5. **题型派生**：选择题（原始）、填空题（答案文本 ≤ 24 字）、应用题（命中应用题标签或题干 ≥ 70 字）。
6. **知识点清洗**：取 `knowledge_point_routes` 末 4 个，剔除「课内体系 / 思想方法」等无信息量黑词。

## 三、切片策略（`src/math_agent/chunking.py` → `chunk_question`）
- **整题原子切片**：把 `【topic·type】+ 题干 + 各选项 + 答案 + 解析 + 知识点` 拼成**一个 chunk**。
- **一个 parsed 题目记录 = 恰好一个 chunk**，不按固定长度切分——这是压低逐题分析幻觉的关键。
- **边界锚点在 ETL**：`parse_row` 仅当 `stem / options / answer / analysis` 四者齐全才返回记录，结构上「切出半个题」不可能发生。
- **同类策略**：教材 → `chunk_by_structure` 按「定义 / 定理 / 例题」等标题切；自由文本 → `semantic_chunk` 按相邻句余弦相似度 < 阈值（0.55）处切，受 `max_chunk_chars=900` 软约束。

## 四、向量化与存储（`vectorstore.py` + `embeddings.py`）
- **Qdrant 本地嵌入式**，`knowledge_base` collection，`Distance.COSINE`，BGE 向量。
- 一题一 upsert，批量 256；元数据：`text / question_id / topic / type / difficulty / knowledge_points / answer`。
- 检索：`top_k` + 可选 `topic` 过滤 + `score_threshold`（默认 0.10）。

## 五、切不准的边界情况与修复
- **几何图注缺失**：TAL 几何图为独立图片文件，文本里只有「如图」二字，图内角度/边长不在文本中 → 合成题做成**全文本自描述**（数值直接写进题干），上传作业则走 GLM-4V 多模态路径补齐图像信息。
- **超长题截断**：`chunk_question` 暂无长度保护，个别「题干 + 解析」超长题会被 BGE 的 `max_length=512` 在 tokenizer 阶段截断尾部解析 → 修法是将解析拆为**附属 chunk** 共享同一 `question_id`，检索主 chunk 命中即连带返回解析（已排入下一轮）。

---

### 简历可用一句话
> 基于 TAL-SCQ5K 开源题库构建 ETL 管线，按「完整题目」原子切片（题干+选项+答案+解析合并为一个 chunk）替代固定长度切分，经 LaTeX 清洗、初中/高中过滤、去重后入库 Qdrant 本地向量库，从源头降低逐题检索的语义割裂与幻觉。
