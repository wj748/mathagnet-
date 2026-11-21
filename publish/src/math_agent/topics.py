"""板块命名单一真源（Single Source of Truth）。

项目里存在两套板块命名：
- 教学板块（CURRICULUM 的 key，讲解/大纲口径）：数与式 / 方程与不等式 / 函数 / 几何 /
  统计与概率 / 锐角三角函数
- 题库板块（SQLite Question.topic / fetch_quiz 的 topic 枚举）：函数 / 二元一次方程 /
  几何图像 / 综合

历史上两边各自维护映射（teaching.py 的 _EXAMPLE_TOPIC 与 nodes.py 的 _NO_BANK_TOPICS），
曾出现同一边缘板块两边映射不一致的漂移（锐角三角函数：一处→几何图像、一处→综合）。
现在统一到本模块：**新增/调整板块只改这里 + CURRICULUM，不再散落打补丁**。

锐角三角函数 → 综合 的依据：题库中三角函数题在 函数/几何图像/二元一次方程 三个板块
均为噪声级分布（2026-09 实测 10/4/9 条），独立出题不达标，回退综合题库最稳。
"""
from __future__ import annotations

from typing import Dict, List

# 题库板块（fetch_quiz 的 topic 枚举；也是 SQLite questions.topic 的合法取值）
BANK_TOPICS: List[str] = ["函数", "二元一次方程", "几何图像", "综合"]

# 题库板块中真正有独立出题能力的（"综合"是兜底桶，不算独立板块）
BANK_CORE_TOPICS: List[str] = ["函数", "二元一次方程", "几何图像"]

# 教学板块（与 teaching.CURRICULUM 的 key 保持同序）
TEACHING_TOPICS: List[str] = ["数与式", "方程与不等式", "函数", "几何",
                              "统计与概率", "锐角三角函数"]

# 教学板块 → 题库板块（讲解配例题 / 教学板块直出刷题时使用）
TEACHING_TO_BANK: Dict[str, str] = {
    "数与式": "综合",
    "方程与不等式": "二元一次方程",
    "函数": "函数",
    "几何": "几何图像",
    "统计与概率": "综合",
    "锐角三角函数": "综合",
}


def to_bank_topic(topic: str) -> str:
    """任意口径的板块名 → 题库板块。未知板块回退「综合」（不静默假装有题）。"""
    if topic in BANK_TOPICS:
        return topic
    return TEACHING_TO_BANK.get(topic, "综合")


__all__ = ["BANK_TOPICS", "BANK_CORE_TOPICS", "TEACHING_TOPICS",
           "TEACHING_TO_BANK", "to_bank_topic"]
