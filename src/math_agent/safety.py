"""未成年人数据保护 + 内容审核过滤（Phase 5 运维侧）。

三层防护，全部**规则层硬约束**（不依赖模型自觉，离线也生效）：

1. **输入审核** :func:`check_input` —— 违法/暴力/色情/自伤/代写作弊等内容直接拦截，
   返回面向初中生的温和引导话术，不进入 Agent 主流程（省 token 也省风险）。
2. **PII 脱敏** :func:`mask_pii` —— 手机号/身份证/邮箱/QQ/住址/学校-班级等个人信息，
   在进入日志、trace、长期记忆之前统一打码（数据最小化）。
3. **输出审核** :func:`sanitize_output` —— 回复同样脱敏，并拦掉模型可能出现的
   "留下学校/电话/加微信" 这类索取未成年人隐私的话术（防模型越界）。

设计取舍：
  * 词表用**词组**而非单字，避免误伤数学语境（如"求根""解方程""交点"）；
  * 命中即拦截，但**不记录原始违规文本**（只记类别），避免日志二次泄漏；
  * 所有函数都是纯函数，无 IO、无锁，可安全在任意节点调用。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# ------------------------------------------------------------------ 归一化
# 变体绕过防护：全角→半角、去空白/分隔符、常见谐音字替换后再匹配
_FW_TABLE = {c: chr(c - 0xFEE0) for c in range(0xFF01, 0xFF5F)}
_FW_TABLE[0x3000] = " "
# 只清分隔类字符，保留字母数字汉字（数学运算符保留，避免误合并产生新词）
_SEP_CHARS = re.compile(r"[\s、，。．,\.!！?？;；:：~～·\-_*×÷()\[\]{}（）【】《》\"'“”‘’]+")
_VARIANT_MAP = {"自鲨": "自杀", "自殺": "自杀", "薇信": "微信", "微伈": "微信",
                "v信": "微信", "绿泡泡": "微信", "扣扣": "QQ", "企鹅号": "QQ"}


def _normalize_for_match(text: str) -> str:
    # 不做 lower()：审核词表含大写（QQ 等），小写化会让全角转半角后的变体反而失配
    t = str(text or "").translate(_FW_TABLE)
    t = _SEP_CHARS.sub("", t)
    for a, b in _VARIANT_MAP.items():
        t = t.replace(a, b)
    return t


# ------------------------------------------------------------------ PII
_PII_RULES: List[Tuple[str, re.Pattern, str]] = [
    # 相邻是数学运算符/等号的长数字串视为算式，不脱敏（避免数学答案被打码）
    ("phone", re.compile(r"(?<![\d=+\-*/^×÷±.])1[3-9]\d{9}(?![\d=+\-*/^×÷±.])"), "📱[手机号]"),
    ("idcard", re.compile(r"(?<![\d=+\-*/^×÷±.])\d{17}[\dXx](?![\d=+\-*/^×÷±.])"), "🪪[身份证]"),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "✉️[邮箱]"),
    # 变体：QQ／全角ＱＱ／Q Q／薇信／微伈／v信
    ("qq", re.compile(r"(?i)(?:[QqＱｑ]\s*[QqＱｑ]|微信|薇信|微伈|v信|wechat|vx)\s*[:：]?\s*[\w-]{5,}"),
     "💬[联系方式]"),
]
# 住址/学校：带明确后缀才判定，避免误伤"小明家到学校"这类应用题
_ADDR_RE = re.compile(r"[\u4e00-\u9fa5]{2,8}(?:省|市|区|县|镇|街道|村)[\u4e00-\u9fa5]{0,10}"
                      r"(?:路|街|巷|号|栋|单元|室|小区|花园|苑)")
_SCHOOL_RE = re.compile(r"[\u4e00-\u9fa5]{2,10}(?:中学|小学|初中|附中|一中|二中|三中|实验小学)"
                        r"[\u4e00-\u9fa5]{0,4}(?:[\d一二三四五六七八九十]+年级)?"
                        r"(?:[\d一二三四五六七八九十]+班)?")


def mask_pii(text: Any) -> str:
    """脱敏：手机号/身份证/邮箱/联系方式/详细住址/学校班级 → 占位符。"""
    s = str(text or "")
    if not s:
        return ""
    for _name, pat, repl in _PII_RULES:
        s = pat.sub(repl, s)
    s = _ADDR_RE.sub("🏠[住址]", s)
    s = _SCHOOL_RE.sub("🏫[学校]", s)
    return s


def has_pii(text: Any) -> bool:
    return mask_pii(text) != str(text or "")


# ------------------------------------------------------------------ 输入审核
# 词表：只放明确违规的**词组**，绝不放单字（数学题里"杀""死""解"都常见）
_BLOCK_RULES: List[Tuple[str, re.Pattern, str]] = [
    ("violence", re.compile(r"(打架|打人|砍人|捅人|杀人|自杀|自残|伤害自己|跳楼|跳河|"
                            r"怎么报复|打群架|校园霸凌|勒索|抢劫)"), "violence"),
    ("sexual", re.compile(r"(色情|黄色影片|淫秽|裸照|援交|开房|约炮|性交)"), "sexual"),
    ("self_harm", re.compile(r"(不想活|活着没意思|割腕|吞药|轻生|结束生命)"), "self_harm"),
    ("illegal", re.compile(r"(毒品|吸毒|冰毒|走私|贩卖|赌博|赌场|黑产|洗钱|"
                           r"黑客攻击|入侵(?:他人|别人|系统|网站|服务器|账号)|"
                           r"破解(?:密码|账号)|盗号|盗取(?:账号|密码)|木马|病毒传播)"), "illegal"),
    ("cheating", re.compile(r"(代写|代考|替考|枪手|作弊器|考试作弊|帮我直接写答案不要过程|"
                            r"入侵教务|改成绩|篡改成绩)"), "cheating"),
    ("privacy_probe", re.compile(r"(你的?(?:住址|家庭地址)|你?(?:在哪个学校|是哪个学校的)|"
                                 r"留个?(?:电话|微信|QQ)|加我?(?:微信|QQ)|"
                                 r"(?:微信|QQ|电话|联系方式)(?:是)?(?:多少|号码|什么|几)|"
                                 r"加(?:你|下|个)?(?:微信|QQ))"), "privacy_probe"),
]

_HINTS: Dict[str, str] = {
    "violence": "这个问题涉及伤害自己或他人，我不能回答哦。如果正在遇到困扰，"
                "可以告诉爸爸妈妈、老师，或拨打 12355 青少年服务热线寻求帮助。",
    "sexual": "这个话题不适合在这里讨论，我们还是回到数学学习吧～",
    "self_harm": "听起来你现在很难过。请一定告诉信任的家长或老师，也可以拨打 "
                 "12355 青少年服务热线，有人愿意帮助你。数学题我们随时可以再聊。",
    "illegal": "这类内容我不能提供帮助。我们还是一起把数学题搞懂吧～",
    "cheating": "作业和考试要自己做才有用哦。我可以给你讲思路、出类似的题练习，"
                "但不会替你写答案。",
    "privacy_probe": "为了保护你的隐私，请不要在对话里留电话、住址、学校等个人信息，"
                     "也不要向他人索要哦。我们继续学数学吧～",
}


def check_input(text: Any) -> Dict[str, Any]:
    """输入审核。返回 ``{allowed, category, reply}``；allowed=False 时 reply 为引导话术。

    匹配前先做归一化（全角→半角、去空白分隔符、谐音变体替换），
    防"自 杀 / 薇信 / 全角ＱＱ"这类变体绕过；命中与否均不记录原文。
    """
    s = str(text or "")
    if not s.strip():
        return {"allowed": True, "category": "", "reply": ""}
    s_norm = _normalize_for_match(s)
    for _name, pat, cat in _BLOCK_RULES:
        if pat.search(s) or pat.search(s_norm):
            return {"allowed": False, "category": cat,
                    "reply": _HINTS.get(cat, "这个话题不适合在这里讨论，我们回到数学学习吧～")}
    return {"allowed": True, "category": "", "reply": ""}


# ------------------------------------------------------------------ 输出审核
_SOLICIT_RE = re.compile(r"(留下(?:你的)?(?:电话|微信|QQ|地址|学校)|"
                         r"(?:把你的?|把联系方式的?)(?:电话|微信|QQ|地址|学校)(?:留下来|留下|发给我|给我)?|"
                         r"告诉我(?:你的)?(?:家庭住址|学校名称|班级)|"
                         r"加(?:我|老师|你)(?:微信|QQ))")


def sanitize_output(text: Any) -> str:
    """输出审核：脱敏 + 移除向未成年人索取隐私的话术。"""
    s = mask_pii(text)
    if _SOLICIT_RE.search(s):
        s = _SOLICIT_RE.sub("（为了保护你的隐私，这里不收集个人信息）", s)
    return s


__all__ = ["mask_pii", "has_pii", "check_input", "sanitize_output"]
