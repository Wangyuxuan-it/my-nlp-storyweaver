"""
玩家输入意图识别模块。

输出四类标签：
- investigate: 调查
- interrogate: 质询
- deduce: 推理
- act: 执行动作
"""

from __future__ import annotations

import re
from typing import Dict


INTENT_LABELS: Dict[str, str] = {
    "investigate": "调查",
    "interrogate": "质询",
    "deduce": "推理",
    "act": "行动",
}


KEYWORDS = {
    "investigate": ["检查", "勘察", "查看", "搜", "线索", "证据", "监控", "记录", "调取", "取证"],
    "interrogate": ["询问", "追问", "质问", "盘问", "对话", "问", "口供", "证词"],
    "deduce": ["推理", "怀疑", "判断", "推断", "猜测", "可能", "动机", "逻辑"],
    "act": ["前往", "进入", "跟踪", "逮捕", "布控", "埋伏", "行动", "执行", "协助"],
}


def detect_intent(user_input: str) -> str:
    """基于关键词重叠的轻量意图识别。"""
    text = (user_input or "").strip().lower()
    if not text:
        return "act"

    score = {label: 0 for label in KEYWORDS}
    for label, words in KEYWORDS.items():
        for w in words:
            if w in text:
                score[label] += 1

    # 若包含明显问句，优先判为质询。
    if re.search(r"\?|？|请问|为什么|谁|何时|哪里", text):
        score["interrogate"] += 2

    best = max(score, key=score.get)
    if score[best] == 0:
        return "act"
    return best
