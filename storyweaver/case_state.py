"""
案件一致性状态机（轻量版）。

维护两类核心结构：
1. timeline: 时间线事件
2. evidence: 证据/线索条目
"""

from __future__ import annotations

import re
from typing import Dict, List


def init_case_state() -> Dict[str, object]:
    """初始化案件状态。"""
    return {
        "timeline": [],
        "evidence": [],
        "intent_counts": {
            "investigate": 0,
            "interrogate": 0,
            "deduce": 0,
            "act": 0,
        },
        "last_intent": "act",
    }


def _extract_section(text: str, section: str) -> str:
    pattern = re.compile(r"\*\*\[\s*(场景|可疑线索|可执行行动)\s*\]\*\*")
    matches = list(pattern.finditer(text))
    if not matches:
        return ""

    for idx, match in enumerate(matches):
        name = match.group(1)
        if name != section:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        return text[start:end].strip()
    return ""


def _clean_line(line: str) -> str:
    return re.sub(r"^[A-Z]\.|^[0-9]+\.|^[\-\*]\s*", "", line).strip()


def update_case_state(
    state: Dict[str, object],
    turn: int,
    user_input: str,
    ai_reply: str,
    intent_label: str,
) -> Dict[str, object]:
    """基于本回合输入输出更新一致性状态。"""
    timeline: List[str] = list(state.get("timeline", []))
    evidence: List[str] = list(state.get("evidence", []))
    intent_counts = dict(state.get("intent_counts", {}))

    scene_text = _extract_section(ai_reply, "场景")
    clue_text = _extract_section(ai_reply, "可疑线索")

    if scene_text:
        timeline.append(f"T{turn}: {scene_text[:120]}")

    for raw in clue_text.splitlines():
        item = _clean_line(raw)
        if item:
            evidence.append(item)

    # 去重并保留最近条目，避免提示词过长。
    dedup_evidence: List[str] = []
    seen = set()
    for item in evidence:
        if item in seen:
            continue
        seen.add(item)
        dedup_evidence.append(item)

    for key in ("investigate", "interrogate", "deduce", "act"):
        intent_counts.setdefault(key, 0)
    intent_counts[intent_label] = intent_counts.get(intent_label, 0) + 1

    return {
        "timeline": timeline[-12:],
        "evidence": dedup_evidence[-15:],
        "intent_counts": intent_counts,
        "last_intent": intent_label,
        "last_user_input": user_input,
    }


def summarize_case_state(state: Dict[str, object]) -> str:
    """将案件状态压缩为提示词可用摘要。"""
    timeline: List[str] = list(state.get("timeline", []))
    evidence: List[str] = list(state.get("evidence", []))
    intent_counts = state.get("intent_counts", {})
    last_intent = state.get("last_intent", "act")

    timeline_text = "；".join(timeline[-4:]) if timeline else "暂无时间线记录"
    evidence_text = "；".join(evidence[-6:]) if evidence else "暂无证据记录"

    return (
        "【案件一致性记忆】"
        f"最近意图={last_intent}；"
        f"意图分布={intent_counts}；"
        f"时间线摘要={timeline_text}；"
        f"证据摘要={evidence_text}。"
        "请保持与上述记录一致，新增信息需与已有线索兼容。"
    )
