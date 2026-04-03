"""
评估指标模块。

提供作业要求中的核心量化项：
1. 玩家选择匹配得分（choice_match_score）
2. 情节连贯性得分（plot_coherence_score）
3. 叙事质量得分（narrative_quality_score）

说明：
- 当前版本采用轻量启发式评分，保证离线可运行、可复现。
- 所有得分均为 1~5（整数），便于课堂展示与实验统计。
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple


SECTION_PATTERN = re.compile(r"\*\*\[\s*(场景|可疑线索|可执行行动)\s*\]\*\*")


def _extract_sections(text: str) -> Dict[str, str]:
    """从标准化回复中提取三段内容。"""
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return {}

    sections: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        section_name = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[section_name] = text[start:end].strip()
    return sections


def _zh_bigrams(text: str) -> set[str]:
    """构造中文双字片段集合，降低分词依赖。"""
    clean = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))
    if len(clean) < 2:
        return set()
    return {clean[i : i + 2] for i in range(len(clean) - 1)}


def _scale_ratio_to_5(ratio: float) -> int:
    """将 [0,1] 的相似度映射为 1~5 分。"""
    if ratio >= 0.60:
        return 5
    if ratio >= 0.40:
        return 4
    if ratio >= 0.25:
        return 3
    if ratio >= 0.10:
        return 2
    return 1


def _normalize_action_line(line: str) -> str:
    return re.sub(r"^[A-Za-z0-9一二三四五六七八九十\(\)（）：:、\.\-\s]+", "", line).strip()


def _extract_action_lines(action_text: str) -> List[str]:
    lines: List[str] = []
    for raw in action_text.splitlines():
        line = _normalize_action_line(raw)
        if line:
            lines.append(line)
    return lines


def _line_similarity(user_input: str, candidate: str) -> float:
    user_grams = _zh_bigrams(user_input)
    cand_grams = _zh_bigrams(candidate)
    if not user_grams or not cand_grams:
        return 0.0

    overlap = len(user_grams & cand_grams)
    cover_ratio = overlap / max(1, len(user_grams))
    jaccard = overlap / max(1, len(user_grams | cand_grams))
    return 0.7 * cover_ratio + 0.3 * jaccard


def score_choice_match(user_input: str, ai_reply: str) -> int:
    """评估玩家输入与 AI 给出的后续行动建议是否匹配。"""
    sections = _extract_sections(ai_reply)
    action_text = sections.get("可执行行动", "")
    if not action_text:
        return 1

    action_lines = _extract_action_lines(action_text)
    if not action_lines:
        return 1

    user_text = user_input.strip()
    option_match = re.match(r"^\s*([A-Za-z])[\.)）:：、\s]", user_text)
    if option_match:
        idx = ord(option_match.group(1).upper()) - ord("A")
        if 0 <= idx < len(action_lines):
            return 5

    compact_user = re.sub(r"\s+", "", user_text)
    compact_action = re.sub(r"\s+", "", action_text)
    if compact_user and compact_user in compact_action:
        return 5

    best_similarity = max(_line_similarity(user_text, line) for line in action_lines)
    return _scale_ratio_to_5(best_similarity)


def score_plot_coherence(history: Sequence[Tuple[str, str]], ai_reply: str) -> int:
    """评估情节连贯性，考虑结构完整度、重复度和文本充分性。"""
    sections = _extract_sections(ai_reply)
    score = 5

    required = {"场景", "可疑线索", "可执行行动"}
    if not required.issubset(set(sections.keys())):
        score -= 2

    text_len = len(re.sub(r"\s+", "", ai_reply))
    if text_len < 100:
        score -= 1

    if history:
        last_reply = history[-1][1]
        now = _zh_bigrams(ai_reply)
        prev = _zh_bigrams(last_reply)
        if now and prev:
            jaccard = len(now & prev) / max(1, len(now | prev))
            if jaccard > 0.80:
                score -= 1

    return max(1, min(5, score))


def score_narrative_quality(ai_reply: str) -> int:
    """评估叙事质量，关注结构、信息密度、可执行性。"""
    sections = _extract_sections(ai_reply)
    score = 1

    if sections:
        score += 1

    if all(key in sections for key in ("场景", "可疑线索", "可执行行动")):
        score += 1

    clue_lines = [line for line in sections.get("可疑线索", "").splitlines() if line.strip()]
    action_lines = [line for line in sections.get("可执行行动", "").splitlines() if line.strip()]
    if len(clue_lines) >= 2:
        score += 1
    if len(action_lines) >= 2:
        score += 1

    return max(1, min(5, score))


def evaluate_turn_metrics(
    user_input: str,
    ai_reply: str,
    history: Sequence[Tuple[str, str]],
) -> Dict[str, int]:
    """计算单回合所有核心指标。"""
    return {
        "choice_match_score": score_choice_match(user_input=user_input, ai_reply=ai_reply),
        "plot_coherence_score": score_plot_coherence(history=history, ai_reply=ai_reply),
        "narrative_quality_score": score_narrative_quality(ai_reply=ai_reply),
    }
