"""
独立对话管理层。

负责：
- 根据模式与回合划分阶段
- 根据用户意图设计分支策略
- 将案件状态记忆、意图结果和对话策略组织成系统提示词
"""

from __future__ import annotations

from dataclasses import dataclass


INTENT_FOCUS = {
    "investigate": "优先输出场景细节、物证发现和可验证线索，强调信息采集。",
    "interrogate": "优先推动人物对话与矛盾追问，强调证词冲突和动机识别。",
    "deduce": "优先收束推理链条，明确假设、证据链与推断结论。",
    "act": "优先执行具体行动，给出清晰可操作的下一步。",
}


@dataclass
class DialoguePlan:
    mode: str
    turn: int
    max_turns: int
    phase: str
    branch_focus: str
    intent_label: str
    intent_confidence: float
    control_prompt: str


class DialogueManager:
    """管理对话阶段、分支策略与生成约束。"""

    def _phase_for_turn(self, mode: str, turn: int, max_turns: int) -> tuple[str, str]:
        if mode != "叙事模式":
            return "自由探索阶段", "保持开放式调查，但仍要围绕当前线索推进。"

        if turn <= 6:
            return "铺垫阶段", "重点建立人物关系、时间线和初始证据。"
        if turn <= 14:
            return "调查阶段", "推动证据链延展与冲突浮现，主动分辨干扰信息。"
        if turn <= 19:
            return "收束阶段", "减少无效分支，聚焦关键证词与物证。"
        return "真相揭示阶段", "必须给出明确真相、证据链和结案结论。"

    def _branch_focus(self, intent_label: str, intent_confidence: float) -> str:
        focus = INTENT_FOCUS.get(intent_label, "优先回应玩家当前动作，并保持剧情逻辑。")
        if intent_confidence < 0.55:
            return focus + " 若意图不确定，请用兼容性较强的方式回复并提供澄清性线索。"
        return focus

    def build_control_prompt(
        self,
        mode: str,
        turn: int,
        max_turns: int,
        intent_label: str,
        intent_confidence: float,
        case_memory: str,
    ) -> DialoguePlan:
        phase, phase_rule = self._phase_for_turn(mode, turn, max_turns)
        branch_focus = self._branch_focus(intent_label, intent_confidence)

        prompt = (
            f"当前游戏模式：{mode}。"
            f"当前回合：第{turn}回合（最多{max_turns}回合）。"
            f"当前阶段：{phase}。"
            f"阶段要求：{phase_rule}"
            f"玩家意图：{intent_label}（置信度 {intent_confidence:.2f}）。"
            f"分支策略：{branch_focus}"
            "对话管理原则：1) 明确回应玩家本轮核心动作；"
            "2) 保持对话状态连续，不引入与案件无关的新主题；"
            "3) 若信息不足，优先抛出可追问的线索或矛盾点；"
            "4) 输出必须维持三段结构：场景、可疑线索、可执行行动；"
            "5) 语言风格简洁、悬疑、可执行。"
        )
        if case_memory:
            prompt += f" 案件状态记忆：{case_memory}"

        return DialoguePlan(
            mode=mode,
            turn=turn,
            max_turns=max_turns,
            phase=phase,
            branch_focus=branch_focus,
            intent_label=intent_label,
            intent_confidence=intent_confidence,
            control_prompt=prompt,
        )
