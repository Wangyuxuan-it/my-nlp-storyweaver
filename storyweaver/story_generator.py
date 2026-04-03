"""
AI 剧情生成模块
负责：
1. 维护系统提示词与对话历史。
2. 调用 DeepSeek API 生成剧情推进文本。
3. 统一处理请求异常，避免前端崩溃。
"""

from __future__ import annotations

from typing import List, Dict, Tuple
import requests

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    MAX_HISTORY,
    MAX_TOKENS,
    TEMPERATURE,
    REQUEST_TIMEOUT,
)


class StoryGenerator:
    """文字冒险剧情生成器。"""

    def __init__(self) -> None:
        # 系统提示词定义游戏规则与叙事风格。
        self.system_prompt = (
            "你是一个现代探案推理文字冒险游戏引擎。"
            "玩家扮演私家侦探林深，正在调查一宗失踪案。"
            "请严格遵循以下规则："
            "1) 使用中文输出；"
            "2) 每次回复必须且只允许包含三个小节，标题依次为：场景、可疑线索、可执行行动；"
            "3) 不要输出任何额外小节标题（如行动建议、技术鉴定失败等）；"
            "4) 在“可执行行动”中给出 2-4 条下一步行动；"
            "5) 对玩家输入做出因果明确的反馈；"
            "6) 保持悬疑氛围与逻辑一致性，不要直接泄露全部真相；"
            "7) 文风简洁有画面感，单次回复控制在 180-320 字左右。"
        )

    def _build_control_prompt(self, mode: str, turn: int, max_turns: int) -> str:
        """根据模式与回合数，构造节奏控制提示词。"""
        if mode == "叙事模式":
            if turn <= 6:
                phase = "铺垫阶段"
                phase_rule = "重点建立人物关系与时间线，给出1-2条可验证线索。"
            elif turn <= 14:
                phase = "调查阶段"
                phase_rule = "推动证据链延展与矛盾浮现，引导玩家排除干扰信息。"
            elif turn <= 19:
                phase = "收束阶段"
                phase_rule = "减少新分支，集中核对关键证词与关键物证。"
            else:
                phase = "真相揭示阶段"
                phase_rule = (
                    "必须明确给出真相，不再留悬念；输出应包含“真相揭示”“证据链”“结案结论”三个小节。"
                )

            return (
                f"当前游戏模式：叙事模式（最多{max_turns}回合）。"
                f"当前回合：第{turn}回合。"
                f"当前阶段：{phase}。"
                f"阶段要求：{phase_rule}"
                "整体节奏要求：信息密度逐步提高，避免重复描述，保证剧情可在限定回合内收束。"
                "输出长度控制：150-240字。"
            )

        return (
            "当前游戏模式：自由模式。"
            "节奏要求：保持悬疑推进，但不过度拖沓。"
            "输出长度控制：160-260字。"
        )

    def _build_messages(
        self,
        player_input: str,
        history: List[Tuple[str, str]],
        control_prompt: str | None = None,
        intent_label: str | None = None,
        case_memory: str | None = None,
    ) -> List[Dict[str, str]]:
        """
        组装发给 DeepSeek 的 messages。

        参数:
        - player_input: 玩家本轮输入。
        - history: 历史轮次 [(user_text, assistant_text), ...]

        返回:
        - OpenAI-compatible messages 列表。
        """
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        if control_prompt:
            messages.append({"role": "system", "content": control_prompt})
        if intent_label:
            messages.append(
                {
                    "role": "system",
                    "content": f"玩家本轮核心意图：{intent_label}。请优先回应该意图并给出可执行推进。",
                }
            )
        if case_memory:
            messages.append({"role": "system", "content": case_memory})

        # 只保留最近 MAX_HISTORY 轮，兼顾上下文与成本。
        recent_history = history[-MAX_HISTORY:] if history else []
        for user_text, assistant_text in recent_history:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})

        messages.append({"role": "user", "content": player_input})
        return messages

    def _call_deepseek(self, messages: List[Dict[str, str]]) -> str:
        """
        调用 DeepSeek Chat Completions 接口并返回文本。

        若请求失败，抛出 RuntimeError，交由上层统一处理。
        """
        if not DEEPSEEK_API_KEY or "请在这里填入" in DEEPSEEK_API_KEY:
            raise RuntimeError("未检测到有效的 DEEPSEEK_API_KEY，请先在 .env 中配置。")

        url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"调用 DeepSeek API 失败：{exc}") from exc

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"DeepSeek 响应格式异常：{data}") from exc

    def generate_next(self, player_input: str, history: List[Tuple[str, str]]) -> str:
        """
        根据玩家输入与历史，生成下一段剧情。

        参数:
        - player_input: 玩家输入。
        - history: 历史轮次列表。

        返回:
        - AI 剧情文本（字符串）。
        """
        messages = self._build_messages(player_input=player_input, history=history)
        return self._call_deepseek(messages)

    def generate_next_with_control(
        self,
        player_input: str,
        history: List[Tuple[str, str]],
        mode: str,
        turn: int,
        max_turns: int,
        intent_label: str | None = None,
        case_memory: str | None = None,
        control_prompt_override: str | None = None,
    ) -> str:
        """根据模式与回合控制生成剧情。"""
        control_prompt = control_prompt_override or self._build_control_prompt(mode=mode, turn=turn, max_turns=max_turns)
        messages = self._build_messages(
            player_input=player_input,
            history=history,
            control_prompt=control_prompt,
            intent_label=intent_label,
            case_memory=case_memory,
        )
        return self._call_deepseek(messages)

    def generate_candidates_with_control(
        self,
        player_input: str,
        history: List[Tuple[str, str]],
        mode: str,
        turn: int,
        max_turns: int,
        intent_label: str | None = None,
        case_memory: str | None = None,
        control_prompt_override: str | None = None,
        num_candidates: int = 3,
    ) -> List[str]:
        """生成多个候选回复，用于后续重排序。"""
        control_prompt = control_prompt_override or self._build_control_prompt(mode=mode, turn=turn, max_turns=max_turns)
        messages = self._build_messages(
            player_input=player_input,
            history=history,
            control_prompt=control_prompt,
            intent_label=intent_label,
            case_memory=case_memory,
        )

        total = max(1, int(num_candidates))
        candidates: List[str] = []
        for _ in range(total):
            candidates.append(self._call_deepseek(messages))
        return candidates

    @staticmethod
    def opening_scene() -> str:
        """返回开场白，用于初始化游戏。"""
        return (
            "**[ 深夜 22:40 | 临江市 ]**\n\n"
            "你，私家侦探林深，收到一通急促电话。\n"
            "“我妹妹苏语昨晚失踪了，警方说还需等待 24 小时……”\n"
            "十分钟后，你抵达报案人公寓。雨水顺着窗沿滴落，客厅茶几上有一只摔裂的手机，"
            "空气里残留着淡淡消毒水味。\n\n"
            "**[ 你可以尝试 ]**\n\n"
            "A.检查摔裂手机\n"
            "B.询问失踪前最后见面的人\n"
            "C.勘察公寓门锁与走廊监控\n"
            "D.查看受害者社交账号动态"
        )
