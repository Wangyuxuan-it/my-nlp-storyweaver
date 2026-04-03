"""日志清洗工具。"""

from __future__ import annotations

from typing import Dict


ERROR_MARKERS = (
    "系统提示：当前无法连接剧情引擎",
    "未检测到有效的 DEEPSEEK_API_KEY",
    "调用 DeepSeek API 失败",
    "DeepSeek 响应格式异常",
)


def is_successful_record(row: Dict) -> bool:
    """判断一条回合日志是否为有效生成结果。"""
    if row.get("generation_success") is False:
        return False

    ai_reply = str(row.get("ai_reply", "") or "")
    return not any(marker in ai_reply for marker in ERROR_MARKERS)