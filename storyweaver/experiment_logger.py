"""
实验日志模块。

按 JSONL 记录每回合评估数据，便于后续统计与画图。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class ExperimentLogger:
    """将实验记录追加写入 JSONL 文件。"""

    def __init__(self, log_dir: str = "experiments", file_name: str = "turn_metrics.jsonl") -> None:
        self.log_path = Path(log_dir) / file_name
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_turn(self, payload: Dict[str, Any]) -> None:
        """写入单回合记录。"""
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
