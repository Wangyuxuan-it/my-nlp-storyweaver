"""
人工评测记录与汇总。

用于记录局末人工评分，支撑课程要求中的主观评估与用户体验评价。
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List


BASE_DIR = Path(__file__).resolve().parent
EVAL_PATH = BASE_DIR / "experiments" / "human_evaluation.csv"
FIELDNAMES = [
    "timestamp_utc",
    "session_id",
    "mode",
    "turn_count",
    "experiment_tag",
    "narrative_quality",
    "response_speed",
    "choice_match",
    "coherence",
    "immersion",
    "overall",
    "comments",
]


def _ensure_file() -> None:
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not EVAL_PATH.exists():
        with EVAL_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def append_human_evaluation(row: Dict) -> None:
    _ensure_file()
    payload = {key: row.get(key, "") for key in FIELDNAMES}
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with EVAL_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(payload)


def load_human_evaluations(path: Path | None = None) -> List[Dict]:
    target = path or EVAL_PATH
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_human_evaluations(path: Path | None = None) -> Dict[str, float]:
    rows = load_human_evaluations(path)
    if not rows:
        return {"num_sessions": 0}

    def values(key: str) -> List[float]:
        result: List[float] = []
        for row in rows:
            raw = row.get(key, "")
            if raw == "":
                continue
            result.append(float(raw))
        return result

    summary: Dict[str, float] = {"num_sessions": float(len(rows))}
    for key in ["narrative_quality", "response_speed", "choice_match", "coherence", "immersion", "overall"]:
        vals = values(key)
        if not vals:
            continue
        summary[f"{key}_mean"] = float(mean(vals))
        summary[f"{key}_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
    return summary


if __name__ == "__main__":
    print(summarize_human_evaluations())
