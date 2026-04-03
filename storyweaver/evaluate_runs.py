"""
离线统计脚本。

使用方法：
python evaluate_runs.py
python evaluate_runs.py --input experiments/turn_metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List


def _safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_std(values: List[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def load_records(path: Path) -> List[Dict]:
    if not path.exists():
        return []

    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize(rows: List[Dict]) -> Dict[str, float]:
    latency = [float(r.get("latency_ms", 0)) for r in rows if "latency_ms" in r]
    coherence = [float(r.get("plot_coherence_score", 0)) for r in rows if "plot_coherence_score" in r]
    narrative = [float(r.get("narrative_quality_score", 0)) for r in rows if "narrative_quality_score" in r]
    choice_match = [float(r.get("choice_match_score", 0)) for r in rows if "choice_match_score" in r]
    immersion = [
        float(r.get("immersion_score", 0))
        for r in rows
        if r.get("immersion_score") is not None
    ]

    return {
        "num_turns": len(rows),
        "latency_ms_mean": _safe_mean(latency),
        "latency_ms_std": _safe_std(latency),
        "plot_coherence_mean": _safe_mean(coherence),
        "narrative_quality_mean": _safe_mean(narrative),
        "choice_match_mean": _safe_mean(choice_match),
        "immersion_mean": _safe_mean(immersion),
        "high_choice_match_rate": (
            sum(1 for x in choice_match if x >= 4) / len(choice_match) if choice_match else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize StoryWeaver evaluation logs")
    parser.add_argument("--input", default="experiments/turn_metrics.jsonl", help="Path to JSONL log")
    args = parser.parse_args()

    rows = load_records(Path(args.input))
    if not rows:
        print("No records found. Run the game first to generate logs.")
        return

    stats = summarize(rows)
    print("=== StoryWeaver Evaluation Summary ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
