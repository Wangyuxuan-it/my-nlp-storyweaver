"""
训练 StoryWeaver 的可学习模型。

生成：
- artifacts/intent_model.joblib
- artifacts/coherence_model.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_models import ARTIFACT_DIR, CoherenceClassifier, IntentClassifier


BASE_DIR = Path(__file__).resolve().parent


def _display_path(path: Path) -> str:
    """优先输出相对项目根目录的路径，避免机器相关绝对路径污染。"""
    try:
        return str(path.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StoryWeaver ML models")
    parser.add_argument("--intent-data", default="data/processed/intent_dataset.jsonl")
    parser.add_argument("--coherence-data", default="data/processed/coherence_dataset.jsonl")
    parser.add_argument("--out-dir", default=str(ARTIFACT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intent_model = IntentClassifier(out_dir / "intent_model.joblib")
    coherence_model = CoherenceClassifier(out_dir / "coherence_model.joblib")

    intent_metrics = intent_model.train(Path(args.intent_data))
    coherence_metrics = coherence_model.train(Path(args.coherence_data))

    summary = {
        "intent": intent_metrics,
        "coherence": coherence_metrics,
        "intent_model_path": _display_path(out_dir / "intent_model.joblib"),
        "coherence_model_path": _display_path(out_dir / "coherence_model.joblib"),
    }

    with (out_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Training Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()