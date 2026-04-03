"""
可训练 NLP 模型封装。

当前包含：
- 意图分类器：四分类（investigate / interrogate / deduce / act）
- 连贯性判别器：二分类（coherent / incoherent）

实现方式：
- 使用 scikit-learn 训练轻量文本分类器
- 以 jsonl/csv 形式的数据集为输入
- 以 joblib 持久化到 artifacts/ 目录
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


@dataclass
class LabelPrediction:
    label: str
    confidence: float


@dataclass
class BinaryPrediction:
    label: int
    probability: float


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _ensure_artifact_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _build_text_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)),
            (
                "classifier",
                LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42),
            ),
        ]
    )


def _split_records(records: Sequence[Dict], label_key: str, text_key: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    train_rows = [row for row in records if row.get("split") == "train"]
    val_rows = [row for row in records if row.get("split") == "val"]
    test_rows = [row for row in records if row.get("split") == "test"]

    if train_rows and test_rows:
        return train_rows, val_rows, test_rows

    texts = [row[text_key] for row in records]
    labels = [row[label_key] for row in records]
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels if len(set(temp_labels)) > 1 else None,
    )

    train_rows = [{text_key: text, label_key: label} for text, label in zip(train_texts, train_labels)]
    val_rows = [{text_key: text, label_key: label} for text, label in zip(val_texts, val_labels)]
    test_rows = [{text_key: text, label_key: label} for text, label in zip(test_texts, test_labels)]
    return train_rows, val_rows, test_rows


class IntentClassifier:
    """四分类意图识别器。"""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or (ARTIFACT_DIR / "intent_model.joblib")
        self.pipeline: Pipeline | None = None

    @classmethod
    def load_or_none(cls, model_path: Path | None = None) -> "IntentClassifier | None":
        path = model_path or (ARTIFACT_DIR / "intent_model.joblib")
        if not path.exists():
            return None
        instance = cls(path)
        payload = joblib.load(path)
        instance.pipeline = payload["pipeline"]
        return instance

    def train(self, dataset_path: Path) -> Dict:
        records = _read_jsonl(dataset_path)
        if not records:
            raise FileNotFoundError(f"未找到意图数据集：{dataset_path}")

        train_rows, val_rows, test_rows = _split_records(records, label_key="intent_label", text_key="user_input")
        fit_rows = train_rows + val_rows if val_rows else train_rows

        self.pipeline = _build_text_pipeline()
        self.pipeline.fit([row["user_input"] for row in fit_rows], [row["intent_label"] for row in fit_rows])

        metrics = self.evaluate(test_rows or val_rows or fit_rows)
        self._save({"pipeline": self.pipeline, "metrics": metrics, "kind": "intent"})
        return metrics

    def evaluate(self, rows: Sequence[Dict]) -> Dict:
        if not rows:
            return {"accuracy": 0.0, "macro_f1": 0.0}
        if self.pipeline is None:
            raise RuntimeError("意图模型尚未训练或加载")

        y_true = [row["intent_label"] for row in rows]
        y_pred = self.pipeline.predict([row["user_input"] for row in rows])
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        }

    def predict(self, text: str) -> LabelPrediction:
        if self.pipeline is None:
            raise RuntimeError("意图模型尚未训练或加载")
        probabilities = self.pipeline.predict_proba([text])[0]
        class_index = int(probabilities.argmax())
        return LabelPrediction(label=str(self.pipeline.classes_[class_index]), confidence=float(probabilities[class_index]))

    def _save(self, payload: Dict) -> None:
        _ensure_artifact_dir()
        joblib.dump(payload, self.model_path)


class CoherenceClassifier:
    """二分类连贯性判别器。"""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or (ARTIFACT_DIR / "coherence_model.joblib")
        self.pipeline: Pipeline | None = None

    @classmethod
    def load_or_none(cls, model_path: Path | None = None) -> "CoherenceClassifier | None":
        path = model_path or (ARTIFACT_DIR / "coherence_model.joblib")
        if not path.exists():
            return None
        instance = cls(path)
        payload = joblib.load(path)
        instance.pipeline = payload["pipeline"]
        return instance

    def train(self, dataset_path: Path) -> Dict:
        records = load_jsonl_pairs(dataset_path)
        if not records:
            raise FileNotFoundError(f"未找到连贯性数据集：{dataset_path}")

        train_rows, val_rows, test_rows = _split_records(records, label_key="coherence_label", text_key="pair_text")
        fit_rows = train_rows + val_rows if val_rows else train_rows

        self.pipeline = _build_text_pipeline()
        self.pipeline.fit([row["pair_text"] for row in fit_rows], [row["coherence_label"] for row in fit_rows])

        metrics = self.evaluate(test_rows or val_rows or fit_rows)
        self._save({"pipeline": self.pipeline, "metrics": metrics, "kind": "coherence"})
        return metrics

    def evaluate(self, rows: Sequence[Dict]) -> Dict:
        if not rows:
            return {"accuracy": 0.0, "macro_f1": 0.0}
        if self.pipeline is None:
            raise RuntimeError("连贯性模型尚未训练或加载")

        y_true = [int(row["coherence_label"]) for row in rows]
        y_pred = self.pipeline.predict([row["pair_text"] for row in rows])
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        }

    def predict(self, history_context: str, candidate_reply: str) -> BinaryPrediction:
        if self.pipeline is None:
            raise RuntimeError("连贯性模型尚未训练或加载")
        text = f"{history_context} [SEP] {candidate_reply}"
        probabilities = self.pipeline.predict_proba([text])[0]
        class_index = int(probabilities.argmax())
        return BinaryPrediction(label=int(self.pipeline.classes_[class_index]), probability=float(probabilities[1]))

    def _save(self, payload: Dict) -> None:
        _ensure_artifact_dir()
        joblib.dump(payload, self.model_path)


def load_jsonl_pairs(dataset_path: Path) -> List[Dict]:
    rows = _read_jsonl(dataset_path)
    for row in rows:
        row["pair_text"] = f"{row['history_context']} [SEP] {row['candidate_reply']}"
    return rows
