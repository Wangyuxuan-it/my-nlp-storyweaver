"""
StoryWeaver 数据预处理脚本。

功能：
1) 读取意图标注与连贯性标注 CSV。
2) 基础清洗与字段校验。
3) 自动或按指定 split 划分 train/val/test。
4) 导出 JSONL 与统计摘要。

用法：
python prepare_dataset.py \
  --intent-csv data/raw/intent_annotations.csv \
  --coherence-csv data/raw/coherence_annotations.csv \
  --out-dir data/processed
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ALLOWED_INTENTS = {"investigate", "interrogate", "deduce", "act"}
ALLOWED_SPLITS = {"train", "val", "test", ""}


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _assign_split_auto(count: int, seed: int) -> List[str]:
    idx = list(range(count))
    rnd = random.Random(seed)
    rnd.shuffle(idx)

    if count <= 2:
        n_train, n_val = count, 0
    elif count <= 5:
        n_train, n_val = count - 1, 0
    else:
        n_train = int(count * 0.8)
        n_val = int(count * 0.1)
        if n_val == 0:
            n_val = 1
        if count - n_train - n_val == 0:
            n_train -= 1

    # 其余给 test
    split_map = ["test"] * count
    for i in idx[:n_train]:
        split_map[i] = "train"
    for i in idx[n_train : n_train + n_val]:
        split_map[i] = "val"
    return split_map


def _assign_split_for_group_sizes(sizes: List[int], seed: int) -> List[str]:
    idx = list(range(len(sizes)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)

    total = sum(sizes)
    n_groups = len(sizes)

    # 若模板簇数量足够，保证 train/val/test 三个集合都非空。
    if n_groups >= 3:
        test_group_idx = idx[-1]
        val_group_idx = idx[-2]
        split_map = ["train"] * n_groups
        split_map[test_group_idx] = "test"
        split_map[val_group_idx] = "val"
        return split_map

    target_train = int(total * 0.8)
    target_val = int(total * 0.1)
    if total > 0 and target_val == 0:
        target_val = 1

    split_map = ["test"] * len(sizes)
    train_acc = 0
    val_acc = 0
    for i in idx:
        size = sizes[i]
        if train_acc < target_train:
            split_map[i] = "train"
            train_acc += size
        elif val_acc < target_val:
            split_map[i] = "val"
            val_acc += size
        else:
            split_map[i] = "test"

    return split_map


def _stratified_auto_split(rows: List[Dict], label_key: str, seed: int) -> None:
    if any((row.get("template_id") or "").strip() for row in rows):
        _group_aware_split_by_label(rows=rows, label_key=label_key, seed=seed)
        return

    grouped: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row[label_key]].append(i)

    # 每类样本过少时，按全局随机划分，避免某些 split 为空。
    min_group = min((len(v) for v in grouped.values()), default=0)
    if min_group < 3:
        indices = [i for i, row in enumerate(rows) if not rows[i].get("split")]
        split_map = _assign_split_auto(len(indices), seed=seed)
        for local_idx, global_idx in enumerate(indices):
            rows[global_idx]["split"] = split_map[local_idx]
        return

    for label, indices in grouped.items():
        split_map = _assign_split_auto(len(indices), seed=seed + hash(label) % 1000)
        for local_idx, global_idx in enumerate(indices):
            if not rows[global_idx].get("split"):
                rows[global_idx]["split"] = split_map[local_idx]


def _group_aware_split_by_label(rows: List[Dict], label_key: str, seed: int) -> None:
    grouped: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        label = row[label_key]
        template_id = (row.get("template_id") or "").strip() or f"_singleton_{i}"
        grouped[label][template_id].append(i)

    for label, template_groups in grouped.items():
        template_ids = list(template_groups.keys())
        sizes = [len(template_groups[tid]) for tid in template_ids]
        split_map = _assign_split_for_group_sizes(sizes=sizes, seed=seed + hash(label) % 1000)

        for local_idx, template_id in enumerate(template_ids):
            split_name = split_map[local_idx]
            for row_idx in template_groups[template_id]:
                if not rows[row_idx].get("split"):
                    rows[row_idx]["split"] = split_name


def _read_intent_rows(path: Path) -> List[Dict]:
    required = {"sample_id", "user_input", "intent_label", "source", "split"}
    rows: List[Dict] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "intent_annotations.csv 必须包含字段: "
                "sample_id,user_input,intent_label,source,split"
            )
        for raw in reader:
            row = {
                "sample_id": _normalize_text(raw["sample_id"]),
                "template_id": _normalize_text(raw.get("template_id", "")),
                "user_input": _normalize_text(raw["user_input"]),
                "intent_label": _normalize_text(raw["intent_label"]).lower(),
                "source": _normalize_text(raw["source"]),
                "split": _normalize_text(raw["split"]).lower(),
            }
            if row["intent_label"] not in ALLOWED_INTENTS:
                raise ValueError(f"非法 intent_label: {row['intent_label']}")
            if row["split"] not in ALLOWED_SPLITS:
                raise ValueError(f"非法 split: {row['split']}")
            if not row["sample_id"] or not row["user_input"]:
                raise ValueError("sample_id 和 user_input 不能为空")
            rows.append(row)

    return rows


def _read_coherence_rows(path: Path) -> List[Dict]:
    required = {"pair_id", "history_context", "candidate_reply", "coherence_label", "source", "split"}
    rows: List[Dict] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "coherence_annotations.csv 必须包含字段: "
                "pair_id,history_context,candidate_reply,coherence_label,source,split"
            )
        for raw in reader:
            label_text = _normalize_text(raw["coherence_label"])
            if label_text not in {"0", "1"}:
                raise ValueError(f"非法 coherence_label: {label_text}")

            row = {
                "pair_id": _normalize_text(raw["pair_id"]),
                "template_id": _normalize_text(raw.get("template_id", "")),
                "history_context": _normalize_text(raw["history_context"]),
                "candidate_reply": _normalize_text(raw["candidate_reply"]),
                "coherence_label": int(label_text),
                "source": _normalize_text(raw["source"]),
                "split": _normalize_text(raw["split"]).lower(),
            }
            if row["split"] not in ALLOWED_SPLITS:
                raise ValueError(f"非法 split: {row['split']}")
            if not row["pair_id"] or not row["history_context"] or not row["candidate_reply"]:
                raise ValueError("pair_id/history_context/candidate_reply 不能为空")
            rows.append(row)

    return rows


def _export_split_files(rows: Sequence[Dict], prefix: str, out_dir: Path) -> Dict[str, int]:
    split_groups: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = row["split"]
        if split not in split_groups:
            continue
        split_groups[split].append(row)

    for split_name, split_rows in split_groups.items():
        _write_jsonl(out_dir / f"{prefix}_{split_name}.jsonl", split_rows)

    return {k: len(v) for k, v in split_groups.items()}


def prepare_dataset(intent_csv: Path, coherence_csv: Path, out_dir: Path, seed: int) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    intent_rows = _read_intent_rows(intent_csv)
    coherence_rows = _read_coherence_rows(coherence_csv)

    _stratified_auto_split(intent_rows, label_key="intent_label", seed=seed)
    _stratified_auto_split(coherence_rows, label_key="coherence_label", seed=seed)

    _write_jsonl(out_dir / "intent_dataset.jsonl", intent_rows)
    _write_jsonl(out_dir / "coherence_dataset.jsonl", coherence_rows)

    intent_split_stats = _export_split_files(intent_rows, prefix="intent", out_dir=out_dir)
    coherence_split_stats = _export_split_files(coherence_rows, prefix="coherence", out_dir=out_dir)

    summary = {
        "intent_total": len(intent_rows),
        "coherence_total": len(coherence_rows),
        "intent_label_distribution": dict(Counter(row["intent_label"] for row in intent_rows)),
        "coherence_label_distribution": dict(Counter(row["coherence_label"] for row in coherence_rows)),
        "intent_split_distribution": intent_split_stats,
        "coherence_split_distribution": coherence_split_stats,
        "template_cluster_split_enabled": any((row.get("template_id") or "").strip() for row in intent_rows + coherence_rows),
        "seed": seed,
    }

    with (out_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare StoryWeaver labeled datasets")
    parser.add_argument("--intent-csv", default="data/raw/intent_annotations.csv")
    parser.add_argument("--coherence-csv", default="data/raw/coherence_annotations.csv")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = prepare_dataset(
        intent_csv=Path(args.intent_csv),
        coherence_csv=Path(args.coherence_csv),
        out_dir=Path(args.out_dir),
        seed=args.seed,
    )

    print("=== Dataset Preparation Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
