"""
对照实验汇总脚本。

用途：
- 按实验标签与开关组合，对比核心指标。

使用示例：
python ablation_report.py
python ablation_report.py --input experiments/turn_metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

from log_utils import is_successful_record


EXPECTED_CONFIGS = {
    ("full", True, True),
    ("no_intent", False, True),
    ("no_memory", True, False),
}


def _safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_std(values: List[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])

    rnd = random.Random(seed)
    n = len(values)
    means: List[float] = []
    for _ in range(n_boot):
        sample = [values[rnd.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()

    lo_idx = int((alpha / 2) * (n_boot - 1))
    hi_idx = int((1 - alpha / 2) * (n_boot - 1))
    return float(means[lo_idx]), float(means[hi_idx])


def _permutation_pvalue(a: List[float], b: List[float], n_perm: int = 2000, seed: int = 42) -> float:
    if not a or not b:
        return 1.0
    observed = abs(statistics.mean(a) - statistics.mean(b))
    pooled = a + b
    n_a = len(a)
    rnd = random.Random(seed)

    count = 0
    for _ in range(n_perm):
        shuffled = pooled[:]
        rnd.shuffle(shuffled)
        diff = abs(statistics.mean(shuffled[:n_a]) - statistics.mean(shuffled[n_a:]))
        if diff >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def load_records(path: Path) -> List[Dict]:
    if not path.exists():
        return []

    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_group(rows: List[Dict]) -> Dict[str, float]:
    latency = [float(r.get("latency_ms", 0)) for r in rows]
    coherence = [float(r.get("plot_coherence_score", 0)) for r in rows]
    narrative = [float(r.get("narrative_quality_score", 0)) for r in rows]
    choice_match = [float(r.get("choice_match_score", 0)) for r in rows]
    immersion = [float(r.get("immersion_score", 0)) for r in rows if r.get("immersion_score") is not None]

    return {
        "num_turns": len(rows),
        "latency_ms_mean": _safe_mean(latency),
        "latency_ms_std": _safe_std(latency),
        "latency_ms_ci_low": _bootstrap_ci(latency)[0],
        "latency_ms_ci_high": _bootstrap_ci(latency)[1],
        "plot_coherence_mean": _safe_mean(coherence),
        "plot_coherence_ci_low": _bootstrap_ci(coherence)[0],
        "plot_coherence_ci_high": _bootstrap_ci(coherence)[1],
        "narrative_quality_mean": _safe_mean(narrative),
        "narrative_quality_ci_low": _bootstrap_ci(narrative)[0],
        "narrative_quality_ci_high": _bootstrap_ci(narrative)[1],
        "choice_match_mean": _safe_mean(choice_match),
        "choice_match_ci_low": _bootstrap_ci(choice_match)[0],
        "choice_match_ci_high": _bootstrap_ci(choice_match)[1],
        "immersion_mean": _safe_mean(immersion),
        "immersion_ci_low": _bootstrap_ci(immersion)[0],
        "immersion_ci_high": _bootstrap_ci(immersion)[1],
        "high_choice_match_rate": (
            sum(1 for x in choice_match if x >= 4) / len(choice_match) if choice_match else 0.0
        ),
    }


def _metric_values(rows: List[Dict], metric: str) -> List[float]:
    if metric == "latency_ms":
        return [float(r.get("latency_ms", 0)) for r in rows]
    if metric == "plot_coherence_score":
        return [float(r.get("plot_coherence_score", 0)) for r in rows]
    if metric == "narrative_quality_score":
        return [float(r.get("narrative_quality_score", 0)) for r in rows]
    if metric == "choice_match_score":
        return [float(r.get("choice_match_score", 0)) for r in rows]
    if metric == "immersion_score":
        return [float(r.get("immersion_score", 0)) for r in rows if r.get("immersion_score") is not None]
    return []


def _session_metric_values(rows: List[Dict], metric: str) -> List[float]:
    """按 session 聚合后返回每个 session 的指标均值。"""
    bucket: Dict[str, List[float]] = {}
    for row in rows:
        session_id = str(row.get("session_id", "")).strip()
        if not session_id:
            continue
        value_list = _metric_values([row], metric)
        if not value_list:
            continue
        bucket.setdefault(session_id, []).append(value_list[0])

    return [statistics.mean(values) for values in bucket.values() if values]


def _find_full_group_keys(groups: Dict[Tuple[str, bool, bool, str], List[Dict]]) -> Dict[str, Tuple[str, bool, bool, str]]:
    """按 mode 找 full baseline 组。"""
    result: Dict[str, Tuple[str, bool, bool, str]] = {}
    for key in groups:
        tag, use_intent, use_memory, mode = key
        if tag == "full" and use_intent and use_memory:
            result[mode] = key
    return result


def key_of(row: Dict) -> Tuple[str, bool, bool, str]:
    tag = str(row.get("experiment_tag", "untagged"))
    use_intent = bool(row.get("used_intent_prompt", False))
    use_memory = bool(row.get("used_case_memory", False))
    mode = str(row.get("mode", "unknown"))
    return tag, use_intent, use_memory, mode


def key_text(key: Tuple[str, bool, bool, str]) -> str:
    tag, use_intent, use_memory, mode = key
    return f"tag={tag} | intent={use_intent} | memory={use_memory} | mode={mode}"


def report_coverage(groups: Dict[Tuple[str, bool, bool, str], List[Dict]]) -> None:
    """检查每个 mode 的消融分组覆盖情况。"""
    modes = sorted({key[3] for key in groups.keys()})
    if not modes:
        print("No mode groups found for coverage check.")
        return

    print("\n=== Ablation Coverage Check ===")
    for mode in modes:
        present = {(tag, use_intent, use_memory) for tag, use_intent, use_memory, m in groups.keys() if m == mode}
        missing = [cfg for cfg in sorted(EXPECTED_CONFIGS) if cfg not in present]
        coverage = (len(EXPECTED_CONFIGS) - len(missing)) / len(EXPECTED_CONFIGS)
        print(f"mode={mode} | coverage={coverage:.2f}")
        if not missing:
            print("  missing: none")
            continue

        for tag, use_intent, use_memory in missing:
            print(f"  missing: tag={tag} | intent={use_intent} | memory={use_memory}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize StoryWeaver ablation results")
    parser.add_argument("--input", default="experiments/turn_metrics.jsonl", help="Path to JSONL log")
    parser.add_argument(
        "--mode",
        default="",
        help="仅统计指定模式（如：自由模式 / 叙事模式）；为空则统计全部。",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="包含失败回合（默认会自动过滤失败记录）。",
    )
    args = parser.parse_args()

    rows = load_records(Path(args.input))
    if not rows:
        print("No records found. Run experiments first.")
        return

    raw_count = len(rows)
    if not args.include_failed:
        rows = [row for row in rows if is_successful_record(row)]
        if not rows:
            print("No successful records found after filtering failed turns.")
            return

    if args.mode:
        rows = [row for row in rows if str(row.get("mode", "")) == args.mode]
        if not rows:
            print(f"No records found for mode={args.mode}.")
            return

    groups: Dict[Tuple[str, bool, bool, str], List[Dict]] = {}
    for row in rows:
        groups.setdefault(key_of(row), []).append(row)

    print("=== StoryWeaver Ablation Report ===")
    print(f"num_turns_raw: {raw_count}")
    print(f"num_turns_used: {len(rows)}")
    for group_key, group_rows in sorted(groups.items(), key=lambda x: key_text(x[0])):
        stats = summarize_group(group_rows)
        session_count = len({str(r.get("session_id", "")).strip() for r in group_rows if str(r.get("session_id", "")).strip()})
        print(f"\n[{key_text(group_key)}]")
        print(f"num_sessions: {session_count}")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    report_coverage(groups)

    full_group_keys = _find_full_group_keys(groups)
    if not full_group_keys:
        print("\nNo full baseline group found; skip significance tests.")
        return

    print("\n=== Session-Level Significance vs Full Baseline (Mode-Matched) ===")
    metrics = [
        "latency_ms",
        "plot_coherence_score",
        "narrative_quality_score",
        "choice_match_score",
        "immersion_score",
    ]

    for group_key, group_rows in sorted(groups.items(), key=lambda x: key_text(x[0])):
        tag, use_intent, use_memory, mode = group_key
        if tag == "full" and use_intent and use_memory:
            continue

        baseline_key = full_group_keys.get(mode)
        if baseline_key is None:
            print(f"\n[{key_text(group_key)}]")
            print("No mode-matched full baseline; skip.")
            continue

        baseline_rows = groups[baseline_key]
        print(f"\n[{key_text(group_key)}] vs [{key_text(baseline_key)}]")
        for metric in metrics:
            current_values = _session_metric_values(group_rows, metric)
            baseline_values = _session_metric_values(baseline_rows, metric)
            if not current_values or not baseline_values:
                print(f"{metric}: insufficient data")
                continue

            diff = _safe_mean(current_values) - _safe_mean(baseline_values)
            p_value = _permutation_pvalue(current_values, baseline_values)
            print(
                f"{metric}: diff={diff:.4f}, p_value={p_value:.4f}, "
                f"n_session_current={len(current_values)}, n_session_baseline={len(baseline_values)}"
            )


if __name__ == "__main__":
    main()
