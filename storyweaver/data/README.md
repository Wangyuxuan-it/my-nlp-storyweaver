# Data Preparation Guide

该目录用于满足课程项目中的 Data preparation 要求。

## 目录结构

```text
data/
├── raw/
│   ├── intent_annotations.csv
│   └── coherence_annotations.csv
└── processed/
```

## 1) 意图标注数据（intent_annotations.csv）

字段：

- sample_id: 唯一样本编号
- user_input: 玩家输入文本
- intent_label: 取值为 investigate / interrogate / deduce / act
- source: 数据来源说明（manual / script / dialogue_corpus 等）
- split: 可选，train / val / test；留空则由脚本自动划分

建议样本规模：

- 演示版最低：120 条（每类至少 30）
- 报告版建议：200-400 条

## 2) 连贯性标注数据（coherence_annotations.csv）

字段：

- pair_id: 唯一对编号
- history_context: 历史剧情片段（1-3 回合摘要）
- candidate_reply: 候选回复
- coherence_label: 1=连贯，0=不连贯
- source: 数据来源
- split: 可选，train / val / test；留空则由脚本自动划分

建议样本规模：

- 演示版最低：80 对
- 报告版建议：150-300 对

## 3) 预处理与切分

在 storyweaver 目录执行：

```bash
python prepare_dataset.py \
  --intent-csv data/raw/intent_annotations.csv \
  --coherence-csv data/raw/coherence_annotations.csv \
  --out-dir data/processed
```

输出：

- intent_dataset.jsonl
- intent_train.jsonl / intent_val.jsonl / intent_test.jsonl
- coherence_dataset.jsonl
- coherence_train.jsonl / coherence_val.jsonl / coherence_test.jsonl
- dataset_summary.json

可在报告中引用 dataset_summary.json 的样本统计作为 Data preparation 证据。
