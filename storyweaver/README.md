# StoryWeaver - AI-Powered Text Adventure Game

一个基于 DeepSeek API 的现代探案推理文字冒险游戏。玩家扮演私家侦探 **林深**，通过输入行动与推理推进剧情，逐步揭开失踪案真相。

## 1. 项目结构

```text
storyweaver/
├── app.py                # Gradio 主程序
├── story_generator.py    # DeepSeek 调用与剧情生成
├── config.py             # 配置文件
├── requirements.txt      # 依赖列表
└── README.md             # 使用说明
```

## 2. 环境准备

- Python 3.10+
- 可用网络环境
- DeepSeek API Key

## 3. 安装依赖

在 `storyweaver` 目录执行：

```bash
pip install -r requirements.txt
```

## 4. 配置 API Key

新建 `.env` 文件（与 `config.py` 同级），写入：

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
# 以下可选
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
MAX_HISTORY=5
MAX_TOKENS=500
TEMPERATURE=0.8
REQUEST_TIMEOUT=60
```

## 5. 启动项目

```bash
python app.py
```

启动后在浏览器访问 Gradio 地址（默认 http://127.0.0.1:7860 ）即可开始游戏。

## 6. 游戏玩法建议

- 输入“动作”与“推理”结合的指令，例如：
  - 检查案发现场门锁和猫眼痕迹。
  - 调取失踪者昨晚 20:00-23:00 的通话记录。
  - 我怀疑报案人隐瞒了时间线，继续追问不在场证明。
- 记录关键线索，注意前后证词是否矛盾。

## 7. 游戏模式说明

- 自由模式：无回合上限，适合自由探索剧情。
- 叙事模式：最多 20 回合。
- 叙事模式节奏：
  - 第 1-6 回合：铺垫阶段（建立关系与时间线）
  - 第 7-14 回合：调查阶段（扩展证据链与矛盾）
  - 第 15-19 回合：收束阶段（聚焦关键证词与物证）
  - 第 20 回合：真相揭示阶段（输出真相、证据链、结案结论）

超过 20 回合后，系统会提示案件已结案，请重新开始。

## 8. 评估与实验记录（课程要求对应）

项目已内置实验记录与离线统计能力，用于支撑以下指标：

- 交互响应速度：`latency_ms`
- 玩家选择匹配准确度（近似）：`choice_match_score`（1-5）
- 情节连贯性得分：`plot_coherence_score`（1-5）
- 叙事质量得分：`narrative_quality_score`（1-5）
- 沉浸感满意度：`immersion_score`（1-5，来自玩家提交时滑块）

### 8.1 日志记录

启动游戏并进行多轮交互后，会自动生成：

```text
experiments/turn_metrics.jsonl
```

每一行是一条 JSON 记录，包含 `session_id`、`turn`、`mode`、`latency_ms` 与各项评分，便于后续统计与报告复现。

### 8.2 离线汇总

在 `storyweaver` 目录执行：

```bash
python evaluate_runs.py
```

可得到：

- 平均响应时间与标准差
- 平均连贯性得分
- 平均叙事质量得分
- 平均选择匹配得分
- 平均沉浸感得分
- 高匹配率（`choice_match_score >= 4`）

你也可指定日志路径：

```bash
python evaluate_runs.py --input experiments/turn_metrics.jsonl
```

## 9. 核心 NLP 模块补充

为对齐课程要求，项目新增了两个轻量模块：

- 意图识别（`intent_recognizer.py`）：将玩家输入识别为“调查/质询/推理/行动”四类。
- 一致性状态机（`case_state.py`）：维护 `timeline`（时间线）与 `evidence`（证据库），每回合更新并回注到模型提示词。
- 可训练模型（`train_models.py` + `ml_models.py`）：基于 `scikit-learn` 训练意图分类器和连贯性判别器。

集成方式：

- 每回合先识别玩家意图。
- 将“意图标签 + 案件状态摘要”作为系统提示词传给生成器，增强上下文一致性。
- 生成后更新状态机并写入实验日志（含 `intent_label`、`timeline_size`、`evidence_size`）。
- 训练后，app 会优先加载 `artifacts/intent_model.joblib` 和 `artifacts/coherence_model.joblib`。

## 10. 对照实验（Ablation）

项目支持通过环境变量快速切换模块开关，用于方法对比：

- `ENABLE_INTENT_PROMPT`：是否启用“意图识别注入提示词”（默认 `1`）
- `ENABLE_CASE_MEMORY`：是否启用“案件状态记忆注入提示词”（默认 `1`）
- `EXPERIMENT_TAG`：实验标签（写入日志用于分组）

建议至少运行以下三组实验：

```bash
# Full（完整系统）
EXPERIMENT_TAG=full ENABLE_INTENT_PROMPT=1 ENABLE_CASE_MEMORY=1 python app.py

# No-Intent（去掉意图注入）
EXPERIMENT_TAG=no_intent ENABLE_INTENT_PROMPT=0 ENABLE_CASE_MEMORY=1 python app.py

# No-Memory（去掉状态记忆注入）
EXPERIMENT_TAG=no_memory ENABLE_INTENT_PROMPT=1 ENABLE_CASE_MEMORY=0 python app.py
```

实验结束后运行：

```bash
python ablation_report.py
```

脚本会按 `experiment_tag + 开关组合 + mode` 自动分组输出核心指标，避免不同模式混杂比较。

并且显著性检验默认采用 **session 级** 统计（先按 `session_id` 聚合，再做置换检验），用于降低 turn 级样本相关性带来的偏差。

如需只看某一模式（推荐用于课程报告的公平对比）：

```bash
python ablation_report.py --mode 叙事模式
```

## 11. 数据准备（Data Preparation）

为满足课程要求，仓库新增了可复现的数据准备流程：

- 标注模板：`data/raw/intent_annotations.csv`、`data/raw/coherence_annotations.csv`
- 数据说明：`data/README.md`
- 预处理脚本：`prepare_dataset.py`

执行方式：

```bash
python prepare_dataset.py \
  --intent-csv data/raw/intent_annotations.csv \
  --coherence-csv data/raw/coherence_annotations.csv \
  --out-dir data/processed
```

输出文件：

- `data/processed/intent_dataset.jsonl`
- `data/processed/intent_train.jsonl` / `intent_val.jsonl` / `intent_test.jsonl`
- `data/processed/coherence_dataset.jsonl`
- `data/processed/coherence_train.jsonl` / `coherence_val.jsonl` / `coherence_test.jsonl`
- `data/processed/dataset_summary.json`

`dataset_summary.json` 可直接作为报告中 Data preparation 的样本统计证据。

## 12. 模型训练与人工评测

### 12.1 训练模型

在 `storyweaver` 目录执行：

```bash
python train_models.py
```

会生成：

- `artifacts/intent_model.joblib`
- `artifacts/coherence_model.joblib`
- `artifacts/training_summary.json`

### 12.2 人工评测

游戏页面右侧新增“人工评测”折叠区，可填写局末主观评分。提交后会写入：

```text
experiments/human_evaluation.csv
```

也可以在命令行汇总：

```bash
python human_evaluation.py
```

## 13. 常见问题

### Q1: 提示未检测到 API Key
- 检查 `.env` 是否与 `config.py` 同级。
- 检查变量名是否为 `DEEPSEEK_API_KEY`。

### Q2: 接口调用失败
- 检查网络连通性。
- 检查密钥配额是否充足。
- 检查 `DEEPSEEK_BASE_URL` 是否正确。

## 14. 课程项目可扩展方向

- 引入“案件状态机”（嫌疑人关系图、时间线、证据库）。
- 增加“多结局系统”（真相结局、误判结局、未破案结局）。
- 增加“难度模式”（提示强度、线索隐蔽度、时间限制）。
- 增加“存档/读档”与“案件报告自动总结”。
