"""
配置文件
存储 API 密钥、模型参数、游戏设置。

说明：
1. 通过 python-dotenv 从 .env 加载环境变量。
2. 所有关键参数都可通过环境变量覆盖，便于本地开发和部署。
"""

import os
from pathlib import Path
from dotenv import load_dotenv


def _env_to_bool(name: str, default: str = "1") -> bool:
	"""将环境变量解析为布尔值。"""
	value = os.environ.get(name, default).strip().lower()
	return value in {"1", "true", "yes", "on"}

# 按顺序加载环境变量：.env -> .env.local（后者覆盖前者）
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
load_dotenv(dotenv_path=BASE_DIR / ".env.local", override=True)

# =========================
# DeepSeek API 配置
# =========================
# 建议在 .env 中配置：DEEPSEEK_API_KEY=你的真实密钥
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "请在这里填入你的API密钥")

# DeepSeek OpenAI-Compatible 接口基础地址
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 默认模型：deepseek-chat
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# =========================
# 游戏与生成参数配置
# =========================
# 保留最近多少轮“用户+助手”历史，用于控制上下文长度。
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", 5))

# 单次生成最大 token 数。
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 500))

# 温度参数：数值越大越有创造性，但也更发散。
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.8))

# 请求超时时间（秒）
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 60))

# =========================
# 实验与对照开关
# =========================
# 是否启用“意图识别注入提示词”
ENABLE_INTENT_PROMPT = _env_to_bool("ENABLE_INTENT_PROMPT", "1")

# 是否启用“案件状态记忆注入提示词”
ENABLE_CASE_MEMORY = _env_to_bool("ENABLE_CASE_MEMORY", "1")

# 当前实验标签，用于日志分组对比
EXPERIMENT_TAG = os.environ.get("EXPERIMENT_TAG", "full")
