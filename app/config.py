"""全局配置（环境变量可覆盖）。"""

import os

# DeepSeek（OpenAI 兼容接口）
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")

# ---- scLinformer 模型运行 ----
# 训练轮数（train.py 里演示用 epochs=1；真实训练可调大后重启）
MODEL_EPOCHS = int(os.environ.get("MODEL_EPOCHS", "1"))
# 强制指定 batch_size（None=按显存+数据量自动判断；OOM 时可用它手动调小）
MODEL_BATCH_SIZE = int(os.environ["MODEL_BATCH_SIZE"]) if os.environ.get("MODEL_BATCH_SIZE") else None
