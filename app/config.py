"""全局配置（环境变量可覆盖）。"""

import os

# DeepSeek（OpenAI 兼容接口）
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")

# ---- 模型运行（与具体模型无关）----
# 默认模型（必须是 app/tools/models/registry.py 中已注册的模型 key）
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "sclinformer")
# 训练轮数（默认 100；演示想快速出结果可设小，如 MODEL_EPOCHS=1）
MODEL_EPOCHS = int(os.environ.get("MODEL_EPOCHS", "100"))
# 是否显式设置过 MODEL_EPOCHS（显式设置时优先于模型 SKILL.md 中声明的默认值）
MODEL_EPOCHS_EXPLICIT = os.environ.get("MODEL_EPOCHS") is not None
# 强制指定 batch_size（None=按显存+数据量自动判断；OOM 时可用它手动调小）
MODEL_BATCH_SIZE = int(os.environ["MODEL_BATCH_SIZE"]) if os.environ.get("MODEL_BATCH_SIZE") else None

# scLinformer 源码根目录（train.py 所在目录，内含 scLinformer/ 包）
SCLINFORMER_DIR = os.environ.get(
    "SCLINFORMER_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model", "scLinformer-main")),
)

# ---- execute_code Docker 沙箱 ----
# 沙箱镜像（首次执行时若本地缺失，会按 docker/sandbox.Dockerfile 自动构建）
SANDBOX_IMAGE = os.environ.get("SANDBOX_IMAGE", "rna-sandbox:latest")
# 沙箱执行超时（秒）：超时强杀容器
SANDBOX_TIMEOUT = int(os.environ.get("SANDBOX_TIMEOUT", "300"))
# 沙箱资源限额（内存 MB / CPU 核数）
SANDBOX_MEM_MB = int(os.environ.get("SANDBOX_MEM_MB", "4096"))
SANDBOX_CPUS = os.environ.get("SANDBOX_CPUS", "2")
