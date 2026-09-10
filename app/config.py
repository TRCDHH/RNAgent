"""全局配置（环境变量可覆盖）。"""

import os

# DeepSeek（OpenAI 兼容接口）
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")

# ---- 模型运行（与具体模型无关）----
# 默认模型（必须是 app/tools/models/registry.py 中已注册的模型 key）
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "sclinformer")

# 模型参数的默认值与覆盖**完全按模型隔离**：
#   1) 默认值：由各模型自己的 SKILL.md 的 params 声明（如 epochs / max_epochs）
#   2) 覆盖：环境变量 {模型KEY}_{参数名}，例如
#        SCLINFORMER_EPOCHS=200      scLinformer 训练轮数
#        SCLINFORMER_BATCH_SIZE=64   scLinformer 批大小
#        SCVI_MAX_EPOCHS=400         scVI 最大训练轮数
#        SCVI_BATCH_SIZE=128         scVI 批大小
#   3) 单次覆盖：API/前端传入的 params
# 这里**不再提供全局的 MODEL_EPOCHS / MODEL_BATCH_SIZE** —— 一个全局变量同时影响
# 多个模型会造成"改了 A 却动了 B"，因此已废弃（见 LEGACY_MODEL_ENV 的启动告警）。
LEGACY_MODEL_ENV = {
    k: os.environ[k] for k in ("MODEL_EPOCHS", "MODEL_BATCH_SIZE") if k in os.environ
}

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
