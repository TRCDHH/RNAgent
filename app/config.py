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

# ---- 跨模态（RNA ↔ ATAC）模型源码目录 ----
# scButterfly：指向包目录本身（内含 train_model.py / model_component.py）
SCBUTTERFLY_DIR = os.environ.get(
    "SCBUTTERFLY_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model", "scButterfly", "scButterfly")),
)
# scCrossTalk：指向仓库根目录（内含 scCrossTalk/ 包与 gene_vocab.json）
SCROSSTALK_DIR = os.environ.get(
    "SCROSSTALK_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model", "scCrossTalk-main")),
)

# ---- 预处理阶段模式（ReAct 决策层）----
# pipeline：现有确定性链路（默认，行为与历史完全一致）
# react   ：始终由 LLM 通过 skill 描述与工具调用自主决策
# auto    ：judge 无 FAIL 时跳过 LLM（快路径），出现 FAIL 才升级为 ReAct
# 未配置 LLM_API_KEY 时，react/auto 一律自动降级为 pipeline（保证不配 key 也能出结果）。
PREPROCESS_MODE = os.environ.get("PREPROCESS_MODE", "pipeline").strip().lower()

# ReAct 预算护栏：任一超限即强制结束决策（收尾仍由确定性代码完成）
PREPROCESS_MAX_STEPS = int(os.environ.get("PREPROCESS_MAX_STEPS", "20"))
PREPROCESS_TIME_BUDGET = int(os.environ.get("PREPROCESS_TIME_BUDGET", "600"))

# 无进展检测：连续 N 轮写操作后 FAIL 集合完全不变 => 判定卡住，强制结束
PREPROCESS_NO_PROGRESS_LIMIT = int(os.environ.get("PREPROCESS_NO_PROGRESS_LIMIT", "3"))

# ReAct 决策用的模型与温度（与报告/助手的 LLM 相互独立，便于单独换更强的模型）
PREPROCESS_MODEL = os.environ.get("PREPROCESS_MODEL", os.environ.get("LLM_MODEL", "deepseek-v4-flash"))
PREPROCESS_TEMPERATURE = float(os.environ.get("PREPROCESS_TEMPERATURE", "0"))

# ---- 训练阶段模式（ReAct 参数决策与失败恢复）----
# default：完全由手册规则决定参数，失败即终止（历史行为，默认值）
# react  ：**首次尝试仍用规则默认值**（不花 LLM 成本），失败后由 LLM 诊断原因、
#          调整参数并重试（有界）。理由是规则表在正常路径上已足够可靠，
#          LLM 的价值集中在"出错了怎么办"——这正是 workflow 与 agent 的分水岭。
TRAINING_MODE = os.environ.get("TRAINING_MODE", "default").strip().lower()

# 训练尝试总次数上限（含首次）。恢复循环绝不会超过这个次数（防止无限烧 GPU）
TRAINING_MAX_ATTEMPTS = int(os.environ.get("TRAINING_MAX_ATTEMPTS", "3"))

# 恢复循环的时长预算（秒），从进入恢复循环开始计时（不含首次训练耗时）
TRAINING_RECOVER_BUDGET = int(os.environ.get("TRAINING_RECOVER_BUDGET", "3600"))

# 恢复决策用的模型与温度（与预处理/报告相互独立）
TRAINING_MODEL = os.environ.get("TRAINING_MODEL", os.environ.get("LLM_MODEL", "deepseek-v4-flash"))
TRAINING_TEMPERATURE = float(os.environ.get("TRAINING_TEMPERATURE", "0"))

# ---- execute_code Docker 沙箱 ----
# 沙箱镜像（首次执行时若本地缺失，会按 docker/sandbox.Dockerfile 自动构建）
SANDBOX_IMAGE = os.environ.get("SANDBOX_IMAGE", "rna-sandbox:latest")
# 沙箱执行超时（秒）：超时强杀容器
SANDBOX_TIMEOUT = int(os.environ.get("SANDBOX_TIMEOUT", "300"))
# 沙箱资源限额（内存 MB / CPU 核数）
SANDBOX_MEM_MB = int(os.environ.get("SANDBOX_MEM_MB", "4096"))
SANDBOX_CPUS = os.environ.get("SANDBOX_CPUS", "2")
