"""模型后端抽象：能力声明（Capabilities）+ 参数 Schema（ParamSpec）+ 生命周期钩子。

设计要点（扩展性）：
- pipeline / 前端 / API **永不出现具体模型名**，只根据 `Capabilities` 能力位决定做什么，
  按 `ParamSpec` 自动渲染参数表单、校验入参、展示报告；
- 参数默认值与决策表来自 `app/skills/registry/models/<key>/SKILL.md`（机器可读），
  代码只声明参数的类型/范围/文案，二者由 `registry.validate()` 做一致性自检；
- 新增模型 = 新增 `backends/<key>.py`（@register_model）+ 一份 SKILL.md，其余零改动。

生命周期：judge_extra -> prepare_data（预处理阶段） -> resolve_config -> train -> evaluate（训练阶段）。
"""

import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from app import config
from app.event_emitter import emit
from app.skills.library import get_library
from app.tools.common import json_safe, read_summary_metrics

SUMMARY_CSV = "summary_metrics.csv"


# ---------------------------------------------------------------------------
# 参数 Schema：一处声明，供前端表单 / API 校验 / 环境变量覆盖 / 报告展示四处复用
# ---------------------------------------------------------------------------
@dataclass
class ParamSpec:
    name: str
    type: str = "str"            # int / float / bool / enum / str
    default: Any = None
    label: str = ""              # 中文展示名
    desc: str = ""               # 表单与报告里的说明
    min: Any = None
    max: Any = None
    choices: list = None         # enum 可选取值
    auto: bool = False           # True：留空时由环境自动决策
    advanced: bool = False       # True：前端折叠为「高级参数」

    def coerce(self, raw: Any) -> Any:
        """按类型强转并做范围/枚举校验（非法值抛 ValueError）。"""
        if raw is None:
            return None
        if self.type == "int":
            v = int(raw)
        elif self.type == "float":
            v = float(raw)
        elif self.type == "bool":
            if isinstance(raw, str):
                s = raw.strip().lower()
                if s in ("1", "true", "yes", "y", "on"):
                    v = True
                elif s in ("0", "false", "no", "n", "off"):
                    v = False
                else:
                    # "auto" 等非布尔字面量：抛错让调用方保留 None（交给环境自动决策），
                    # 切勿静默当成 False，否则 use_batch / use_cell_type 会被永久关闭
                    raise ValueError(f"{self.name} 需要布尔值，收到 {raw!r}")
            else:
                v = bool(raw)
        elif self.type == "enum":
            v = str(raw)
            if self.choices and v not in self.choices:
                raise ValueError(f"{self.name} 取值必须是 {self.choices} 之一，收到 {v!r}")
        else:
            v = str(raw)
        if self.min is not None and v < self.min:
            raise ValueError(f"{self.name} 不能小于 {self.min}")
        if self.max is not None and v > self.max:
            raise ValueError(f"{self.name} 不能大于 {self.max}")
        return v

    def to_dict(self, default: Any = None, include_schema: bool = True) -> dict:
        d = {
            "name": self.name,
            "type": self.type,
            "default": self.default if default is None else default,
            "label": self.label or self.name,
            "desc": self.desc,
            "auto": self.auto,
            "advanced": self.advanced,
        }
        if include_schema:
            d.update({"min": self.min, "max": self.max, "choices": self.choices})
        return d


# ---------------------------------------------------------------------------
# 能力声明：用「能不能 / 需不需要」代替 if model == "xxx"
# ---------------------------------------------------------------------------
@dataclass
class Capabilities:
    needs_counts_layer: bool = False      # 是否需要 layers["counts"]（scVI=True）
    owns_preprocessing: bool = True       # 是否模型内部做归一化/HVG（scVI 为 False，流水线侧做）
    supports_batch: bool = True           # 能否利用 batch 列
    supports_celltype: bool = True        # 能否利用 cell_type 列
    evaluates_internally: bool = False    # 训练是否自带评测（scLinformer.test_model 自带）
    embedding_key: str = "latent"         # 评测读取的 obsm key
    artifacts: tuple = ()                 # 产物说明，供报告展示

    def to_dict(self) -> dict:
        d = asdict(self)
        d["artifacts"] = list(self.artifacts)
        return d


# ---------------------------------------------------------------------------
# 模型后端基类
# ---------------------------------------------------------------------------
class ModelBackend:
    key: str = ""
    name: str = ""
    description: str = ""
    capabilities: Capabilities = Capabilities()
    param_specs: list = []

    # ---------------- 依赖探测 ----------------
    def is_available(self) -> tuple:
        """返回 (是否可用, 说明)。不可用时前端禁用该选项、运行期给出清晰报错。"""
        return True, ""

    # ---------------- 参数 ----------------
    def skill(self):
        return get_library().load_model(self.key)

    def spec(self, name: str):
        for sp in self.param_specs:
            if sp.name == name:
                return sp
        return None

    def default_params(self) -> dict:
        """默认值优先级：环境变量 {KEY}_{PARAM} > SKILL.md 声明 > 代码 ParamSpec。"""
        declared = self.skill().param_defaults() if self.skill() else {}
        out = {}
        for sp in self.param_specs:
            value = sp.default
            if sp.name in declared:
                try:
                    value = sp.coerce(declared[sp.name])
                except Exception:
                    pass
            env_value = os.environ.get(f"{self.key.upper()}_{sp.name.upper()}")
            if env_value:
                try:
                    value = sp.coerce(env_value)
                except Exception:
                    pass
            out[sp.name] = value
        return out

    def merge_overrides(self, overrides: dict) -> dict:
        """按 Schema 过滤 + 强转用户传入参数（未知参数直接丢弃，None/空串视为未设置）。"""
        out = {}
        if not overrides:
            return out
        for sp in self.param_specs:
            if sp.name not in overrides:
                continue
            raw = overrides[sp.name]
            if raw is None or raw == "":
                continue
            out[sp.name] = sp.coerce(raw)
        return out

    def decide_batch_size(self, n_cells: int, env: dict) -> int:
        """batch_size 决策：MODEL_BATCH_SIZE 强制覆盖 > SKILL 查表规则。"""
        if config.MODEL_BATCH_SIZE:
            return max(1, int(config.MODEL_BATCH_SIZE))
        env = env or {}
        return get_library().model_batch_size(
            self.key, int(n_cells), env.get("vram_mb"), bool(env.get("gpu_available"))
        )

    def resolve_config(self, obs_columns, n_cells: int, env: dict, overrides: dict = None) -> dict:
        """合并「默认值 -> 用户覆盖 -> 环境自动决策」，产出最终训练配置。"""
        obs = [str(c) for c in (obs_columns or [])]
        cfg = self.default_params()
        cfg.update(self.merge_overrides(overrides))
        if self.capabilities.supports_batch and self.spec("use_batch") and cfg.get("use_batch") is None:
            cfg["use_batch"] = "batch" in obs
        if self.capabilities.supports_celltype and self.spec("use_cell_type") and cfg.get("use_cell_type") is None:
            cfg["use_cell_type"] = "cell_type" in obs
        if not cfg.get("batch_size"):
            cfg["batch_size"] = self.decide_batch_size(n_cells, env)
        return cfg

    # ---------------- 预处理阶段钩子 ----------------
    def judge_extra(self, adata) -> list:
        """模型专属判定项（追加到通用判定契约之后）。"""
        return []

    def prepare_data(self, adata, params: dict):
        """按模型要求加工数据（能力位驱动，非模型名驱动）。"""
        return adata

    def impact_sentences(self, verdicts: list, result: dict) -> list:
        """「对模型配置的影响」句式（写入 01_数据状况.md）。"""
        return []

    # ---------------- 训练阶段 ----------------
    def train(self, adata, output_dir: str, cfg: dict) -> dict:
        """训练并把嵌入写入 adata.obsm[capabilities.embedding_key]；返回附加信息 dict。"""
        raise NotImplementedError

    def train_and_eval(self, adata, output_dir: str, cfg: dict) -> dict:
        """训练 + 评测（在子进程中执行）。产物格式与模型无关。"""
        emit("training_log", {"message": f"初始化 {self.name} 模型"})
        info = self.train(adata, output_dir, cfg) or {}

        if not self.capabilities.evaluates_internally:
            emit("training_log", {"message": "训练完成，开始计算嵌入评测指标"})
            self.evaluate(adata, output_dir)

        metrics = read_summary_metrics(os.path.join(output_dir, SUMMARY_CSV))
        emit("training_log", {"message": "评测完成", "metrics": metrics})

        out = {
            "status": "success",
            "model": self.key,
            "model_name": self.name,
            "metrics": json_safe(metrics),
        }
        out.update({k: v for k, v in info.items() if v is not None})
        out.setdefault("model_dir", os.path.join(output_dir, "model"))
        return out

    def evaluate(self, adata, output_dir: str):
        """统一评测：复用 scLinformer 的 evaluate_sc_embedding，产出 3 个 CSV + UMAP。

        任何模型只要把嵌入写进 obsm[embedding_key]，下游（报告/前端/AI 助手）就无需改动。
        """
        if config.SCLINFORMER_DIR not in sys.path:
            sys.path.insert(0, config.SCLINFORMER_DIR)
        from scLinformer.utils import evaluate_sc_embedding  # noqa: E402

        evaluate_sc_embedding(
            adata,
            embedding_key=self.capabilities.embedding_key,
            outdir=output_dir,
        )

    # ---------------- 对外描述（API / 前端）----------------
    def to_dict(self, with_schema: bool = True) -> dict:
        ok, reason = self.is_available()
        defaults = self.default_params()
        try:
            defaults = {k: json_safe(v) for k, v in defaults.items()}
        except Exception:
            pass
        d = {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "available": ok,
            "reason": reason,
            "has_skill": self.skill() is not None,
            "capabilities": self.capabilities.to_dict(),
        }
        if with_schema:
            d["params"] = [
                sp.to_dict(default=defaults.get(sp.name)) for sp in self.param_specs
            ]
        return d


# 让 dataclass 的 field 导入保持可用（子模块声明 param_specs 时更方便）
__all__ = ["ParamSpec", "Capabilities", "ModelBackend", "field"]
