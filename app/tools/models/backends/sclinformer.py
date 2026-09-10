"""scLinformer 模型后端。

特点：
- 模型内部完成 normalize_total + log1p + HVG（owns_preprocessing=True，流水线不做预处理）；
- test_model() 自带 evaluate_sc_embedding（evaluates_internally=True，训练阶段不再重复评测）；
- use_batch / use_cell_type 由 obs 是否含对应列自动决定（缺失列会让模型崩溃，故必须关）。
"""

import os
import sys

from app import config
from app.event_emitter import emit
from app.tools.models.base import Capabilities, ModelBackend, ParamSpec
from app.tools.models.registry import register_model


@register_model
class ScLinformerBackend(ModelBackend):
    key = "sclinformer"
    name = "scLinformer"
    description = "Transformer 自编码器，内部自带归一化与高变基因筛选。"  # 兜底，实际取 SKILL.md

    capabilities = Capabilities(
        needs_counts_layer=False,
        owns_preprocessing=True,
        supports_batch=True,
        supports_celltype=True,
        evaluates_internally=True,   # test_model() 内部已调用 evaluate_sc_embedding
        embedding_key="X_emb",
        artifacts=("model/*.pth", "processed_data/*.npy"),
    )

    param_specs = [
        ParamSpec("epochs", "int", default=None, auto=True,
                  label="训练轮数", desc="默认取 SKILL.md 声明值；可用 SCLINFORMER_EPOCHS 覆盖"),
        ParamSpec("batch_size", "int", default=None, auto=True,
                  label="批大小", desc="留空=按显存与数据量自动决定"),
        ParamSpec("use_batch", "bool", default=None, auto=True,
                  label="使用 batch", desc="留空=obs 含 batch 列时自动开启"),
        ParamSpec("use_cell_type", "bool", default=None, auto=True,
                  label="使用 cell_type", desc="留空=obs 含 cell_type 列时自动开启"),
        ParamSpec("process_data", "bool", default=True, advanced=True,
                  label="模型内部预处理", desc="关闭则直接使用输入矩阵"),
        ParamSpec("use_hvg", "bool", default=True, advanced=True,
                  label="使用高变基因", desc="模型内部做 HVG 筛选"),
        ParamSpec("n_genes", "int", default=2000, min=100, advanced=True,
                  label="高变基因数", desc="use_hvg=True 时生效"),
        ParamSpec("use_universal_model", "bool", default=False, advanced=True,
                  label="通用基因词表对齐", desc="跨数据集基因 ID 对齐，默认关闭"),
    ]

    # ---------------- 依赖探测 ----------------
    def is_available(self) -> tuple:
        if not os.path.isdir(config.SCLINFORMER_DIR):
            return False, f"scLinformer 源码目录不存在：{config.SCLINFORMER_DIR}"
        return True, f"源码目录 {config.SCLINFORMER_DIR}"

    # ---------------- 参数 ----------------
    def resolve_config(self, obs_columns, n_cells: int, env: dict, overrides: dict = None) -> dict:
        """默认值全部来自本模型 SKILL.md，覆盖用 SCLINFORMER_* 环境变量或单次 params。"""
        cfg = super().resolve_config(obs_columns, n_cells, env, overrides)
        cfg["use_universal_model"] = bool(cfg.get("use_universal_model"))
        return cfg

    # ---------------- 预处理阶段 ----------------
    def judge_extra(self, adata) -> list:
        """scLinformer 专属判定项：universal 基因覆盖率。"""
        from app.tools.preprocess_tools import _judge_universal_coverage  # 惰性导入避免循环

        return [_judge_universal_coverage(adata)]

    def impact_sentences(self, verdicts: list, result: dict) -> list:
        """「对模型配置的影响」句式（见 skills/registry/models/sclinformer/SKILL.md）。"""
        by_item = {v["item"]: v for v in (verdicts or [])}
        sentences = []

        batch = by_item.get("batch", {})
        if batch.get("status") == "PASS":
            n = _category_count(result, "batch")
            sentences.append(
                f"检测到 batch（{n} 类），use_batch=True，RNADecoder 按 {n} 类批次条件化，"
                f"启用 batch 评测（Batch_ASW / Graph_Connectivity）。"
            )
        else:
            sentences.append("未检测到 batch，use_batch=False，RNADecoder 无批次条件化，batch 评测项为 nan。")

        ct = by_item.get("cell_type", {})
        if ct.get("status") == "PASS":
            n = _category_count(result, "cell_type")
            sentences.append(f"检测到 cell_type（{n} 类），use_cell_type=True，启用判别头与聚类评测（ARI/ASW）。")
        else:
            sentences.append("未检测到 cell_type，use_cell_type=False，跳过判别与 ARI/ASW 评测。")

        uni = by_item.get("universal_coverage", {})
        if uni.get("status") == "PASS":
            sentences.append("universal 基因覆盖率达到阈值，可启用 use_universal_model=True 进行跨数据集对齐。")
        elif uni.get("status") == "WARNING":
            sentences.append("universal 基因覆盖率偏低，建议使用 use_universal_model=False，或确认基因命名与模型词表一致。")

        return sentences

    # ---------------- 训练 ----------------
    def train(self, adata, output_dir: str, cfg: dict) -> dict:
        if not os.path.isdir(config.SCLINFORMER_DIR):
            raise FileNotFoundError(f"scLinformer 源码目录不存在：{config.SCLINFORMER_DIR}")
        if config.SCLINFORMER_DIR not in sys.path:
            sys.path.insert(0, config.SCLINFORMER_DIR)

        from scLinformer.scLinformerModel import Model  # noqa: E402 惰性导入

        model = Model(
            RNAData=adata,
            output_path=output_dir,
            process_data=cfg["process_data"],
            use_universal_model=cfg["use_universal_model"],
            use_cell_type=cfg["use_cell_type"],
            use_batch=cfg["use_batch"],
        )

        emit("training_log", {
            "message": f"开始训练：epochs={cfg['epochs']}, batch_size={cfg['batch_size']}"
        })
        model.train_model(epochs=cfg["epochs"], batch_size=cfg["batch_size"])

        emit("training_log", {"message": "训练完成，开始测试并计算嵌入评测指标"})
        # use_test=True：内部执行 evaluate_sc_embedding（embedding_key="X_emb"），写 CSV + UMAP
        model.test_model(batch_size=cfg["batch_size"])

        return {
            "epochs": cfg["epochs"],
            "model_dir": os.path.join(output_dir, "model"),
            "processed_dir": os.path.join(output_dir, "processed_data"),
        }


def _category_count(result: dict, col: str):
    comp = (result or {}).get("composition", {}).get(col) or {}
    return comp.get("n_categories", "?")
