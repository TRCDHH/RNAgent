"""scVI 模型后端（极简接入版）。

只暴露最常用的几个参数（训练轮数 / 批大小 / batch 与 cell_type 字段名 / 高变基因数 / 设备），
网络结构（n_latent=30, n_hidden=128, n_layers=2, gene_likelihood="nb"）一律用 scvi-tools 推荐默认。

与 scLinformer 的关键差异（由 capabilities 声明，pipeline 不出现模型名）：
- 需要原始计数存入 layers["counts"]（needs_counts_layer=True）
- 流水线侧做高变基因筛选，模型不再做归一化/HVG（owns_preprocessing=False）
  —— scVI 用 NB 似然直接建模计数，不做 log1p，这与 scLinformer 的「模型内部归一化」相反
- 训练后自行写入 obsm["latent"]，由统一评测产出指标（evaluates_internally=False）
"""

import os

from app.event_emitter import emit
from app.tools.models.base import Capabilities, ModelBackend, ParamSpec
from app.tools.models.registry import register_model

# scvi-tools 推荐默认结构（不暴露给用户）
DEFAULT_STRUCTURE = {
    "n_latent": 30,
    "n_hidden": 128,
    "n_layers": 2,
    "gene_likelihood": "nb",
    "dropout_rate": 0.1,
}


@register_model
class ScVIBackend(ModelBackend):
    key = "scvi"
    name = "scVI"
    description = "单细胞变分自编码器：以负二项似然直接建模原始计数，擅长批次校正与低维表征；要求原始计数存入 layers['counts']。"

    capabilities = Capabilities(
        needs_counts_layer=True,
        owns_preprocessing=False,
        supports_batch=True,
        supports_celltype=True,
        evaluates_internally=False,
        embedding_key="latent",
        artifacts=("model/scvi/",),
    )

    param_specs = [
        ParamSpec("max_epochs", "int", default=100, min=1, max=2000,
                  label="最大训练轮数", desc="开启 early stopping，收敛后自动停止"),
        ParamSpec("batch_size", "int", default=None, auto=True,
                  label="批大小", desc="留空=按显存与数据量自动决定"),
        ParamSpec("batch_key", "str", default="batch",
                  label="批次字段名", desc="obs 中的批次列名；不存在则关闭批次校正"),
        ParamSpec("cell_type_key", "str", default="cell_type",
                  label="细胞类型字段名", desc="obs 中的细胞类型列名；用于评测与 UMAP 着色"),
        ParamSpec("n_top_genes", "int", default=2000, min=0,
                  label="高变基因数", desc="0 表示不做高变基因筛选；需小于总基因数"),
        ParamSpec("accelerator", "enum", default="auto", choices=["auto", "cpu", "gpu"], advanced=True,
                  label="计算设备", desc="auto=有 GPU 则用 GPU"),
    ]

    # ---------------- 依赖探测 ----------------
    def is_available(self) -> tuple:
        try:
            import scvi  # noqa: F401 惰性导入
        except Exception as e:
            return False, f"未安装 scvi-tools（pip install scvi-tools>=1.1）：{e}"
        return True, f"scvi-tools {getattr(scvi, '__version__', '?')}"

    # ---------------- 参数 ----------------
    def resolve_config(self, obs_columns, n_cells: int, env: dict, overrides: dict = None) -> dict:
        obs = [str(c) for c in (obs_columns or [])]
        cfg = super().resolve_config(obs_columns, n_cells, env, overrides)
        # 字段名是「用户可能想改」的核心项：记录命中情况，供日志与报告展示
        cfg["has_batch"] = cfg.get("batch_key") in obs
        cfg["has_cell_type"] = cfg.get("cell_type_key") in obs
        return cfg

    # ---------------- 预处理阶段 ----------------
    def judge_extra(self, adata) -> list:
        """scVI 专属判定项：是否已存在原始计数层 layers['counts']。"""
        if "counts" in adata.layers:
            return [{
                "item": "counts_layer", "status": "PASS",
                "evidence": "已存在 layers['counts']，scVI 将直接以 NB 似然建模原始计数",
                "action": "无需处理",
            }]
        return [{
            "item": "counts_layer", "status": "WARNING",
            "evidence": f"缺少 layers['counts']（现有 layers：{list(adata.layers.keys()) or '无'}）",
            "action": "prepare_data 自动将 X 复制为 layers['counts']（要求 X 为原始计数）",
        }]

    def prepare_data(self, adata, params: dict):
        """scVI 的数据加工：counts 层 + 基因名唯一 + 分类 dtype + 高变基因筛选。"""
        import scanpy as sc  # noqa: E402 惰性导入

        params = params or {}
        adata.var_names_make_unique()

        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
            emit("preprocess_log", {"message": "已按 scVI 要求将 X 复制为 layers['counts']（原始计数）"})

        for key in (params.get("batch_key"), params.get("cell_type_key")):
            if key and key in adata.obs.columns:
                adata.obs[key] = adata.obs[key].astype("category")

        n_top = int(params.get("n_top_genes") or 0)
        if 0 < n_top < adata.n_vars:
            try:
                sc.pp.highly_variable_genes(
                    adata, layer="counts", n_top_genes=n_top, flavor="seurat_v3", subset=True
                )
                emit("preprocess_log", {"message": f"已筛选高变基因 {n_top} 个（seurat_v3）"})
            except Exception as e:  # seurat_v3 对数据有要求，失败则回退普通 flavor
                emit("preprocess_log", {"message": f"seurat_v3 高变基因筛选失败（{e}），回退 seurat flavor"})
                try:
                    sc.pp.highly_variable_genes(
                        adata, layer="counts", n_top_genes=n_top, flavor="seurat", subset=True
                    )
                except Exception as e2:
                    emit("preprocess_log", {"message": f"高变基因筛选失败，使用全部基因：{e2}"})
        return adata

    def impact_sentences(self, verdicts: list, result: dict) -> list:
        by_item = {v["item"]: v for v in (verdicts or [])}
        sentences = []

        batch = by_item.get("batch", {})
        if batch.get("status") == "PASS":
            n = _category_count(result, "batch")
            sentences.append(
                f"检测到 batch（{n} 类），scVI 以 batch_key 作为协变量建模，"
                f"启用 batch 评测（Batch_ASW / Graph_Connectivity）。"
            )
        else:
            sentences.append("未检测到 batch，scVI 不做批次校正（batch_key=None），batch 评测项为 nan。")

        ct = by_item.get("cell_type", {})
        if ct.get("status") == "PASS":
            n = _category_count(result, "cell_type")
            sentences.append(f"检测到 cell_type（{n} 类），启用聚类评测（ARI/AMI/NMI/HOM）与 Cell_ASW，并生成 cell_type UMAP。")
        else:
            sentences.append("未检测到 cell_type，跳过聚类评测与 Cell_ASW，仅生成无标注 UMAP。")

        counts = by_item.get("counts_layer", {})
        if counts.get("status") == "PASS":
            sentences.append("原始计数已存入 layers['counts']，scVI 以负二项似然直接建模计数，不做 log1p 归一化。")
        else:
            sentences.append("已自动将 X 复制为 layers['counts']；若 X 非原始计数，scVI 的计数建模会失真，建议提供原始计数。")

        return sentences

    # ---------------- 训练 ----------------
    def train(self, adata, output_dir: str, cfg: dict) -> dict:
        import numpy as np  # noqa: E402 惰性导入
        import scvi  # noqa: E402

        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()

        batch_key = cfg.get("batch_key") if cfg.get("batch_key") in adata.obs.columns else None
        emit("training_log", {"message": f"setup_anndata: layer=counts, batch_key={batch_key}"})
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)

        model = scvi.model.SCVI(adata, **DEFAULT_STRUCTURE)

        accelerator = cfg.get("accelerator") or "auto"
        if accelerator == "auto":
            import torch  # noqa: E402
            accelerator = "gpu" if torch.cuda.is_available() else "cpu"

        emit("training_log", {
            "message": f"开始训练：max_epochs={cfg['max_epochs']}, "
                       f"batch_size={cfg['batch_size']}, accelerator={accelerator}"
        })
        kwargs = {"max_epochs": int(cfg["max_epochs"]),
                  "batch_size": int(cfg["batch_size"]),
                  "accelerator": accelerator,
                  "early_stopping": True}
        if accelerator == "gpu":
            kwargs["devices"] = 1
        model.train(**kwargs)

        latent = np.asarray(model.get_latent_representation())
        adata.obsm[self.capabilities.embedding_key] = latent
        emit("training_log", {"message": f"得到低维表征：{latent.shape[0]} 细胞 × {latent.shape[1]} 维"})

        model_dir = os.path.join(output_dir, "model", "scvi")
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_dir)

        return {
            "epochs": cfg["max_epochs"],
            "model_dir": model_dir,
            "processed_dir": None,
        }


def _category_count(result: dict, col: str):
    comp = (result or {}).get("composition", {}).get(col) or {}
    return comp.get("n_categories", "?")
