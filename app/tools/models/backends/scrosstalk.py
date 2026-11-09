"""scCrossTalk 模型后端（RNA ↔ ATAC 跨模态翻译，双模态接入口）。

模型定位
--------
scCrossTalk 用「单模型自编码 + 多模型翻译」三阶段在 RNA 与 ATAC 之间对齐表征：
    1) train_rna_model  ：RNA 自编码（重构 + GAN + DCL 对比学习）
    2) train_atac_model ：ATAC 自编码（按染色体切块编码）
    3) train_multi_model：跨模态翻译对齐（R2A / A2R + MMD）

与既有模型的关键差异（全部由 capabilities 声明，流水线不出现模型名）
------------------------------------------------------------------
- needs_second_modality=True：**必须**提供与 RNA 配对的 ATAC（同细胞、同顺序）
- 原模型不产出 obsm：训练后由本后端从 RNA 编码器 / ATAC 编码器抽 128 维 latent，
  分别写入 obsm['latent']（统一评测）与 obsm['latent_atac']（跨模态评测）
- 原仓库 process_data=True 时会连带处理 ATAC（TF-IDF / 按染色体分块），且
  n_genes 与 HVG 后列数的断言被注释掉（crossTalk.py:79）——尺寸不匹配只会在
  前向时报一句无关的矩阵乘错误。因此这里统一传 process_data=False，
  由模态层（modalities.py）预处理好之后再喂给模型。
- 原仓库 num_workers=4 硬编码，Windows spawn 会递归，统一改为 0
"""

import importlib.util
import os
import sys

from app import config
from app.event_emitter import emit
from app.tools.models.base import Capabilities, ModelBackend, ParamSpec
from app.tools.models.registry import register_model

# 与官方默认一致的结构（不暴露给用户）
LATENT_DIM = 128
RNA_ENCODER_HID = 128
RNA_DECODER_HID = 256
TRANSLATOR_HID = 128
ATAC_TOKEN_DIM = 32


def _install_path():
    """把仓库根目录加入 sys.path（`from scCrossTalk.crossTalk import ...` 需要）。"""
    src = config.SCROSSTALK_DIR
    if src and src not in sys.path:
        sys.path.insert(0, src)


def _patch_runtime():
    """运行时补丁（只作用于训练子进程，不改动模型源码）。

    1) num_workers 强制 0：原仓库 `DataLoader(..., num_workers=4)` 硬编码，Windows spawn 会无限重启；
    2) 补回 `np.Inf`/`np.NaN` 等 NumPy 2.0 移除的别名（当前环境为 numpy 2.x），
       避免早停器等辅助代码在运行时 AttributeError。
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader as _TorchDataLoader

    import scCrossTalk.crossTalk as ct

    if not hasattr(np, "Inf"):
        np.Inf = np.inf
    if not hasattr(np, "NaN"):
        np.NaN = np.nan

    if getattr(ct, "__num_workers_patched__", False):
        return

    def _loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        return _TorchDataLoader(*args, **kwargs)

    ct.DataLoader = _loader
    ct.__num_workers_patched__ = True


@register_model
class ScCrossTalkBackend(ModelBackend):
    key = "scrosstalk"
    name = "scCrossTalk"
    description = "RNA 与 ATAC 双向跨模态翻译框架，通过对比学习与对抗训练对齐两种模态。"  # 兜底，实际取 SKILL.md

    capabilities = Capabilities(
        needs_counts_layer=False,
        owns_preprocessing=True,
        supports_batch=False,
        supports_celltype=True,
        evaluates_internally=False,
        embedding_key="latent",
        artifacts=("model/*.pth", "cross_modal_metrics.csv"),
        needs_second_modality=True,
        second_modality="atac",
        second_embedding_key="latent_atac",
        cross_modal_eval=True,
    )

    param_specs = [
        ParamSpec("epochs", "int", default=100, min=1, max=2000,
                  label="单模态预训练轮数",
                  desc="RNA 自编码与 ATAC 自编码各训练这么多轮（官方默认 100）"),
        ParamSpec("translator_epochs", "int", default=200, min=1, max=2000,
                  label="跨模态翻译轮数",
                  desc="train_multi_model 的轮数（官方默认 200）"),
        ParamSpec("batch_size", "int", default=None, auto=True, min=2,
                  label="批大小", desc="留空=按显存与数据量自动决定"),
        ParamSpec("n_genes", "int", default=3000, min=200,
                  label="高变基因数", desc="RNA 侧 HVG 筛选后作为模型输入维度"),
        ParamSpec("patience", "int", default=50, min=5,
                  label="早停耐心值", desc="验证集损失连续多少轮不下降则停止该阶段"),
        ParamSpec("atac_fpeaks", "float", default=0.005, min=0.0, max=1.0, advanced=True,
                  label="ATAC 峰过滤比例",
                  desc="过滤在少于该比例细胞中出现的峰（官方默认 0.005）"),
    ]

    # ---------------- 依赖探测 ----------------
    def is_available(self) -> tuple:
        if not os.path.isdir(config.SCROSSTALK_DIR):
            return False, f"scCrossTalk 源码目录不存在：{config.SCROSSTALK_DIR}"
        if not os.path.isdir(os.path.join(config.SCROSSTALK_DIR, "scCrossTalk")):
            return False, f"目录 {config.SCROSSTALK_DIR} 下缺少 scCrossTalk/ 包"
        try:
            if importlib.util.find_spec("torch") is None:
                return False, "未安装 torch"
            if importlib.util.find_spec("linformer") is None:
                return False, "未安装 linformer（pip install linformer）"
        except Exception as e:
            return False, f"依赖探测失败：{e}"
        return True, f"源码目录 {config.SCROSSTALK_DIR}"

    # ---------------- 预处理阶段 ----------------
    def judge_extra(self, adata) -> list:
        verdicts = [{
            "item": "scrosstalk_input",
            "status": "PASS",
            "evidence": "RNA 侧将做 normalize_total + log1p + HVG，ATAC 侧做二值化 + TF-IDF",
            "action": "无需处理",
        }]
        if "cell_type" not in adata.obs.columns:
            verdicts.append({
                "item": "cross_modal_cell_type", "status": "WARNING",
                "evidence": "缺少 cell_type 列",
                "action": "跨模态一致性指标需要 cell_type 标签，缺失时只输出 Modality_Align",
            })
        return verdicts

    def impact_sentences(self, verdicts: list, result: dict) -> list:
        return [
            "本模型为 RNA↔ATAC 跨模态模型：主嵌入取 RNA 编码器输出（统一评测口径），"
            "ATAC 编码器输出用于跨模态一致性指标。",
            "训练分三段：RNA 自编码 → ATAC 自编码 → 跨模态翻译对齐（R2A/A2R）；"
            "跨模态指标（ATAC_ARI / Cross_KNN_Acc / Modality_Align）见报告「跨模态一致性」一节。",
        ]

    # ---------------- 训练阶段 ----------------
    def train_cross_modal(self, adata, second, output_dir: str, cfg: dict) -> dict:
        import numpy as np
        import torch
        from scipy import sparse

        from app.tools.models.modalities import prepare_rna

        if second is None:
            raise ValueError("scCrossTalk 需要配对的 ATAC 数据（第二模态）")
        if not os.path.isdir(config.SCROSSTALK_DIR):
            raise FileNotFoundError(f"scCrossTalk 源码目录不存在：{config.SCROSSTALK_DIR}")

        _install_path()
        try:
            from scCrossTalk.crossTalk import scCrossTalkModel  # noqa: E402
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"scCrossTalk 依赖缺失（{e}）。请先执行 pip install linformer") from e

        _patch_runtime()

        emit("training_log", {"message": "RNA 预处理：normalize_total + log1p + 高变基因筛选"})
        rna = prepare_rna(adata, cfg, self.name)
        atac = second

        n_genes, n_peaks = int(rna.n_vars), int(atac.n_vars)
        if n_genes < 50 or n_peaks < 50:
            raise ValueError(f"输入维度过小：RNA {n_genes} 基因 / ATAC {n_peaks} 峰")

        # 原模型按 batch 逐条 dense 化（SingleDataset），不会整表 toarray，
        # 但仍给出规模提示，便于排查 OOM
        emit("training_log", {
            "message": f"输入规模：{rna.n_obs} 细胞 × {n_genes} 基因；ATAC {n_peaks} 峰"})

        emit("training_log", {"message": "初始化 scCrossTalk（RNA / ATAC 双编码器）"})
        model = scCrossTalkModel(
            RNAData=rna,
            ATACData=atac,
            output_path=output_dir,
            n_genes=n_genes,
            lantent_dim=LATENT_DIM,
            rna_encoder_hid_dim=RNA_ENCODER_HID,
            rna_decoder_hid_dim=RNA_DECODER_HID,
            atac_token_dim=ATAC_TOKEN_DIM,
            translator_hid_dim=TRANSLATOR_HID,
            process_data=False,       # 预处理已由 modalities 完成
            save_processed_data=False,
            use_hvg=False,            # 同上：HVG 已在 prepare_rna 里做
        )

        epochs = int(cfg.get("epochs") or 100)
        trans_epochs = int(cfg.get("translator_epochs") or 200)
        batch_size = int(cfg.get("batch_size") or 64)
        patience = int(cfg.get("patience") or 50)

        emit("training_log", {"message": f"[1/3] RNA 自编码预训练：{epochs} 轮"})
        model.train_rna_model(epochs=epochs, batch_size=batch_size, patience=patience)

        emit("training_log", {"message": f"[2/3] ATAC 自编码预训练：{epochs} 轮"})
        model.train_atac_model(epochs=epochs, batch_size=batch_size, patience=patience)

        emit("training_log", {"message": f"[3/3] 跨模态翻译训练：{trans_epochs} 轮"})
        model.train_multi_model(epochs=trans_epochs, batch_size=batch_size, patience=patience)

        # 抽取嵌入（模型本身不写 obsm）
        emit("training_log", {"message": "训练完成，抽取 RNA / ATAC 编码器嵌入"})
        model.set_eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _encode(module, mat):
            X = mat.toarray() if sparse.issparse(mat) else np.asarray(mat)
            outs = []
            with torch.no_grad():
                for i in range(0, X.shape[0], 256):
                    xb = torch.from_numpy(np.ascontiguousarray(X[i:i + 256])).float().to(device)
                    outs.append(module(xb).detach().cpu().numpy())
            return np.vstack(outs) if outs else np.zeros((0, LATENT_DIM), dtype=np.float32)

        rna_lat = _encode(model.rna_encoder, rna.X)
        atac_lat = _encode(model.atac_encoder, atac.X)
        adata.obsm[self.capabilities.embedding_key] = rna_lat
        second.obsm[self.capabilities.second_embedding_key] = atac_lat
        emit("training_log", {
            "message": f"嵌入完成：RNA {rna_lat.shape} / ATAC {atac_lat.shape}"})

        return {
            "epochs": epochs,
            "translator_epochs": trans_epochs,
            "model_dir": os.path.join(output_dir, "model"),
            "processed_dir": os.path.join(output_dir, "processed_data"),
            "n_genes": n_genes,
            "n_peaks": n_peaks,
        }
