"""scButterfly 模型后端（RNA ↔ ATAC 跨模态翻译，双模态接入口）。

模型定位
--------
scButterfly 用**双对齐变分自编码器**在 RNA 与 ATAC 之间做跨模态翻译：
    RNA --RNA_encoder--> latent --translator--> ATAC --ATAC_decoder--> ATAC
    ATAC --ATAC_encoder--> latent --translator--> RNA --RNA_decoder--> RNA
训练分三段（与官方 train.py 一致）：
    1) R2R 预训练   ：RNA 自编码 + KL + 判别器
    2) A2A 预训练   ：ATAC 自编码
    3) 翻译器训练   ：两个 modality 的 latent 对齐（GAN + KL）

与既有模型的关键差异（全部由 capabilities 声明，流水线不出现模型名）
------------------------------------------------------------------
- needs_second_modality=True：**必须**提供与 RNA 配对的 ATAC（同细胞、同顺序）
- 原模型不产出 obsm：训练后由本后端从 RNA_encoder / ATAC_encoder 抽 128 维 latent，
  分别写入 obsm['latent']（统一评测）与 obsm['latent_atac']（跨模态评测）
- 依赖 episcanpy（ATAC 预处理）：Windows 上装不了 pybedtools，因此本后端
  用 app/tools/models/modalities.py 复刻同一套预处理，import 侧用 stub 顶住
- 原仓库 num_workers=4 硬编码，Windows spawn 会递归，统一改为 0
"""

import importlib.util
import os
import sys
import types

from app import config
from app.event_emitter import emit
from app.tools.models.base import Capabilities, ModelBackend, ParamSpec
from app.tools.models.registry import register_model

# 与官方 construct_model 一致的网络骨架（只暴露少量开关，其余走模型默认）
R_ENCODER_HIDDEN = 256
LATENT_DIM = 128
A_TOKEN_DIM = 32


def _episcanpy_stub():
    """伪造 episcanpy：data_processing.py 在模块级 `import episcanpy.api as epi`。

    真正的 episcanpy 依赖 pybedtools（Windows 无法安装）。我们只在 RNA/ATAC
    预处理被调用时才会用到它 —— 而本接入口的预处理由 modalities.py 承担，
    所以这里只需要一个能通过 import 的占位对象。
    """
    if "episcanpy" in sys.modules:
        return
    try:
        if importlib.util.find_spec("episcanpy") is not None:  # 真装了就用真的
            return
    except Exception:
        pass

    epi = types.ModuleType("episcanpy")
    api = types.ModuleType("episcanpy.api")

    class _Placeholder:
        def __getattr__(self, name):
            def _unavailable(*a, **k):
                raise RuntimeError(
                    "episcanpy 未安装（本接入口的 ATAC 预处理由 modalities.prepare_atac 承担，"
                    "不应调用到这里）")
            return _unavailable

    api.pp = _Placeholder()
    api.tl = _Placeholder()
    epi.api = api
    sys.modules["episcanpy"] = epi
    sys.modules["episcanpy.api"] = api


def _import_train_model():
    """导入 scButterfly.train_model.Model。

    手工注册包对象（带 __path__）可跳过 `scButterfly/__init__.py` ——
    它会连锁 import train_model_cite / perturb，进而依赖 adjustText、pot 等
    与基础模型无关的库。
    """
    if "scButterfly" not in sys.modules:
        _episcanpy_stub()
        pkg = types.ModuleType("scButterfly")
        pkg.__path__ = [config.SCBUTTERFLY_DIR]
        sys.modules["scButterfly"] = pkg

    from scButterfly.train_model import Model  # noqa: E402

    return Model


def _patch_runtime():
    """运行时补丁（只作用于训练子进程，不改动模型源码）。

    1) num_workers 强制 0：原仓库 `DataLoader(..., num_workers=4)` 是硬编码的，
       Windows 走 spawn 会在训练子进程里反复重启自身。数据是内存里的 numpy 矩阵，
       单进程加载没有额外开销。
    2) CPU 环境下屏蔽 `Tensor.cuda()`：`forward_discriminator` 里有一处无条件
       `.cuda()`（train_model.py:373），无 GPU 会直接抛
       "Torch not compiled with CUDA enabled"。模型参数本身是按
       `torch.cuda.is_available()` 决定是否上 GPU 的，这里只需让那个张量留在 CPU。
    3) 补回 `np.Inf`：model_utlis.py:117 用了该别名，NumPy 2.0 已移除
       （环境为 numpy 2.x，不补会直接 AttributeError 崩在早停器构造处）。
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader as _TorchDataLoader

    import scButterfly.train_model as bt

    if not hasattr(np, "Inf"):
        np.Inf = np.inf

    if getattr(bt, "__runtime_patched__", False):
        return

    def _loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["pin_memory"] = False
        return _TorchDataLoader(*args, **kwargs)

    bt.DataLoader = _loader

    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *a, **k: self  # noqa: E731

    bt.__runtime_patched__ = True


@register_model
class ScButterflyBackend(ModelBackend):
    key = "scbutterfly"
    name = "scButterfly"
    description = "双对齐变分自编码器，在 RNA 与 ATAC 之间做跨模态翻译与联合表征。"  # 兜底，实际取 SKILL.md

    capabilities = Capabilities(
        needs_counts_layer=False,
        owns_preprocessing=True,        # 归一化/HVG 在训练子进程内完成（对齐 scLinformer 的做法）
        supports_batch=False,           # 基础模型不使用 batch 信息
        supports_celltype=True,         # 仅用于评测与 UMAP 着色
        evaluates_internally=False,     # 评测统一由基类完成
        embedding_key="latent",
        artifacts=("model/*.pt", "cross_modal_metrics.csv"),
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
                  desc="latent 对齐阶段轮数（官方默认 200）"),
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
        """只查目录与 torch 是否可导入，**不真正 import 模型**（避免拖慢门户启动）。"""
        if not os.path.isdir(config.SCBUTTERFLY_DIR):
            return False, f"scButterfly 源码目录不存在：{config.SCBUTTERFLY_DIR}"
        if not os.path.exists(os.path.join(config.SCBUTTERFLY_DIR, "train_model.py")):
            return False, f"目录 {config.SCBUTTERFLY_DIR} 下缺少 train_model.py"
        try:
            if importlib.util.find_spec("torch") is None:
                return False, "未安装 torch"
        except Exception as e:
            return False, f"torch 探测失败：{e}"
        return True, f"源码目录 {config.SCBUTTERFLY_DIR}"

    # ---------------- 预处理阶段 ----------------
    def judge_extra(self, adata) -> list:
        """scButterfly 对 RNA 侧的要求：原始计数 + 细胞类型列（用于评测与 UMAP）。"""
        verdicts = [{
            "item": "scbutterfly_input",
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
            "训练分三段：RNA 自重建 → ATAC 自重建 → 跨模态翻译对齐；"
            "跨模态指标（ATAC_ARI / Cross_KNN_Acc / Modality_Align）见报告「跨模态一致性」一节。",
        ]

    # ---------------- 训练阶段 ----------------
    def train_cross_modal(self, adata, second, output_dir: str, cfg: dict) -> dict:
        import numpy as np
        import torch
        from scipy import sparse

        from app.tools.models.modalities import chrom_list_of, prepare_rna

        if second is None:
            raise ValueError("scButterfly 需要配对的 ATAC 数据（第二模态）")

        Model = _import_train_model()
        _patch_runtime()

        # 1) RNA 侧预处理（官方在内部做同样三步，这里做是为了掌握 HVG 后的真实列数）
        emit("training_log", {"message": "RNA 预处理：normalize_total + log1p + 高变基因筛选"})
        rna = prepare_rna(adata, cfg, self.name)
        atac = second

        n_genes, n_peaks = int(rna.n_vars), int(atac.n_vars)
        if n_genes < 50 or n_peaks < 50:
            raise ValueError(f"输入维度过小：RNA {n_genes} 基因 / ATAC {n_peaks} 峰")

        # 2) 稠密化前的内存评估：官方实现会把两份矩阵全部 toarray()
        dense_gb = rna.n_obs * (n_genes + n_peaks) * 4 / 1024 ** 3
        emit("training_log", {
            "message": f"待稠密化矩阵约 {dense_gb:.2f} GB（{rna.n_obs} 细胞 × "
                       f"{n_genes} 基因 + {n_peaks} 峰）"})
        if dense_gb > 6:
            raise MemoryError(
                f"dense 矩阵预计 {dense_gb:.1f} GB，超过安全上限。"
                f"请增大 atac_fpeaks（当前 {cfg.get('atac_fpeaks')}）过滤更多峰，"
                f"或减少 n_genes（当前 {n_genes}），或改用细胞数更少的数据集")

        chrom = chrom_list_of(atac)
        if not chrom or sum(chrom) != n_peaks:
            raise ValueError(f"ATAC 峰无法按染色体切分：chrom_list 合计 {sum(chrom) if chrom else 0}，"
                             f"实际峰数 {n_peaks}（峰名需为 chr1:1000-2000 形式）")
        k = len(chrom)

        # 3) 构造模型（维度推导与官方 butterfly.construct_model 完全一致）
        model = Model(
            RNA_data=rna,
            ATAC_data=atac,
            chrom_list=chrom,
            logging_path=output_dir,
            R_encoder_dim_list=[n_genes, R_ENCODER_HIDDEN, LATENT_DIM],
            A_encoder_dim_list=[n_peaks, A_TOKEN_DIM * k, LATENT_DIM],
            R_decoder_dim_list=[LATENT_DIM, R_ENCODER_HIDDEN, n_genes],
            A_decoder_dim_list=[LATENT_DIM, A_TOKEN_DIM * k, n_peaks],
            R_encoder_nlayer=2,
            A_encoder_nlayer=2,
            R_decoder_nlayer=2,
            A_decoder_nlayer=2,
            translator_embed_dim=LATENT_DIM,
            translator_input_dim_r=LATENT_DIM,
            translator_input_dim_a=LATENT_DIM,
            # 显式传入新列表：原实现会对默认值 append(1)，复用默认列表会污染后续调用
            discriminator_dim_list_R=[128],
            discriminator_dim_list_A=[128],
        )

        # 4) 划分训练/验证集（RNA 与 ATAC 行已对齐，同一份索引即可）
        n = int(rna.n_obs)
        rng = np.random.RandomState(19193)
        perm = rng.permutation(n)
        n_valid = max(1, int(round(n * 0.2)))
        valid_ids = perm[:n_valid].tolist()
        train_ids = perm[n_valid:].tolist()
        if not train_ids:
            raise ValueError(f"细胞数过少（{n}），无法划分训练集")

        r_kl = 20.0 / max(1, n_genes)
        a_kl = 20.0 / max(1, n_peaks)
        loss_weight = [1, 2, 1, r_kl, a_kl, r_kl + a_kl]

        epochs = int(cfg.get("epochs") or 100)
        trans_epochs = int(cfg.get("translator_epochs") or 200)
        batch_size = int(cfg.get("batch_size") or 64)
        emit("training_log", {
            "message": f"开始训练：R2R/A2A 各 {epochs} 轮，翻译器 {trans_epochs} 轮，"
                       f"batch_size={batch_size}，染色体块数={k}"})

        model.train(
            loss_weight=loss_weight,
            train_id_r=train_ids,
            train_id_a=train_ids,
            validation_id_r=valid_ids,
            validation_id_a=valid_ids,
            R2R_pretrain_epoch=epochs,
            A2A_pretrain_epoch=epochs,
            translator_epoch=trans_epochs,
            patience=int(cfg.get("patience") or 50),
            batch_size=batch_size,
            r_loss=torch.nn.MSELoss(),
            a_loss=torch.nn.BCELoss(),
            d_loss=torch.nn.BCELoss(),
            output_path=output_dir,
            logging_path=output_dir,
        )

        # 5) 抽取嵌入（模型本身不写 obsm）
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

        rna_lat = _encode(model.RNA_encoder, rna.X)
        atac_lat = _encode(model.ATAC_encoder, atac.X)
        adata.obsm[self.capabilities.embedding_key] = rna_lat
        second.obsm[self.capabilities.second_embedding_key] = atac_lat
        emit("training_log", {
            "message": f"嵌入完成：RNA {rna_lat.shape} / ATAC {atac_lat.shape}"})

        return {
            "epochs": epochs,
            "translator_epochs": trans_epochs,
            "model_dir": os.path.join(output_dir, "model"),
            "processed_dir": None,
            "n_genes": n_genes,
            "n_peaks": n_peaks,
            "n_chrom_blocks": k,
        }
