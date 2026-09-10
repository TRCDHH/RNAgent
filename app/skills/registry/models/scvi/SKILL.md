---
name: scvi
type: model
display_name: scVI
description: 单细胞变分自编码器，以负二项似然直接建模原始计数，擅长批次校正与低维表征，要求原始计数存入 layers counts
params:
  max_epochs: 100
  batch_size: auto
  batch_key: batch
  cell_type_key: cell_type
  n_top_genes: 2000
  accelerator: auto
batch_size_table: cpu=64; 0=128; 8000=256; 16000=512; 24000=1024
batch_size_caps: 500=32; 2000=128
---

# scVI 运行指南（极简接入）

## 一、数据要求（与 scLinformer 的关键差异）

| 项 | scVI |
|---|------|
| 计数存放位置 | **必须** `adata.layers["counts"]`（缺失时流水线自动从 `X` 复制） |
| 归一化 | **不做** `log1p` / `normalize_total`，scVI 用 NB 似然直接建模计数 |
| 高变基因 | 流水线侧做（`n_top_genes`，seurat_v3），模型内部不做 |
| batch | `setup_anndata(batch_key=...)`，作为协变量参与批次校正 |
| cell_type | 仅用于评测与 UMAP 着色（本接入口不启用 SCANVI 半监督） |

## 二、参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `max_epochs` | 100 | 开启 early stopping，收敛后自动停止 |
| `batch_size` | auto | 见 frontmatter 查表；scVI 是轻量 VAE，可比 scLinformer 取更大值 |
| `batch_key` | `batch` | obs 中的批次列名，不存在则关闭批次校正 |
| `cell_type_key` | `cell_type` | obs 中的细胞类型列名 |
| `n_top_genes` | 2000 | 0 表示不做高变基因筛选 |
| `accelerator` | `auto` | auto / cpu / gpu |

网络结构固定使用 scvi-tools 推荐默认：`n_latent=30, n_hidden=128, n_layers=2,
gene_likelihood=nb, dropout_rate=0.1`（不暴露，避免过度调参）。

## 三、流程

```text
prepare_data: 写 counts 层 -> 基因名唯一 -> 分类 dtype -> HVG 筛选
     ↓
setup_anndata(layer="counts", batch_key=...)
     ↓
SCVI(...).train(max_epochs, batch_size, accelerator, early_stopping)
     ↓
adata.obsm["latent"] = get_latent_representation()
     ↓
evaluate_sc_embedding(embedding_key="latent")   ← 与 scLinformer 同一套评测
     ↓
summary / cluster / batch metrics + UMAP
```

## 四、异常兜底

- 未安装 `scvi-tools` → 前端该模型置灰；强行提交时训练阶段报明确错误
- 无 `batch` 列 → `batch_key=None`，正常训练，Batch_ASW / Graph_Connectivity 为 N/A
- seurat_v3 高变基因筛选失败 → 回退 `seurat` flavor，再失败则使用全部基因
- CUDA OOM → 调小 `batch_size`（scVI 显存占用远低于 Transformer 类模型）

## 五、产物

`<output_dir>/model/scvi/`（scvi 模型存档）、`adata.obsm["latent"]`、
3 个指标 CSV、UMAP 图。
