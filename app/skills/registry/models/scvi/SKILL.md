---
name: scvi
type: model
display_name: scVI
description: 变分自编码器，擅长批次校正与低维表征
params:
  max_epochs: auto
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
| `max_epochs` | auto | **不写死**：留空即用 scvi-tools 官方启发式，随细胞数自适应（见下）；开启 early stopping，收敛后自动停止 |
| `batch_size` | auto | 见 frontmatter 查表；scVI 是轻量 VAE，可比 scLinformer 取更大值 |
| `batch_key` | `batch` | obs 中的批次列名，不存在则关闭批次校正 |
| `cell_type_key` | `cell_type` | obs 中的细胞类型列名 |
| `n_top_genes` | 2000 | 0 表示不做高变基因筛选 |
| `accelerator` | `auto` | auto / cpu / gpu |

网络结构固定使用 scvi-tools 推荐默认：`n_latent=30, n_hidden=128, n_layers=2,
gene_likelihood=nb, dropout_rate=0.1`（不暴露，避免过度调参）。

### max_epochs 走官方启发式，不要写死

scvi-tools 在 `model/base/_training_mixin.py` 里的逻辑是：

```python
if max_epochs is None:
    max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
# scvi/model/_utils.py
max_epochs = min(round(20000 / n_obs * 400), 400)   # 下限 1
```

即**细胞越多、轮数越少，上限 400**。所以本模型默认不设固定值，实际取值：

| 细胞数 | max_epochs |
|--------|-----------|
| 100 | 400 |
| 500 | 400 |
| 5,000 | 400 |
| 20,000 | 400 |
| 69,249 | 116 |
| 200,000 | 40 |

想手动固定就用 `SCVI_MAX_EPOCHS=<n>`。本仓库在主进程里复刻了同一公式
（`scvi.py::_max_epochs_heuristic`），目的是让日志与报告能显示真实轮数，
同时避免主进程 import scvi（冷启动约 7 秒）。

### 参数覆盖（与 scLinformer 完全隔离）

本模型的参数**只认自己的前缀 `SCVI_`**，改动不会影响 scLinformer：

| 想改什么 | 环境变量 |
|---------|---------|
| 最大训练轮数 | `SCVI_MAX_EPOCHS=400` |
| 批大小 | `SCVI_BATCH_SIZE=128` |
| 批次列名 | `SCVI_BATCH_KEY=Sample` |
| 细胞类型列名 | `SCVI_CELL_TYPE_KEY=celltype` |
| 高变基因数 | `SCVI_N_TOP_GENES=3000` |
| 计算设备 | `SCVI_ACCELERATOR=cpu` |

> 历史全局变量 `MODEL_EPOCHS` / `MODEL_BATCH_SIZE` 已废弃（会同时影响多个模型），
> 服务启动时若检测到会打印告警。

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
