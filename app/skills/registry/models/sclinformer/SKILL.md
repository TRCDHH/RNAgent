---
name: sclinformer
type: model
display_name: scLinformer
description: 基于 Transformer 的自编码模型，模型内部完成归一化与高变基因筛选，适用于通用单细胞表征学习
params:
  epochs: 100
  batch_size: auto
  use_batch: auto
  use_cell_type: auto
  process_data: true
  use_hvg: true
  n_genes: 2000
  use_universal_model: false
batch_size_table: cpu=32; 0=32; 4000=64; 8000=128; 16000=256; 24000=512
batch_size_caps: 500=32; 2000=64
---

# scLinformer 运行指南

## 一、数据要求

- `adata.X` 必须是**原始计数**：模型内部会执行 `normalize_total + log1p + HVG`，
  若输入已归一化会导致二次归一化失真（预处理阶段的 `raw_counts` 判定项负责把关）。
- `var_names` 必须唯一，模型内部会再执行一次 `var_names_make_unique()`。
- `obs` 可含 `cell_type`（判别头 + 聚类评测）与 `batch`（解码器批次条件化 + 去批次评测），
  缺失则对应能力自动关闭——**不能强行开启，否则模型访问缺失列会崩溃**。

## 二、参数决定规则

| 参数 | 怎么定 |
|------|--------|
| `use_batch` | `'batch' in obs.columns`，留空（auto）即按此自动决定 |
| `use_cell_type` | `'cell_type' in obs.columns`，同上 |
| `batch_size` | 见下方查表规则；环境变量 `MODEL_BATCH_SIZE` 可强制覆盖 |
| `epochs` | 默认 1（演示）；`MODEL_EPOCHS` 或 `SCLINFORMER_EPOCHS` 可覆盖 |
| `use_universal_model` | 固定 False（不启用跨数据集基因对齐） |
| `process_data` / `use_hvg` / `n_genes` | 保持默认 True / True / 2000 |

### batch_size 查表（显存优先，数据量收窄）

见 frontmatter `batch_size_table`。含义：显存 ≥ 阈值时取对应 batch_size；无 GPU 取 `cpu` 值。

- 小样本：`n_cells=800`，8GB 显存 → 表值 128，收窄后 `min(128,64)=64`
- 中样本：`n_cells=20000`，16GB 显存 → 256
- 大样本：`n_cells=120000`，24GB 显存 → 512（再被 `min(base, n_cells)` 兜底）
- CPU：`n_cells=5000` → 32（慢但能出结果）

## 三、评测

`test_model(use_test=True)` 内部调用 `evaluate_sc_embedding(embedding_key="X_emb")`，
在测试集划分上产出 `summary_metrics.csv` / `cluster_metrics.csv` / `batch_metrics.csv`
与 `umap_cell_type.png` / `umap_batch.png`。**训练阶段不再重复评测。**

## 四、异常兜底

- CUDA out of memory → `batch_size` 减半重试，或设 `MODEL_BATCH_SIZE` 强制小 batch
- 找不到 `processed_rna.h5ad` → 报错并提示先完成预处理阶段
- 无 GPU → 自动用 CPU，`batch_size` 取 `cpu` 值，提示训练较慢
- 找不到源码目录 → 前端该模型置灰，提示配置 `SCLINFORMER_DIR`

## 五、产物

`<output_dir>/model/*.pth`、`<output_dir>/processed_data/*.npy`、3 个指标 CSV、UMAP 图。
