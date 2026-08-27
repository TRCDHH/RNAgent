---
name: model-running
description: scLinformer 模型运行技能：分析上一步(预处理)结果与本机环境，决定 use_batch / batch_size，传参执行训练+评测工具。不启用 universal_model，其余用 train.py 默认值。
---

# scLinformer 模型运行技能

本阶段 Agent 只做两件事：**分析上一阶段结果 + 本机环境 → 决定参数**，以及**把参数传给
训练工具执行**。模型（scLinformer）内部会自己做 `normalize_total + log1p + HVG`，本阶段不重复预处理。

## 一、先分析什么

1. 上一步（预处理）结果：预处理产物 `processed_rna.h5ad` 的 `obs.columns` 里有没有
   `cell_type` / `batch` 列。这等价于判定契约的结论：
   - 有 `cell_type` → `use_cell_type=True`；没有 → `False`
   - 有 `batch` → `use_batch=True`；没有 → `False`
2. 本机环境：是否有 GPU、显存多大（`torch.cuda.get_device_properties(0).total_memory`），
   以及数据规模 `n_cells`（决定 batch 与显存占用）。

## 二、参数决定规则

| 参数 | 怎么定 |
|------|--------|
| use_batch | `'batch' in obs.columns`。没有就 False，否则模型访问缺失列会崩 |
| use_cell_type | `'cell_type' in obs.columns`。同上 |
| batch_size | 按显存 + n_cells 查下面的表 |
| use_universal_model | **固定 False**（不启用） |
| 其他 | 用 train.py / Model 默认值：process_data=True、use_hvg=True、n_genes=2000 |

### batch_size 查表（显存优先，数据量微调）

| 显存 | batch_size |
|------|-----------|
| 无 GPU / CPU | 32 |
| < 4GB | 32 |
| 4 ~ 8GB | 64 |
| 8 ~ 16GB | 128 |
| 16 ~ 24GB | 256 |
| ≥ 24GB | 512 |

数据量微调（在表值基础上再收窄）：

- `n_cells < 500` → 取 `min(表值, 32)`
- `500 ≤ n_cells < 2000` → 取 `min(表值, 64)`
- 始终保证 `1 ≤ batch_size ≤ n_cells`

### 数据量案例

- 小样本：`n_cells=800`，8GB 显存 → 表值 64，收窄后 `min(64,64)=64` → batch_size=64
- 中样本：`n_cells=20000`，16GB 显存 → batch_size=256
- 大样本：`n_cells=120000`，24GB 显存 → batch_size=512
- CPU 环境：`n_cells=5000`、无 GPU → batch_size=32（慢但能出结果）
- 无 batch 列：`use_batch=False`（即使 Model 默认 True 也要关掉）

## 三、异常兜底

- CUDA out of memory → batch_size 减半重试（或设环境变量 `MODEL_BATCH_SIZE` 强制小 batch）
- 找不到 `processed_rna.h5ad` → 报错并提示先完成预处理阶段
- 无 GPU → 自动用 CPU，batch_size 取小型（32），提示训练较慢

## 四、产出

- 模型权重：`<output_dir>/model/*.pth`
- 评测指标：`<output_dir>/summary_metrics.csv`（ARI/AMI/NMI/HOM/Cell_ASW/Batch_ASW/Graph_Connectivity）
- UMAP 图：`<output_dir>/umap_cell_type.png`、`<output_dir>/umap_batch.png`（有对应列时）
- 运行参数与指标写入任务 process，供结果分析阶段读取