---
name: scrosstalk
type: model
display_name: scCrossTalk
description: RNA 与 ATAC 双向跨模态翻译框架，通过对比学习与对抗训练对齐两种模态
params:
  epochs: 100
  translator_epochs: 200
  batch_size: auto
  n_genes: 3000
  patience: 50
  atac_fpeaks: 0.005
batch_size_table: cpu=32; 0=64; 8000=128; 16000=256
batch_size_caps: 500=32; 2000=64; 8000=128
---

# scCrossTalk 运行指南（双模态接入口）

## 一、这是什么模型

scCrossTalk 是 **RNA ↔ ATAC 跨模态翻译**框架，用「单模型自编码 + 多模型翻译」三阶段
把两种模态对齐到同一 latent 空间：

```text
[1] train_rna_model    RNA 自编码：重构 + GAN 判别 + DCL 对比学习
[2] train_atac_model   ATAC 自编码：按染色体切块编码（chrom_list）
[3] train_multi_model  跨模态翻译：R2A / A2R + MMD 对齐
```

**关键：必须提供与 RNA 配对的 ATAC 数据**（同一批细胞、同一顺序）。
只给 RNA 数据时该模型不可运行 —— 数据集登记时必须在「配对 ATAC 路径」里指到 ATAC 文件。

## 二、数据要求

| 项 | RNA | ATAC |
|---|---|---|
| 格式 | `.h5ad`（AnnData） | `.h5ad`（AnnData，细胞 × 峰） |
| 细胞 | 原始计数最好 | 与 RNA **同细胞、同顺序** |
| 特征名 | 基因名（自动去重） | **必须是 `chr1:1000-2000` 形式**，按染色体 + 坐标排序 |
| 预处理 | 模型侧做 `normalize_total` + `log1p` + HVG（`n_genes`） | 流水线侧做二值化 + 过滤低出现峰（`atac_fpeaks`）+ TF-IDF + 缩放到 [0,1] |
| cell_type | 可选，缺失时跨模态指标只剩 `Modality_Align` | 不需要 |

`chrom_list` 由峰的染色体连续块统计得出，模型首层按块切列；
**峰乱序会导致切列错位**，因此 prepare 阶段会自动按 `chr` + 起点排序。

> 注意：通用判定契约要求 obs 含 `batch` 列（所有模型一致）。本模型并不使用 batch 信息，
> 但缺失时判定会标记 FAIL 并进入修复流程 —— 补一列常量 `batch` 即可让判定变 PASS。

## 三、参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `epochs` | 100 | RNA 自编码与 ATAC 自编码各训练轮数（官方默认） |
| `translator_epochs` | 200 | `train_multi_model` 轮数（官方默认） |
| `batch_size` | auto | 见 frontmatter 查表；三轮训练共用 |
| `n_genes` | 3000 | RNA 侧 HVG 数，即 `RNAEncoder` 的输入维度 |
| `patience` | 50 | 每个阶段的早停耐心值 |
| `atac_fpeaks` | 0.005 | 过滤在少于该比例细胞中出现的峰 |

网络结构固定使用官方默认：`lantent_dim=128`、`rna_encoder_hid_dim=128`、
`rna_decoder_hid_dim=256`、`atac_token_dim=32`、`translator_hid_dim=128`
（不暴露，避免过度调参）。

### 参数覆盖（与其他模型完全隔离）

| 想改什么 | 环境变量 |
|---------|---------|
| 预训练轮数 | `SCROSSTALK_EPOCHS=50` |
| 翻译轮数 | `SCROSSTALK_TRANSLATOR_EPOCHS=100` |
| 批大小 | `SCROSSTALK_BATCH_SIZE=32` |
| 高变基因数 | `SCROSSTALK_N_GENES=2000` |
| 早停耐心 | `SCROSSTALK_PATIENCE=20` |
| 峰过滤比例 | `SCROSSTALK_ATAC_FPEAKS=0.01` |

## 四、流程

```text
预处理阶段
  load RNA -> 通用 judge -> 修复 -> load ATAC -> ATAC 判定 -> 对齐细胞顺序
    -> ATAC 二值化/过滤峰/TF-IDF/归一 -> processed_atac.h5ad
训练阶段（独立子进程）
  prepare_rna：normalize_total + log1p + HVG
     ↓
  scCrossTalkModel(process_data=False, use_hvg=False, n_genes=RNA 实际列数)
     ↓
  [1] train_rna_model -> [2] train_atac_model -> [3] train_multi_model
     ↓
  obsm['latent']      <- rna_encoder 输出 128 维（统一评测口径）
  obsm['latent_atac'] <- atac_encoder 输出 128 维（跨模态对照）
     ↓
  evaluate_sc_embedding(embedding_key='latent')  ← 与 scLinformer 同一套评测
  cross_modal_metrics.csv                        ← 跨模态一致性
```

> 为什么传 `process_data=False`：官方在 `process_data=True` 时会连带处理 ATAC，
> 且 `n_genes` 与 HVG 后列数的一致性断言被注释掉了（`crossTalk.py:79`），
> 尺寸不匹配只会在前向报一句无关的矩阵乘错误。这里把预处理收归模态层，
> 由我们自己掌握 HVG 后的真实列数。

## 五、跨模态指标口径

`cross_modal_metrics.csv` 是**辅助对照**，不覆盖 `summary_metrics.csv`：

| 指标 | 含义 |
|------|------|
| `RNA_ARI/AMI/NMI/HOM` | 用 RNA 嵌入做 KMeans（k=细胞类型数）后与 `cell_type` 的一致性 |
| `ATAC_ARI/AMI/NMI/HOM` | 同上，但只用 ATAC 嵌入 |
| `Modality_Align` | 配对细胞在两个 latent 空间的余弦相似度均值（越高越对齐） |
| `Cross_KNN_Acc` | 用 RNA 嵌入 + `cell_type` 训练 KNN，在 ATAC 嵌入上预测标签的准确率 |

刻意用确定性的 KMeans 而不是 leiden，避免分辨率参数引入额外噪声，
因此这些 ARI 与主表的 leiden-ARI 不完全等价，只用于跨模态对照。

## 六、异常兜底

- ATAC 与 RNA 细胞数不一致 → 预处理阶段 FAIL，任务终止（不会静默错位）
- ATAC 峰名不是 `chr:start-end` → 预处理阶段 FAIL
- ATAC 未按染色体排序 → WARNING，prepare 阶段自动排序
- 未安装 `linformer` → 前端该模型置灰（`pip install linformer` 即可）
- 未安装 torch / 源码目录不存在 → 前端该模型置灰

## 七、产物

`<output_dir>/model/*.pth`（rna/atac 编解码器 + translator + discriminator）、
`<output_dir>/processed_data/*.npy`（五折划分索引）、3 个统一指标 CSV、
UMAP 图、`cross_modal_metrics.csv`。

## 八、已知取舍

- 原仓库 `DataLoader(num_workers=4)` 硬编码，在 Windows 会递归 spawn，本接入口统一改为 0。
- 原仓库 `test_model()` 会把 A2R/R2A 重建结果写成 tSNE.pdf 并另算一套聚类指标，
  本接入口**不调用它**（依赖 `obs['cell_type']`、产物是 PDF、且指标针对翻译后的表达谱），
  改用统一的 `cross_modal_metrics.csv` 口径，保证与 scButterfly 可直接对比。
