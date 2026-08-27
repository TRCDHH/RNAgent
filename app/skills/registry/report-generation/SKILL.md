---
name: report-generation
description: 结果分析阶段：汇总前两阶段（预处理 + 模型运行）产物，生成最终 HTML 报告；说明有哪些产物、分析什么、指标怎么解读。
---

# 结果分析技能

本阶段 Agent 根据**预处理阶段**和**模型运行阶段**留下的产物，生成一份最终 HTML 报告。
只分析真实存在的数据，缺什么就写「跳过」，不编造。

## 一、模型最终会留下什么（产物清单）

运行目录为 `<output_dir>`（即任务输出目录），两阶段产物如下：

| 产物 | 路径 | 说明 |
|------|------|------|
| 原始计数数据（预处理交训练用） | `<output_dir>/processed_rna.h5ad` | 修复后的原始计数矩阵 |
| 数据状况/分析/审计（预处理） | `<output_dir>/数据处理结果分析/01_数据状况.md`、`02_数据分析报告.md`、`03_操作审计.md` | 判定结论与 QC |
| 评测总指标 | `<output_dir>/summary_metrics.csv` | ARI/AMI/NMI/HOM/Cell_ASW/Batch_ASW/Graph_Connectivity（1 行） |
| 聚类指标 | `<output_dir>/cluster_metrics.csv` | 每个 resolution 一行 ARI/AMI/NMI/HOM（有 cell_type 才有） |
| 批次/ASW 指标 | `<output_dir>/batch_metrics.csv` | Cell_ASW/Batch_ASW/Graph_Connectivity |
| UMAP 图 | `<output_dir>/umap_cell_type.png`、`<output_dir>/umap_batch.png` | 有对应列才生成 |
| 模型权重 | `<output_dir>/model/{rna_encoder,rna_decoder,cell_type_discriminator}.pth` | 训练保存的权重 |
| 数据划分 | `<output_dir>/processed_data/{train_ids,valid_ids,test_ids}.npy` | 64/16/20 五折划分 |

## 二、分析什么（对应上文产物）

1. **数据集概览**：n_cells × n_genes、总 UMI、稀疏度 sparsity、obs 有哪些列。
2. **数据状况**：有没有 cell_type / batch 列 → `use_cell_type` / `use_batch`；
   判定结论（PASS/WARNING/FAIL）。这是模型是否做判别 / 去批次的前提。
3. **训练配置与资源**：use_batch、use_cell_type、batch_size、epochs、是否 universal、GPU/显存。
4. **模型评测**（核心，来自 summary_metrics.csv）：
   - ARI / AMI / NMI / HOM：聚类与真实 cell_type 的一致性（有 cell_type 才有）
   - Cell_ASW：细胞类型分离度
   - Batch_ASW：去批次效果（有 batch 才有）
   - Graph_Connectivity：批次内连通性（有 batch 才有）
5. **UMAP 可视化**：cell_type 着色图 / batch 着色图，解读是否分群清晰、是否残留批次效应。

## 三、指标解读规范

- ARI/AMI/NMI/HOM 越接近 1 聚类越好；< 0 说明甚至不如随机
- Cell_ASW 越接近 1 细胞类型分离越好
- Batch_ASW 越接近 1（或绝对值低）去批次效果越好
- Graph_Connectivity 越接近 1 越好

## 四、缺列时的固定说法

- 无 cell_type：`use_cell_type=False`，跳过 ARI/AMI/NMI/HOM/Cell_ASW 与 cell_type UMAP，指标为 N/A
- 无 batch：`use_batch=False`，跳过 Batch_ASW / Graph_Connectivity 与 batch UMAP，指标为 N/A
- 不要为空缺项编造任何数值。

## 五、报告结构（最终 HTML）

1. 数据集概览
2. 数据状况（有无 cell_type/batch、judge 结论）
3. 训练配置与资源占用
4. 模型评测（指标表 + 逐项解读 + UMAP 图）
5. 结论与建议

- 直接输出 HTML（`<!DOCTYPE html>` 起），内联 CSS，中文。
- 图片用相对路径 `<img src="umap_cell_type.png">` / `<img src="umap_batch.png">`，只引用存在的图。