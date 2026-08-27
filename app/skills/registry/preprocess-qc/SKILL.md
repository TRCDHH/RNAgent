---
name: preprocess-qc
description: scLinformer 数据预处理技能：先按判定契约(judge)逐项质检，再按三级分流 SOP 修复到全 PASS，产出数据处理结果分析（01/02/03 + figures）并交训练阶段。
---

# scLinformer 数据预处理技能

把任意来源的原始数据加工成 scLinformer 模型可接受的输入。模型会在其内部执行
`normalize_total + log1p + HVG`，因此本技能的第一原则是：**先确认原始计数，再谈其它**，
避免输入已被归一化导致模型二次归一化失真。

---

## A. 判定契约（judge 清单）

调用 `judge(adata)` 逐项输出 PASS / FAIL / WARNING + 证据 + 建议动作。清单与修复映射如下：

| # | 检查项 item | PASS 标准 | 修复动作（FAIL/WARNING 时） |
|---|-------------|-----------|------------------------------|
| 1 | x_exists | X 存在且非空 | 提供有效表达矩阵 |
| 2 | raw_counts | X 为非负整数计数 | 从 `.raw` / `layers['counts']` 顶替；否则警告继续 |
| 3 | cell_type | obs 含 `cell_type` 列 | 有同义列（cell.type/celltype/annotation）则 `rename_column`；无则标记缺失 |
| 4 | batch | obs 含 `batch` 列 | 有同义列（sample/donor/sample_id）则 `rename_column`；无则标记缺失 |
| 5 | gene_name_unique | var_names 全部唯一 | `var_names_make_unique()` 去重 |
| 6 | numeric_valid | 无 NaN/Inf、非负 | 清理非法值（置 0 或过滤） |
| 7 | scale_min | n_cells≥100 且 n_genes≥200 | 规模过小，警告并建议扩充 |
| 8 | universal_coverage | universal 基因覆盖率≥50% | 覆盖率低：不用 universal 模型，或对齐基因命名 |

**核心顺序**：永远先判 `raw_counts`（第 2 项）。若 X 已被归一化/对数化，后续
per-cell QC、HVG 选择、总 UMI 统计都会失真，必须先谈清楚原始计数。

## B. 修复 SOP（三级分流）

对 judge 出的 FAIL/WARNING，按以下三级处理，每修一处都要 `judge` 重跑确认：

1. **简单改名 → `rename_column`**
   仅当存在「无歧义的同义列」时使用（如 `cell.type -> cell_type`、`Sample -> batch`）。
   返回改动前后对照并自动重跑 judge。

2. **升级 → `execute_code`（兜底）**
   命中以下任一情况，升级为受控执行自写代码：
   - 命名有歧义（多个候选列，无法确定哪个是 cell_type/batch）；
   - 多级列需拆分（如 `cell.type = "T cell:CD4"`，需拆成 cell_type 与 subtype）；
   - 异常对象（非标准 AnnData / 嵌套结构）；
   - 需统计推断（如从基因表达反推 cell_type）；
   - 格式转换（loom/csv/mtx -> h5ad）；
   - 需访问模型 vocab（多数据集基因对齐）。
   `execute_code` 在受限环境执行（白名单 builtins + numpy/pandas/scanpy/anndata/scipy），
   执行内容写入操作审计（03_操作审计.md）。

3. **有歧义就停下问用户，不瞎猜**
   若无法从数据本身无歧义推断意图（例如 cell_type 候选列有多个、拆分维度不清），
   停下并向用户确认，而不是擅自假设。

**结束条件**：重跑 `judge` 直到无 FAIL（WARNING 需人工确认是否可接受）；
然后用 `qc_stats` / `composition_stats` / `plot_qc` 产 QC 与组成图；
`generate_report` 汇总出「数据处理结果分析」；`save_dataset` 写 `processed_rna.h5ad` 交训练阶段。

## 产物结构（数据处理结果分析）

```
<output_dir>/
├── processed_rna.h5ad        # save_dataset 交集（交训练阶段）
└── 数据处理结果分析/
    ├── 01_数据状况.md         # 判断结论 + 对模型配置影响（固定句式）
    ├── 02_数据分析报告.md     # 单细胞数据集通用模板
    ├── 03_操作审计.md         # 跑了哪些脚本/代码/参数（可追溯）
    └── figures/               # QC 与组成图
```

## 固定句式（写入 01_数据状况.md）

- 无 batch：`未检测到 batch，use_batch=False，RNADecoder 无批次条件化，batch 评测项为 nan`
- 无 cell_type：`未检测到 cell_type，use_cell_type=False，跳过判别与 ARI/ASW`
- X 非原始计数：`检测到 X 非原始计数（证据），已从 .raw/layers 顶替，否则警告继续`
- 有 batch/cell_type：`检测到 X（N 类），use_X=True … 启用对应评测项`