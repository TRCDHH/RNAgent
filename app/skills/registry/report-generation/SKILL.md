---
name: report-generation
description: 结果分析阶段：汇总前两阶段（预处理 + 模型运行）产物，按固定模板 report_template.html 生成最终 HTML 报告；说明有哪些产物、分析什么、指标怎么解读、以及模板占位符。
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

固定六个章节，顺序与版式不可变（**核心结果前置**：先看评测，再看依据）：

1. **模型评测**（综合指标条、逐项解读表、综合判断、分辨率表、批次指标、UMAP 图）
2. 所用模型与关键参数（模型名称/KEY、训练参数、能力表、产物路径）
3. 数据集概览（细胞数/基因数/总 UMI/稀疏度、obs 字段标签）
4. 数据状况与判定结果（cell_type/batch 可用性、judge 状态）
5. 训练配置与资源占用（超参数表、GPU/显存/PyTorch）
6. 结论与建议

> 顺序设计意图：读者打开报告第一屏即看到指标与 UMAP，无需滚动到底。因此
> 「模型 / 数据 / 配置」这类**背景信息统一后置**。

图片用相对路径 `<img src="umap_cell_type.png">` / `<img src="umap_batch.png">`，只引用存在的图。

## 六、版式由模板固定（重要）

报告**不再由 LLM 自由生成 HTML**（那样每次样式都不一样）。改为：

```text
app/skills/registry/report-generation/
├── SKILL.md                 ← 本文件
└── report_template.html     ← 唯一版式来源（CSS + 六章节结构 + 占位符）
```

渲染流程（`app/tools/report_tools.py`）：

```text
collect_report_context()      汇总产物为结构化 context
        ↓
_build_fragments()            把 context 渲染成 HTML 片段
        ↓
模板占位符替换 {{XXX}}         样式与结构完全来自模板
        ↓
_narrative()                  仅「结论 / 建议」两条列表交给 LLM 撰写
                              （LLM 不可用/失败 → _rule_narrative() 规则兜底）
```

- **数值、表格、图片、徽章全部由代码确定性生成**，不会因 LLM 随机性而变化；
- LLM 只被要求输出两段纯文本列表（`###CONCLUSIONS###` / `###SUGGESTIONS###` 分隔），
  **不产出 HTML**，因此无法影响版式；
- 模板缺失时回退 `_mock_html()`（简易兜底模板），保证报告非空。

### 模板占位符清单

| 占位符 | 内容 |
|--------|------|
| `{{PAGE_TITLE}}` `{{HERO_TITLE}}` `{{HERO_SUBTITLE}}` `{{CHIPS}}` | 标题与首屏信息胶囊 |
| `{{MODEL_NAME}}` `{{MODEL_DESC}}` `{{MODEL_PARAMS}}` | 模型简介与参数卡片 |
| `{{CAPABILITY_ROWS}}` `{{ARTIFACT_TAGS}}` `{{ARTIFACT_PATHS}}` | 能力表与产物 |
| `{{DATASET_STATS}}` `{{DATASET_HINT}}` `{{OBS_COUNT}}` `{{OBS_TAGS}}` | 数据集概览 |
| `{{CONDITION_STATS}}` `{{CONDITION_HINT}}` | 数据状况 |
| `{{TRAIN_ROWS}}` `{{ENV_STATS}}` `{{ENV_HINT}}` | 训练配置与资源 |
| `{{SUMMARY_BARS}}` `{{METRIC_ROWS}}` `{{SUMMARY_WARN}}` | 综合指标与解读 |
| `{{CLUSTER_ROWS}}` `{{CLUSTER_HINT}}` | 分辨率指标 |
| `{{BATCH_STATS}}` `{{BATCH_HINT}}` `{{UMAP_IMAGES}}` `{{UMAP_HINT}}` | 批次指标与 UMAP |
| `{{CONCLUSION_LIST}}` `{{SUGGESTION_LIST}}` `{{FOOTER}}` | 结论与建议 |

### 改版式的正确姿势

直接编辑 `report_template.html`（改 CSS 或调整章节）即可，**不需要改 Python 代码**；
新增数据字段时，在 `_build_fragments()` 里多返回一个占位符并在模板中引用。

## 七、文案规范（简洁高效）

**总原则：一屏能看完，不看第二遍。** 数值已经在表格里，文字只负责「结论」，不复述数字。

### 硬性约束

| 位置 | 约束 |
|------|------|
| `hint` / `warn-box` 提示块 | **不超过 2 句 / 60 字**；只写「结论 + 一句原因」 |
| 指标逐项解读 | **每项 1 句**，格式固定：`<档位徽章> {含义}；本次 {数值}，属于{档位}水平。` |
| 结论列表 | **3–5 条**，每条 **≤ 40 字** |
| 建议列表 | **3–5 条**，每条 **≤ 40 字**，必须是可执行动作 |

### 写法要求

- **禁止复述表格里已有的数字**：不要写「Batch_ASW 为 0.9028，Graph_Connectivity 为 1.0，Cell_ASW 为 0.4149」，
  改成「批次整合良好，细胞类型分离不足」。
- **禁止空话**：删掉「综上所述」「值得注意的是」「由此可见」这类连接词。
- **禁止重复**：同一结论不在「综合判断」「结论」「建议」里说三遍，各自只出现一次。
  综合判断 = 一句话定性；结论 = 分维度事实；建议 = 下一步动作。
- **禁止编造**：指标为 `null` 就写「已跳过」，不推测、不补齐。
- **每条建议必须可执行**：带明确动作（调什么参数、做什么对比），不写「建议进一步分析」。
- 用陈述句，不用疑问句、不用感叹号。

### 反例 → 正例

| ❌ 冗长 | ✅ 简洁 |
|--------|--------|
| 本次实验使用 scLinformer 模型对 492 个细胞和 3000 个基因的数据进行了训练，训练轮数为 1，batch_size 为 32。 | 已训练 scLinformer（1 epoch / batch 32），轮数偏少，表征未收敛。 |
| Batch_ASW 为 0.9028，说明批次效应被很好地消除了；Graph_Connectivity 为 1.0，说明图连通性很好。 | 批次整合良好（Batch_ASW、Graph_Connectivity 均高）。 |
| 建议可以考虑适当增加训练轮数，以便让模型更好地收敛，从而可能提升指标。 | 提高 epochs 至 50+，对比 ARI 是否上升。 |

> LLM 只负责写「结论 / 建议」两条列表，提示词中已注入上述约束；其余文案由
> `_build_fragments()` 按同一规范确定性生成，不经过 LLM。