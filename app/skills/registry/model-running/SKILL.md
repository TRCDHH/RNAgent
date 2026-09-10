---
name: model-running
description: 模型运行阶段（与具体模型无关）：读取预处理产物决定能力开关、按模型 SKILL 决定参数、子进程训练并统一评测。各模型专属规则见 registry/models/<key>/SKILL.md。
---

# 模型运行技能（模型无关）

本阶段只做三件事：**读上一步结果 → 决定参数 → 传参执行工具**。
**具体模型名不出现在本流程中**：所有差异都由后端声明的 `Capabilities` 能力位与
`ParamSpec` 参数 Schema 表达，各模型的规则与默认值写在
`skills/registry/models/<key>/SKILL.md`。

## 一、分析什么

1. 预处理产物 `processed_rna.h5ad` 的 `obs.columns`：决定 `use_batch` / `use_cell_type`
   这类能力开关（有列则开，无列则关，避免模型访问缺失列崩溃）。
2. 本机环境：是否有 GPU、显存大小、数据规模 `n_cells`（决定 `batch_size` 与设备）。

## 二、参数决定流程（与模型无关）

```text
ParamSpec 默认值（代码声明类型/范围）
        ↓  覆盖
SKILL.md params（声明默认值 —— 调参只改 SKILL，不改代码）
        ↓  覆盖
环境变量 {MODEL_KEY}_{PARAM}（如 SCVI_MAX_EPOCHS）
        ↓  覆盖
用户传入 params（API / 前端表单，按 Schema 强转与校验）
        ↓  补全
环境自动决策（batch_size 按 SKILL 的 batch_size_table 查表）
```

## 三、按能力位分流（而不是按模型名）

| 能力位 | 为真时流水线做什么 |
|--------|------------------|
| `needs_counts_layer` | 预处理阶段把原始计数写入 `layers["counts"]` |
| `owns_preprocessing` | 为真：模型内部做归一化/HVG，流水线不动；为假：流水线侧做 HVG 筛选 |
| `supports_batch` / `supports_celltype` | 按 obs 列自动开关对应的能力 |
| `evaluates_internally` | 为真：模型自带评测，训练阶段不重复评测；为假：统一调 `evaluate_sc_embedding` |

## 四、统一评测（关键）

任何模型只要把嵌入写进 `adata.obsm[embedding_key]`，就调用同一个
`evaluate_sc_embedding` 产出 `summary_metrics.csv` / `cluster_metrics.csv` /
`batch_metrics.csv` 与 UMAP 图。**因此结果分析、报告、前端、AI 助手完全不需要区分模型。**

## 五、执行与兜底

- 训练在独立子进程执行（`app/tools/train_worker.py`），崩溃 / OOM / 段错误不连累主服务；
  父进程轮询 `progress.jsonl` 与 `train_stdout.log` 转发进度。
- 模型不可用时（源码缺失 / 依赖未装）前端置灰该选项，运行期给出明确报错。
- 找不到 `processed_rna.h5ad` → 报错并提示先完成预处理阶段。
