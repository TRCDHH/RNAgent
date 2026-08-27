<div align="center">

# 🧬 RNAgent

### 面向单细胞 RNA 测序分析的自主智能 Agent 系统

**从原始数据到可复现分析 —— 自动化 · 可观测 · 可解释**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-1C3C3C?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-4D6BFE?style=for-the-badge)](https://www.deepseek.com/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://www.mysql.com/)

<br/>

**LangGraph 智能编排 · DeepSeek 推理 · scLinformer 建模 · 全链路可观测**

</div>

---

## ✨ RNAgent 是什么？

RNAgent 是一个**面向单细胞转录组学分析的多阶段智能 Agent 系统**。

传统的单细胞分析流程通常需要用户手动串联数据预处理、质量控制、模型训练、指标评测以及结果分析等多个环节。

RNAgent 将这些步骤统一纳入 Agent 驱动的自动化流水线：

```text
原始单细胞 RNA 测序数据
          │
          ▼
┌─────────────────────┐
│  01  数据预处理      │
│  检测 · 判定 · 修复 · QC │
└──────────┬──────────┘
           │ 结构化产物
           ▼
┌─────────────────────┐
│  02  模型运行        │
│  环境 · 配置 · 训练 · 评测 │
└──────────┬──────────┘
           │ 指标 + 嵌入
           ▼
┌─────────────────────┐
│  03  结果分析        │
│  汇总 · 解读 · 报告   │
└──────────┬──────────┘
           │
           ▼
      📄 HTML 分析报告
```

> **核心理念：让 Agent 决定“下一步做什么”，同时让每一个关键操作都留下可追溯的结构化产物。**

---

## 🖼️ 系统架构

<p align="center">
  <img src="imgs/a.png" alt="RNAgent 系统架构图" width="920"/>
</p>

RNAgent 按职责划分为多个层次：

| 层级 | 实现 | 职责 |
|:---|:---|:---|
| **前端** | `app/static/` | 任务创建 · 进度查看 · 日志 · 报告 · AI 问答 |
| **API** | `app/api.py` | REST · SSE · 静态资源托管 |
| **Agent / 编排** | `app/graph/` | StateGraph · 条件路由 · 错误处理 |
| **工具** | `app/tools/` | 数据预处理 · 模型运行 · 结果分析 |
| **知识** | `app/skills/` | `SKILL.md` 决策知识库 |
| **持久化** | `app/db.py` | MySQL 数据集 / 任务状态 |

---

# 🔄 三阶段智能分析流水线

## 01 · 数据预处理

<p align="center">
  <img src="imgs/b.png" alt="RNAgent 数据预处理流程图" width="920"/>
</p>

RNAgent 不假设输入数据已经满足模型要求，而是首先对数据进行自动检查和判定。

```text
加载数据
   ↓
格式 & 元数据检查
   ↓
PASS ──────────────────→ 继续分析
   │
   ├─ WARNING → 标准化 / 重命名
   │
   └─ FAIL    → 自动修复 / 执行代码
                    ↓
                  重新检查
                    ↓
                   PASS
                    ↓
                  QC 分析
                    ↓
               结构化分析产物
```

典型场景：

**10x 输出目录 + 包含 `cell.type` 与 `Sample` 的 `annotations.csv`**

Agent 可以自动完成：

- 检测输入格式与字段情况
- 判断 `cell_type` / `batch` 是否满足要求
- 自动合并 annotation 信息
- 对齐 barcode 的 `-1` 后缀
- 重命名同义字段
- 对多级标签进行拆分，例如 `"T cell:CD4"`
- 重新执行判定流程
- 生成 QC 与数据分析结果
- 将结构化产物交给下一阶段

### 核心原则

> **先确认原始计数，再进行后续处理。**

避免重复归一化等可能造成数据失真的操作。

---

## 02 · 模型运行

模型 Agent 将经过验证的数据连接至真实的 **scLinformer** 模型实现。

```text
环境探测
   ↓
GPU / 显存 / 数据规模
   ↓
参数自动决策
   ↓
scLinformer 训练
   ↓
嵌入评测
   ↓
指标 + UMAP
```

Agent 根据数据和运行环境自动完成关键决策：

- 根据 `obs` 列判断 `use_batch`
- 根据 `obs` 列判断 `use_cell_type`
- 根据 GPU 显存与数据规模确定 `batch_size`
- 生成模型运行配置
- 调用真实 scLinformer 进行训练
- 训练完成后执行嵌入评测
- 生成指标 CSV 与 UMAP
- OOM 时提供 batch size 降级策略

---

## 03 · 结果分析

结果分析 Agent 自动汇总前两个阶段产生的全部产物：

```text
数据规模
   +
判定结果
   +
QC 分析
   +
训练配置
   +
运行环境
   +
评测指标
   +
UMAP
   +
产物路径
   │
   ▼
 DeepSeek
   │
   ▼
自包含 HTML 分析报告
```

最终报告保存至：

```text
runtime/task/{task_id}/report.html
```

当 LLM 不可用时，系统提供本地报告模板作为兜底。

---

# ⭐ 核心特性

<div align="center">

| 🧩 三阶段 Agent | 🛡️ 智能数据修复 | ⚙️ 自动参数决策 |
|:---:|:---:|:---:|
| LangGraph StateGraph | 判定契约 | GPU + 数据规模 |
| 条件路由 | 多级修复策略 | 自动 batch size |
| 错误处理 | 修复后重新检查 | OOM 兜底 |

| 📡 全链路可观测 | 🧠 Skill 知识系统 | 💬 AI 分析助手 |
|:---:|:---:|:---:|
| SSE 实时事件 | 渐进式知识加载 | DeepSeek 报告 |
| 前端时间线 | `SKILL.md` | Function Calling |
| `events.jsonl` 审计 | 单细胞领域知识 | 查询真实指标 |

</div>

---

## 📊 任务产物与可复现性

每次分析任务都会独立保存至：

```text
runtime/task/{task_id}/
```

完整产物包括：

```text
├── report.html                  # 最终 HTML 分析报告
├── processed_rna.h5ad           # 修复后的原始计数矩阵
├── summary_metrics.csv          # 总体评测指标
├── cluster_metrics.csv          # 不同 resolution 的聚类指标
├── batch_metrics.csv            # 批次 / ASW 指标
├── umap_cell_type.png           # Cell Type UMAP
├── umap_batch.png               # Batch UMAP
│
├── model/
│   ├── rna_encoder.pth
│   ├── rna_decoder.pth
│   └── cell_type_discriminator.pth
│
├── processed_data/
│   ├── train_ids.npy
│   ├── valid_ids.npy
│   └── test_ids.npy
│
└── 数据处理结果分析/
    ├── 01_数据状况.md
    ├── 02_数据分析报告.md
    ├── 03_操作审计.md
    └── figures/
```

### 📈 评测指标

RNAgent 当前支持：

`ARI` · `AMI` · `NMI` · `HOM` · `Cell_ASW` · `Batch_ASW` · `Graph_Connectivity`

| 指标 | 含义 |
|:---|:---|
| **ARI / AMI / NMI / HOM** | 聚类结果与真实标签的一致程度 |
| **Cell_ASW** | 细胞类型之间的分离程度 |
| **Batch_ASW** | 批次效应相关指标 |
| **Graph_Connectivity** | 批次内图连通性 |

如果数据缺少 `cell_type` 或 `batch`，对应指标会自动跳过并标记为 **N/A**。

> **RNAgent 不编造任何评测数据。**

---

# 🧠 Skill 系统

RNAgent 将领域决策知识沉淀为模块化的 `SKILL.md`：

```text
app/
└── skills/
    └── registry/
        ├── preprocess-qc
        ├── model-running
        ├── report-generation
        ├── sc-domain-qa
        ├── training-troubleshoot
        └── h5ad-conversion
```

| Skill | 作用 |
|:---|:---|
| `preprocess-qc` | 判定契约 + 数据修复 SOP |
| `model-running` | 参数决策 + batch size + OOM 兜底 |
| `report-generation` | 报告产物 + 指标解读 |
| `sc-domain-qa` | 单细胞领域知识问答 |
| `training-troubleshoot` | 训练故障排查 |
| `h5ad-conversion` | 数据格式转换 |

RNAgent 使用**渐进式披露（Progressive Disclosure）**机制：

```text
SkillLibrary.disclose()
        ↓
仅暴露 Skill 名称与描述
        ↓
       load(name)
        ↓
需要时加载完整知识
```

这样可以避免所有领域知识长期占用 Agent 上下文。

---

# 🚀 快速开始

## 环境要求

- Python **≥ 3.10**
- MySQL **8.0**
- GPU 可选
- CPU 模式同样支持运行
- DeepSeek API Key（启用 LLM 报告 / AI 问答时需要）

## 安装

```bash
# 1. 克隆项目
git clone <repo-url>
cd z-newRnagent

# 2. 创建虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 如果默认源较慢，可以使用阿里镜像
# pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 4. 初始化 MySQL
mysql -u root -p < init.sql

# 5. 配置环境变量
copy .env.example .env

# Linux / macOS
# cp .env.example .env

# 编辑 .env：
# LLM_API_KEY=sk-xxx

# 6. 启动 RNAgent
python server.py
```

启动后访问：

**http://localhost:8000**

完整使用流程：

```text
创建数据集
   ↓
开始分析
   ↓
实时查看 Agent 进度
   ↓
查看任务日志
   ↓
查看分析结果
   ↓
阅读 HTML 报告
```

---

# ⚙️ 配置项

所有主要配置均支持通过环境变量覆盖：

| 变量 | 默认值 | 说明 |
|:---|:---|:---|
| `LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容接口地址 |
| `LLM_API_KEY` | — | DeepSeek API Key |
| `LLM_MODEL` | `deepseek-v4-flash` | LLM 模型名称 |
| `MODEL_EPOCHS` | `1` | 模型训练轮数 |
| `MODEL_BATCH_SIZE` | 自动 | 根据 GPU / 数据规模自动决定 |
| `SCLINFORMER_DIR` | `../../model/scLinformer-main` | scLinformer 源码目录 |
| `SANDBOX_IMAGE` | `rna-sandbox:latest` | `execute_code` 沙箱镜像（首次运行按 `docker/sandbox.Dockerfile` 自动构建） |
| `SANDBOX_TIMEOUT` | `300` | 沙箱执行超时（秒），超时强杀容器 |
| `SANDBOX_MEM_MB` | `4096` | 沙箱内存上限（MB） |
| `SANDBOX_CPUS` | `2` | 沙箱 CPU 限额 |

---

# 📡 API

| 方法 | 接口 | 说明 |
|:---:|:---|:---|
| `GET` | `/api/datasets` | 获取数据集列表 |
| `POST` | `/api/datasets` | 创建数据集 |
| `DELETE` | `/api/datasets/{id}` | 删除数据集 |
| `POST` | `/api/tasks/run` | 启动分析任务 |
| `GET` | `/api/tasks` | 获取任务列表 |
| `GET` | `/api/tasks/{id}/state` | 获取任务状态与结果 |
| `GET` | `/api/tasks/{id}/stream` | SSE 实时日志 / 进度 |
| `GET` | `/output/{id}/report.html` | 查看 HTML 分析报告 |
| `POST` | `/api/chat` | AI 助手对话 |
| `POST` | `/api/chat/stream` | AI 助手流式对话 |

---

# 📁 项目结构

```text
z-newRnagent/
├── server.py
├── run_demo.py
├── init.sql
├── requirements.txt
│
└── app/
    ├── api.py
    ├── main.py
    ├── state.py
    ├── config.py
    ├── db.py
    ├── llm.py
    ├── event_emitter.py
    │
    ├── static/
    │   └── index.html
    │
    ├── graph/
    │   ├── builder.py
    │   └── nodes.py
    │
    ├── tools/
    │   ├── preprocess_tools.py
    │   ├── model_tools.py
    │   ├── report_tools.py
    │   ├── assistant_tools.py
    │   └── local_tools.py
    │
    └── skills/
        ├── library.py
        └── registry/
```

---

# 🛣️ Roadmap

## ✅ 已完成

- [x] LangGraph 三阶段 Agent 编排
- [x] 条件路由与错误处理
- [x] 数据判定契约
- [x] 三级分流修复机制
- [x] QC 分析与审计产物
- [x] 接入真实 scLinformer
- [x] 自动参数与 batch size 决策
- [x] DeepSeek HTML 报告生成
- [x] AI Assistant Function Calling
- [x] SSE 实时进度推送
- [x] MySQL 数据持久化
- [x] `execute_code` Docker 沙箱隔离（断网、只读根文件系统、非 root、资源限额、超时强杀）

## 🚧 下一步

- [ ] Redis Checkpoint 与断点续跑
- [ ] Langfuse 全链路 Trace
- [ ] Vue 3 前端迁移

---

# 🔬 设计理念

### 01 · 先验证，再行动

不假设输入数据已经符合模型要求。

Agent 首先检查数据格式、字段和关键约束，再决定下一步操作。

### 02 · Agent 决策，但不编造

Agent 负责**调用工具、进行决策和组织流程**。

真正的训练结果、评测指标和分析产物来自实际执行，而不是由 LLM 虚构。

### 03 · 每一步都应该可观测

任务进度、Agent 动作、工具执行和最终产物都应该能够被查看、记录和追溯。

### 04 · 将知识变成可复用能力

领域知识通过 `SKILL.md` 模块化管理，而不是全部硬编码在 Agent 流程中。

---

<div align="center">

# 🧬 RNAgent

### 让单细胞分析流程自主运转

<br/>

**LangGraph × DeepSeek × scLinformer × FastAPI**

<br/>

*From Raw Data to Analysis — Autonomously.*

</div>