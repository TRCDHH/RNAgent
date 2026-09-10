<div align="center">

# 🧬 RNAgent

**一个方便用各种模型处理各种数据集的单细胞分析 Agent 系统**

### *选个数据集 → 选个模型 → 拿报告。*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-1C3C3C?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/%E6%A8%A1%E5%9E%8B-scLinformer%20%7C%20scVI%20%7C%20...-6D28D9?style=for-the-badge)](https://docs.scvi-tools.org/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-4D6BFE?style=for-the-badge)](https://www.deepseek.com/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://www.mysql.com/)

<br/>

**主打简单 · 不挑模型 · 全程可观测**

<sub>[English](README.md) · 中文</sub>

</div>

---

## 🧭 介绍

RNAgent 只做一件事：

> **让你不用成为流水线工程师，也能把单细胞分析模型用起来。**

不用写脚本串联、不用调参数、不用考古环境依赖。一份数据进，一份 HTML 报告出：

```text
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ ① 选个数据集 │ ───▶ │ ② 选个模型   │ ───▶ │ ③ 看报告     │
└──────────────┘      └──────────────┘      └──────────────┘
```

这份「简单」的背后，是一条 **LangGraph** 三阶段流水线（预处理 → 模型训练 → 结果分析），凡是能自动决定的它都自己决定了：

| 你不需要做的事 | RNAgent 替你做的事 |
|:---|:---|
| 配参数 | 探测 GPU / 显存 / 数据规模，自动决定 batch_size、训练轮数与设备 |
| 清洗元数据 | 按**判定契约**校验，自动修复列名错配与多级标签（如 `T cell:CD4`） |
| 判断看哪些指标 | 跑统一评测，并逐项解释每个数值的含义与好坏 |
| 写报告 | DeepSeek 生成自包含 HTML 报告（未配置 LLM 时回退本地模板） |
| 为新模型重搭流水线 | 模型插件只需 **2 个文件**，API / DB / 前端 / 报告自动适配 |

### 🧩 支持的模型

| 模型 | 擅长 | 数据要求 |
|:---|:---|:---|
| `scLinformer` | 通用表征学习（内部自带归一化） | `adata.X` 为原始计数 |
| `scVI` | 批次校正与低维表征 | `layers["counts"]` 为原始计数 |
| *更多接入中* | 见下方 **未来计划** | — |

> 任何模型只要把嵌入写进 `obsm["latent"]`，就自动获得同一套评测、同一种报告格式和一个前端入口 —— 因此**不同模型的结果可以横向对比**。

---

## 🖼️ 结果展示

### 1 · 系统架构

<p align="center">
  <img src="imgs/a.png" alt="RNAgent 系统架构" width="940"/>
</p>

<p align="center"><sub>分层设计：<b>用户层</b> → <b>接入层 / 后台服务</b> → <b>LangGraph 核心编排</b>（含失败路由）→ <b>工具层 + 技能知识库</b> → <b>存储层</b>；MySQL、DeepSeek 与模型运行时为外部依赖。</sub></p>

### 2 · Web 界面

<p align="center">
  <img src="imgs/bg.png" alt="RNAgent 页面" width="940"/>
</p>

<p align="center"><sub><b>左</b> 选择模型并启动任务 · <b>中</b> 三阶段进度 + SSE 实时日志时间线 · <b>右</b> 直接向 AI 助手询问任务状态与报告。</sub></p>

模型可用性在运行时探测 —— 未安装的模型直接**置灰**，而不是跑到一半才失败。参数默认全部自动决策，不填任何配置也能出结果。

### 3 · 报告 · 模型评测

<p align="center">
  <img src="imgs/report1.png" alt="报告指标部分" width="940"/>
</p>

<p align="center"><sub>报告开头即给出运行上下文（<b>模型 · 细胞数 · 基因数 · 判定状态 · 设备</b>），随后是综合指标条形图与逐项解读表 —— 每个指标都标注了<b>解读方向</b>并对本次表现给出评级（<i>优秀 / 中等</i>）。</sub></p>

### 4 · 报告 · 批次整合与 UMAP

<p align="center">
  <img src="imgs/report2.png" alt="报告 UMAP 部分" width="940"/>
</p>

<p align="center"><sub>批次指标以卡片呈现，并自动生成一句话结论；下方并排给出按 <code>cell_type</code> 与按 <code>batch</code> 着色的 UMAP —— 批次图中各样本细胞已<b>充分混合</b>，正对应 <code>Batch_ASW = 0.9574</code>。</sub></p>

---

## 🚀 使用

### 环境要求

| | |
|:---|:---|
| **Python** | ≥ 3.10 |
| **MySQL** | 8.0 |
| **GPU** | 可选 —— 支持 CPU，但推荐 `torch` + CUDA |
| **Docker** | 可选 —— 用于预处理兜底修复的 `execute_code` 沙箱 |
| **DeepSeek Key** | 可选 —— 未配置时报告回退本地模板 |

### 安装与启动

```bash
# 1. 克隆
git clone https://github.com/TRCDHH/RNAgent.git
cd z-newRnagent

# 2. 环境（conda 或 venv 任选）
conda create -n rnagent python=3.10 -y && conda activate rnagent
# python -m venv .venv && .venv\Scripts\activate     # Windows
# python -m venv .venv && source .venv/bin/activate  # Linux / macOS

# 3. 安装依赖（版本已锁定为实测环境）
pip install -r requirements.txt
# GPU 用户（PyPI 在 Windows 上的 wheel 是 CPU-only）：
# pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 4. 初始化数据库
mysql -u root -p < init.sql

# 5. 配置环境变量
#    Windows (PowerShell)
$env:LLM_API_KEY="sk-xxx"
$env:SCLINFORMER_DIR="D:\code\RNAgent\model\scLinformer-main"
#    Linux / macOS
# export LLM_API_KEY=sk-xxx
# export SCLINFORMER_DIR=/opt/rnagent/model/scLinformer-main

# 6. 启动（必须单进程，切勿加 --workers）
python server.py
```

打开 **http://localhost:8000** → 添加数据集 → 选择模型 → **开始分析** → 观察实时日志 → 查看 `report.html`。

也可以用命令行直接发起：

```bash
curl -X POST http://localhost:8000/api/tasks/run \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id":1,"model":"scvi","params":{"max_epochs":100}}'
```

### 配置项

| 变量 | 默认值 | 说明 |
|:---|:---|:---|
| `LLM_API_KEY` | — | DeepSeek API Key（报告 + 助手） |
| `LLM_MODEL` | `deepseek-v4-flash` | LLM 模型名 |
| `LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容接口地址 |
| `DEFAULT_MODEL` | `sclinformer` | 请求未指定 `model` 时使用的模型 |
| `MODEL_EPOCHS` | `100` | 全局训练轮数（模型 `SKILL.md` 可单独覆盖） |
| `{模型KEY}_{参数名}` | 见各模型 `SKILL.md` | 按模型隔离地覆盖单个参数，如 `SCVI_MAX_EPOCHS=200` |
| `SCLINFORMER_DIR` | `<仓库>/../model/scLinformer-main` | scLinformer 源码目录 |
| `SANDBOX_IMAGE` / `SANDBOX_TIMEOUT` / `SANDBOX_MEM_MB` / `SANDBOX_CPUS` | `rna-sandbox:latest` / `300` / `4096` / `2` | `execute_code` 沙箱镜像与资源上限 |

参数优先级：**模型 `SKILL.md` 默认值 → `{模型KEY}_{参数名}` 环境变量 → 请求 `params` → 运行时自动决策。**

### 接口

| 方法 | 路径 | 用途 |
|:---:|:---|:---|
| `GET` | `/api/models` | 可用模型 **+ 参数 Schema**（前端据此渲染表单） |
| `GET` / `POST` / `DELETE` | `/api/datasets` | 数据集管理 |
| `POST` | `/api/tasks/run` | 启动任务 —— `{dataset_id, model, params}` |
| `GET` | `/api/tasks` · `/api/tasks/{id}/state` | 任务列表 · 单任务状态与结果 |
| `GET` | `/api/tasks/{id}/stream` | SSE 实时日志 |
| `GET` | `/output/{id}/report.html` | 查看报告 |
| `POST` | `/api/chat` · `/api/chat/stream` | AI 助手（普通 / 流式） |

### 产物

每个任务完全隔离：

```text
runtime/task/{task_id}/
├── report.html                 # LLM 生成的报告
├── processed_rna.h5ad
├── summary_metrics.csv · cluster_metrics.csv · batch_metrics.csv
├── umap_cell_type.png · umap_batch.png
├── model/                      # 模型权重 / scVI 存档
├── processed_data/             # train / valid / test 划分
└── 数据处理结果分析/            # 01_数据状况 · 02_数据分析报告 · 03_操作审计
```

### 接入你自己的模型

```text
app/tools/models/backends/<key>.py          # Capabilities + ParamSpec + train()
app/skills/registry/models/<key>/SKILL.md   # 参数默认值 + batch_size 决策表
```

两个文件。注册表自动发现，`GET /api/models` 自动暴露，前端按 Schema 自动渲染参数表单。

---

## 🔭 未来计划

> 两条主线是**模型**和**简单**，下面所有事情都服务于其中之一。

### 🚧 进行中

| | |
|:---|:---|
| **更多模型支持** | `scGPT` · `Geneformer` · `UCE` · `scFoundation` —— 插件层就是为这件事建的：一个模型 2 个文件，其它地方零改动。 |
| **部署流程简化** | 一键 `docker compose up`，把 MySQL、沙箱镜像和应用打包起来 —— 不用再手工装 MySQL / Docker、不用再对路径。 |

### 🗺️ 规划中

- **模型横向对比** —— 同一份数据集跑多个模型，指标并排比
- **参数预设一键复用** —— 把调好的配置存下来，换个数据集直接套用
- **任务队列与并发** —— 目前刻意保持单进程，后续引入 Redis 支持多 worker
- **登录与多用户** —— 账号、工作空间隔离与配额
- **可复现清单** —— 每个任务附带环境指纹，与报告一起归档

### ✅ 已完成

- [x] LangGraph 三阶段流水线 + 条件路由 + 错误处理
- [x] 模型插件架构 —— 内置 `scLinformer` 与 `scVI`
- [x] 判定契约 + 元数据自动修复（多级标签、列名错配）
- [x] 环境感知的参数与 batch_size 自动决策
- [x] 所有模型共用同一套评测
- [x] DeepSeek 生成 HTML 报告（含本地模板兜底）
- [x] SSE 实时日志 + 追加写 `events.jsonl` 审计留痕
- [x] Docker 隔离的 `execute_code` 沙箱
- [x] 依赖精确锁定（实测环境验证）

---

<div align="center">

### 🧬 RNAgent

**任何模型，任何数据集，一键出结果。**

<sub>LangGraph × DeepSeek × scLinformer / scVI × FastAPI</sub>

</div>
