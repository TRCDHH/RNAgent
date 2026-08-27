"""结果分析阶段：汇总前两阶段（预处理 + 模型运行）产物，生成最终 HTML 报告。

数据来源（模型最终留下的东西）：
- 预处理：output_dir/processed_rna.h5ad + 数据处理结果分析/{01_数据状况,02_数据分析报告,03_操作审计}.md + figures
- 训练：output_dir/summary_metrics.csv / cluster_metrics.csv / batch_metrics.csv
        output_dir/umap_cell_type.png / umap_batch.png
        output_dir/model/{rna_encoder,rna_decoder,cell_type_discriminator}.pth
        output_dir/processed_data/{train_ids,valid_ids,test_ids}.npy

分析什么、怎么解读，见 skill: skills/registry/report-generation/SKILL.md
"""

import json
import os
import re

from app.event_emitter import emit
from app.llm import get_llm, llm_configured
from app.tools.preprocess_tools import _json_safe

# scLinformer 评测产物文件名（与 utils.evaluate_sc_embedding 输出一致）
SUMMARY_CSV = "summary_metrics.csv"
CLUSTER_CSV = "cluster_metrics.csv"
BATCH_CSV = "batch_metrics.csv"
UMAP_CELL = "umap_cell_type.png"
UMAP_BATCH = "umap_batch.png"


# ---------------------------------------------------------------------------
# 产物读取
# ---------------------------------------------------------------------------
def _read_csv(csv_path):
    """读取评测 CSV：单行返回 dict，多行返回 list[dict]；不存在/空返回 None；NaN -> None。"""
    if not os.path.exists(csv_path):
        return None
    import math  # noqa: E402
    import pandas as pd  # noqa: E402

    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    rows = []
    for _, r in df.iterrows():
        d = {}
        for k in df.columns:
            try:
                v = float(r[k])
            except (TypeError, ValueError):
                v = r[k]
            if isinstance(v, float) and math.isnan(v):
                v = None
            d[str(k)] = v
        rows.append(d)
    return rows[0] if len(rows) == 1 else rows


def collect_report_context(output_dir, preprocess, training, dataset_id) -> dict:
    """汇总预处理 + 模型运行产物，返回结构化报告上下文（喂给 LLM / 本地模板）。"""
    preprocess = preprocess or {}
    training = training or {}

    meta = preprocess.get("metadata") or {}
    cfg = training.get("config") or {}
    env = training.get("env") or {}
    data = training.get("data") or {}

    dataset = {
        "dataset_id": dataset_id,
        "n_cells": meta.get("n_cells", data.get("n_cells")),
        "n_genes": meta.get("n_genes", data.get("n_genes")),
        "total_umi": meta.get("total_umi"),
        "sparsity": meta.get("sparsity"),
        "obs_columns": meta.get("obs_columns") or [],
    }

    summary = training.get("metrics") or _read_csv(os.path.join(output_dir, SUMMARY_CSV)) or {}
    cluster = _read_csv(os.path.join(output_dir, CLUSTER_CSV))
    batch = _read_csv(os.path.join(output_dir, BATCH_CSV)) or {}

    images = {}
    if os.path.exists(os.path.join(output_dir, UMAP_CELL)):
        images["cell_type"] = UMAP_CELL
    if os.path.exists(os.path.join(output_dir, UMAP_BATCH)):
        images["batch"] = UMAP_BATCH

    return _json_safe({
        "dataset": dataset,
        "data_condition": {
            "use_cell_type": cfg.get("use_cell_type"),
            "use_batch": cfg.get("use_batch"),
            "judge_status": preprocess.get("judge_status"),
        },
        "train_config": cfg,
        "env": env,
        "metrics": {"summary": summary, "cluster": cluster, "batch": batch},
        "images": images,
        "artifacts": {
            "model_dir": training.get("model_dir"),
            "processed_dir": training.get("processed_dir"),
            "processed_path": training.get("processed_path"),
            "analysis_dir": (preprocess.get("report") or {}).get("analysis_dir"),
        },
    })


# ---------------------------------------------------------------------------
# 生成 HTML（优先 LLM，失败回退本地模板）
# ---------------------------------------------------------------------------
def generate_report_html(dataset_id, output_dir, preprocess, training) -> str:
    """结果分析阶段入口：生成最终 HTML 报告（自包含，图片用相对路径引用）。"""
    context = collect_report_context(output_dir, preprocess, training, dataset_id)
    if llm_configured():
        html = _llm_report(context)
        if html:
            return html
    return _mock_html(context)


def _llm_report(context: dict) -> str:
    """让 LLM（DeepSeek）基于完整结果生成报告；失败返回空串。"""
    llm = get_llm()
    prompt = f"""你是单细胞 RNA 测序（scRNA-seq）分析助手。下面是本次 scLinformer 流水线的完整结果（JSON）。
请基于它生成一份完整、美观的中文 HTML 分析报告。

要求：
- 直接输出 HTML 代码（从 <!DOCTYPE html> 开始），不要输出任何解释性文字或 markdown 代码块标记。
- 报告至少覆盖：数据集概览、数据状况（有无 cell_type/batch、judge 结论）、训练配置与资源占用、
  模型评测（逐项解读指标 + UMAP 可视化）、结论与建议。
- 指标解读：ARI/AMI/NMI/HOM 越接近 1 聚类越好；Cell_ASW/Batch_ASW 越接近 1 越好；
  Graph_Connectivity 越接近 1 越好。
- UMAP 图用相对路径引用（<img src="umap_cell_type.png"> / <img src="umap_batch.png">），
  只引用 images 字段里存在的图片。
- 严格基于给定数据，指标为 null（无 cell_type/batch 而跳过）时，明确写「已跳过」，不要编造数值。
- 使用内联 CSS 美化，风格专业。

结果 JSON：
{json.dumps(context, ensure_ascii=False, default=str)}
"""
    try:
        raw = llm.invoke(prompt).content or ""
        html = _extract_html(raw)
        return html or ""
    except Exception as e:  # LLM 失败：回退本地模板，保证非空
        emit("analyze_warn", {"error": f"LLM 生成报告失败，回退本地模板：{e}"})
        return ""


def _extract_html(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:html)?\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# 本地兜底模板（无 LLM 或 LLM 失败时使用）
# ---------------------------------------------------------------------------
def _fmt(v):
    if v is None:
        return '<span style="color:#9ca3af">N/A（跳过）</span>'
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _kv_table(d):
    if not d:
        return "<p>—</p>"
    rows = "".join(
        f"<tr><th>{k}</th><td>{_fmt(v)}</td></tr>" for k, v in d.items()
    )
    return f'<table class="tbl">{rows}</table>'


def _list_table(lst):
    if not lst:
        return "<p>—</p>"
    keys = list(lst[0].keys())
    head = "".join(f"<th>{k}</th>" for k in keys)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(r.get(k))}</td>" for k in keys) + "</tr>" for r in lst
    )
    return f'<table class="tbl"><tr>{head}</tr>{body}</table>'


def _img_block(src, caption):
    return (
        f'<div class="img"><img src="{src}" alt="{caption}">'
        f'<p>{caption}</p></div>'
    )


def _mock_html(context: dict) -> str:
    ds = context.get("dataset") or {}
    cond = context.get("data_condition") or {}
    cfg = context.get("train_config") or {}
    env = context.get("env") or {}
    m = context.get("metrics") or {}
    images = context.get("images") or {}
    arts = context.get("artifacts") or {}

    imgs = ""
    if images.get("cell_type"):
        imgs += _img_block(images["cell_type"], "UMAP（按 cell_type 着色）")
    if images.get("batch"):
        imgs += _img_block(images["batch"], "UMAP（按 batch 着色）")
    if not imgs:
        imgs = "<p>无 UMAP 图。</p>"

    summary_rows = _kv_table(m.get("summary"))
    cluster_tbl = _list_table(m.get("cluster"))
    batch_rows = _kv_table(m.get("batch"))
    cond_rows = _kv_table(cond)
    cfg_rows = _kv_table(cfg)
    env_rows = _kv_table(env)

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>scLinformer 分析报告</title>
<style>
body{{font-family:-apple-system,'Segoe UI',sans-serif;margin:0;background:#f5f7fa;color:#111}}
.wrap{{max-width:960px;margin:0 auto;padding:32px 20px}}
h1{{color:#2563eb;margin-bottom:4px}} .sub{{color:#6b7280;margin-top:0}}
.card{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:20px;margin:16px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.card h2{{margin-top:0;color:#1f2937;font-size:17px;border-bottom:1px solid #f0f0f0;padding-bottom:8px}}
.tbl{{width:100%;border-collapse:collapse;font-size:14px}}
.tbl th,.tbl td{{border:1px solid #e5e7eb;padding:8px 10px;text-align:left}}
.tbl th{{background:#f9fafb;color:#374151}}
.img img{{max-width:100%;border:1px solid #e5e7eb;border-radius:8px}}
.img p{{color:#6b7280;font-size:13px;margin:6px 0}}
.note{{background:#fefce8;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;color:#92400e;font-size:13px}}
</style></head><body><div class="wrap">
<h1>scLinformer 单细胞分析报告</h1><p class="sub">数据集 #{ds.get('dataset_id')} · {ds.get('n_cells')} 细胞 × {ds.get('n_genes')} 基因</p>

<div class="card"><h2>数据集概览</h2>{_kv_table(ds)}</div>
<div class="card"><h2>数据状况</h2>{cond_rows}</div>
<div class="card"><h2>训练配置</h2>{cfg_rows}</div>
<div class="card"><h2>运行环境</h2>{env_rows}</div>
<div class="card"><h2>模型评测指标（summary）</h2>{summary_rows}</div>
<div class="card"><h2>聚类指标（按 resolution）</h2>{cluster_tbl}</div>
<div class="card"><h2>批次 / ASW 指标</h2>{batch_rows}</div>
<div class="card"><h2>UMAP 可视化</h2>{imgs}</div>
<div class="card"><h2>产物</h2>{_kv_table(arts)}</div>
<div class="note">本报告由本地模板生成（未配置 LLM_API_KEY 或 LLM 调用失败）。指标为 N/A 表示对应评测项因缺少 cell_type/batch 而跳过。</div>
</div></body></html>"""