"""结果分析阶段：汇总前两阶段（预处理 + 模型运行）产物，生成最终 HTML 报告。

数据来源（模型最终留下的东西）：
- 预处理：output_dir/processed_rna.h5ad + 数据处理结果分析/{01_数据状况,02_数据分析报告,03_操作审计}.md + figures
- 训练：output_dir/summary_metrics.csv / cluster_metrics.csv / batch_metrics.csv
        output_dir/umap_cell_type.png / umap_batch.png
        output_dir/model/{rna_encoder,rna_decoder,cell_type_discriminator}.pth
        output_dir/processed_data/{train_ids,valid_ids,test_ids}.npy

分析什么、怎么解读，见 skill: skills/registry/report-generation/SKILL.md
"""

import html
import json
import os
import re

from app.event_emitter import emit
from app.llm import get_llm, llm_configured
from app.tools.common import json_safe as _json_safe
from app.tools.models import get_backend

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

    # 模型信息（本次分析用的是哪个模型、参数是什么）——与具体模型无关，从注册表取
    model_key = training.get("model") or preprocess.get("model")
    backend = get_backend(model_key)
    model_info = {
        "key": backend.key,
        "name": training.get("model_name") or preprocess.get("model_name") or backend.name,
        "description": backend.description,
        "params": training.get("model_params") or preprocess.get("model_params") or {},
        "capabilities": backend.capabilities.to_dict(),
    }

    return _json_safe({
        "model": model_info,
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
# 固定版式渲染（模板来自 skill：registry/report-generation/report_template.html）
#
# 样式与章节结构完全由模板决定，只有数据（数值/表格/图片）与「结论与建议」文案是动态的，
# 因此不同任务生成的报告风格一致，不会因为 LLM 自由发挥而每次不同。
# ---------------------------------------------------------------------------
SKILL_NAME = "report-generation"
TEMPLATE_FILE = "report_template.html"

CAPABILITY_LABELS = {
    "needs_counts_layer": "需要 counts layer",
    "owns_preprocessing": "自带预处理",
    "supports_batch": "支持 batch 信息",
    "supports_celltype": "支持 cell_type 信息",
    "evaluates_internally": "内部评测",
    "embedding_key": "嵌入键",
}

METRIC_META = {
    "ARI": ("ARI（调整兰德指数）", "越接近 1 越好", "反映聚类划分与真实细胞类型标签的一致性"),
    "AMI": ("AMI（调整互信息）", "越接近 1 越好", "校正偶然性后，聚类与标签之间的共享信息"),
    "NMI": ("NMI（归一化互信息）", "越接近 1 越好", "未校正随机性的标签共享信息"),
    "HOM": ("HOM（同质性）", "越接近 1 越好", "衡量每个聚类簇内部的细胞类型纯度"),
    "Cell_ASW": ("Cell_ASW（细胞类型轮廓系数）", "越接近 1 越好",
                 "衡量嵌入空间中同类型细胞的聚集程度与类间边界清晰度"),
    "Batch_ASW": ("Batch_ASW（批次轮廓系数）", "越接近 1 越好",
                  "衡量批次混合程度，越高说明批次效应消除得越彻底"),
    "Graph_Connectivity": ("Graph_Connectivity（图连通性）", "越接近 1 越好",
                           "衡量 KNN 近邻图中各细胞群的连通性"),
}


def _esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=False)


def _num(v, nd: int = 4):
    """数值格式化：None 显示 N/A，float 保留 nd 位，int 加千分位。"""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return _esc(v)


def _level(v):
    """按取值返回 (样式 class, 中文档位)。"""
    if v is None:
        return "skip", "已跳过"
    if v >= 0.7:
        return "good", "优秀"
    if v >= 0.4:
        return "warn", "中等"
    if v >= 0.2:
        return "bad", "偏低"
    return "bad", "很低"


def _stat(label: str, value, note: str = None, small: bool = False, raw: bool = False) -> str:
    """一个统计卡片。raw=True 表示 value 已是 HTML（如徽章），不再转义。"""
    cls = "value small" if small else "value"
    note_html = f'<div class="note">{_esc(note)}</div>' if note else ""
    shown = str(value) if raw else _esc(value)
    return (f'<div class="stat"><div class="label">{_esc(label)}</div>'
            f'<div class="{cls}">{shown}</div>{note_html}</div>')


def _chip(text: str) -> str:
    return f'<span class="chip">{_esc(text)}</span>'


def _tags(items) -> str:
    items = [str(i) for i in (items or [])]
    return "".join(f'<span class="tag">{_esc(i)}</span>' for i in items) or "—"


def _metric_bar(name: str, v) -> str:
    cls, _ = _level(v)
    width = 0 if v is None else max(0.0, min(1.0, float(v))) * 100
    return (f'<div class="metric-row"><div class="metric-name">{_esc(name)}</div>'
            f'<div class="bar-track"><div class="bar-fill {cls}" style="width:{width:.1f}%;"></div></div>'
            f'<div class="metric-val">{_num(v)}</div></div>')


def _metric_row(name: str, v) -> str:
    cn, direction, clause = METRIC_META.get(name, (name, "越接近 1 越好", ""))
    cls, label = _level(v)
    badge_cls = {"good": "badge-high", "warn": "badge-mid", "bad": "badge-low"}.get(cls, "badge-skip")
    if v is None:
        text = '<span class="badge badge-skip">已跳过</span> 缺少对应标签，未计算。'
    else:
        # 结论不复述数值（数值已在左侧「数值」列），只给档位判断
        text = f'<span class="badge {badge_cls}">{label}</span> {_esc(clause)}，本次{label}。'
    return (f'<tr><td>{_esc(cn)}</td><td class="num">{_num(v)}</td>'
            f'<td>{_esc(direction)}</td><td>{text}</td></tr>')


def _bool_cn(v) -> str:
    return "是" if v else "否"


def _build_fragments(context: dict) -> dict:
    """把报告上下文渲染成一段段 HTML 片段（供模板占位符替换）。"""
    ds = context.get("dataset") or {}
    cond = context.get("data_condition") or {}
    cfg = context.get("train_config") or {}
    env = context.get("env") or {}
    m = context.get("metrics") or {}
    images = context.get("images") or {}
    arts = context.get("artifacts") or {}
    model = context.get("model") or {}

    backend = get_backend(model.get("key"))
    spec_labels = {sp.name: sp.label for sp in backend.param_specs}
    spec_descs = {sp.name: sp.desc for sp in backend.param_specs}

    summary = m.get("summary") or {}
    cluster = m.get("cluster") or []
    batch = m.get("batch") or {}

    # ---- Hero ----
    chips = "".join([
        _chip(f"模型：{model.get('name') or backend.name}"),
        _chip(f"细胞数：{_num(ds.get('n_cells'))}"),
        _chip(f"基因数：{_num(ds.get('n_genes'))}"),
        _chip(f"数据判定：{cond.get('judge_status') or 'N/A'}"),
        _chip(f"设备：{'CUDA / ' + env['gpu_name'] if env.get('gpu_available') and env.get('gpu_name') else ('CUDA' if env.get('gpu_available') else 'CPU')}"),
    ])

    # ---- 1. 模型 ----
    model_params = "".join(
        _stat(spec_labels.get(k, k), _num(v), note=k, small=not isinstance(v, int))
        for k, v in list(cfg.items())[:9]
    ) or _stat("参数", "无")
    caps = model.get("capabilities") or {}
    cap_rows = "".join(
        f"<tr><td>{_esc(CAPABILITY_LABELS.get(k, k))}</td><td>"
        f"{('<code>' + _esc(v) + '</code>') if k == 'embedding_key' else (_bool_cn(v) if isinstance(v, bool) else _esc(v))}"
        f"</td></tr>"
        for k, v in caps.items() if k != "artifacts"
    )
    artifact_tags = _tags(caps.get("artifacts"))
    artifact_paths = "<br>".join(
        f"{_esc(k)}：<code>{_esc(v)}</code>" for k, v in arts.items() if v
    ) or "—"

    # ---- 2. 数据集 ----
    sparsity = ds.get("sparsity")
    dataset_stats = "".join([
        _stat("细胞数（n_cells）", _num(ds.get("n_cells"))),
        _stat("基因数（n_genes）", _num(ds.get("n_genes"))),
        _stat("总 UMI（total_umi）", _num(ds.get("total_umi"))),
        _stat("稀疏度（sparsity）", _num(sparsity),
              note=f"约 {sparsity * 100:.2f}% 为零值" if isinstance(sparsity, (int, float)) else None),
    ])
    dataset_hint = (
        '<div class="hint">'
        + (f"规模 {int(ds['n_cells']):,} 细胞 × {int(ds['n_genes']):,} 基因"
           if ds.get("n_cells") and ds.get("n_genes") else "规模未知")
        + (f"，稀疏度 {sparsity:.1%}" if isinstance(sparsity, (int, float)) else "")
        + "。小样本下指标波动大，需结合 UMAP 判读。</div>"
    )
    obs_cols = ds.get("obs_columns") or []

    # ---- 3. 数据状况 ----
    use_ct, use_bt = cond.get("use_cell_type"), cond.get("use_batch")
    judge = cond.get("judge_status") or "N/A"
    cond_stats = "".join([
        _stat("cell_type 注释", "可用" if use_ct else "不可用",
              note="use_cell_type = true → 聚类指标正常计算" if use_ct else "use_cell_type = false → 聚类指标跳过",
              small=True),
        _stat("batch 信息", "可用" if use_bt else "不可用",
              note="use_batch = true → 批次整合指标正常计算" if use_bt else "use_batch = false → 批次整合指标跳过",
              small=True),
        _stat("判定状态（judge_status）",
              f'<span class="badge badge-pass">{_esc(judge)}</span>'
              if judge == "PASS" else f'<span class="badge badge-skip">{_esc(judge)}</span>',
              note="数据满足流水线评测前置条件" if judge == "PASS" else "存在 WARNING/FAIL 项，请查看 01_数据状况.md",
              small=True, raw=True),
    ])
    if use_ct and use_bt:
        cond_hint = ('<div class="hint"><code>cell_type</code> 与 <code>batch</code> 齐全，'
                     '全部指标已计算，无跳过项。</div>')
    else:
        missed = [n for n, ok in (("cell_type", use_ct), ("batch", use_bt)) if not ok]
        cond_hint = (f'<div class="warn-box">缺少 {" / ".join(missed)}，相关指标已跳过（N/A）。'
                     f'补齐字段即可获得完整评测。</div>')

    # ---- 4. 训练配置 ----
    train_rows = "".join(
        f"<tr><td>{_esc(k)}</td><td class=\"num\">{_esc(v)}</td><td>{_esc(spec_descs.get(k, ''))}</td></tr>"
        for k, v in cfg.items()
    ) or '<tr><td colspan="3">无</td></tr>'
    vram = env.get("vram_mb")
    env_stats = "".join([
        _stat("GPU 可用", "是" if env.get("gpu_available") else "否",
              note=f"device = {env.get('device')}", small=True),
        _stat("GPU 型号", env.get("gpu_name") or "—", small=True),
        _stat("显存（VRAM）", f"{vram:,} MB" if vram else "—",
              note=f"约 {vram / 1024:.0f} GB" if vram else None),
        _stat("PyTorch 版本", env.get("torch_version") or "—", small=True),
    ])
    env_hint = (
        '<div class="hint">运行于 '
        + ("GPU" + (f"（{vram:,} MB 显存）" if vram else "") if env.get("gpu_available") else "CPU")
        + (f" · PyTorch {env['torch_version']}" if env.get("torch_version") else "")
        + ("，资源非瓶颈。" if env.get("gpu_available") else "，训练速度明显慢于 GPU。")
        + "</div>"
    )

    # ---- 5. 评测 ----
    summary_bars = "".join(_metric_bar(k, summary.get(k)) for k in METRIC_META) or "<p>无指标。</p>"
    metric_rows = "".join(_metric_row(k, summary.get(k)) for k in METRIC_META)

    best = None
    for r in cluster:
        if isinstance(r, dict) and isinstance(r.get("ARI"), (int, float)):
            if best is None or r["ARI"] > best["ARI"]:
                best = r
    cluster_rows = ""
    for r in cluster:
        if not isinstance(r, dict):
            continue
        strong = best is not None and r.get("resolution") == best.get("resolution")
        style = ' style="background:#f2f8fe;"' if strong else ""
        cells = "".join(
            f'<td class="num">{("<strong>" + _num(r.get(k)) + "</strong>") if strong else _num(r.get(k))}</td>'
            for k in ("ARI", "AMI", "NMI", "HOM")
        )
        cluster_rows += (f'<tr{style}><td class="num">'
                         f'{("<strong>" + _num(r.get("resolution")) + "</strong>") if strong else _num(r.get("resolution"))}'
                         f"</td>{cells}</tr>")
    if not cluster_rows:
        cluster_rows = '<tr><td colspan="5">无 cell_type，跳过聚类评测。</td></tr>'
    cluster_hint = (
        f'<div class="hint">共 {len(cluster)} 档分辨率，最优 ARI 在 '
        f'resolution = {best.get("resolution")}（粗体行）；各档均接近 0 时，'
        f'问题在嵌入质量。</div>'
        if best else ""
    )

    batch_stats = "".join(
        _stat(k, _num(batch.get(k)), note=METRIC_META.get(k, (k, "", ""))[2])
        for k in ("Cell_ASW", "Batch_ASW", "Graph_Connectivity")
    )
    b_asw, c_asw = batch.get("Batch_ASW"), batch.get("Cell_ASW")
    gc = batch.get("Graph_Connectivity")
    _bh = []
    if b_asw is None and gc is None:
        _bh.append("批次评测已跳过")
    else:
        _bh.append("批次整合良好" if (b_asw or 0) >= 0.7 else "批次仍有分离")
        if (gc or 0) >= 0.9:
            _bh.append("图结构连通")
    if c_asw is not None:
        _bh.append("细胞类型分离不足" if c_asw < 0.6 else "细胞类型分离良好")
    batch_hint = f'<div class="hint">{"，".join(_bh)}。</div>'

    img_blocks = []
    if images.get("cell_type"):
        img_blocks.append(
            f'<div class="img-wrap"><img src="{_esc(images["cell_type"])}" alt="UMAP 按细胞类型着色">'
            f'<div class="img-caption">图 1：UMAP 嵌入按 <code>cell_type</code> 着色</div></div>')
    if images.get("batch"):
        img_blocks.append(
            f'<div class="img-wrap"><img src="{_esc(images["batch"])}" alt="UMAP 按批次着色">'
            f'<div class="img-caption">图 2：UMAP 嵌入按 <code>batch</code> 着色</div></div>')
    umap_images = "".join(img_blocks) or (
        '<div class="img-wrap"><div class="img-caption">本次未生成 UMAP 图（缺少 cell_type / batch 列）。</div></div>')
    if images.get("cell_type") and images.get("batch"):
        umap_hint = '<div class="hint">左图看细胞类型分离，右图看批次是否混合。</div>'
    elif images.get("cell_type"):
        umap_hint = '<div class="hint">观察细胞类型在二维嵌入中的分离程度。</div>'
    elif images.get("batch"):
        umap_hint = '<div class="hint">观察各批次是否均匀混合。</div>'
    else:
        umap_hint = ""

    # 综合判断：一句话定性，不复述数字、不重复结论与建议
    _w = []
    if b_asw is None and gc is None:
        _w.append("批次评测已跳过")
    else:
        _w.append("批次整合优秀" if (b_asw or 0) >= 0.7 else "批次整合一般")
    ari = summary.get("ARI")
    if ari is None:
        _w.append("聚类评测已跳过")
    else:
        _w.append("聚类一致性良好" if ari >= 0.5 else "聚类一致性偏弱")
    _ep = cfg.get("epochs") or cfg.get("max_epochs")
    if isinstance(_ep, int) and _ep < 20 and ari is not None:
        _w.append("疑与训练轮数不足有关")
    warn = f'<div class="warn-box"><strong>综合判断：</strong>{"，".join(_w)}。</div>'

    conclusions, suggestions = _narrative(context)

    return {
        "PAGE_TITLE": f"{model.get('name') or '单细胞'} 单细胞 RNA 测序分析报告",
        "HERO_TITLE": f"{model.get('name') or '单细胞'} 单细胞 RNA 测序分析报告",
        "HERO_SUBTITLE": "单细胞表征学习流水线 · 聚类与批次整合评测",
        "CHIPS": chips,
        "MODEL_NAME": model.get("name") or backend.name,
        "MODEL_DESC": model.get("description") or backend.description,
        "MODEL_PARAMS": model_params,
        "CAPABILITY_ROWS": cap_rows,
        "ARTIFACT_TAGS": artifact_tags,
        "ARTIFACT_PATHS": artifact_paths,
        "DATASET_STATS": dataset_stats,
        "DATASET_HINT": dataset_hint,
        "OBS_COUNT": len(obs_cols),
        "OBS_TAGS": _tags(obs_cols),
        "CONDITION_STATS": cond_stats,
        "CONDITION_HINT": cond_hint,
        "TRAIN_ROWS": train_rows,
        "ENV_STATS": env_stats,
        "ENV_HINT": env_hint,
        "SUMMARY_BARS": summary_bars,
        "METRIC_ROWS": metric_rows,
        "SUMMARY_WARN": warn,
        "CLUSTER_ROWS": cluster_rows,
        "CLUSTER_HINT": cluster_hint,
        "BATCH_STATS": batch_stats,
        "BATCH_HINT": batch_hint,
        "UMAP_IMAGES": umap_images,
        "UMAP_HINT": umap_hint,
        "CONCLUSION_LIST": "".join(f"<li>{c}</li>" for c in conclusions),
        "SUGGESTION_LIST": "".join(f"<li>{s}</li>" for s in suggestions),
        "FOOTER": ("本报告由 RNAgent 流水线自动生成 · 版式由 skills/registry/report-generation/"
                   "report_template.html 固定；所有指标直接取自流水线产物，未做任何推测性修改"),
    }


def _narrative(context: dict) -> tuple:
    """生成「结论 / 建议」文案：优先 LLM，失败或不可用时用规则兜底（保证结构一致）。"""
    llm_pairs = _llm_narrative(context) if llm_configured() else None
    if llm_pairs:
        return llm_pairs
    return _rule_narrative(context)


def _llm_narrative(context: dict):
    """让 LLM 只写「结论 / 建议」两条列表（不产出 HTML），返回 (conclusions, suggestions)。"""
    llm = get_llm()
    model = context.get("model") or {}
    prompt = f"""你是单细胞 RNA 测序分析助手。下面是本次分析的结构化结果（JSON）。

只输出「结论」与「建议」的纯文字，不要 HTML、不要 CSS、不要任何解释：

###CONCLUSIONS###
- ...
###SUGGESTIONS###
- ...

文案规范（严格遵守，总原则：简洁高效，一屏看完）：
- 结论 3-5 条、建议 3-5 条，每条一行、以 "- " 开头。
- 每条不超过 40 字，一句话讲完，不换行。
- 不复述表格里已有的数字，直接给判断结论。
- 不使用「综上所述」「值得注意的是」「由此可见」等连接词。
- 同一件事只说一次，不重复其他章节的内容。
- 建议必须可执行：写清动作与对象（调什么参数、做什么对比）。
- 指标为 null 表示已跳过，就写「已跳过」，绝不编造数值。
- 中文陈述句，不用疑问句、不用感叹号。

指标方向：ARI/AMI/NMI/HOM 越接近 1 越好；Cell_ASW/Batch_ASW/Graph_Connectivity 越接近 1 越好。

结果 JSON：
{json.dumps(context, ensure_ascii=False, default=str)}
"""
    try:
        raw = llm.invoke(prompt).content or ""
    except Exception as e:
        emit("analyze_warn", {"error": f"LLM 生成结论失败：{e}"})
        return None

    parts = re.split(r"###CONCLUSIONS###|###SUGGESTIONS###", raw)
    if len(parts) < 3:
        return None

    def _lines(block: str):
        out = []
        for ln in block.splitlines():
            ln = ln.strip().lstrip("-•").strip()
            if ln:
                out.append(html.escape(ln, quote=False))
        return out

    conclusions, suggestions = _lines(parts[1]), _lines(parts[2])
    if not conclusions and not suggestions:
        return None
    return conclusions, suggestions


def _rule_narrative(context: dict) -> tuple:
    """无 LLM 时的规则化结论与建议（遵循「简洁高效」规范：每条 ≤ 40 字、不复述数字）。"""
    cfg = context.get("train_config") or {}
    model = context.get("model") or {}
    summary = (context.get("metrics") or {}).get("summary") or {}
    cond = context.get("data_condition") or {}

    name = _esc(model.get("name") or "模型")
    epochs = cfg.get("epochs") or cfg.get("max_epochs")
    bs = cfg.get("batch_size")
    b_asw, c_asw, ari = summary.get("Batch_ASW"), summary.get("Cell_ASW"), summary.get("ARI")

    train_desc = "，".join(
        [f"{epochs} epoch" if epochs else "", f"batch {bs}" if bs else ""]
    ).strip("，") or "配置见报告第五章"
    conclusions = [f"<strong>{name}</strong> 训练 {train_desc}。"]

    if cond.get("use_cell_type") and cond.get("use_batch"):
        conclusions.append("cell_type 与 batch 齐全，全部指标已计算。")
    else:
        missed = [n for n, ok in (("cell_type", cond.get("use_cell_type")),
                                  ("batch", cond.get("use_batch"))) if not ok]
        conclusions.append(f"缺少 {' / '.join(missed)}，相关指标已跳过。")

    if b_asw is not None:
        conclusions.append("批次整合良好。" if b_asw >= 0.7 else "批次间仍有分离。")
    if ari is not None:
        conclusions.append("聚类与真实标签一致性良好。" if ari >= 0.5 else "聚类与真实标签一致性偏弱。")
    if (cond.get("judge_status") or "").upper() not in ("", "PASS"):
        conclusions.append(f"数据判定为 {_esc(cond.get('judge_status'))}，建议先处理告警项。")

    suggestions = []
    ep = epochs if isinstance(epochs, int) else None
    if ep is not None and ep < 20:
        suggestions.append("提高 epochs 至 50 以上，对比 ARI 是否上升。")
    if ari is not None and ari < 0.5:
        suggestions.append("核查 HVG 数量与归一化方式是否匹配数据稀疏度。")
    if (b_asw or 0) >= 0.9 and (c_asw is not None and c_asw < 0.5):
        suggestions.append("做 use_batch 消融实验，排查是否过度去除生物信号。")
    if not cond.get("use_batch"):
        suggestions.append("补齐 batch 列，以评估去批次效果。")
    if not cond.get("use_cell_type"):
        suggestions.append("补齐 cell_type 注释，以启用聚类评测。")
    suggestions.append("与 PCA / Harmony 基线对比，确认模型增益。")

    return conclusions, suggestions[:5]


def generate_report_html(dataset_id, output_dir, preprocess, training) -> str:
    """结果分析阶段入口：用固定模板渲染 HTML 报告（自包含，图片用相对路径引用）。"""
    context = collect_report_context(output_dir, preprocess, training, dataset_id)
    html_str = _render_template(context)
    if html_str:
        return html_str
    return _mock_html(context)


def _render_template(context: dict) -> str:
    """读取 skill 中的 HTML 模板并注入数据；模板缺失/渲染失败返回空串（调用方回退本地模板）。"""
    from app.skills.library import get_library

    template = get_library().load_asset(SKILL_NAME, TEMPLATE_FILE)
    if not template:
        emit("analyze_warn", {"error": f"未找到报告模板 {SKILL_NAME}/{TEMPLATE_FILE}，回退本地模板"})
        return ""
    try:
        fragments = _build_fragments(context)
    except Exception as e:
        emit("analyze_warn", {"error": f"报告片段渲染失败，回退本地模板：{e}"})
        return ""

    for key, value in fragments.items():
        template = template.replace("{{" + key + "}}", str(value))
    # 清理未替换的占位符，避免把 {{XXX}} 直接暴露给用户
    return re.sub(r"\{\{[A-Z_]+\}\}", "—", template)


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
    model = context.get("model") or {}
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
    model_rows = _kv_table({
        "模型": model.get("name"),
        "标识": model.get("key"),
        "说明": model.get("description"),
        **({"用户参数": json.dumps(model.get("params") or {}, ensure_ascii=False)}
           if model.get("params") else {}),
    })
    model_name = model.get("name") or "模型"

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{model_name} 分析报告</title>
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
<h1>{model_name} 单细胞分析报告</h1><p class="sub">数据集 #{ds.get('dataset_id')} · {ds.get('n_cells')} 细胞 × {ds.get('n_genes')} 基因</p>

<div class="card"><h2>所用模型</h2>{model_rows}</div>
<div class="card"><h2>数据集概览</h2>{_kv_table(ds)}</div>
<div class="card"><h2>数据状况</h2>{cond_rows}</div>
<div class="card"><h2>训练配置</h2>{cfg_rows}</div>
<div class="card"><h2>运行环境</h2>{env_rows}</div>
<div class="card"><h2>模型评测指标（summary）</h2>{summary_rows}</div>
<div class="card"><h2>聚类指标（按 resolution）</h2>{cluster_tbl}</div>
<div class="card"><h2>批次 / ASW 指标</h2>{batch_rows}</div>
<div class="card"><h2>UMAP 可视化</h2>{imgs}</div>
<div class="card"><h2>产物</h2>{_kv_table(arts)}</div>
<div class="note">本报告由本地模板生成（未配置 LLM_API_KEY 或 LLM 调用失败）。指标为 N/A 表示对应评测项因缺少 cell_type/batch 而跳过。所有模型共用同一套评测（evaluate_sc_embedding），指标可直接横向对比。</div>
</div></body></html>"""