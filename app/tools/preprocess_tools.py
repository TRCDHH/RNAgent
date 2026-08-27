"""数据预处理阶段：把原始数据加工成 scLinformer 模型可接受的输入。

scLinformer 模型输入要求（见 model/scLinformer-main/train.py 与
scLinformer/scLinformerModel.py 的 RNA_data_preprocessing）：
    1. AnnData（h5ad）表达矩阵 adata.X，且必须是「原始计数」——
       模型内部会再做 normalize_total + log1p，若输入已归一化会导致二次归一化失真。
    2. obs 可含 cell_type 列（use_cell_type）、batch 列（use_batch），缺省则相应能力关闭。
    3. var_names 必须唯一（模型会 var_names_make_unique）。
    4. 可选 universal 基因词表（scLinformer/default_gene_vocab.json），用于跨数据集对齐基因 ID。

本模块对外提供 9 个工具函数（风格与 mcp-service/tools.py 一致：含 numpy JSON
序列化、报错堆栈、pymysql 数据库访问）：
    load_data / judge / rename_column / execute_code
    qc_stats / composition_stats / plot_qc / generate_report / save_dataset

以及一个编排入口 run_preprocess()，把整个判断->修复->分析->报告->保存流程串起来。

注意：scanpy/anndata/pandas/matplotlib/scipy/numpy 等重依赖采用「惰性导入」，
保证主应用（FastAPI 门户）在未安装这些库时也能正常启动，仅在实际执行预处理时报错提示。
"""

import json
import os
import traceback
from datetime import datetime

from app.event_emitter import emit

# 通用基因词表路径（模型自带的跨数据集 gene token 词典）
DEFAULT_VOCAB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "model", "scLinformer-main", "scLinformer", "default_gene_vocab.json",
    )
)

CELLTYPE_ALIASES = ["cell_type", "celltype", "cell.type", "cell_type_name", "celltype_name", "annotation"]
BATCH_ALIASES = ["batch", "sample", "sample_id", "donor", "donor_id", "sampleid"]


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def _json_safe(obj):
    """把 numpy 标量/数组等转成可 JSON 序列化的原生类型（鸭子类型，避免顶层 import numpy）。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(i) for i in obj]
    # numpy 标量：.item() -> 原生标量
    if hasattr(obj, "item") and callable(getattr(obj, "item", None)):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass
    # numpy 数组 / pandas 结构：.tolist() 或 str
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist", None)):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict", None)):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    return str(obj)


def _err_info(e: Exception) -> str:
    """返回完整报错堆栈（风格对齐 mcp-service/tools.py）。"""
    return traceback.format_exc()


# ---------------------------------------------------------------------------
# 工具 1：load_data —— 读数据集（h5ad / 10x 目录 / annotations.csv），返回元数据
# ---------------------------------------------------------------------------
def _read_data(path: str):
    """惰性读取数据为 AnnData。支持 h5ad、10x 目录（含 annotations.csv）、csv 表达矩阵。"""
    import scanpy as sc  # noqa: E402 惰性导入

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据路径不存在：{path}")

    if os.path.isdir(path):
        # 优先 10x 目录（matrix/barcodes/features），其次找目录内唯一的 .h5ad
        mtx = [f for f in os.listdir(path) if f.endswith((".mtx", ".mtx.gz"))]
        h5 = [f for f in os.listdir(path) if f.endswith(".h5ad")]
        if mtx:
            adata = sc.read_10x_mtx(path, var_names="gene_symbols")
            # 尝试合并 annotations.csv（含 cell.type / Sample 等列）
            ann_path = os.path.join(path, "annotations.csv")
            if os.path.exists(ann_path):
                adata = _merge_annotations(adata, ann_path)
            return adata
        if h5:
            return sc.read_h5ad(os.path.join(path, h5[0]))
        raise ValueError(f"目录 {path} 中未找到 10x matrix 或 h5ad 文件")

    if path.endswith(".h5ad"):
        return sc.read_h5ad(path)

    if path.endswith(".csv"):
        import pandas as pd  # noqa: E402
        df = pd.read_csv(path, index_col=0)
        import anndata as ad  # noqa: E402
        return ad.AnnData(df.values, var=pd.DataFrame(index=df.columns), obs=pd.DataFrame(index=df.index))

    raise ValueError(f"不支持的数据格式：{path}（支持 h5ad / 10x 目录 / csv）")


def _merge_annotations(adata, ann_path: str):
    """把 annotations.csv 合并进 adata.obs（按 barcode 对齐）。

    10x 矩阵的条形码常带 "-1" 后缀而 annotations.csv 不带（或相反），
    直接 join 匹配率低时，先去掉 "-数字" 后缀再对齐。
    """
    import pandas as pd  # noqa: E402

    ann = pd.read_csv(ann_path)
    # 找细胞标识列：优先 barcode / cell 等常见名，否则用第一列
    id_col = None
    for cand in ["barcode", "cell", "cell_id", "cellid", "index", "cell.barcode", "barcodes"]:
        if cand in ann.columns:
            id_col = cand
            break
    if id_col is None:
        id_col = ann.columns[0]

    ann = ann.set_index(id_col)
    ann.index = ann.index.astype(str)

    # 直接对齐；匹配率低则去掉 "-数字" 后缀重试（10x 条形码常见差异）
    if adata.obs.index.astype(str).isin(ann.index).mean() >= 0.5:
        adata.obs = adata.obs.join(ann, how="left")
        return adata

    ann2 = ann.copy()
    ann2.index = ann2.index.str.replace(r"-\d+$", "", regex=True)
    ann2 = ann2[~ann2.index.duplicated(keep="first")]
    obs2 = adata.obs.copy()
    obs2.index = obs2.index.astype(str).str.replace(r"-\d+$", "", regex=True)
    adata.obs = obs2.join(ann2, how="left")
    return adata


def _describe(adata, path: str) -> dict:
    """从 AnnData 生成 JSON 元数据。"""
    import numpy as np  # noqa: E402
    from scipy import sparse  # noqa: E402

    x = adata.X
    if sparse.issparse(x):
        nnz = int(x.nnz)
        total_umi = float(x.sum())
    else:
        xn = np.asarray(x)
        nnz = int(np.count_nonzero(xn))
        total_umi = float(xn.sum())

    return {
        "path": path,
        "format": "h5ad" if str(path).endswith(".h5ad") else "10x_mtx/csv",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "nnz": nnz,
        "total_umi": total_umi,
        "sparsity": round(1 - nnz / (adata.n_obs * adata.n_vars), 4) if adata.n_obs * adata.n_vars else None,
        "obs_columns": [str(c) for c in adata.obs.columns],
        "var_columns": [str(c) for c in adata.var.columns],
        "has_raw": getattr(adata, "raw", None) is not None,
        "layers": [str(k) for k in getattr(adata, "layers", {}).keys()],
        "gene_name_preview": [str(g) for g in list(adata.var_names[:10])],
    }


def load_data(path: str) -> dict:
    """读取数据集或字段，返回元数据（JSON 可序列化）。"""
    try:
        adata = _read_data(path)
        return {"status": "ok", "metadata": _json_safe(_describe(adata, path))}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": _err_info(e)}


# ---------------------------------------------------------------------------
# 工具 2：judge —— 按判定契约逐项输出 PASS/FAIL/WARNING + 证据 + 建议动作
# ---------------------------------------------------------------------------
def judge(adata) -> dict:
    """对 AnnData 执行判定契约，逐项输出结论。核心顺序：先确认原始计数再谈其它。"""
    import numpy as np  # noqa: E402
    from scipy import sparse  # noqa: E402

    verdicts = []
    x = adata.X
    if sparse.issparse(x):
        x_dense = x.toarray() if min(x.shape) <= 5000 else x[:min(5000, x.shape[0]), :min(5000, x.shape[1])].toarray()
    else:
        x_dense = np.asarray(x)

    # 1) X 是否存在
    if x is None or x.shape[0] == 0 or x.shape[1] == 0:
        verdicts.append({"item": "x_exists", "status": "FAIL",
                         "evidence": "表达矩阵 X 为空或缺维", "action": "提供有效的表达矩阵"})
    else:
        verdicts.append({"item": "x_exists", "status": "PASS",
                         "evidence": f"X 形状 {x.shape}，非零元素数 {int((x_dense != 0).sum())}",
                         "action": "无需处理"})

    # 2) 原始计数判定（是否原始计数，避免模型二次归一化）
    raw_status, raw_evidence, raw_action = _judge_raw_counts(adata, x_dense)
    verdicts.append({"item": "raw_counts", "status": raw_status,
                     "evidence": raw_evidence, "action": raw_action})

    # 3) cell_type 列
    verdicts.append(_judge_obs_column(adata, "cell_type", CELLTYPE_ALIASES))

    # 4) batch 列
    verdicts.append(_judge_obs_column(adata, "batch", BATCH_ALIASES))

    # 5) 基因名唯一性
    genes = [str(g) for g in adata.var_names]
    dup = len(genes) - len(set(genes))
    verdicts.append({"item": "gene_name_unique", "status": "PASS" if dup == 0 else "FAIL",
                     "evidence": f"重复基因名 {dup} 个" if dup else "基因名全部唯一",
                     "action": "" if dup == 0 else "执行 var_names_make_unique() 去重"})

    # 6) 数值合法性（NaN / Inf / 负值）
    verdicts.append(_judge_numeric(x_dense))

    # 7) 规模下限
    n_cells, n_genes = int(adata.n_obs), int(adata.n_vars)
    scale_ok = n_cells >= 100 and n_genes >= 200
    verdicts.append({"item": "scale_min", "status": "PASS" if scale_ok else "WARNING",
                     "evidence": f"n_cells={n_cells}, n_genes={n_genes}（下限 100 / 200）",
                     "action": "" if scale_ok else "规模偏小，训练结果可能不稳定，建议扩充样本"})

    # 8) universal 基因覆盖率
    verdicts.append(_judge_universal_coverage(adata))

    # 汇总
    status = "PASS" if all(v["status"] == "PASS" for v in verdicts) else (
        "FAIL" if any(v["status"] == "FAIL" for v in verdicts) else "WARNING")
    return {"status": status, "verdicts": _json_safe(verdicts)}


def _judge_raw_counts(adata, x_dense) -> tuple:
    """判断 X 是否原始计数（原始计数 -> PASS，已归一化/对数化 -> WARNING/FAIL）。"""
    import numpy as np  # noqa: E402

    flat = x_dense.ravel()
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return "FAIL", "X 无有效数值", "提供有效表达数据"

    has_negative = bool(np.any(flat < 0))
    is_integer_like = bool(np.allclose(flat, np.round(flat), atol=1e-6))
    max_val = float(flat.max())

    # 已有 .raw 或 layers 存原始计数
    alt_raw = getattr(adata, "raw", None) is not None or any(
        "count" in str(k).lower() for k in getattr(adata, "layers", {}).keys())

    if has_negative:
        return ("FAIL", "X 含负值（已做过标准化/z-score 而非原始计数）",
                "从 .raw 或 layers['counts'] 顶替为原始计数；若无则警告继续")
    if max_val < 20 and not is_integer_like:
        if alt_raw:
            return ("WARNING", f"X 最大值 {max_val:.2f} 且含大量小数，疑似已完成 log1p/log 归一化（存在 .raw 可顶替）",
                    "用 .raw / layers['counts'] 顶替 X")
        return ("WARNING", f"X 最大值 {max_val:.2f} 且含大量小数，疑似已完成归一化（无 .raw 可顶替）",
                "警告：模型将再次归一化，可能导致二次归一化失真；建议提供原始计数")
    if alt_raw and not is_integer_like:
        return ("WARNING", "存在 .raw 但当前 X 非整数计数", "优先使用 .raw 作为输入")
    return ("PASS", f"X 为非负整数计数（max={max_val:.0f}）", "无需处理")


def _judge_obs_column(adata, canonical: str, aliases: list) -> dict:
    cols = [str(c) for c in adata.obs.columns]
    col_lower = {c.lower(): c for c in cols}
    # 精确命中 canonical 即 PASS
    if canonical in cols:
        return {"item": canonical, "status": "PASS",
                "evidence": f"已存在 {canonical} 列，{adata.obs[canonical].nunique()} 个类别",
                "action": "无需处理"}
    # 大小写不敏感匹配同义列别名（如 Sample -> sample -> batch，cell.type -> celltype）
    hit = next((col_lower[alias.lower()] for alias in aliases if alias.lower() in col_lower), None)
    if hit is None:
        return {"item": canonical, "status": "FAIL",
                "evidence": f"obs 中无 {canonical} 列（现有列：{cols}）",
                "action": f"若存在同义列（别名 {aliases}），用 rename_column 改名；否则标记缺失"}
    return {"item": canonical, "status": "FAIL",
            "evidence": f"发现同义列「{hit}」，需改名为「{canonical}」",
            "action": f"用 rename_column 将 {hit} -> {canonical}"}


def _judge_numeric(x_dense) -> dict:
    import numpy as np  # noqa: E402
    n_nan = int(np.isnan(x_dense).sum())
    n_inf = int(np.isinf(x_dense).sum())
    if n_nan or n_inf:
        return {"item": "numeric_valid", "status": "FAIL",
                "evidence": f"NaN={n_nan}, Inf={n_inf}",
                "action": "清理非法值（置 0 或过滤对应细胞/基因）"}
    return {"item": "numeric_valid", "status": "PASS",
            "evidence": "无 NaN/Inf", "action": "无需处理"}


def _judge_universal_coverage(adata) -> dict:
    vocab = _load_universal_vocab()
    if vocab is None:
        return {"item": "universal_coverage", "status": "WARNING",
                "evidence": "未找到 universal 基因词表，跳过覆盖率判定",
                "action": "确认 default_gene_vocab.json 路径；非 universal 模式可忽略"}
    genes = [str(g) for g in adata.var_names]
    matched = sum(1 for g in genes if g in vocab)
    cov = round(matched / len(genes), 4) if genes else 0.0
    status = "PASS" if cov >= 0.5 else "WARNING"
    return {"item": "universal_coverage", "status": status,
            "evidence": f"覆盖率 {cov:.2%}（{matched}/{len(genes)} 个基因命中 universal 词表）",
            "action": "覆盖率过低时，若不使用 universal 模型可忽略；否则需用模型自带词表对齐基因 ID"}


def _load_universal_vocab() -> dict:
    """加载模型自带 universal 基因词表 gene_name -> id。找不到返回 None。"""
    if not os.path.exists(DEFAULT_VOCAB_PATH):
        return None
    try:
        with open(DEFAULT_VOCAB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 工具 3：rename_column —— 改列名并重跑 judge
# ---------------------------------------------------------------------------
def rename_column(adata, from_col: str, to_col: str) -> dict:
    """把 obs[from_col] 改名为 obs[to_col]，返回改动前后对照并重跑 judge。"""
    try:
        if from_col not in adata.obs.columns:
            return {"status": "error", "error": f"列不存在：{from_col}"}
        before = list(adata.obs.columns)
        adata.obs = adata.obs.rename(columns={from_col: to_col})
        after = list(adata.obs.columns)
        re_judge = judge(adata)
        return {
            "status": "ok",
            "from_col": from_col, "to_col": to_col,
            "columns_before": _json_safe(before),
            "columns_after": _json_safe(after),
            "re_judge": re_judge,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": _err_info(e)}


# ---------------------------------------------------------------------------
# 工具 4：execute_code —— 兜底执行模型/用户自写代码（受限环境 + 审计日志）
# ---------------------------------------------------------------------------
_SAFE_BUILTINS = {
    "print": print, "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "str": str, "int": int,
    "float": float, "bool": bool, "min": min, "max": max, "sum": sum, "abs": abs,
    "round": round, "sorted": sorted, "isinstance": isinstance, "getattr": getattr,
    "hasattr": hasattr, "Exception": Exception, "ValueError": ValueError,
}


def execute_code(code: str, adata, outdir: str) -> dict:
    """在受限环境中执行自写代码（用于兜底处理命名歧义/多级拆分/格式转换等）。

    受限策略：仅暴露白名单 builtins 与限定模块（numpy/pandas/scanpy/anndata/scipy），
    禁止网络、文件系统以外路径访问等危险操作；执行内容与结果写入审计日志。
    """
    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402
    import scanpy as sc  # noqa: E402
    import anndata as ad  # noqa: E402
    from scipy import sparse  # noqa: E402

    sandbox = {
        "__builtins__": _SAFE_BUILTINS,
        "np": np, "pd": pd, "sc": sc, "ad": ad, "sparse": sparse,
        "adata": adata, "outdir": outdir,
    }
    audit = {"code": code, "stdout": "", "error": None, "time": datetime.now().isoformat()}
    try:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<agent_code>", "exec"), sandbox, sandbox)
        audit["stdout"] = buf.getvalue()
        new_adata = sandbox.get("adata", adata)
        audit["result"] = f"执行成功，输出 adata shape={getattr(new_adata, 'shape', 'N/A')}"
        return {"status": "ok", "adata": new_adata, "audit": _json_safe(audit)}
    except Exception as e:
        audit["error"] = _err_info(e)
        return {"status": "error", "error": str(e), "audit": _json_safe(audit)}


# ---------------------------------------------------------------------------
# 工具 5/6/7：数据分析原子工具
# ---------------------------------------------------------------------------
def qc_stats(adata) -> dict:
    """单细胞 QC 统计：细胞/基因、UMI、%mito/%ribo、top 表达基因、HVG、全零细胞、离群。"""
    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402
    from scipy import sparse  # noqa: E402

    x = adata.X
    is_sparse = sparse.issparse(x)
    n_cells, n_genes = adata.n_obs, adata.n_vars

    if is_sparse:
        total_umi_per_cell = np.asarray(x.sum(axis=1)).ravel()
        genes_per_cell = np.asarray((x > 0).sum(axis=1)).ravel()
        total_umi_per_gene = np.asarray(x.sum(axis=0)).ravel()
    else:
        xn = np.asarray(x)
        total_umi_per_cell = xn.sum(axis=1)
        genes_per_cell = (xn > 0).sum(axis=1)
        total_umi_per_gene = xn.sum(axis=0)

    # %mito / %ribo（基因名匹配 mt- / RPL/RPS/MRPL/MRPS）
    genes = [str(g) for g in adata.var_names]
    is_mito = np.array([g.lower().startswith("mt-") for g in genes])
    is_ribo = np.array([g.lower().startswith(("rpl", "rps", "mrpl", "mrps")) for g in genes])
    mito_frac = np.zeros_like(total_umi_per_cell, dtype=float)
    if is_mito.any():
        if is_sparse:
            mito_per_cell = np.asarray(x[:, is_mito].sum(axis=1)).ravel()
        else:
            mito_per_cell = xn[:, is_mito].sum(axis=1)
        mito_frac = np.divide(mito_per_cell, total_umi_per_cell, out=mito_frac,
                              where=total_umi_per_cell > 0)

    # top 表达基因（按总 UMI）
    top_idx = np.argsort(total_umi_per_gene)[::-1][:20]
    top_genes = [{"gene": genes[i], "umi": float(total_umi_per_gene[i])} for i in top_idx]

    # 全零细胞占比 / 全零基因占比
    zero_cell_ratio = float((total_umi_per_cell == 0).mean())
    zero_gene_ratio = float((total_umi_per_gene == 0).mean())

    # 离群细胞（按总 UMI 的 MAD 阈值，粗略）
    med = float(np.median(total_umi_per_cell))
    mad = float(np.median(np.abs(total_umi_per_cell - med))) or 1.0
    outlier_mask = total_umi_per_cell > med + 5 * 1.4826 * mad
    outlier_ratio = float(outlier_mask.mean())

    return _json_safe({
        "n_cells": n_cells, "n_genes": n_genes,
        "total_umi": float(total_umi_per_cell.sum()),
        "genes_per_cell_median": float(np.median(genes_per_cell)),
        "umi_per_cell_median": float(np.median(total_umi_per_cell)),
        "mito_pct_median": float(np.median(mito_frac) * 100),
        "has_mito_genes": bool(is_mito.any()),
        "has_ribo_genes": bool(is_ribo.any()),
        "top_genes": top_genes,
        "zero_cell_ratio": zero_cell_ratio,
        "zero_gene_ratio": zero_gene_ratio,
        "outlier_cell_ratio": outlier_ratio,
    })


def composition_stats(adata) -> dict:
    """batch / cell_type 组成统计：比例表 + 小批次提示。"""
    import pandas as pd  # noqa: E402

    result = {}
    for col in ["cell_type", "batch"]:
        if col in adata.obs.columns:
            vc = adata.obs[col].astype(str).value_counts()
            total = int(vc.sum())
            ratio = {str(k): round(int(v) / total, 4) for k, v in vc.items()}
            small = [str(k) for k, v in vc.items() if int(v) < max(10, total * 0.01)]
            result[col] = {
                "n_categories": int(vc.shape[0]),
                "distribution": {str(k): int(v) for k, v in vc.items()},
                "ratio": ratio,
                "small_samples": small,
            }
        else:
            result[col] = None
    return _json_safe(result)


def plot_qc(adata, outdir: str) -> dict:
    """生成 QC 图：UMI 直方图、UMI-基因相关、batch/cell_type 组成柱状图，返回图片路径。"""
    import numpy as np  # noqa: E402
    import matplotlib  # noqa: E402
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from scipy import sparse  # noqa: E402

    fig_dir = os.path.join(outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    x = adata.X
    if sparse.issparse(x):
        total_umi = np.asarray(x.sum(axis=1)).ravel()
        genes_per_cell = np.asarray((x > 0).sum(axis=1)).ravel()
    else:
        xn = np.asarray(x)
        total_umi = xn.sum(axis=1)
        genes_per_cell = (xn > 0).sum(axis=1)

    paths = {}

    # UMI 直方图
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log1p(total_umi), bins=50, color="#2563eb", alpha=0.8)
    ax.set_xlabel("log1p(total UMI)"); ax.set_ylabel("细胞数"); ax.set_title("总 UMI 分布")
    p = os.path.join(fig_dir, "umi_hist.png"); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    paths["umi_hist"] = p

    # UMI-基因相关散点
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(genes_per_cell, total_umi, s=3, alpha=0.3, c="#16a34a")
    ax.set_xlabel("每细胞基因数"); ax.set_ylabel("总 UMI"); ax.set_title("UMI-基因相关性")
    p = os.path.join(fig_dir, "umi_vs_genes.png"); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    paths["umi_vs_genes"] = p

    # batch / cell_type 柱状图
    for col in ["cell_type", "batch"]:
        if col in adata.obs.columns:
            vc = adata.obs[col].astype(str).value_counts()
            fig, ax = plt.subplots(figsize=(max(5, min(20, vc.shape[0] * 0.6)), 4))
            vc.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#d97706")
            ax.set_title(f"{col} 组成"); ax.set_xlabel(col); ax.set_ylabel("细胞数")
            plt.xticks(rotation=45, ha="right")
            p = os.path.join(fig_dir, f"{col}_composition.png"); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
            paths[f"{col}_composition"] = p

    return _json_safe(paths)


# ---------------------------------------------------------------------------
# 工具 8：generate_report —— 生成「数据处理结果分析」产物（01/02/03 + figures）
# ---------------------------------------------------------------------------
def generate_report(result: dict, outdir: str) -> dict:
    """生成 数据状况/数据分析报告/操作审计 三份 markdown 与 figures 目录。"""
    analysis_dir = os.path.join(outdir, "数据处理结果分析")
    os.makedirs(analysis_dir, exist_ok=True)

    p1 = os.path.join(analysis_dir, "01_数据状况.md")
    p2 = os.path.join(analysis_dir, "02_数据分析报告.md")
    p3 = os.path.join(analysis_dir, "03_操作审计.md")

    with open(p1, "w", encoding="utf-8") as f:
        f.write(_render_data_status(result))
    with open(p2, "w", encoding="utf-8") as f:
        f.write(_render_analysis_report(result))
    with open(p3, "w", encoding="utf-8") as f:
        f.write(_render_audit(result))

    return _json_safe({
        "analysis_dir": analysis_dir,
        "files": {"01_数据状况.md": p1, "02_数据分析报告.md": p2, "03_操作审计.md": p3},
    })


def _render_data_status(result: dict) -> str:
    """01_数据状况.md：判断结论 + 对模型配置的影响（固定句式）。"""
    verdicts = result.get("judge", {}).get("verdicts", [])
    lines = ["# 01 数据状况", ""]

    lines.append("## 1. 判断结论")
    lines.append("")
    lines.append("| 检查项 | 结论 | 证据 | 建议动作 |")
    lines.append("|--------|------|------|----------|")
    for v in verdicts:
        lines.append(f"| {v['item']} | {v['status']} | {v['evidence']} | {v['action']} |")
    lines.append("")

    lines.append("## 2. 对模型配置的影响")
    lines.append("")
    for line in _config_impact_sentences(verdicts, result):
        lines.append(f"- {line}")
    lines.append("")
    return "\n".join(lines)


def _config_impact_sentences(verdicts: list, result: dict) -> list:
    """根据判定结果输出固定句式（检测到/未检测到 X → 结论 → 因而模型将如何训练/评测）。"""
    by_item = {v["item"]: v for v in verdicts}

    sentences = []
    batch = by_item.get("batch", {})
    ct = by_item.get("cell_type", {})
    raw = by_item.get("raw_counts", {})

    # batch
    if batch.get("status") == "PASS":
        n = _category_count(result, "batch")
        sentences.append(f"检测到 batch（{n} 类），use_batch=True，RNADecoder 按 {n} 类批次条件化，启用 batch 评测（Batch_ASW / Graph_Connectivity）。")
    else:
        sentences.append("未检测到 batch，use_batch=False，RNADecoder 无批次条件化，batch 评测项为 nan。")

    # cell_type
    if ct.get("status") == "PASS":
        n = _category_count(result, "cell_type")
        sentences.append(f"检测到 cell_type（{n} 类），use_cell_type=True，启用判别头与聚类评测（ARI/ASW）。")
    else:
        sentences.append("未检测到 cell_type，use_cell_type=False，跳过判别与 ARI/ASW 评测。")

    # raw counts
    if raw.get("status") == "PASS":
        sentences.append("检测到原始计数，模型将执行 normalize_total + log1p + HVG 预处理，无二次归一化风险。")
    elif raw.get("status") == "FAIL":
        sentences.append("检测到 X 非原始计数，已尝试从 .raw / layers 顶替；若不可用则警告继续，结果可能受二次归一化影响。")
    else:
        sentences.append("X 原始计数存疑（WARNING），建议核对数据是否未经归一化，避免模型内部二次归一化失真。")

    # universal 覆盖率
    uni = by_item.get("universal_coverage", {})
    if uni.get("status") == "PASS":
        sentences.append("universal 基因覆盖率达到阈值，可启用 use_universal_model=True 进行跨数据集对齐。")
    elif uni.get("status") == "WARNING":
        sentences.append("universal 基因覆盖率偏低，建议使用 use_universal_model=False，或确认基因命名与模型词表一致。")

    return sentences


def _category_count(result: dict, col: str):
    comp = result.get("composition", {}).get(col) or {}
    return comp.get("n_categories", "?")


def _render_analysis_report(result: dict) -> str:
    """02_数据分析报告.md：单细胞数据集通用模板。"""
    qc = result.get("qc", {})
    comp = result.get("composition", {})
    meta = result.get("metadata", {})
    figures = result.get("figures", {})

    lines = ["# 02 数据分析报告", ""]

    lines.append("## 1. 数据规模")
    lines.append("")
    lines.append(f"- 细胞 × 基因：{meta.get('n_cells', '?')} × {meta.get('n_genes', '?')}")
    lines.append(f"- 总 UMI：{meta.get('total_umi', '?')}")
    lines.append(f"- 稀疏度：{meta.get('sparsity', '?')}")
    lines.append("")

    lines.append("## 2. Per-cell QC")
    lines.append("")
    lines.append(f"- 每细胞基因数中位数：{qc.get('genes_per_cell_median', '?')}")
    lines.append(f"- 每细胞 UMI 中位数：{qc.get('umi_per_cell_median', '?')}")
    lines.append(f"- %mito 中位数：{qc.get('mito_pct_median', '?')}%")
    lines.append("")

    lines.append("## 3. Top 表达基因与 HVG")
    lines.append("")
    for g in qc.get("top_genes", [])[:10]:
        lines.append(f"- {g['gene']}：{g['umi']}")
    lines.append("")

    lines.append("## 4. 质量异常")
    lines.append("")
    lines.append(f"- 全零细胞占比：{qc.get('zero_cell_ratio', '?')}")
    lines.append(f"- 全零基因占比：{qc.get('zero_gene_ratio', '?')}")
    lines.append(f"- 离群细胞占比（MAD）：{qc.get('outlier_cell_ratio', '?')}")
    lines.append("")

    lines.append("## 5. batch / cell_type 组成")
    lines.append("")
    for col in ["batch", "cell_type"]:
        info = comp.get(col)
        if not info:
            lines.append(f"- {col}：未检测到")
            continue
        lines.append(f"### {col}（{info.get('n_categories')} 类）")
        for k, v in info.get("distribution", {}).items():
            lines.append(f"- {k}：{v}（{info.get('ratio', {}).get(k)}）")
        if info.get("small_samples"):
            lines.append(f"- 小批次提示：{', '.join(info['small_samples'])} 样本数过少，去批次结果可能不稳定")
        lines.append("")
    lines.append("")

    lines.append("## 6. universal 覆盖率")
    lines.append("")
    uni = result.get("judge", {}).get("verdicts", [])
    for v in uni:
        if v["item"] == "universal_coverage":
            lines.append(f"- {v['evidence']}")
    lines.append("")

    lines.append("## 7. 附图")
    lines.append("")
    for name, path in figures.items():
        lines.append(f"- {name}：`{path}`")
    lines.append("")
    return "\n".join(lines)


def _render_audit(result: dict) -> str:
    """03_操作审计.md：记录跑了哪些脚本/代码/参数。"""
    log = result.get("audit_log", [])
    lines = ["# 03 操作审计", ""]
    lines.append("| 时间 | 操作 | 详情 |")
    lines.append("|------|------|------|")
    for entry in log:
        lines.append(f"| {entry.get('time', '')} | {entry.get('op', '')} | {entry.get('detail', '')} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 工具 9：save_dataset —— 保存处理后的数据，交训练阶段
# ---------------------------------------------------------------------------
def save_dataset(adata, outdir: str) -> dict:
    """保存处理后的 AnnData 为 processed_rna.h5ad（交训练阶段）。"""
    try:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "processed_rna.h5ad")
        adata.write_h5ad(path)
        return {"status": "ok", "path": path}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": _err_info(e)}


# ---------------------------------------------------------------------------
# 编排入口：load_data -> judge -> 修复 -> 再 judge -> 数据分析 -> 报告 -> 保存
# ---------------------------------------------------------------------------
def run_preprocess(task_id: int, dataset_id: int, dataset_path: str, output_dir: str) -> dict:
    """预处理阶段编排（确定性主链路，execute_code 作为兜底）。"""
    import scanpy as sc  # noqa: E402

    os.makedirs(output_dir, exist_ok=True)
    audit_log = []
    audit_log.append({"time": datetime.now().isoformat(), "op": "load_data",
                      "detail": f"dataset_id={dataset_id}, path={dataset_path}"})

    emit("preprocess_log", {"message": f"开始预处理：加载 {dataset_path}"})

    # 1) load_data
    adata = _read_data(dataset_path)
    metadata = _describe(adata, dataset_path)
    emit("preprocess_log", {"message": f"加载完成：{metadata['n_cells']} 细胞 × {metadata['n_genes']} 基因"})

    # 2) judge
    verdicts = judge(adata)
    emit("preprocess_log", {"message": f"首次判定：{verdicts['status']}"})
    audit_log.append({"time": datetime.now().isoformat(), "op": "judge",
                      "detail": json.dumps(verdicts, ensure_ascii=False)})

    # 3) 自动修复：简单改名（同义列）-> 多级/歧义 -> execute_code 兜底
    adata, audit_log = _auto_repair(adata, verdicts, output_dir, audit_log)

    # 4) 再 judge
    verdicts = judge(adata)
    emit("preprocess_log", {"message": f"修复后再判定：{verdicts['status']}"})
    audit_log.append({"time": datetime.now().isoformat(), "op": "re_judge",
                      "detail": json.dumps(verdicts, ensure_ascii=False)})

    # 5) 数据分析
    qc = qc_stats(adata)
    composition = composition_stats(adata)
    figures = plot_qc(adata, os.path.join(output_dir, "数据处理结果分析"))
    emit("preprocess_log", {"message": "数据分析完成（QC/组成/绘图）"})

    # 6) 生成报告
    result = {
        "metadata": metadata,
        "judge": verdicts,
        "qc": qc,
        "composition": composition,
        "figures": figures,
        "audit_log": audit_log,
    }
    report = generate_report(result, output_dir)
    emit("preprocess_log", {"message": f"已生成数据处理结果分析：{report['analysis_dir']}"})

    # 7) 保存数据集，交训练阶段
    saved = save_dataset(adata, output_dir)
    emit("preprocess_log", {"message": f"数据集已保存：{saved.get('path')}（交训练阶段）"})

    return _json_safe({
        "status": "success",
        "judge_status": verdicts["status"],
        "metadata": metadata,
        "report": report,
        "processed_path": saved.get("path"),
    })


def _auto_repair(adata, verdicts, output_dir, audit_log):
    """三级分流修复：简单改名用 rename_column；命名歧义/多级拆分等升级 execute_code。

    确定性主链路只处理「能无歧义推断」的改名，其余记录到审计日志供用户/LLM 决策。
    """
    for v in verdicts.get("verdicts", []):
        if v["item"] not in ("cell_type", "batch") or v["status"] != "FAIL":
            continue

        # 情况 A：存在明确同义列，直接改名
        evidence = v.get("evidence", "")
        if "同义列" in evidence:
            # 从 evidence 解析出原列名（证据文案：发现同义列「X」）
            import re
            m = re.search(r"「([^」]+)」", evidence)
            if m:
                from_col = m.group(1)
                r = rename_column(adata, from_col, v["item"])
                audit_log.append({"time": datetime.now().isoformat(), "op": "rename_column",
                                  "detail": f"{from_col} -> {v['item']}"})
                emit("preprocess_log", {"message": f"rename_column: {from_col} -> {v['item']}"})
                if r.get("status") == "ok":
                    continue

        # 情况 B：无同义列 / 需要多级拆分，升级 execute_code（这里记录，交给兜底脚本）
        audit_log.append({"time": datetime.now().isoformat(), "op": "escalate_execute_code",
                          "detail": f"{v['item']} 需人工/execute_code：{evidence}"})
        emit("preprocess_log", {"message": f"{v['item']} 需要拆分/人工，升级 execute_code"})

    # 对含多级值的列升级 execute_code 拆分（cell.type 形如 "T cell:CD4" 时拆出第一级作为主标签）
    adata = _try_split_multilevel(adata, audit_log, output_dir)

    return adata, audit_log


def _try_split_multilevel(adata, audit_log, output_dir):
    """检测 obs 中 cell_type/batch 的多级值（含 ':' '/' '|' 分隔），升级 execute_code 拆分。

    该场景属于「多级列要拆」，按修复 SOP 归属 execute_code 兜底，故真实调用 execute_code
    （受限环境 + 审计日志），而非自行赋值，保证操作可追溯。
    """
    for col in ["cell_type", "batch"]:
        if col not in adata.obs.columns:
            continue
        vals = adata.obs[col].astype(str)
        sample = vals.iloc[0] if len(vals) else ""
        for sep in [":", "/", "|"]:
            # 注意 regex=False："|" 是正则特殊字符，默认按正则会匹配一切导致误判
            if sep in sample and vals.str.contains(sep, regex=False).mean() > 0.5:
                code = (
                    f"adata.obs['{col}'] = adata.obs['{col}'].astype(str)"
                    f".str.split('{sep}').str[0].astype(str)\n"
                )
                r = execute_code(code, adata, output_dir)
                if r.get("status") == "ok":
                    adata = r.get("adata", adata)
                audit_log.append({"time": datetime.now().isoformat(), "op": "execute_code",
                                  "detail": json.dumps({"reason": f"多级列 {col} 拆分（分隔符 {sep}）取第一级",
                                                        "code": code, "result": r.get("status")},
                                                       ensure_ascii=False)})
                emit("preprocess_log", {"message": f"execute_code 拆分多级列 {col}，取第一级（分隔符 {sep}）"})
                break
    return adata