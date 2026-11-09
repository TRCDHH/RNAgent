"""模态数据处理与跨模态评测（模型无关的共享层）。

为什么单独一个模块：
- scButterfly / scCrossTalk 这类 **RNA ↔ ATAC 跨模态模型** 需要两份配对数据，
  它们的「数据要求」是共通的（峰名 `chr:start-end`、二值化、过滤低出现峰、TF-IDF、按染色体排序），
  但这些要求与具体模型无关 —— 放在 backend 里会导致两份重复实现、改一处漏一处；
- 同时这里不 import 任何模型源码，主进程（FastAPI 门户）可以安全调用
  `judge_atac / prepare_atac / describe_atac`，重依赖（torch）只在训练子进程里出现。

评测口径说明：
- 单模态嵌入仍走 scLinformer 的 evaluate_sc_embedding（ARI/AMI/NMI/HOM/ASW/UMAP），保证与既有模型可比；
- 这里的跨模态指标是**辅助对照**，刻意用确定性的 KMeans（k=细胞类型数）而不是 leiden，
  避免分辨率参数引入额外噪声；结果写入 cross_modal_metrics.csv，不覆盖 summary_metrics.csv。
"""

import math
import os
import re

# 峰名格式：chr1:1000-2000 / 1:1000-2000 / chrX:100-200
PEAK_PATTERN = re.compile(r"^([0-9A-Za-z_.]+):(\d+)-(\d+)$")

# 过滤低出现峰的下限比例（与 scButterfly / scCrossTalk 的默认 fpeaks 一致）
DEFAULT_FPEAKS = 0.005

# 染色体自然排序用（chr1..chr22, chrX, chrY, 其余按字典序）
_CHROM_ORDER = {f"chr{i}": i for i in range(1, 23)}
_CHROM_ORDER.update({"chrX": 23, "chrY": 24, "chrM": 25})


# ---------------------------------------------------------------------------
# 峰名解析与排序
# ---------------------------------------------------------------------------
def _chrom_rank(chrom: str) -> tuple:
    key = chrom if chrom.startswith("chr") else f"chr{chrom}"
    if key in _CHROM_ORDER:
        return (0, _CHROM_ORDER[key], "")
    return (1, 0, key)


def _parse_peaks(var_names) -> list:
    """解析峰名，返回 [(chrom, start, end) | None]，None 表示无法解析。"""
    out = []
    for v in var_names:
        m = PEAK_PATTERN.match(str(v))
        out.append((m.group(1), int(m.group(2)), int(m.group(3))) if m else None)
    return out


def _sort_peaks(atac):
    """按染色体自然序 + 起点排序，保证同一染色体的峰连续（模型按 chrom_list 切列的前提）。"""
    import numpy as np

    parsed = _parse_peaks(atac.var_names)
    if any(p is None for p in parsed):
        return atac  # 无法全部解析时不排序，交给 judge 报错

    order = sorted(
        range(atac.n_vars),
        key=lambda i: (_chrom_rank(parsed[i][0]), parsed[i][1], parsed[i][2]),
    )
    if order == list(range(atac.n_vars)):
        return atac
    return atac[:, np.asarray(order)].copy()


def chrom_list_of(atac) -> list:
    """按「连续块」统计每条染色体的峰数，供模型首层按染色体切分使用。"""
    parsed = _parse_peaks(atac.var_names)
    out = []
    last = None
    for p in parsed:
        if p is None:
            continue
        chrom = p[0] if p[0].startswith("chr") else f"chr{p[0]}"
        if chrom == last:
            out[-1] += 1
        else:
            out.append(1)
            last = chrom
    return out


# ---------------------------------------------------------------------------
# 判定（供 preprocess 阶段合并进判定契约）
# ---------------------------------------------------------------------------
def judge_atac(atac, rna=None) -> list:
    """ATAC（第二模态）判定项。任一 FAIL 会让预处理阶段整体失败。"""
    import numpy as np
    from scipy import sparse

    verdicts = []

    # 1) 矩阵存在
    if atac is None or atac.n_obs == 0 or atac.n_vars == 0:
        return [{"item": "atac_exists", "status": "FAIL",
                 "evidence": "ATAC 数据为空或缺维", "action": "提供有效的 ATAC peak 矩阵（h5ad）"}]
    verdicts.append({"item": "atac_exists", "status": "PASS",
                     "evidence": f"X 形状 {tuple(atac.shape)}（细胞 × 峰）", "action": "无需处理"})

    # 2) 峰名格式：模型要求 chr:start-end
    parsed = _parse_peaks(atac.var_names)
    bad = sum(1 for p in parsed if p is None)
    if bad == 0:
        verdicts.append({"item": "atac_peak_format", "status": "PASS",
                         "evidence": f"{atac.n_vars} 个峰名全部符合 chr:start-end", "action": "无需处理"})
    else:
        example = [str(v) for v, p in zip(atac.var_names, parsed) if p is None][:3]
        verdicts.append({"item": "atac_peak_format",
                         "status": "FAIL" if bad == atac.n_vars else "WARNING",
                         "evidence": f"{bad}/{atac.n_vars} 个峰名无法解析，示例：{example}",
                         "action": "把 adata.var_names 改为 chr1:1000-2000 形式（并用 var 提供坐标）"})

    # 3) 与 RNA 配对：细胞数
    if rna is not None:
        same_n = int(atac.n_obs) == int(rna.n_obs)
        verdicts.append({"item": "atac_paired",
                         "status": "PASS" if same_n else "FAIL",
                         "evidence": f"ATAC {atac.n_obs} 细胞 / RNA {rna.n_obs} 细胞",
                         "action": "无需处理" if same_n else "提供与 RNA 同一批细胞（数量与顺序一致）的 ATAC 数据"})

        # 4) 细胞名是否一致（不一致时按 obs_names 对齐）
        if same_n:
            same_names = list(map(str, atac.obs_names)) == list(map(str, rna.obs_names))
            if same_names:
                verdicts.append({"item": "atac_obs_names", "status": "PASS",
                                 "evidence": "obs_names 与 RNA 完全一致", "action": "无需处理"})
            else:
                shared = len(set(map(str, atac.obs_names)) & set(map(str, rna.obs_names)))
                verdicts.append({"item": "atac_obs_names",
                                 "status": "PASS" if shared == same_n else "WARNING",
                                 "evidence": f"obs_names 顺序不同，交集 {shared}/{rna.n_obs}",
                                 "action": "按 obs_names 重排 ATAC 行；交集不足则按位置对应"})

    # 5) 是否已二值化
    x = atac.X if not sparse.issparse(atac.X) else atac.X.data
    vals = np.asarray(x).ravel()
    if vals.size:
        binary = bool(np.all((vals == 0) | (vals == 1)))
        verdicts.append({"item": "atac_binary",
                         "status": "PASS" if binary else "WARNING",
                         "evidence": "X 已是 0/1" if binary else
                                     f"X 非二值（max={float(vals.max()):.2f}）",
                         "action": "无需处理" if binary else "prepare 阶段自动二值化"})

    # 6) 规模下限
    ok = atac.n_obs >= 100 and atac.n_vars >= 500
    verdicts.append({"item": "atac_scale_min",
                     "status": "PASS" if ok else "WARNING",
                     "evidence": f"n_cells={atac.n_obs}, n_peaks={atac.n_vars}（下限 100 / 500）",
                     "action": "" if ok else "峰数偏少，跨模态翻译效果可能不稳定"})

    # 7) 染色体连续性（模型首层按连续块切列，乱序会维度错位）
    ranks = [_chrom_rank(p[0]) for p in parsed if p is not None]
    contiguous = all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1))
    verdicts.append({"item": "atac_chrom_order",
                     "status": "PASS" if contiguous else "WARNING",
                     "evidence": "峰已按染色体 + 坐标排序" if contiguous else "峰未按染色体排序",
                     "action": "无需处理" if contiguous else "prepare 阶段自动按 chr + 起点排序"})

    return verdicts


def describe_atac(atac) -> dict:
    """ATAC 规模摘要（供报告展示）。"""
    from scipy import sparse

    if atac is None:
        return {}
    nnz = int(atac.X.nnz) if sparse.issparse(atac.X) else int((atac.X != 0).sum())
    total = max(1, int(atac.n_obs) * int(atac.n_vars))
    chrs = {p[0] for p in _parse_peaks(atac.var_names) if p is not None}
    return {
        "n_cells": int(atac.n_obs),
        "n_peaks": int(atac.n_vars),
        "n_chroms": len(chrs),
        "sparsity": round(1.0 - nnz / total, 6),
    }


# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------
def align_atac_to_rna(atac, rna):
    """把 ATAC 的细胞顺序对齐到 RNA（模型用位置索引配对，顺序错了等于配错细胞）。"""
    if rna is None or int(atac.n_obs) != int(rna.n_obs):
        return atac
    atac_names = list(map(str, atac.obs_names))
    rna_names = list(map(str, rna.obs_names))
    if atac_names == rna_names:
        return atac
    index = {n: i for i, n in enumerate(atac_names)}
    if len(index) == len(atac_names) and all(n in index for n in rna_names):
        import numpy as np

        return atac[np.asarray([index[n] for n in rna_names])].copy()
    return atac  # 名字对不上：按位置对应（在上游 judge 里已给 WARNING）


def prepare_atac(atac, params: dict, name: str = "", rna=None):
    """标准 ATAC 预处理：对齐细胞 -> 排序 -> 二值化 -> 过滤低出现峰 -> TF-IDF -> 缩放到 [0,1]。

    与 scButterfly.data_processing.ATAC_data_preprocessing 及 scCrossTalk 的同名函数口径一致，
    但不依赖 episcanpy（Windows 上装不了 pybedtools），因此可以在任何环境复现。
    """
    import numpy as np
    from scipy import sparse

    params = params or {}
    fpeaks = float(params.get("atac_fpeaks") or DEFAULT_FPEAKS)
    use_tfidf = bool(params.get("atac_tfidf", True))

    atac = align_atac_to_rna(atac, rna)
    atac = _sort_peaks(atac.copy())
    atac.var_names_make_unique()

    # 1) 二值化（ATAC 只关心「峰是否可及」）
    x = atac.X
    x = sparse.csr_matrix(x) if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x))
    x = x.astype(np.float32)
    x.data = np.ones_like(x.data)

    # 2) 过滤低出现率峰（至少保留 1 个，否则后面除零）
    if 0 < fpeaks < 1:
        min_cells = max(1, int(math.ceil(fpeaks * atac.n_obs)))
        keep = np.asarray((x > 0).sum(axis=0)).ravel() >= min_cells
        if keep.sum() >= 100:
            x = x[:, keep]
            atac = atac[:, keep].copy()
        else:
            print(f"[warn] {name}：fpeaks={fpeaks} 会过滤到只剩 {int(keep.sum())} 个峰，已跳过过滤")

    # 3) TF-IDF：tf = 峰计数 / 该细胞总计数；idf = log(1 + 细胞数 / 该峰出现细胞数)
    if use_tfidf:
        n_cells = x.shape[0]
        cell_total = np.asarray(x.sum(axis=1)).ravel()
        peak_total = np.asarray(x.sum(axis=0)).ravel()
        inv_cell = sparse.diags(1.0 / np.maximum(cell_total, 1.0))
        x = (inv_cell @ x).tocsr()
        idf = np.log1p(n_cells / np.maximum(peak_total, 1.0))
        x = x.multiply(idf[None, :]).tocsr()

    # 4) 缩放到 [0,1]（与 scButterfly 的 normalize 一致）
    max_v = float(x.max()) if x.nnz else 1.0
    if max_v > 0:
        x = x / max_v

    atac.X = x.astype(np.float32)
    return atac


def prepare_rna(adata, params: dict, name: str = ""):
    """RNA 预处理：基因名唯一 -> normalize_total -> log1p -> HVG 筛选。

    两个跨模态模型的官方实现都在内部做这三步，但它们把 HVG 数写死成 3000 且
    无法在「已知最终列数」之外做校验（scCrossTalk 的 n_genes 断言甚至被注释掉了）。
    这里统一在 backend 侧做，好处是 HVG 后的真实列数由我们自己掌握。
    """
    import scanpy as sc

    params = params or {}
    n_top = int(params.get("n_genes") or 0)

    adata = adata.copy()
    adata.var_names_make_unique()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if n_top <= 0 or n_top >= adata.n_vars:
        return adata
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat")
    except Exception as e:  # 数据过小等场景：退回全基因，不阻断训练
        print(f"[warn] {name}：高变基因筛选失败（{e}），使用全部 {adata.n_vars} 个基因")
        return adata
    return adata[:, adata.var["highly_variable"]].copy()


# ---------------------------------------------------------------------------
# 跨模态评测
# ---------------------------------------------------------------------------
def _kmeans_labels(emb, k: int):
    import numpy as np
    from sklearn.cluster import KMeans

    k = max(2, min(int(k), emb.shape[0]))
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(np.asarray(emb))


def _cluster_scores(emb, labels) -> dict:
    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                 homogeneity_score, normalized_mutual_info_score)

    k = len(set(labels))
    pred = _kmeans_labels(emb, k)
    return {
        "ARI": float(adjusted_rand_score(labels, pred)),
        "AMI": float(adjusted_mutual_info_score(labels, pred)),
        "NMI": float(normalized_mutual_info_score(labels, pred)),
        "HOM": float(homogeneity_score(labels, pred)),
        "n_clusters": int(k),
    }


def cross_modal_metrics(adata, second, rna_key: str, second_key: str, outdir: str) -> dict:
    """计算并落盘跨模态一致性指标（cross_modal_metrics.csv），返回 dict。"""
    import numpy as np
    import pandas as pd

    if rna_key not in adata.obsm:
        raise ValueError(f"RNA 嵌入缺失：obsm['{rna_key}']")
    if second_key not in second.obsm:
        raise ValueError(f"第二模态嵌入缺失：obsm['{second_key}']")

    rna_emb = np.asarray(adata.obsm[rna_key], dtype=np.float32)
    atac_emb = np.asarray(second.obsm[second_key], dtype=np.float32)

    # 两模态嵌入维度可能不同（例如 ATAC 侧 head 不同），对齐度只在同维时计算
    align = None
    if rna_emb.shape == atac_emb.shape and rna_emb.size:
        num = (rna_emb * atac_emb).sum(axis=1)
        den = np.linalg.norm(rna_emb, axis=1) * np.linalg.norm(atac_emb, axis=1)
        align = float(np.mean(num / np.maximum(den, 1e-8)))

    out = {"Modality_Align": align}

    labels = None
    if "cell_type" in adata.obs.columns:
        col = adata.obs["cell_type"]
        if len(set(col.astype(str))) > 1:
            labels = col.astype(str).to_numpy()

    if labels is not None:
        try:
            out.update({f"RNA_{k}": v for k, v in _cluster_scores(rna_emb, labels).items()
                        if k != "n_clusters"})
            out.update({f"ATAC_{k}": v for k, v in _cluster_scores(atac_emb, labels).items()
                        if k != "n_clusters"})
        except Exception as e:
            print(f"[warn] 跨模态聚类指标计算失败：{e}")

        # 跨模态标签迁移：RNA 嵌入上训练 KNN，在 ATAC 嵌入上预测细胞类型
        try:
            from sklearn.neighbors import KNeighborsClassifier

            n = min(15, max(1, len(labels) // 10))
            knn = KNeighborsClassifier(n_neighbors=n).fit(rna_emb, labels)
            out["Cross_KNN_Acc"] = float((knn.predict(atac_emb) == labels).mean())
        except Exception as e:
            print(f"[warn] 跨模态 KNN 迁移计算失败：{e}")

    path = os.path.join(outdir, "cross_modal_metrics.csv")
    try:
        pd.DataFrame([out]).to_csv(path, index=False)
    except Exception as e:
        print(f"[warn] 跨模态指标落盘失败：{e}")

    result = {k: (None if v is None else float(v)) for k, v in out.items()}
    result["csv_path"] = path
    return result
