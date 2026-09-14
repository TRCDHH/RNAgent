"""预处理 ReAct 工具集：把 preprocess_tools 的原子能力暴露给 LLM（LangChain @tool）。

与原实现的三个关键差异（见技术方案第 4 节）：
1. **补上「观察类」工具**（list_obs_columns / value_counts / sample_values）——
   LLM 必须先看清数据长什么样，才谈得上决定怎么修。这是原实现最大的缺口。
2. **写操作后强制自动 judge**（_after_write）—— 不依赖 LLM 自觉，把原来
   「judge -> 修复 -> re-judge」的确定性闭环保留下来。
3. **显式终止**（finish）—— 不能靠「LLM 没有 tool_calls」隐式结束，否则无法区分
   「已完成」与「模型放弃」。

所有工具都通过线程局部会话（preprocess_session.get_session）取数据，
因此函数签名里不出现 adata —— 这也恰好满足 LangChain 工具参数必须可序列化的要求。
"""

import json

from langchain_core.tools import tool

from app.event_emitter import emit
from app.skills.library import get_library
from app.tools import preprocess_tools as pt
from app.tools.preprocess_session import get_session, now_iso


# ---------------------------------------------------------------------------
# 会话与审计
# ---------------------------------------------------------------------------
def _require_session():
    """取当前会话；未初始化（或数据没加载）时给可读错误而不是 AttributeError。"""
    s = get_session()
    if s is None or s.adata is None:
        raise RuntimeError("预处理会话未初始化或数据尚未加载")
    return s


def _audit(op: str, detail: str):
    s = get_session()
    if s is not None:
        s.audit_log.append({"time": now_iso(), "op": op, "detail": detail})


def _obs_columns(s) -> list:
    return [str(c) for c in s.adata.obs.columns]


def _after_write(result: dict) -> dict:
    """写操作后置：自动 judge + 行数守恒校验。

    这是把原确定性链路的「修复后必须复检」固化进工具层——LLM 不用（也不应该）记得
    手动调 judge_data，引擎保证每次改动后判定结果一定出现在它眼前。
    """
    s = _require_session()
    try:
        verdicts = pt.judge(s.adata, s.backend)
    except Exception as e:  # judge 自身失败不应吞掉本次修复结果
        result["judge_error"] = f"{type(e).__name__}: {e}"
        return result

    s.verdicts = verdicts
    fail_items = sorted(v["item"] for v in verdicts.get("verdicts", []) if v["status"] == "FAIL")
    s.judge_history.append(tuple(fail_items))

    result["judge"] = verdicts.get("status")
    result["fail_items"] = fail_items
    result["n_cells_now"] = int(s.adata.n_obs)

    # 行数守恒：改名/拆列/删列都不应改变细胞数；execute_code 例外，仅告警不阻断
    if s.n_cells_0 is not None and int(s.adata.n_obs) != int(s.n_cells_0):
        result["row_conservation_violated"] = True
        msg = (f"细胞数由 {s.n_cells_0} 变为 {int(s.adata.n_obs)}，"
               f"违反行数守恒，请确认这是必要的（否则应回退该操作）")
        result["note"] = msg
        emit("preprocess_warn", {"message": msg})
    return result


# ---------------------------------------------------------------------------
# 观察类工具（只读）—— 决策的前提
# ---------------------------------------------------------------------------
@tool
def get_metadata() -> dict:
    """查看当前数据的总体元信息（规模、稀疏度、obs/var 列名、layers、是否含 raw 等）。

    这是了解数据的第一步；但要判断某一列到底是什么，请用 list_obs_columns /
    value_counts / sample_values。
    """
    s = _require_session()
    return pt._describe(s.adata, s.dataset_path)


@tool
def list_obs_columns() -> dict:
    """列出 obs 的所有列，含每列的 dtype 与唯一值个数。

    用途：快速判断数据是否缺少 cell_type / batch，以及有哪些疑似同义列
    （如 cell.type、Sample）。唯一值个数能帮你识别「这列是标签还是 ID」。
    """
    s = _require_session()
    cols = []
    for c in s.adata.obs.columns:
        try:
            n_unique = int(s.adata.obs[c].nunique())
        except Exception:
            n_unique = None
        cols.append({
            "column": str(c),
            "dtype": str(s.adata.obs[c].dtype),
            "n_unique": n_unique,
        })
    return {"n_cells": int(s.adata.n_obs), "n_columns": len(cols), "obs_columns": cols}


@tool
def value_counts(col: str, top: int = 20) -> dict:
    """查看 obs 中某一列的取值分布（按数量降序，最多返回 top 个）。

    用途：决定改名 / 拆分 / 合并之前，先看清这列到底是什么。例如判断
    'cell.type' 是单级标签（'T cell'）还是多级标签（'T cell:CD4'）。
    列不存在时返回 error 与现有列名。
    """
    s = _require_session()
    if col not in s.adata.obs.columns:
        return {"error": f"列不存在：{col}", "obs_columns": _obs_columns(s)}
    vc = s.adata.obs[col].astype(str).value_counts()
    n = max(1, int(top))
    return {
        "column": col,
        "n_categories": int(vc.shape[0]),
        "total": int(vc.sum()),
        "top": {str(k): int(v) for k, v in vc.head(n).items()},
    }


@tool
def sample_values(col: str, n: int = 10) -> dict:
    """查看某列前 n 个原始字符串值（不做任何统计处理）。

    用途：判断形态问题——分隔符（':'/'/'/'|'）、大小写不一致、多余空格、
    条形码后缀（-1）等。这些细节决定该用 rename_column 还是 split_column 还是 execute_code。
    """
    s = _require_session()
    if col not in s.adata.obs.columns:
        return {"error": f"列不存在：{col}", "obs_columns": _obs_columns(s)}
    vals = s.adata.obs[col].astype(str).head(max(1, int(n))).tolist()
    return {"column": col, "samples": vals}


@tool
def judge_data() -> dict:
    """按判定契约检查当前数据，逐项返回 PASS/FAIL/WARNING + 证据 + 建议动作。

    这是判断「是否已经修好」的唯一依据。注意：任何写操作后本工具的结果会**自动**
    附带在返回值里，所以你通常不必手动调用；主动调用用于开工前确认现状。
    """
    s = _require_session()
    verdicts = pt.judge(s.adata, s.backend)
    s.verdicts = verdicts
    return verdicts


@tool
def qc_summary() -> dict:
    """计算当前数据的 QC 统计（每细胞基因数/UMI 中位数、%mito、全零细胞/基因比例、离群比例）。

    用于确认修复没有引入质量问题（例如误删列导致的空值、误过滤导致的分布畸变）。
    """
    s = _require_session()
    return pt.qc_stats(s.adata)


# ---------------------------------------------------------------------------
# 修复类工具（写）—— 决策的落地
# ---------------------------------------------------------------------------
@tool
def rename_column(from_col: str, to_col: str) -> dict:
    """把 obs 的列 from_col 改名为 to_col（用于同义列对齐，如 cell.type -> cell_type）。

    不改变数据内容与细胞数。只在**语义无歧义**时使用：拿不准时先用 sample_values
    确认该列确实是目标语义，不要凭列名猜测。
    """
    s = _require_session()
    r = pt.rename_column(s.adata, from_col, to_col, run_judge=False)
    if r.get("status") != "ok":
        return {"error": r.get("error", "改名失败"), "obs_columns": _obs_columns(s)}
    _audit("rename_column", f"{from_col} -> {to_col}")
    return _after_write({"ok": True, "from_col": from_col, "to_col": to_col})


@tool
def split_column(col: str, sep: str, keep: str = "first") -> dict:
    """把 obs 的 col 按分隔符 sep 拆分后保留其中一段（用于多级标签）。

    keep 可取 "first"（第一段，主类型）、"last"（最后一段，亚型）或下标 0/1/2。
    例：split_column("cell_type", ":", "first") 把 'T cell:CD4' 变成 'T cell'。
    拆分会在**原列上原地替换**，因此不需要时请先确认语义顺序。
    """
    s = _require_session()
    if col not in s.adata.obs.columns:
        return {"error": f"列不存在：{col}", "obs_columns": _obs_columns(s)}
    if not sep:
        return {"error": "分隔符 sep 不能为空"}

    parts = s.adata.obs[col].astype(str).str.split(sep, regex=False)

    def _pick(x):
        if not isinstance(x, list) or not x:
            return ""
        if keep == "first":
            return x[0]
        if keep == "last":
            return x[-1]
        try:
            i = int(keep)
            return x[i] if -len(x) <= i < len(x) else x[0]
        except (TypeError, ValueError):
            return x[0]

    before = s.adata.obs[col].astype(str).head(5).tolist()
    s.adata.obs[col] = parts.map(_pick).astype(str)
    after = s.adata.obs[col].astype(str).head(5).tolist()
    _audit("split_column", json.dumps(
        {"col": col, "sep": sep, "keep": keep}, ensure_ascii=False))
    return _after_write({"ok": True, "column": col, "before_sample": before, "after_sample": after})


@tool
def drop_obs_columns(cols: str) -> dict:
    """删除 obs 中指定的列，多个列名用英文逗号分隔（用于清理无用/污染列）。

    慎用：删除后不可恢复。若列里含有有效信息，优先考虑改名或拆分而不是删除。
    """
    s = _require_session()
    names = [c.strip() for c in str(cols).split(",") if c.strip()]
    if not names:
        return {"error": "未提供列名"}
    missing = [c for c in names if c not in s.adata.obs.columns]
    if missing:
        return {"error": f"列不存在：{missing}", "obs_columns": _obs_columns(s)}
    s.adata.obs = s.adata.obs.drop(columns=names)
    _audit("drop_obs_columns", ",".join(names))
    return _after_write({"ok": True, "dropped": names, "obs_columns": _obs_columns(s)})


@tool
def execute_code(code: str) -> dict:
    """在 Docker 沙箱中执行你自写的 Python 修复代码（万能兜底，非必要不用）。

    沙箱已预置：np、pd、sc(scanpy)、ad(anndata)、sparse，变量 adata 为当前数据，
    输出目录为 outdir。直接在代码里修改 adata（如 adata.obs['x'] = ...），
    沙箱会把结果写回。仅允许通过本工具改数据，禁止臆造数据或绕过它。

    安全约束：断网、根文件系统只读、非 root、丢弃全部 capabilities、内存/CPU 限额、超时强杀。
    注意：Docker 不可用时本工具会**拒绝执行**（不会回退到进程内 exec），
    此时请改用 rename_column / split_column，或调用 finish(reason="unsupported") 交人工处理。
    """
    s = _require_session()
    r = pt.execute_code(code, s.adata, s.output_dir)
    audit = r.get("audit") or {}
    _audit("execute_code", json.dumps(
        {"code": code, "result": r.get("status"), "error": r.get("error")},
        ensure_ascii=False)[:2000])
    if r.get("status") != "ok":
        return {"error": r.get("error", "沙箱执行失败"),
                "stdout": (audit.get("stdout") or "")[-1500:]}

    new_adata = r.get("adata")
    if new_adata is not None:
        s.adata = new_adata
    return _after_write({"ok": True, "stdout": (audit.get("stdout") or "")[-1500:]})


# ---------------------------------------------------------------------------
# 控制类工具
# ---------------------------------------------------------------------------
@tool
def load_skill(name: str) -> dict:
    """加载某个技能的完整操作手册（SKILL.md 正文）。

    可用技能名见系统提示中的清单；**动手之前建议先加载 "preprocess-qc"**，
    它包含完整的判定契约表与三级分流修复 SOP。
    技能不存在时返回 error 与可用技能名列表。
    """
    lib = get_library()
    body = lib.load(name)
    if not body:
        return {"error": f"技能不存在：{name}", "available": lib.names()}
    _audit("load_skill", name)
    return {"name": name, "content": body}


@tool
def finish(reason: str, notes: str = "") -> dict:
    """结束预处理决策。所有需要修复的项已处理，或确认无法继续时调用。

    reason 请用以下之一：
      all_pass     全部检查项通过（或仅剩可接受的 WARNING）
      no_fail      剩余 FAIL 无法修复但不影响训练
      ambiguous    数据信息不足/语义有歧义，需要人工确认——**不要猜**
      unsupported  非标准结构，沙箱内无法修复
      budget       已尽力，到达步数/时间上限
    notes 写给用户的说明（会进入审计日志与报告）。

    调用后决策立即结束，不会再有工具可用，请在此之前完成所有修复。
    """
    s = _require_session()
    s.finished = True
    s.finish_reason = str(reason or "").strip() or "unspecified"
    s.finish_notes = str(notes or "")
    _audit("finish", json.dumps({"reason": s.finish_reason, "notes": s.finish_notes},
                                ensure_ascii=False))
    emit("preprocess_log", {"message": f"ReAct 决策结束（{s.finish_reason}）"})
    return {"finished": True, "reason": s.finish_reason, "notes": s.finish_notes}


# 暴露给 LLM 的工具清单（顺序即提示顺序：先观察、再修复、最后收尾）
def build_tools() -> list:
    return [
        get_metadata,
        list_obs_columns,
        value_counts,
        sample_values,
        judge_data,
        qc_summary,
        rename_column,
        split_column,
        drop_obs_columns,
        execute_code,
        load_skill,
        finish,
    ]


def execute_by_name(name: str, args: dict = None) -> dict:
    """按工具名执行（公开入口，供轨迹重放与外部调用）。

    语义与 ReAct 循环内部的分发一致：未知工具、参数错误、执行异常一律转成可读 error
    返回，而不是抛异常中断调用方。

    注意：本函数**不创建会话**。调用方必须先用 preprocess_session.set_session() 建立
    会话并加载数据，否则工具会返回「预处理会话未初始化」。高层封装见
    app/tools/preprocess_replay.py::run_replay()。
    """
    tools_by_name = {t.name: t for t in build_tools()}
    t = tools_by_name.get(name)
    if t is None:
        return {"error": f"未知工具 {name}", "available": list(tools_by_name.keys())}
    try:
        return t.invoke(args or {})
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
