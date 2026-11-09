"""训练阶段 ReAct 工具集：把「诊断失败 + 调整参数 + 重新训练」暴露给 LLM（LangChain @tool）。

与预处理工具集一致的设计取舍，但有一处关键差异：
- 观察类合并成一个 get_training_context（一次给够上下文，减少往返）；
- **行动类只有一个 retrain** —— 因为这里的"行动"是真实启动一次训练子进程（昂贵、不可撤回），
  所以把「校验参数」与「执行」合成一步：参数非法时**在训练前**就返回错误、不消耗尝试次数；
- 控制类含显式 finish —— 放弃恢复也必须说清原因，便于审计与人工接手。

护栏（都在 retrain 里，不依赖 LLM 自觉）：
尝试次数上限、恢复时长预算、拒绝与上次完全相同的参数（无进展防护）。
"""

import json
import time

from langchain_core.tools import tool

from app.event_emitter import emit
from app.skills.library import get_library
from app.tools.model_tools import attempt_training
from app.tools.training_session import get_training_session, now_iso


# ---------------------------------------------------------------------------
# 会话与审计
# ---------------------------------------------------------------------------
def _require_session():
    s = get_training_session()
    if s is None:
        raise RuntimeError("训练会话未初始化")
    return s


def _audit(op: str, detail: str):
    s = get_training_session()
    if s is not None:
        s.audit_log.append({"time": now_iso(), "op": op, "detail": detail})


def _as_dict(v):
    """LLM 有时把对象参数序列化成 JSON 字符串传进来，这里统一兼容。"""
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip():
        try:
            d = json.loads(v)
            return d if isinstance(d, dict) else None
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# 观察类
# ---------------------------------------------------------------------------
@tool
def get_training_context() -> dict:
    """查看训练上下文：数据规模、obs 列、GPU/显存、模型参数 Schema、当前参数与历史尝试。

    这是你判断该如何调整参数的**唯一依据**，开始前必须先调用它。
    - param_specs：每个可调参数的 type / min / max / 默认值 / 说明，取值必须落在其范围内；
    - attempts：每次尝试的参数、结果与失败原因（error_kind: oom / diverged / dependency ...）；
    - attempts_left：还剩几次重试机会（用尽后 retrain 会直接拒绝）。
    """
    s = _require_session()
    return {
        "dataset": {
            "n_cells": s.n_cells,
            "n_genes": s.n_genes,
            "obs_columns": s.obs_columns,
        },
        "environment": s.env,
        "model": {
            "key": s.backend.key,
            "name": s.backend.name,
            "description": s.backend.get_description(),
            "capabilities": s.backend.capabilities.to_dict(),
        },
        "param_specs": [
            sp.to_dict(default=s.base_config.get(sp.name)) for sp in s.backend.param_specs
        ],
        "base_config": s.base_config,
        "current_params": s.current_params,
        "attempts": s.attempts,
        "attempts_used": len(s.attempts),
        "attempts_max": s.max_attempts,
        "attempts_left": s.attempts_left(),
    }


# ---------------------------------------------------------------------------
# 行动类
# ---------------------------------------------------------------------------
@tool
def retrain(params: dict, reason: str) -> dict:
    """用调整后的参数重新训练一次（会**真实启动训练子进程**，可能耗时数分钟至数小时）。

    params 只需给出要改的参数，会与当前参数合并，例如 {"batch_size": 32}。
    参数按模型 Schema 强转与校验：非法值会被拒绝，且**不消耗尝试次数**。
    与上一次尝试完全相同的参数也会被拒绝（结果不会变，纯属浪费）。

    调用前请先 get_training_context() 并 load_skill("training-troubleshoot")。
    成功时无需再调用 finish —— 本工具会自动结束恢复循环。
    返回 error 表示"这次尝试也失败了"，请继续分析；若已无可行方案请调用 finish()。
    """
    s = _require_session()

    # ---- 护栏：尝试次数 / 时长预算 ----
    if s.attempts_left() <= 0:
        return {"error": f"已达训练尝试上限（{s.max_attempts} 次），无法再重试。请调用 finish()。"}
    if s.recover_budget_exceeded():
        return {"error": "已超出恢复时长预算，无法再重试。请调用 finish()。"}

    # ---- 校验参数（在训练之前，不消耗尝试次数）----
    overrides_in = _as_dict(params)
    if overrides_in is None:
        return {"error": 'params 必须是对象，例如 {"batch_size": 32}'}
    try:
        overrides = s.backend.merge_overrides(overrides_in)
    except ValueError as e:
        return {"error": f"参数校验失败：{e}",
                "param_specs": [sp.to_dict() for sp in s.backend.param_specs]}
    if not overrides:
        return {"error": "没有识别到任何有效参数（未知参数会被丢弃）",
                "valid_names": [sp.name for sp in s.backend.param_specs]}

    merged = {**s.current_params, **overrides}

    # ---- 无进展防护 ----
    # current_params 就是上一次尝试实际使用的参数，因此「合并后无变化」等价于
    # 「重复上一次的参数」。用这一条同时覆盖两种情形（也覆盖 attempts 尚为空时
    # 提交与默认值相同的参数 —— 那同样是浪费一次昂贵训练）。
    changes = [
        {"name": k, "from": s.current_params.get(k), "to": v}
        for k, v in merged.items() if s.current_params.get(k) != v
    ]
    if not changes:
        return {"error": "参数与当前配置完全相同，重新训练不会改变结果；请调整后再试",
                "current_params": s.current_params}

    attempt_no = len(s.attempts) + 1

    emit("training_attempt", {
        "attempt": attempt_no, "attempt_max": s.max_attempts,
        "status": "running", "changes": changes, "reason": reason,
    })
    emit("training_log", {
        "message": f"第 {attempt_no}/{s.max_attempts} 次训练尝试：调整 "
                   + "、".join(f"{c['name']}={c['to']}" for c in changes)
                   + f"（依据：{reason}）"})

    # ---- 真实执行一次训练尝试 ----
    s.current_params = merged
    t0 = time.time()
    outcome = attempt_training(s.rna_path, s.output_dir, merged, model=s.backend.key,
                               atac_path=getattr(s, "atac_path", None))
    elapsed = round(time.time() - t0, 1)

    s.record_attempt(merged, outcome, elapsed)
    s.trace.append({
        "step": len(s.trace), "attempt": attempt_no, "reason": reason,
        "changes": changes, "params": merged, "ok": outcome["ok"],
        "error_kind": outcome.get("error_kind"), "elapsed_s": elapsed,
    })
    _audit("retrain", json.dumps(
        {"attempt": attempt_no, "changes": changes, "reason": reason,
         "ok": outcome["ok"], "error_kind": outcome.get("error_kind")},
        ensure_ascii=False))

    emit("training_attempt", {
        "attempt": attempt_no, "attempt_max": s.max_attempts,
        "status": "success" if outcome["ok"] else "failed",
        "changes": changes, "error_kind": outcome.get("error_kind"),
        "elapsed_s": elapsed,
    })

    if outcome["ok"]:
        s.finished = True
        s.finish_reason = s.finish_reason or "recovered"
        return {"ok": True, "attempt": attempt_no, "elapsed_s": elapsed,
                "message": "训练成功，恢复完成"}

    return {
        "ok": False, "attempt": attempt_no, "elapsed_s": elapsed,
        "error_kind": outcome.get("error_kind"),
        "error": (outcome.get("error") or "")[:1500],
        "attempts_left": s.attempts_left(),
        "hint": "请分析失败原因并调整参数；若已无可行方案请调用 finish()",
    }


# ---------------------------------------------------------------------------
# 控制类
# ---------------------------------------------------------------------------
@tool
def load_skill(name: str) -> dict:
    """加载某个技能的完整操作手册（SKILL.md 正文）。

    可用技能名见系统提示中的清单；**动手之前建议先加载 "training-troubleshoot"**，
    它给出常见异常信号（OOM / loss NaN / 显存告警）与对应的处理动作。
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
    """放弃恢复并结束训练阶段（训练将按失败上报，由人工接手）。

    reason 请用以下之一：
      retries_exhausted 尝试次数用尽仍失败
      no_solution       没有可行的参数调整方案（如依赖缺失、数据本身有问题）
      environment_issue 环境问题（无 GPU / 依赖未装 / 显存根本不够）
      need_manual       需要人工介入
    notes 写给用户的说明（会写入审计与失败原因，请写清你观察到什么、建议怎么处理）。

    成功场景**不需要**调用本工具 —— retrain 成功会自动结束。
    """
    s = _require_session()
    s.finished = True
    s.finish_reason = str(reason or "").strip() or "unspecified"
    s.finish_notes = str(notes or "")
    _audit("finish", json.dumps({"reason": s.finish_reason, "notes": s.finish_notes},
                                ensure_ascii=False))
    emit("training_log", {"message": f"训练恢复决策结束（{s.finish_reason}）"})
    return {"finished": True, "reason": s.finish_reason, "notes": s.finish_notes}


def build_training_tools() -> list:
    """暴露给 LLM 的工具清单（顺序即提示顺序）。"""
    return [get_training_context, load_skill, retrain, finish]
