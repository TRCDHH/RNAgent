"""训练阶段 ReAct 恢复子图：LLM 诊断训练失败原因并调整参数重试（有界）。

设计要点（对齐"先默认、主要是 batch_size、OOM 改完重训"的诉求）：

- **先默认**：第一次尝试一律用规则默认参数（`resolve_config()`），不花 LLM 成本 ——
  因为 batch_size 查表规则在正常路径上已足够可靠，让 LLM 决定"正常情况"的参数
  只会引入不确定性而没有信息增量；
- **失败才 ReAct**：只有训练真的失败时，才让 LLM 拿到失败原因、环境、参数 Schema
  与历史尝试，去决定「改哪个参数、改成多少」；
- **有界重试**：尝试次数上限 + 恢复时长预算 + 拒绝重复参数，三重护栏防止无限烧 GPU；
- **契约不变**：LLM 只碰**参数值**，`backend.train()` 仍是确定性代码 ——
  模型插件抽象（哪个模型需要 counts 层、要不要内部归一化）绝不被 LLM 绕过。

架构意义：这是 workflow → agent 的关键一步 —— 系统从「失败即上报」变成
「失败 → 诊断 → 调整 → 重试」。补上这条回路，整个系统才具备自主恢复能力。
"""

import json
import os
import time
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app import config
from app.event_emitter import emit
from app.llm import get_llm, llm_configured
from app.skills.library import get_library
from app.tools.common import json_safe as _json_safe
from app.tools.model_tools import attempt_training, prepare_training_context
from app.tools.training_agent_tools import build_training_tools
from app.tools.training_session import (
    TrainingSession, clear_training_session, now_iso, set_training_session,
)


class TrainingReactState(TypedDict):
    """恢复循环状态。messages 必须用 add_messages（追加而非覆盖），否则会丢失上下文。"""
    messages: Annotated[list, add_messages]
    finished: bool
    finish_reason: str


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------
def build_system_prompt(session: TrainingSession) -> str:
    library = get_library()
    skills = library.disclose()
    skill_lines = "\n".join(f"- {s['name']}：{s['description']}" for s in skills) or "- （无）"

    return f"""你是单细胞模型训练的参数调优助手。当前 {session.backend.name} 的一次训练尝试已经失败，
你的任务是**判断失败原因、调整参数、让它跑通**。

# 可用技能（调用 load_skill(name) 获取完整手册）
{skill_lines}

# 可用工具
- get_training_context()：数据规模、环境、参数 Schema、当前参数、历史尝试（**先调用它**）
- load_skill(name)：加载手册，建议先读 training-troubleshoot
- retrain(params, reason)：用调整后的参数重新训练一次（真实启动训练，昂贵）
- finish(reason, notes)：放弃恢复

# 工作纪律（严格遵守）
1. **先看再改**：先 get_training_context() 与 load_skill("training-troubleshoot")，
   不要凭感觉改参数。
2. **只改最相关的那一个**：针对当前 error_kind 改最可能解决它的参数
   （如 oom → 调小 batch_size；diverged → 调小学习率类参数）。
   一次改太多会无法归因。
3. **取值必须合法**：参数值要落在 param_specs 的 min/max 内、类型正确，
   名字必须是 param_specs 里出现过的。
4. **珍惜尝试次数**：总共只剩 {session.attempts_left()} 次机会，每次 retrain 都真实消耗一次，
   不要提交无意义的重复参数。
5. **该放弃就放弃**：依赖缺失、数据本身有问题、硬件根本不够时，
   立即 finish() 并写清原因与建议，不要浪费尝试次数。
6. 成功**不需要**手动 finish —— retrain 成功后会自动结束。
"""


def _kickoff_prompt(session: TrainingSession) -> str:
    last = session.last_failure()
    return (
        f"训练尝试 #{last.get('n', '?')} 失败。\n\n"
        f"失败分类：{last.get('error_kind') or 'unknown'}\n"
        f"错误信息：\n{(last.get('error') or '（无）')[:1200]}\n\n"
        f"本次尝试使用的参数：{json.dumps(last.get('params') or {}, ensure_ascii=False)}\n"
        f"剩余尝试次数：{session.attempts_left()} / {session.max_attempts}\n\n"
        f"请先调用 get_training_context() 看清全貌，再 load_skill(\"training-troubleshoot\")，"
        f"然后决定如何调整参数并 retrain。若判断无法修复，请调用 finish() 说明原因。"
    )


# ---------------------------------------------------------------------------
# 护栏
# ---------------------------------------------------------------------------
def _budget_exceeded(session: TrainingSession) -> bool:
    if session.attempts_left() <= 0:
        return True
    return session.recover_budget_exceeded()


def _execute_tool(tools_by_name: dict, name: str, args: dict) -> dict:
    """执行工具：异常转成可读 error 返回给 LLM，而不是中断恢复循环。"""
    t = tools_by_name.get(name)
    if t is None:
        return {"error": f"未知工具 {name}", "available": list(tools_by_name.keys())}
    try:
        return t.invoke(args or {})
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _summarize(result) -> dict:
    """轨迹摘要：只保留结论性字段，避免轨迹文件膨胀。"""
    if not isinstance(result, dict):
        return {"raw": str(result)[:300]}
    keys = ("ok", "error", "error_kind", "attempt", "elapsed_s", "attempts_left",
            "finished", "reason", "notes", "message")
    out = {k: _json_safe(v) for k, v in result.items() if k in keys}
    if "content" in result:
        out["content"] = str(result["content"])[:500]
    return out or {"keys": list(result.keys())[:10]}


# ---------------------------------------------------------------------------
# 恢复循环
# ---------------------------------------------------------------------------
def _run_recovery_loop(session: TrainingSession):
    tools = build_training_tools()
    tools_by_name = {t.name: t for t in tools}
    llm = get_llm(temperature=config.TRAINING_TEMPERATURE,
                  model=config.TRAINING_MODEL).bind_tools(tools)

    def agent_node(state: TrainingReactState):
        if session.finished:
            return {"finished": True, "finish_reason": session.finish_reason or "finished"}
        if _budget_exceeded(session):
            session.finished = True
            session.finish_reason = session.finish_reason or (
                "retries_exhausted" if session.attempts_left() <= 0 else "budget")
            emit("training_warn", {
                "message": f"训练恢复达到护栏上限（剩余尝试 {session.attempts_left()} 次），结束恢复循环"})
            return {"finished": True, "finish_reason": session.finish_reason}

        try:
            resp = llm.invoke(state["messages"])
        except Exception as e:
            session.finished = True
            session.finish_reason = session.finish_reason or "llm_error"
            emit("training_warn", {"message": f"LLM 调用失败，结束恢复循环：{e}"})
            return {"finished": True, "finish_reason": session.finish_reason}

        calls = [tc["name"] for tc in (getattr(resp, "tool_calls", None) or [])]
        emit("agent_thought", {
            "stage": "training",
            "step": len(session.trace) + 1,
            "content": (resp.content or "")[:600] if isinstance(resp.content, str) else "",
            "tools": calls,
        })
        if session.finished:
            return {"messages": [resp], "finished": True,
                    "finish_reason": session.finish_reason}
        return {"messages": [resp]}

    def tools_node(state: TrainingReactState):
        last = state["messages"][-1]
        outs = []
        for tc in (getattr(last, "tool_calls", None) or []):
            name = tc["name"]
            args = tc.get("args") or {}
            # retrain 自己会往 session.trace 里记带 attempt 的条目，这里只记其它工具
            result = _execute_tool(tools_by_name, name, args)
            if name != "retrain":
                session.trace.append({
                    "step": len(session.trace), "tool": name,
                    "args": _json_safe(args), "result": _summarize(result),
                })
            emit("agent_tool_result", {
                "stage": "training", "tool": name,
                "ok": not (isinstance(result, dict) and result.get("error")),
            })
            outs.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=tc["id"],
            ))
        return {"messages": outs}

    g = StateGraph(TrainingReactState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent",
        lambda s: "end" if s.get("finished") else "tools",
        {"tools": "tools", "end": END},
    )
    g.add_edge("tools", "agent")
    compiled = g.compile()

    init_state: TrainingReactState = {
        "messages": [
            SystemMessage(content=build_system_prompt(session)),
            HumanMessage(content=_kickoff_prompt(session)),
        ],
        "finished": False,
        "finish_reason": "",
    }
    try:
        compiled.invoke(init_state)
    except Exception as e:
        session.finish_reason = session.finish_reason or "graph_error"
        emit("training_warn", {"message": f"训练恢复子图异常：{e}"})
    finally:
        if not session.finished:
            session.finished = True
            session.finish_reason = session.finish_reason or "no_finish_call"
            emit("training_warn", {"message": "恢复循环未显式结束（未调用 retrain 成功或 finish）"})


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
def _finalize(session: TrainingSession, result: dict, ctx: dict, mode: str = "react") -> dict:
    """组装最终训练结果：在既有结果上补充参数、环境、尝试历史与模式信息。"""
    result.update({
        "config": _json_safe(session.current_params or ctx["base_config"]),
        "env": ctx["env"],
        "data": {"n_cells": ctx["n_cells"], "n_genes": ctx["n_genes"]},
        "processed_path": ctx["rna_path"],
        "model": ctx["backend"].key,
        "model_name": ctx["backend"].name,
        "model_params": _json_safe(ctx["model_params"]),
        "training_mode": mode,
        "training_attempts": session.attempts,
        "training_finish_reason": session.finish_reason,
        "training_recovered": len(session.attempts) > 1,
    })
    return _json_safe(result)


def run_model_stage_react(task_id: int, output_dir: str, preprocess: dict = None,
                          model: str = None, model_params: dict = None) -> dict:
    """训练阶段 ReAct 入口：默认参数起手；失败后由 LLM 诊断、调参、重试。

    成功（首次或重试后）返回训练结果；耗尽尝试次数仍失败则 raise ——
    与 default 路径的失败语义保持一致，由 training_node 转成 training_status=failed。
    """
    ctx = prepare_training_context(output_dir, preprocess, model, model_params)
    session = TrainingSession(task_id, output_dir, ctx,
                              max_attempts=config.TRAINING_MAX_ATTEMPTS)
    set_training_session(session)
    session.audit_log.append({
        "time": now_iso(), "op": "prepare_training",
        "detail": f"model={ctx['backend'].key}, n_cells={ctx['n_cells']}, "
                  f"base_config={json.dumps(ctx['base_config'], ensure_ascii=False, default=str)}",
    })

    try:
        # ---- 1) 首次尝试：规则默认参数，不经过 LLM（"先默认"）----
        emit("training_attempt", {
            "attempt": 1, "attempt_max": session.max_attempts,
            "status": "running", "changes": [], "reason": "default_params",
        })
        emit("training_log", {
            "message": f"第 1/{session.max_attempts} 次训练尝试：使用规则默认参数（未调用 LLM）"})
        t0 = time.time()
        first = attempt_training(ctx["rna_path"], output_dir, ctx["base_config"],
                                 model=ctx["backend"].key, atac_path=ctx.get("atac_path"))
        elapsed = round(time.time() - t0, 1)
        session.record_attempt(ctx["base_config"], first, elapsed)
        emit("training_attempt", {
            "attempt": 1, "attempt_max": session.max_attempts,
            "status": "success" if first["ok"] else "failed",
            "changes": [], "error_kind": first.get("error_kind"), "elapsed_s": elapsed,
        })

        if first["ok"]:
            session.finish_reason = "first_attempt_ok"
            emit("training_log", {"message": "首次尝试即成功，无需 ReAct 恢复"})
            return _finalize(session, first["result"], ctx)

        # ---- 2) 失败 -> 有界 ReAct 恢复 ----
        if not llm_configured():
            emit("training_warn", {
                "message": "未配置 LLM_API_KEY，无法进行 ReAct 恢复，按失败上报"})
            raise RuntimeError(f"训练失败：{first['error']}")

        if session.attempts_left() <= 0:
            raise RuntimeError(
                f"训练失败且无剩余尝试机会（上限 {session.max_attempts} 次）：{first['error']}")

        emit("training_warn", {
            "message": f"训练失败（{first.get('error_kind')}），进入 ReAct 恢复循环，"
                       f"剩余 {session.attempts_left()} 次尝试机会"})
        session.recover_start_ts = time.time()
        _run_recovery_loop(session)

        if session.best_result is not None:
            return _finalize(session, session.best_result, ctx)

        # ---- 3) 恢复失败：把尝试历史与 Agent 说明一并抛出（便于前端与人工定位）----
        last = session.last_failure()
        detail = f"训练在 {len(session.attempts)} 次尝试后仍失败（{last.get('error_kind')}）：{last.get('error')}"
        if session.finish_notes:
            detail += f"｜Agent 说明：{session.finish_notes}"
        raise RuntimeError(detail)
    finally:
        clear_training_session()
