"""预处理 ReAct 子图：LLM 通过 skill 描述与工具调用，自主决定如何把数据修对。

架构边界（技术方案第 0 节，务必遵守）：
- LLM 决定「**如何把数据修对**」（改名/拆列/写修复代码/判定何时收手）；
- 「修对之后产什么交付物」（backend.prepare_data / QC / 报告 / 存档）仍是确定性代码，
  见 preprocess_tools.finalize_preprocess()。模型契约永远不经过 LLM。

图结构（两个节点 + 一条条件边，刻意不用预构建件以获得预算与终止的完全控制）：

    START -> agent -> (有 tool_calls 且未 finish) -> tools -> agent ...
                   -> (已 finish / 无 tool_calls / 超预算) -> END

护栏（方案第 6 节）：
  步数预算、时间预算、无进展检测、显式 finish 终止、LLM 异常降级、写操作后强制 judge
  （最后一条在 preprocess_agent_tools._after_write 里）。
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
from app.llm import get_llm
from app.skills.library import get_library
from app.tools import preprocess_tools as pt
from app.tools.common import json_safe as _json_safe
from app.tools.preprocess_agent_tools import build_tools
from app.tools.preprocess_replay import save_trace
from app.tools.preprocess_session import PreprocessSession, now_iso, set_session

# 会产生数据变更的工具（用于轨迹标记与重放筛选）
WRITE_TOOLS = {"rename_column", "split_column", "drop_obs_columns", "execute_code"}


class ReactState(TypedDict):
    """ReAct 子图状态。

    注意 messages 必须用 add_messages reducer —— 否则每轮会**覆盖**而不是追加历史，
    LLM 就会丢失上下文（这是 LangGraph 里最常见的坑）。
    """
    messages: Annotated[list, add_messages]
    finished: bool
    finish_reason: str


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------
def build_system_prompt(session) -> str:
    """系统提示：技能清单（渐进式披露，只给 name + description）+ 工作纪律。"""
    library = get_library()
    skills = library.disclose()
    skill_lines = "\n".join(f"- {s['name']}：{s['description']}" for s in skills) or "- （无）"

    return f"""你是单细胞 RNA 测序（scRNA-seq）数据预处理助手，为后续的 {session.backend.name} 训练准备输入。

# 你的目标
让判定契约（judge）的 FAIL 项清零。WARNING 需判断是否可接受，并在 finish 的 notes 里说明。

# 可用技能（调用 load_skill(name) 获取完整操作手册）
{skill_lines}

**动手之前先调用 load_skill("preprocess-qc")** —— 它包含完整的判定契约表与三级分流修复 SOP。

# 可用工具
- 观察（只读）：get_metadata / list_obs_columns / value_counts / sample_values / judge_data / qc_summary
- 修复（写）  ：rename_column / split_column / drop_obs_columns / execute_code
- 控制        ：load_skill / finish

# 工作纪律（严格遵守）
1. **先观察，再动手**：不清楚某列是什么，就用 value_counts / sample_values 看清楚，不要凭列名猜测语义。
2. **写操作后系统会自动 judge**，判定结果就在工具返回值里；请据此决定下一步，不要无视 fail_items。
3. **不确定就停**：无法从数据本身无歧义推断时，调用 finish(reason="ambiguous", notes="...")，
   说明你观察到了什么、需要用户确认什么。**绝不臆造数据、绝不猜测标签语义。**
4. **不许伪造进度**：不要为了"看起来完成"而删列、改值。任何数据变更都必须有依据。
5. **细胞数守恒**：改名/拆列/删列都不应改变细胞数；若必须过滤细胞，请在 notes 里说明理由。
6. execute_code 是万能兜底，但**只在其它工具无法表达你的修复意图时使用**；
   优先用语义明确的 rename_column / split_column。
7. 完成后**必须**调用 finish()。只返回文字而不调用 finish 视为异常。
"""


def _kickoff_prompt(session) -> str:
    """首条用户消息：把已探明的数据事实与首次判定结果交给 LLM。"""
    v = session.verdicts or {}
    rows = "\n".join(
        f"- [{x['status']}] {x['item']}：{x['evidence']} → 建议动作：{x['action']}"
        for x in (v.get("verdicts") or [])
    ) or "- （判定结果为空）"

    meta = session.metadata or {}
    return (
        f"数据已加载：{meta.get('n_cells', '?')} 细胞 × {meta.get('n_genes', '?')} 基因，"
        f"目标模型为 {session.backend.name}，稀疏度 {meta.get('sparsity', '?')}。\n\n"
        f"当前判定契约结果：\n{rows}\n\n"
        f"请先加载 preprocess-qc 手册，然后用观察类工具把需要修复的项看清楚，"
        f"再动手修复。完成后调用 finish()。"
    )


# ---------------------------------------------------------------------------
# 护栏
# ---------------------------------------------------------------------------
def _budget_exceeded(session) -> bool:
    if session.steps >= config.PREPROCESS_MAX_STEPS:
        return True
    return (time.time() - session.start_ts) > config.PREPROCESS_TIME_BUDGET


def _no_progress(session) -> bool:
    """连续 N 轮写操作后 FAIL 集合完全不变 => 判定卡住。

    这是防死循环的关键：LLM 有时会反复做同一种无效修复。
    """
    h = session.judge_history
    limit = max(2, int(config.PREPROCESS_NO_PROGRESS_LIMIT))
    if len(h) < limit:
        return False
    return len(set(h[-limit:])) == 1


def _summarize(result) -> dict:
    """轨迹里的结果摘要：保留结论性字段，截断长文本，避免轨迹文件膨胀。"""
    if not isinstance(result, dict):
        return {"raw": str(result)[:300]}
    keep_keys = ("ok", "status", "error", "judge", "fail_items", "n_cells_now",
                 "row_conservation_violated", "note", "finished", "reason", "notes",
                 "column", "n_categories", "obs_columns")
    out = {}
    for k, v in result.items():
        if k in keep_keys:
            out[k] = _json_safe(v)
        elif k in ("content", "stdout"):
            out[k] = str(v)[:500]
    return out or {"keys": list(result.keys())[:10]}


def _execute_tool(tools_by_name: dict, name: str, args: dict) -> dict:
    """执行单个工具调用：任何异常都转成可读 error 返回给 LLM，而不是中断决策。"""
    t = tools_by_name.get(name)
    if t is None:
        return {"error": f"未知工具 {name}", "available": list(tools_by_name.keys())}
    try:
        return t.invoke(args or {})
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# 决策循环
# ---------------------------------------------------------------------------
def _run_react_loop(session: PreprocessSession):
    """驱动 ReAct 循环，直到 finish / 无工具调用 / 触发护栏 / LLM 异常。"""
    tools = build_tools()
    tools_by_name = {t.name: t for t in tools}
    llm = get_llm(temperature=config.PREPROCESS_TEMPERATURE,
                  model=config.PREPROCESS_MODEL).bind_tools(tools)

    def agent_node(state: ReactState):
        # 1) 已显式 finish
        if session.finished:
            return {"finished": True, "finish_reason": session.finish_reason or "finished"}
        # 2) 预算 / 无进展护栏
        if _budget_exceeded(session):
            session.finished = True
            session.finish_reason = session.finish_reason or "budget"
            emit("preprocess_warn", {
                "message": f"ReAct 达到预算上限（steps={session.steps}，"
                           f"耗时 {round(time.time() - session.start_ts, 1)}s），强制结束决策"})
            return {"finished": True, "finish_reason": session.finish_reason}
        if _no_progress(session):
            session.finished = True
            session.finish_reason = session.finish_reason or "no_progress"
            emit("preprocess_warn", {
                "message": "ReAct 连续多轮修复后 FAIL 集合无变化，判定卡住，强制结束决策"})
            return {"finished": True, "finish_reason": session.finish_reason}

        # 3) 调 LLM
        try:
            resp = llm.invoke(state["messages"])
        except Exception as e:
            session.finished = True
            session.finish_reason = session.finish_reason or "llm_error"
            emit("preprocess_warn", {"message": f"LLM 调用失败，结束决策：{e}"})
            return {"finished": True, "finish_reason": session.finish_reason}

        session.steps += 1
        calls = [tc["name"] for tc in (getattr(resp, "tool_calls", None) or [])]
        emit("agent_thought", {
            "step": session.steps,
            # 600 字足够放下一次决策的说明，同时避免事件流被长文本撑爆
            "content": (resp.content or "")[:600] if isinstance(resp.content, str) else "",
            "tools": calls,
        })
        # finish 工具已置位时立即收敛（避免多绕一轮）
        if session.finished:
            return {"messages": [resp], "finished": True,
                    "finish_reason": session.finish_reason}
        return {"messages": [resp]}

    def tools_node(state: ReactState):
        last = state["messages"][-1]
        outs = []
        for tc in (getattr(last, "tool_calls", None) or []):
            name = tc["name"]
            args = tc.get("args") or {}
            result = _execute_tool(tools_by_name, name, args)
            # 轨迹留痕（重放只取 write=True 的步骤）
            session.trace.append({
                "step": len(session.trace),
                "tool": name,
                "args": _json_safe(args),
                "write": name in WRITE_TOOLS,
                "result": _summarize(result),
            })
            emit("agent_tool_result", {
                "tool": name,
                "ok": not (isinstance(result, dict) and result.get("error")),
            })
            outs.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=tc["id"],
            ))
        return {"messages": outs}

    g = StateGraph(ReactState)
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

    init_state: ReactState = {
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
        session.finished = True
        session.finish_reason = session.finish_reason or "graph_error"
        emit("preprocess_warn", {"message": f"ReAct 子图异常，转入确定性收尾：{e}"})
    finally:
        if not session.finished:
            # 未调用 finish 就自然结束（LLM 只返回文本）：视为异常终止，但流程继续
            session.finish_reason = session.finish_reason or "no_finish_call"
            emit("preprocess_warn", {"message": "LLM 未调用 finish，视为异常终止"})


# ---------------------------------------------------------------------------
# 对外入口（与 run_preprocess 返回结构保持一致）
# ---------------------------------------------------------------------------
def run_preprocess_react(task_id: int, dataset_id: int, dataset_path: str, output_dir: str,
                         model: str = None, model_params: dict = None,
                         fast_path_if_clean: bool = False) -> dict:
    """ReAct 模式预处理：LLM 决策修复 -> 确定性收尾 -> 返回与 pipeline 模式同构的结果。

    fast_path_if_clean=True（对应 PREPROCESS_MODE=auto）：首次判定无 FAIL 时跳过 LLM，
    直接进入确定性收尾 —— 只在真正需要判断时才付 LLM 的代价。
    """
    from app.tools.models import get_backend

    backend = get_backend(model)
    params = {**backend.default_params(), **backend.merge_overrides(model_params)}

    os.makedirs(output_dir, exist_ok=True)
    session = PreprocessSession(task_id=task_id, dataset_id=dataset_id,
                                dataset_path=dataset_path, output_dir=output_dir,
                                backend=backend, params=params)
    set_session(session)

    session.audit_log.append({
        "time": now_iso(), "op": "load_data",
        "detail": f"dataset_id={dataset_id}, path={dataset_path}, model={backend.key}, mode=react",
    })
    emit("preprocess_log", {"message": f"开始预处理（ReAct 模式，模型：{backend.name}）：加载 {dataset_path}"})

    # 1) 确定性加载 + 首次判定（LLM 不参与"读数据"，这部分没有决策空间）
    session.adata = pt._read_data(dataset_path)
    session.n_cells_0 = int(session.adata.n_obs)
    session.metadata = pt._describe(session.adata, dataset_path)
    emit("preprocess_log", {
        "message": f"加载完成：{session.metadata['n_cells']} 细胞 × {session.metadata['n_genes']} 基因"})

    verdicts = pt.judge(session.adata, backend)
    session.verdicts = verdicts
    session.judge_initial = verdicts
    session.audit_log.append({"time": now_iso(), "op": "judge",
                              "detail": json.dumps(verdicts, ensure_ascii=False)})
    fail_items = sorted(v["item"] for v in verdicts.get("verdicts", []) if v["status"] == "FAIL")
    session.judge_history.append(tuple(fail_items))
    emit("preprocess_log", {"message": f"首次判定：{verdicts['status']}（FAIL 项：{fail_items or '无'}）"})

    # 2) 决策：快路径 or ReAct
    if fast_path_if_clean and not fail_items:
        session.finish_reason = "fast_path"
        emit("preprocess_log", {"message": "首次判定无 FAIL，跳过 LLM 决策（快路径），直接进入收尾"})
    else:
        emit("preprocess_log", {
            "message": f"进入 ReAct 决策循环（模型 {config.PREPROCESS_MODEL}，"
                       f"预算 {config.PREPROCESS_MAX_STEPS} 步 / {config.PREPROCESS_TIME_BUDGET}s）"})
        _run_react_loop(session)

    # 3) 决策后复检 + 轨迹归档
    verdicts = pt.judge(session.adata, backend)
    session.verdicts = verdicts
    session.audit_log.append({"time": now_iso(), "op": "re_judge",
                              "detail": json.dumps(verdicts, ensure_ascii=False)})
    emit("preprocess_log", {"message": f"决策结束后判定：{verdicts['status']}"})

    trace_file = save_trace(output_dir, session)
    emit("preprocess_log", {
        "message": f"Agent 轨迹已归档：{trace_file}"
                   f"（{session.steps} 步，结束原因 {session.finish_reason or 'unknown'}）"})

    # 4) 确定性收尾（与 pipeline 模式共用，保证产物一致）
    result = pt.finalize_preprocess(
        adata=session.adata, verdicts=verdicts, output_dir=output_dir, backend=backend,
        params=params, audit_log=session.audit_log, dataset_path=dataset_path,
    )
    result.update({
        "preprocess_mode": "react",
        "agent_trace": trace_file,
        "agent_steps": session.steps,
        "agent_finish_reason": session.finish_reason,
        "agent_finish_notes": session.finish_notes,
    })
    return result
