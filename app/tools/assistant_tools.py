"""面向 AI 助手的可调用本地工具（LLM function calling）。

供聊天助手（/api/chat）通过 tool calling 调用，查询数据库或任务产物：
- view_datasets：查看所有数据集（查 MySQL）
- view_task_data：查看指定任务数据（查 MySQL）
- view_task_result：查看指定任务结果（报告链接 + 真实评测指标）
"""

import json
import os

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from app import db
from app.llm import get_llm


@tool
def view_datasets() -> dict:
    """查看所有数据集（查询数据库，返回 id/name/path/create_time）。"""
    return {"datasets": db.list_datasets()}


@tool
def view_task_data(task_id: int) -> dict:
    """查看指定任务的详细数据（查询数据库，返回任务元信息与 process 状态）。"""
    task = db.get_task(task_id)
    if not task:
        return {"error": f"任务 {task_id} 不存在"}
    task["process"] = _parse_process(task.get("process"))
    return {"task": task}


@tool
def view_task_result(task_id: int) -> dict:
    """查看指定任务的结果（报告链接与真实评测指标 ARI/ASW 等，来自任务产物）。"""
    task = db.get_task(task_id)
    if not task:
        return {"error": f"任务 {task_id} 不存在"}

    process = _parse_process(task.get("process"))
    result = process.get("result") or {}
    if not isinstance(result, dict):
        result = {}

    report_url = (result.get("report") or {}).get("report_url", "")
    metrics = (result.get("training") or {}).get("metrics") or {}

    # 兜底：数据库没存指标时，直接读任务输出目录里的 summary_metrics.csv
    if not metrics and task.get("path"):
        metrics = _read_summary_metrics(task["path"])

    return {
        "task_id": task_id,
        "status": process.get("status", "unknown"),
        "report_url": report_url,
        "metrics": metrics,
    }


def _read_summary_metrics(output_dir: str) -> dict:
    """从任务输出目录读 summary_metrics.csv（不存在/读失败返回空 dict，NaN 转 None）。"""
    path = os.path.join(output_dir, "summary_metrics.csv")
    if not os.path.exists(path):
        return {}
    try:
        import pandas as pd  # 惰性导入

        df = pd.read_csv(path)
        if df.empty:
            return {}
        return {str(k): (None if pd.isna(df.iloc[0][k]) else df.iloc[0][k]) for k in df.columns}
    except Exception:
        return {}


ASSISTANT_TOOLS = [view_datasets, view_task_data, view_task_result]


def run_assistant(message: str) -> str:
    """聊天助手入口：把工具绑定到 LLM，循环执行 tool calling，直至得到最终文本回复。"""
    llm = get_llm().bind_tools(ASSISTANT_TOOLS)
    messages = [HumanMessage(content=message)]

    for _ in range(5):  # 限制轮次，避免死循环
        resp = llm.invoke(messages)
        messages.append(resp)
        if not getattr(resp, "tool_calls", None):
            return resp.content

        for tc in resp.tool_calls:
            tool_result = _execute_tool(tc["name"], tc.get("args") or {})
            messages.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False, default=str),
                    tool_call_id=tc["id"],
                )
            )
    return resp.content


async def stream_assistant(message: str):
    """流式聊天助手：边生成边 yield 文本片段（支持工具调用）。"""
    llm = get_llm().bind_tools(ASSISTANT_TOOLS)
    messages = [HumanMessage(content=message)]

    for _ in range(5):  # 限制轮次，避免死循环
        full = None
        async for chunk in llm.astream(messages):
            full = chunk if full is None else full + chunk
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                yield text
        if full is None:
            return
        messages.append(full)
        if not getattr(full, "tool_calls", None):
            return
        for tc in full.tool_calls:
            tool_result = _execute_tool(tc["name"], tc.get("args") or {})
            messages.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False, default=str),
                    tool_call_id=tc["id"],
                )
            )


def _execute_tool(name: str, args: dict) -> dict:
    for t in ASSISTANT_TOOLS:
        if t.name == name:
            return t.invoke(args)
    return {"error": f"未知工具 {name}"}


def _parse_process(raw):
    if not raw:
        return {"status": "unknown", "events": []}
    try:
        return json.loads(raw)
    except Exception:
        return {"status": "unknown", "events": []}