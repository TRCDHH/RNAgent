"""组装顶层 StateGraph：预处理 -> 训练 -> 结果分析（第二阶段，无 MCP / ReAct）。"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.state import PipelineState
from .nodes import (
    preprocess_node,
    training_node,
    analyze_node,
    handle_error_node,
    get_router_preprocess,
    get_router_training,
    get_router_analyze,
)


def build_app(checkpointer=None):
    g = StateGraph(PipelineState)

    g.add_node("preprocess", preprocess_node)
    g.add_node("training", training_node)
    g.add_node("analyze", analyze_node)
    g.add_node("handle_error", handle_error_node)

    g.add_edge(START, "preprocess")
    g.add_conditional_edges("preprocess", get_router_preprocess(),
                            {"ok": "training", "error": "handle_error"})
    g.add_conditional_edges("training", get_router_training(),
                            {"ok": "analyze", "error": "handle_error"})
    g.add_conditional_edges("analyze", get_router_analyze(),
                            {"ok": END, "error": "handle_error"})
    g.add_edge("handle_error", END)

    # 第一阶段用 MemorySaver；后续可换 RedisSaver 实现断点续跑
    return g.compile(checkpointer=checkpointer or MemorySaver())
