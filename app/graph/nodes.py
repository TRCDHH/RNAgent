"""LangGraph 顶层节点：预处理 -> 模型运行 -> 结果分析（生成 HTML 报告）。

第二阶段：节点为直接函数（不再用 ReAct 子图），全部走本地工具。
"""

from app.event_emitter import emit
from app.tools.local_tools import preprocess_dataset, run_training, save_report
from app.tools.report_tools import generate_report_html


def preprocess_node(state: dict):
    emit("stage_start", {"stage": "preprocess"})
    try:
        task_id = state.get("task_id", 1)
        dataset_id = state.get("dataset_id", 1)
        dataset_path = state.get("dataset_path", "")
        output_dir = state.get("output_dir", ".")
        result = preprocess_dataset(task_id, dataset_id, dataset_path, output_dir)
        emit("stage_end", {"stage": "preprocess", "status": "success"})
        return {"preprocess_status": "success", "preprocess": result, "current_stage": "preprocess"}
    except Exception as e:
        emit("stage_error", {"stage": "preprocess", "error": str(e)})
        return {"preprocess_status": "failed", "error": str(e), "current_stage": "preprocess"}


def training_node(state: dict):
    emit("stage_start", {"stage": "training"})
    try:
        task_id = state.get("task_id", 1)
        output_dir = state.get("output_dir", ".")
        preprocess = state.get("preprocess", {})
        result = run_training(task_id, output_dir, preprocess)
        emit("stage_end", {"stage": "training", "status": "success"})
        return {"training_status": "success", "training": result, "current_stage": "training"}
    except Exception as e:
        emit("stage_error", {"stage": "training", "error": str(e)})
        return {"training_status": "failed", "error": str(e), "current_stage": "training"}


def analyze_node(state: dict):
    emit("stage_start", {"stage": "analyze"})
    try:
        task_id = state.get("task_id", 1)
        dataset_id = state.get("dataset_id")
        output_dir = state.get("output_dir", ".")
        preprocess = state.get("preprocess", {})
        training = state.get("training", {})

        html = generate_report_html(dataset_id, output_dir, preprocess, training)
        report_path = save_report(output_dir, html)
        report_url = f"/output/{task_id}/report.html"

        emit("stage_end", {"stage": "analyze", "status": "success", "report_url": report_url})
        return {
            "report_status": "success",
            "report": {"report_path": report_path, "report_url": report_url},
            "current_stage": "analyze",
        }
    except Exception as e:
        emit("stage_error", {"stage": "analyze", "error": str(e)})
        return {"report_status": "failed", "error": str(e), "current_stage": "analyze"}


def handle_error_node(state: dict):
    emit("pipeline_failed", {
        "error": state.get("error", ""),
        "current_stage": state.get("current_stage", ""),
    })
    return {"error": state.get("error", "")}


def _make_router(status_field: str):
    def route(state: dict):
        return "ok" if state.get(status_field) == "success" else "error"
    return route


def get_router_preprocess():
    return _make_router("preprocess_status")


def get_router_training():
    return _make_router("training_status")


def get_router_analyze():
    return _make_router("report_status")
