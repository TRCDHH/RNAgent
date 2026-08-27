"""LangGraph 顶层编排状态（PipelineState）。

说明：节点为直接函数（本地工具编排），
顶层状态只保留结构化产物与状态标记。
"""

from typing import TypedDict


class PipelineState(TypedDict, total=False):
    # ── 输入 ──
    task_id: int
    dataset_id: int
    dataset_path: str
    output_dir: str       # 任务工作目录（输出结果目录）
    config: dict          # 由 Agent 自动生成，用户不填
    current_stage: str
    trace_id: str

    # ── 阶段1 产物：数据集预处理 ──
    preprocess: dict
    preprocess_status: str   # pending/success/failed

    # ── 阶段2 产物：模型训练 ──
    training: dict
    training_status: str

    # ── 阶段3 产物：结果分析 / 报告 ──
    evaluation: dict
    report: str
    report_status: str

    # ── 通用 ──
    error: str
