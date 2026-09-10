"""本地工具（第二阶段：全部本地函数，不再使用 MCP）。

- preprocess_dataset：预处理（load_data -> judge -> 修复 -> 按模型加工 -> 分析 -> 报告 -> 保存）
- run_training：模型运行（分析环境 + 按模型决定参数 -> 子进程训练 + 统一评测）
- save_report：保存 HTML 报告到输出目录
"""

import os

from app.tools.model_tools import run_model_stage
from app.tools.preprocess_tools import run_preprocess


def preprocess_dataset(task_id: int, dataset_id: int, dataset_path: str, output_dir: str,
                       model: str = None, model_params: dict = None) -> dict:
    """预处理阶段：交给数据预处理工具编排（judge -> 修复 -> 按模型加工 -> 分析 -> 报告 -> 保存）。"""
    return run_preprocess(task_id, dataset_id, dataset_path, output_dir,
                          model=model, model_params=model_params)


def run_training(task_id: int, output_dir: str, preprocess: dict = None,
                 model: str = None, model_params: dict = None) -> dict:
    """模型训练阶段：分析上一步结果 + 本机环境 -> 按所选模型决定参数 -> 训练 + 评测。"""
    return run_model_stage(task_id, output_dir, preprocess,
                           model=model, model_params=model_params)


def save_report(output_dir: str, content: str) -> str:
    """保存 HTML 报告到输出目录，返回报告文件路径。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
