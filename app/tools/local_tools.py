"""本地工具（第二阶段：全部本地函数，不再使用 MCP）。

- preprocess_dataset：预处理入口（按 PREPROCESS_MODE 选择决策层，两条路径共用确定性收尾）
- run_training：模型运行（分析环境 + 按模型决定参数 -> 子进程训练 + 统一评测）
- save_report：保存 HTML 报告到输出目录
"""

import os

from app import config
from app.event_emitter import emit
from app.llm import llm_configured
from app.tools.model_tools import run_model_stage
from app.tools.preprocess_tools import run_preprocess


def _resolve_preprocess_mode() -> tuple:
    """解析预处理模式，返回 (mode, fast_path_if_clean)。

    - pipeline：现有确定性链路；
    - react   ：始终由 LLM 通过 skill 描述 + 工具调用自主决策；
    - auto    ：首次判定无 FAIL 时跳过 LLM（快路径），出现 FAIL 才升级 ReAct；
    未配置 LLM_API_KEY 时一律降级为 pipeline —— 保证「不配 key 也能出结果」这一既有性质不被破坏。
    """
    mode = (config.PREPROCESS_MODE or "pipeline").strip().lower()
    if mode not in ("pipeline", "react", "auto"):
        emit("preprocess_warn", {"message": f"未知 PREPROCESS_MODE={mode}，按 pipeline 处理"})
        return "pipeline", False
    if mode == "pipeline":
        return "pipeline", False
    if not llm_configured():
        emit("preprocess_warn", {
            "message": f"PREPROCESS_MODE={mode} 但未配置 LLM_API_KEY，自动降级为 pipeline 模式"})
        return "pipeline", False
    return mode, (mode == "auto")


def preprocess_dataset(task_id: int, dataset_id: int, dataset_path: str, output_dir: str,
                       model: str = None, model_params: dict = None) -> dict:
    """预处理阶段入口：选决策层 -> 决策 -> 确定性收尾。

    「改数据」的决策由 LLM（ReAct）或确定性规则（pipeline）完成；
    「加工/分析/报告/存档」始终由 preprocess_tools.finalize_preprocess 执行，两条路径产物一致。
    ReAct 链路自身异常时回退 pipeline，任务不会因 LLM 不可用而失败。
    """
    mode, fast_path = _resolve_preprocess_mode()
    if mode == "pipeline":
        return run_preprocess(task_id, dataset_id, dataset_path, output_dir,
                              model=model, model_params=model_params)

    # 惰性导入：未启用 ReAct 时不加载 langgraph / langchain 工具层
    try:
        from app.graph.preprocess_agent import run_preprocess_react
        return run_preprocess_react(task_id, dataset_id, dataset_path, output_dir,
                                    model=model, model_params=model_params,
                                    fast_path_if_clean=fast_path)
    except Exception as e:
        emit("preprocess_warn", {
            "message": f"ReAct 预处理失败，回退确定性链路：{type(e).__name__}: {e}"})
        return run_preprocess(task_id, dataset_id, dataset_path, output_dir,
                              model=model, model_params=model_params)


def _resolve_training_mode() -> str:
    """解析训练模式。

    - default：规则决定参数，失败即上报（历史行为）；
    - react  ：首次尝试仍用规则默认值，失败后由 LLM 诊断、调参、重试（有界）；
    未配置 LLM_API_KEY 时 react 降级为 default —— 无 LLM 则恢复循环无从谈起，
    降级后行为与历史完全一致（失败即上报）。
    """
    mode = (config.TRAINING_MODE or "default").strip().lower()
    if mode not in ("default", "react"):
        emit("training_warn", {"message": f"未知 TRAINING_MODE={mode}，按 default 处理"})
        return "default"
    if mode == "react" and not llm_configured():
        emit("training_warn", {
            "message": "TRAINING_MODE=react 但未配置 LLM_API_KEY，降级为 default 模式"})
        return "default"
    return mode


def run_training(task_id: int, output_dir: str, preprocess: dict = None,
                 model: str = None, model_params: dict = None) -> dict:
    """模型训练阶段入口：按 TRAINING_MODE 选择参数决策层。

    两条路径的**第一次尝试完全相同**（都用规则默认参数）；区别只在训练失败之后：
    - default：直接上报失败；
    - react  ：由 LLM 诊断原因、调整参数并重试（有界）。

    注意：训练失败必须向上抛出（不在这里吞掉），否则 training_node 无法把
    training_status 置为 failed —— 会把失败显示成成功。
    """
    if _resolve_training_mode() == "default":
        return run_model_stage(task_id, output_dir, preprocess,
                               model=model, model_params=model_params)

    # 惰性导入：仅启用时加载 langgraph / 训练工具层。
    # 只有「模块加载失败」才回退 default；恢复循环自身失败走异常上报。
    try:
        from app.graph.training_agent import run_model_stage_react
    except Exception as e:
        emit("training_warn", {
            "message": f"训练 ReAct 模块加载失败，回退默认链路：{type(e).__name__}: {e}"})
        return run_model_stage(task_id, output_dir, preprocess,
                               model=model, model_params=model_params)

    return run_model_stage_react(task_id, output_dir, preprocess,
                                 model=model, model_params=model_params)


def save_report(output_dir: str, content: str) -> str:
    """保存 HTML 报告到输出目录，返回报告文件路径。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
