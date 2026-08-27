"""RNAgent 流水线执行逻辑（CLI 与后端 API 共用）。

- run_pipeline()：跑一条完整 pipeline（预处理 -> 训练 -> 分析），返回结果。
- run()：CLI 入口包装，打印汇总（供 run_demo.py 使用）。
"""

import json

from app.event_emitter import emit, set_context
from app.graph.builder import build_app


def run_pipeline(task_id: int = 1, dataset_id: int = 3,
                 dataset_path: str = None, output_dir: str = None) -> dict:
    set_context(task_id, f"trace_{task_id}")
    emit("pipeline_start", {"task_id": task_id, "dataset_id": dataset_id})

    app = build_app()
    config = {"configurable": {"thread_id": f"task_{task_id}"}}
    initial = {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "dataset_path": dataset_path or "/data/demo.h5ad",
        "output_dir": output_dir or ".",
        "config": {},
        "current_stage": "preprocess",
    }

    try:
        result = app.invoke(initial, config=config)
        emit("pipeline_end", {"status": "success"})
        return {"status": "success", "result": result}
    except Exception as e:
        emit("pipeline_failed", {"error": str(e)})
        return {"status": "failed", "error": str(e)}


def run(task_id: int = 1, dataset_id: int = 3, dataset_path: str = None, output_dir: str = None):
    outcome = run_pipeline(task_id, dataset_id, dataset_path=dataset_path, output_dir=output_dir)
    if outcome["status"] == "success":
        print("\n===== 最终状态（PipelineState）=====")
        print(json.dumps(outcome["result"], ensure_ascii=False, default=str, indent=2))
    else:
        print(f"\n===== 执行失败 =====\n{outcome['error']}")
