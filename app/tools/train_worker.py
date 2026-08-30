"""训练子进程 worker：独立进程运行 scLinformer 训练 + 评测。

设计（对齐架构方案「阶段 2 子进程隔离」）：
- 训练是重负载长任务（占满 GPU/CPU、可能 OOM/段错误），放进独立子进程，
  崩溃不连累 FastAPI 主服务；
- 生命周期里程碑写 progress.jsonl（父进程轮询转发给前端）；
- 模型自身 stdout（epoch 进度等）由父进程重定向到 train_stdout.log 并尾随转发；
- 最终结果（或报错堆栈）写 train_result.json，父进程读取汇总。

调用方式（父进程）：
    python -m app.tools.train_worker <job.json>
其中 job.json 含 rna_path / output_dir / model_config。
"""

import json
import os
import sys
import traceback

PROGRESS_FILE = "progress.jsonl"
RESULT_FILE = "train_result.json"


def _append_progress(prog_path: str, message: str, **extra):
    """追加一条生命周期里程碑（父进程轮询转发为 training_log 事件）。"""
    rec = {"event": "log", "message": str(message)}
    rec.update(extra)
    with open(prog_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def main():
    if len(sys.argv) < 2:
        print("用法: python -m app.tools.train_worker <job.json>", file=sys.stderr)
        sys.exit(2)

    job_path = sys.argv[1]
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)

    output_dir = job["output_dir"]
    prog_path = os.path.join(output_dir, PROGRESS_FILE)
    result_path = os.path.join(output_dir, RESULT_FILE)

    try:
        import scanpy as sc  # noqa: E402 惰性导入

        _append_progress(prog_path, "训练子进程启动，加载数据")
        adata = sc.read_h5ad(job["rna_path"])
        _append_progress(prog_path, "数据加载完成",
                         n_cells=int(adata.n_obs), n_genes=int(adata.n_vars))

        # 复用主模块的训练+评测实现（其中的 emit 会打印到本子进程 stdout，被日志捕获）
        from app.tools.model_tools import _train_and_eval
        result = _train_and_eval(adata, output_dir, job["model_config"])

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, default=str)
        _append_progress(prog_path, "训练子进程正常退出")
    except Exception:
        err = traceback.format_exc()
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"status": "error", "error": err}, f, ensure_ascii=False)
        _append_progress(prog_path, "训练子进程异常退出")
        print(err, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
