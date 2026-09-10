"""模型运行阶段编排：分析上一步（预处理）结果 + 本机环境 -> 决定参数 -> 子进程训练评测。

本文件**不出现任何具体模型名**：具体模型由 `app/tools/models/registry.py` 注册，
通过 `get_backend(model_key)` 取得后端，由后端自己决定
`judge_extra / prepare_data / resolve_config / train / evaluate`。

参数规则与决策表写在各模型的 SKILL.md：
    skills/registry/models/<model_key>/SKILL.md
流程规则见 skills/registry/model-running/SKILL.md。

对外提供：inspect_environment / run_model / run_model_stage

注意：torch、scanpy 等重依赖均惰性导入，保证主应用（FastAPI 门户）
在未安装这些库时也能正常启动，仅在实际执行训练时报错提示。
"""

import json  # noqa: E402
import os
import subprocess
import sys
import time

from app import config
from app.event_emitter import emit
from app.tools.common import json_safe as _json_safe
from app.tools.models import get_backend

# scLinformer 源码根目录（保留别名，兼容旧引用）
MODEL_DIR = config.SCLINFORMER_DIR

# 预处理阶段产出的、供训练读取的原始计数 h5ad 文件名
PROCESSED_H5AD = "processed_rna.h5ad"


# ---------------------------------------------------------------------------
# 工具 1：inspect_environment —— 查询本机 GPU / 显存，返回事实供参数决策
# ---------------------------------------------------------------------------
def inspect_environment() -> dict:
    """查询本机运行环境（GPU 是否可用、显存大小），返回 JSON 事实。"""
    env = {"gpu_available": False, "device": "cpu", "gpu_name": None, "vram_mb": None}
    try:
        import torch  # noqa: E402 惰性导入
    except Exception as e:  # torch 未安装：视为无 GPU
        env["torch_error"] = str(e)
        return _json_safe(env)

    env["torch_version"] = str(getattr(torch, "__version__", "?"))
    env["gpu_available"] = bool(torch.cuda.is_available())
    if env["gpu_available"]:
        env["device"] = "cuda"
        try:
            env["gpu_name"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            env["vram_mb"] = int(total // (1024 * 1024))
        except Exception:
            env["vram_mb"] = None
    return _json_safe(env)


# ---------------------------------------------------------------------------
# 子进程训练 + 父进程监控：训练崩溃不连累主服务，进度实时可见
# ---------------------------------------------------------------------------
def _drain_lines(path: str, start: int, encoding: str = "utf-8") -> tuple:
    """增量读取文件行，返回 (新的行数, 新增行列表)。"""
    if not os.path.exists(path):
        return start, []
    with open(path, encoding=encoding, errors="replace") as f:
        lines = f.readlines()
    new = [ln.rstrip("\n") for ln in lines[start:] if ln.strip()]
    return len(lines), new


def _console_safe(text: str) -> str:
    """按当前控制台编码过滤不可打印字符，避免 GBK 控制台 UnicodeEncodeError。"""
    enc = (sys.stdout.encoding or "utf-8")
    try:
        return text.encode(enc, "ignore").decode(enc)
    except Exception:
        return text.encode("ascii", "ignore").decode()


def _train_and_eval_monitored(rna_path: str, output_dir: str, model_config: dict,
                              model: str = None) -> dict:
    """在独立子进程中执行「训练 + 评测」，父进程轮询监控。

    - 子进程（app.tools.train_worker）按 model 取后端跑训练，崩溃 / OOM / 段错误不影响主服务；
    - 父进程轮询 progress.jsonl 转发生命周期事件、尾随 train_stdout.log 转发模型
      自身输出（epoch 进度等），每 5s 用 psutil 上报子进程 CPU / 内存；
    - 结束后读取 train_result.json 汇总返回；异常则抛出含堆栈的 RuntimeError。
    """
    backend = get_backend(model)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    job_path = os.path.join(output_dir, "train_job.json")
    prog_path = os.path.join(output_dir, "progress.jsonl")
    log_path = os.path.join(output_dir, "train_stdout.log")
    result_path = os.path.join(output_dir, "train_result.json")
    for stale in (prog_path, result_path):
        if os.path.exists(stale):
            os.remove(stale)

    with open(job_path, "w", encoding="utf-8") as f:
        json.dump({"rna_path": rna_path, "output_dir": output_dir, "model": backend.key,
                   "model_config": _json_safe(model_config)}, f, ensure_ascii=False)

    with open(log_path, "w", encoding="utf-8") as logf:
        # PYTHONIOENCODING=utf-8：统一子进程 stdout/stderr 编码，避免 GBK 乱码
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-m", "app.tools.train_worker", job_path],
            cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, env=env,
        )
    emit("training_log", {"message": f"训练子进程已启动（pid={proc.pid}，模型 {backend.name}），父进程监控中"})

    # psutil 可选：装了才有资源监控，没装只跟踪存活与进度
    child = None
    try:
        import psutil  # noqa: E402
        child = psutil.Process(proc.pid)
    except Exception:
        pass

    t0 = time.time()
    last_prog = 0
    last_log = 0
    last_mon = 0.0
    while proc.poll() is None:
        # 1) 子进程生命周期里程碑（progress.jsonl）
        last_prog, new_progs = _drain_lines(prog_path, last_prog)
        for ln in new_progs:
            try:
                rec = json.loads(ln)
                emit("training_log", {k: v for k, v in rec.items() if k != "event"})
            except ValueError:
                pass
        # 2) 模型自身 stdout（跳过 emit 的事件行，避免重复）
        last_log, new_logs = _drain_lines(log_path, last_log)
        for ln in new_logs:
            if not ln.startswith("[event]"):
                emit("training_log", {"message": _console_safe(ln)})
        # 3) 资源监控：每 5s 上报子进程 CPU / 内存
        now = time.time()
        if child is not None and now - last_mon >= 5:
            last_mon = now
            try:
                with child.oneshot():
                    emit("training_monitor", {
                        "elapsed_s": round(now - t0, 1),
                        "cpu_percent": child.cpu_percent(interval=0),
                        "mem_mb": round(child.memory_info().rss / 1048576, 1),
                    })
            except Exception:
                child = None  # 子进程刚退出等场景，停止资源监控
        time.sleep(1)

    # 收尾：再读一次，防止退出与轮询之间的漏读
    last_prog, new_progs = _drain_lines(prog_path, last_prog)
    for ln in new_progs:
        try:
            rec = json.loads(ln)
            emit("training_log", {k: v for k, v in rec.items() if k != "event"})
        except ValueError:
            pass
    last_log, new_logs = _drain_lines(log_path, last_log)
    for ln in new_logs:
        if not ln.startswith("[event]"):
            emit("training_log", {"message": _console_safe(ln)})

    result = None
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)

    if proc.returncode != 0:
        err = (result or {}).get("error") or f"训练子进程异常退出（code={proc.returncode}），详见 {log_path}"
        raise RuntimeError(f"训练子进程失败：{err}")
    if not result or result.get("status") != "success":
        raise RuntimeError((result or {}).get("error", "训练子进程未返回结果"))
    emit("training_log", {"message": "训练子进程正常结束，产物已回收"})
    return result


# ---------------------------------------------------------------------------
# 工具 2：run_model —— 按给定 config 执行训练（Agent 兜底 / 手动调用，当前进程内）
# ---------------------------------------------------------------------------
def run_model(rna_path: str, output_dir: str, model_config: dict, model: str = None) -> dict:
    """执行工具：读取 rna_path 的 AnnData，按 model_config 运行所选模型训练 + 评测。"""
    import scanpy as sc  # noqa: E402 惰性导入
    backend = get_backend(model)
    ok, reason = backend.is_available()
    if not ok:
        raise RuntimeError(f"模型 {backend.name} 不可用：{reason}")
    adata = sc.read_h5ad(rna_path)
    return _json_safe(backend.train_and_eval(adata, output_dir, model_config))


# ---------------------------------------------------------------------------
# 编排入口：分析环境 + 上一步结果 -> 决策参数 -> 训练评测
# ---------------------------------------------------------------------------
def run_model_stage(task_id: int, output_dir: str, preprocess: dict = None,
                    model: str = None, model_params: dict = None) -> dict:
    """训练阶段编排：如同一个只「传参 + 执行工具」的 Agent，把决策落到模型运行。"""
    os.makedirs(output_dir, exist_ok=True)
    preprocess = preprocess or {}
    backend = get_backend(model)
    rna_path = preprocess.get("processed_path") or os.path.join(output_dir, PROCESSED_H5AD)
    if not os.path.exists(rna_path):
        raise FileNotFoundError(f"找不到预处理产物：{rna_path}（请先完成数据预处理阶段）")

    ok, reason = backend.is_available()
    if not ok:
        raise RuntimeError(f"模型 {backend.name} 不可用：{reason}")

    # 1) 本机环境
    emit("training_log", {"message": "检查本机运行环境（GPU/显存）"})
    env = inspect_environment()

    # 2) 读取上一步（预处理）产物，由后端决定参数（use_batch / use_cell_type / batch_size ...）
    import scanpy as sc  # noqa: E402 惰性导入
    adata = sc.read_h5ad(rna_path)
    n_cells, n_genes = int(adata.n_obs), int(adata.n_vars)
    obs_columns = [str(c) for c in adata.obs.columns]
    model_config = backend.resolve_config(obs_columns, n_cells, env, overrides=model_params)

    emit("training_log", {
        "message": f"模型 {backend.name} 参数决定：" +
                   ", ".join(f"{k}={model_config[k]}" for k in list(model_config)[:6]) +
                   f"（显存 {env.get('vram_mb')}MB, n_cells={n_cells}）"
    })

    # 3) 子进程训练 + 评测（父进程监控进度与资源）
    result = _train_and_eval_monitored(rna_path, output_dir, model_config, model=backend.key)
    result.update({
        "config": _json_safe(model_config),
        "env": env,
        "data": {"n_cells": n_cells, "n_genes": n_genes},
        "processed_path": rna_path,
        "model": backend.key,
        "model_name": backend.name,
        "model_params": _json_safe(model_params or {}),
    })
    return _json_safe(result)
