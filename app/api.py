"""Python FastAPI 后端（第二阶段，本地 MySQL 持久化 + DeepSeek LLM）。

职责（纯业务门户，无 AI 逻辑，AI 全在 LangGraph 引擎）：
- 数据集管理（CRUD）
- 任务管理（触发 pipeline、查状态、SSE 推送、过程持久化）
- AI 聊天（DeepSeek）
- 输出目录静态托管（/output/{task_id}/report.html）
"""

import asyncio
import json
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, db
from app.event_emitter import load_events, snapshot
from app.llm import llm_configured
from app.main import run_pipeline
from app.tools.assistant_tools import run_assistant, stream_assistant
from app.tools.models import all_backends, default_key, get_backend, list_models, validate
from app.tools.preprocess_replay import run_replay, trace_path

app = FastAPI(title="RNAgent")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 任务输出目录（模型结果目录，含 report.html），静态托管给前端
RUNTIME_TASK_DIR = Path(__file__).resolve().parent.parent / "runtime" / "task"
RUNTIME_TASK_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=RUNTIME_TASK_DIR), name="output")

_running: set = set()
_running_lock = threading.Lock()


@app.on_event("startup")
def _startup():
    try:
        db.init_schema()
    except Exception as e:
        print(f"[warn] 数据库初始化失败（请先执行 init.sql）：{e}")
    # 模型注册表自检：模型实现与 SKILL.md 是否一致（不一致只告警，不影响启动）
    for problem in validate():
        print(f"[warn] 模型一致性自检：{problem}")
    # 运行模式自述：让人一眼看出本次启动到底启用了哪些 Agent 能力
    print(f"[info] LLM={'已配置' if llm_configured() else '未配置'}"
          f" | 预处理 PREPROCESS_MODE={config.PREPROCESS_MODE}（react/auto 需 LLM，否则降级 pipeline）"
          f" | 训练 TRAINING_MODE={config.TRAINING_MODE}"
          f"（最多 {config.TRAINING_MAX_ATTEMPTS} 次尝试）")
    # 模型参数已按模型隔离：旧的全局变量不再生效，提示改用 {模型KEY}_{参数名}
    for k, v in config.LEGACY_MODEL_ENV.items():
        print(f"[warn] 环境变量 {k}={v} 已废弃且不再生效（参数已按模型隔离）；"
              f"请改用 SCLINFORMER_EPOCHS / SCVI_MAX_EPOCHS / SCVI_BATCH_SIZE 等 模型KEY_参数名 形式")
    try:
        reconcile_tasks()
    except Exception as e:
        print(f"[warn] 任务状态对账失败：{e}")


def reconcile_tasks():
    """启动时对账，做两件事（都幂等）：

    ① **状态对账**：DB 是 running、内存里又没有对应运行线程 => 上次进程残留的僵尸。
       结果写回数据库只在流水线收尾时执行一次，中途重启（或那次写库异常）就会永远停在
       running。这里按磁盘产物判断真实结果并修正。
    ② **日志补救**：DB 的 process.events 为空时，从 events.jsonl 把完整日志读回来。
       （注意不能只在①里做——已经对过账、状态已是 success 的任务同样缺日志。）
    """
    for t in db.list_tasks():
        task_id = t["id"]
        proc = _parse_process(t.get("process"))
        changed = False

        if (proc.get("status") or "") == "running":
            with _running_lock:
                still_running = task_id in _running
            if not still_running:
                out_dir = t.get("path") or ""
                has_report = bool(out_dir) and (Path(out_dir) / "report.html").exists()
                proc["status"] = "success" if has_report else "failed"
                if has_report:
                    res = proc.get("result")
                    if not isinstance(res, dict):
                        res = {}
                        proc["result"] = res
                    res.setdefault("report", {})["report_url"] = f"/output/{task_id}/report.html"
                    res.setdefault("report_status", "success")
                else:
                    proc["error"] = proc.get("error") or "任务未完成（服务进程在收尾前中断）"
                changed = True
                print(f"[warn] 任务状态对账：#{task_id} running -> {proc['status']}")

        # 日志补救：DB 里没有事件就从落盘文件读回
        if not (proc.get("events") or []):
            events = load_events(task_id)
            if events:
                proc["events"] = events
                changed = True
                print(f"[warn] 任务日志补救：#{task_id} 从 events.jsonl 恢复 {len(events)} 条")

        if changed:
            db.update_task_process(task_id, json.dumps(proc, ensure_ascii=False, default=str))


# ---------- 模型（注册表驱动，新增模型自动出现在这里） ----------
@app.get("/api/models")
def api_models():
    """返回可选模型列表 + 参数 Schema（前端据此动态渲染参数表单）。"""
    return {
        "models": list_models(with_schema=True),
        "default": default_key(),
        "warnings": validate(),
        # 运行时模式：前端/调用方可据此判断当前是否启用了 ReAct 决策与失败恢复
        "runtime": {
            "llm_configured": llm_configured(),
            "preprocess_mode": config.PREPROCESS_MODE,
            "training_mode": config.TRAINING_MODE,
            "training_max_attempts": config.TRAINING_MAX_ATTEMPTS,
        },
    }


@app.get("/")
def index():
    # no-store：避免浏览器按 ETag/Last-Modified 缓存 index.html，
    # 否则改完前端刷新看到的是旧版本（"改了没变化"的根因）
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"},
    )


# ---------- 数据集 ----------
@app.get("/api/datasets")
def list_datasets():
    return {"datasets": db.list_datasets()}


class DatasetIn(BaseModel):
    name: str
    path: str
    atac_path: str = None      # 跨模态模型的配对 ATAC 数据（可选；单模态模型忽略）


@app.post("/api/datasets")
def create_dataset(body: DatasetIn):
    dataset_id = db.create_dataset(body.name, body.path, body.atac_path)
    return {"id": dataset_id}


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: int):
    db.delete_dataset(dataset_id)
    return {"ok": True}


# ---------- 任务 ----------
@app.get("/api/tasks")
def list_tasks():
    tasks = []
    for t in db.list_tasks():
        proc = _parse_process(t["process"])
        tasks.append({
            "id": t["id"],
            "name": t["name"],
            "path": t["path"],
            "model": t.get("model") or default_key(),
            "dataset_id": t.get("dataset_id"),
            # 是否留有 Agent 轨迹：决定前端是否显示「重放」按钮
            "has_trace": _has_trace(t.get("path")),
            # 列表只回状态摘要：events 可能很大（单任务数百条），详情走 /api/tasks/{id}/state
            "process": {
                "status": proc.get("status") or "unknown",
                "error": proc.get("error"),
                "event_count": len(proc.get("events") or []),
            },
            "create_time": str(t["create_time"]),
        })
    return {"tasks": tasks}


class RunIn(BaseModel):
    dataset_id: int
    model: str = None         # 模型 key，空=默认模型
    params: dict = None       # 覆盖模型参数（按该模型 ParamSpec 强转与校验）


@app.post("/api/tasks/run")
def run_task(body: RunIn):
    ds = db.get_dataset(body.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="数据集不存在")

    if body.model and body.model not in all_backends():
        raise HTTPException(status_code=400, detail=f"未知模型：{body.model}")
    backend = get_backend(body.model)
    ok, reason = backend.is_available()
    if not ok:
        raise HTTPException(status_code=400, detail=f"模型 {backend.name} 不可用：{reason}")

    # 跨模态模型必须先登记配对 ATAC：早失败比跑到训练阶段再崩清楚得多
    atac_path = (ds.get("atac_path") or "").strip() or None
    if backend.capabilities.needs_second_modality:
        if not atac_path:
            raise HTTPException(
                status_code=400,
                detail=f"模型 {backend.name} 是跨模态模型，需要数据集的「配对 ATAC 路径」；"
                       f"请在数据集 #{body.dataset_id} 上补充 ATAC 数据后重试")
        if not Path(atac_path).exists():
            raise HTTPException(status_code=400, detail=f"配对 ATAC 文件不存在：{atac_path}")

    try:
        model_params = backend.merge_overrides(body.params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"参数校验失败：{e}")

    task_id = db.create_task(name=ds["name"], path="", model=backend.key,
                             dataset_id=body.dataset_id)
    task_name = f"{ds['name']}_{task_id}"
    output_dir = str(RUNTIME_TASK_DIR / str(task_id))
    db.update_task_meta(task_id, task_name, output_dir)
    db.update_task_process(task_id, json.dumps({"status": "running", "events": []}, ensure_ascii=False))

    with _running_lock:
        _running.add(task_id)

    threading.Thread(
        target=_run_in_thread,
        args=(task_id, body.dataset_id, ds["path"], output_dir, backend.key, model_params,
              atac_path),
        daemon=True,
    ).start()
    return {"task_id": task_id, "model": backend.key}


def _run_in_thread(task_id: int, dataset_id: int, dataset_path: str, output_dir: str,
                   model: str = None, model_params: dict = None, atac_path: str = None):
    try:
        outcome = run_pipeline(task_id, dataset_id, dataset_path=dataset_path, output_dir=output_dir,
                               model=model, model_params=model_params, atac_path=atac_path)
    finally:
        with _running_lock:
            _running.discard(task_id)

    process = {
        "status": outcome["status"],
        "result": outcome.get("result"),
        "error": outcome.get("error"),
        "events": snapshot(task_id),
    }
    try:
        db.update_task_process(task_id, json.dumps(process, ensure_ascii=False, default=str))
    except Exception as e:
        # 落库失败绝不能静默吞掉：异常会让这个 daemon 线程直接死掉，
        # 任务状态就永远停在启动时写的 running（历史上正是如此）。
        print(f"[warn] 任务 #{task_id} 结果落库失败：{e}")
        process.pop("events", None)     # 事件可从 events.jsonl 恢复，优先保住状态与结果
        try:
            db.update_task_process(task_id, json.dumps(process, ensure_ascii=False, default=str))
        except Exception as e2:
            print(f"[error] 任务 #{task_id} 状态落库仍然失败：{e2}")


@app.get("/api/tasks/{task_id}/state")
def task_state(task_id: int):
    t = db.get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="任务不存在")

    process = _parse_process(t["process"])
    # events 有三个来源，取最全的一份：
    #   ① DB process.events —— 只在流水线收尾时写过一次，进程中途挂掉就为空
    #   ② 内存 BUFFERS     —— 运行中最新，但随进程消失
    #   ③ events.jsonl     —— 追加落盘，进程重启后仍能完整恢复（日志的持久化来源）
    candidates = [process.get("events") or [], snapshot(task_id), load_events(task_id)]
    best = max(candidates, key=len)
    if best:
        process["events"] = best

    with _running_lock:
        process["running"] = task_id in _running
    return process


@app.get("/api/tasks/{task_id}/stream")
async def task_stream(task_id: int):
    async def gen():
        sent = 0
        while True:
            events = snapshot(task_id)
            while sent < len(events):
                yield f"data: {json.dumps(events[sent], ensure_ascii=False, default=str)}\n\n"
                sent += 1
            with _running_lock:
                running = task_id in _running
            if not running and sent >= len(events):
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: int):
    """删除任务：连同其输出目录（report/产物）一起清理。

    - 运行中的任务拒绝删除（409），避免删到一半还在写文件；
    - 只删 runtime/task 之下的目录，防止历史脏数据里的 path 指向别处被误删。
    """
    t = db.get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="任务不存在")

    with _running_lock:
        if task_id in _running:
            raise HTTPException(status_code=409, detail="任务正在运行，请等待结束后再删除")

    out_dir = (t.get("path") or "").strip()
    if out_dir:
        try:
            p = Path(out_dir).resolve()
            if RUNTIME_TASK_DIR.resolve() in p.parents:
                shutil.rmtree(p, ignore_errors=True)
        except Exception as e:
            print(f"[warn] 删除任务输出目录失败（仅删库记录）：{e}")

    db.delete_task(task_id)
    return {"ok": True, "deleted_files": bool(out_dir)}


# ---------- Agent 轨迹重放（不使用 LLM） ----------
class ReplayIn(BaseModel):
    model: str = None          # 缺省用轨迹里记录的模型
    output_dir: str = None     # 缺省写到 <任务目录>/replay，避免覆盖原始产物


@app.post("/api/tasks/{task_id}/replay")
def replay_task(task_id: int, body: ReplayIn = None):
    """按 Agent 轨迹重放一次预处理（**全程不调用 LLM**）。

    仅适用于 ReAct 模式产生的任务（需存在 04_Agent轨迹.json）。
    默认输出到 <任务目录>/replay，不覆盖原任务的报告与产物。

    注意：本接口是**同步**执行的（重放不含 LLM 与训练，通常远快于原任务，
    但仍需加载数据并重算 QC/报告），大任务请耐心等待。
    """
    t = db.get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="任务不存在")

    trace_dir = (t.get("path") or "").strip()
    if not trace_dir:
        raise HTTPException(status_code=400, detail="该任务没有输出目录，无法重放")
    if not _has_trace(trace_dir):
        raise HTTPException(
            status_code=400,
            detail="该任务没有 Agent 轨迹（04_Agent轨迹.json）——"
                   "重放仅适用于 ReAct 模式（PREPROCESS_MODE=react/auto）运行的任务")

    # 优先用数据集登记的路径；数据集已删除时回退到轨迹里记录的那份
    dataset_path = None
    atac_path = None
    if t.get("dataset_id"):
        ds = db.get_dataset(t["dataset_id"])
        dataset_path = (ds or {}).get("path")
        atac_path = (ds or {}).get("atac_path")

    body = body or ReplayIn()
    try:
        return run_replay(trace_dir, dataset_path=dataset_path, model=body.model,
                          output_dir=body.output_dir, atac_path=atac_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重放失败：{type(e).__name__}: {e}")


# ---------- AI 聊天（DeepSeek） ----------
class ChatIn(BaseModel):
    message: str


@app.post("/api/chat")
def chat(body: ChatIn):
    if llm_configured():
        reply = run_assistant(body.message)
    else:
        reply = (
            f"（未配置 LLM_API_KEY）收到你的问题：「{body.message}」\n\n"
            "请设置环境变量 LLM_API_KEY（DeepSeek）后即可使用真实问答。"
        )
    return {"reply": reply}


@app.post("/api/chat/stream")
async def chat_stream(body: ChatIn):
    """AI 聊天流式输出（SSE）。"""
    async def gen():
        if not llm_configured():
            fallback = (
                f"（未配置 LLM_API_KEY）收到你的问题：「{body.message}」\n\n"
                "请设置环境变量 LLM_API_KEY（DeepSeek）后即可使用真实问答。"
            )
            yield f"data: {json.dumps({'delta': fallback}, ensure_ascii=False)}\n\n"
            return
        async for chunk in stream_assistant(body.message):
            yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _parse_process(raw: str):
    if not raw:
        return {"status": "unknown", "events": []}
    try:
        return json.loads(raw)
    except Exception:
        return {"status": "unknown", "events": []}


def _has_trace(path) -> bool:
    """任务目录下是否存在 Agent 轨迹（决定前端是否提供「重放」入口）。"""
    d = (path or "").strip()
    return bool(d) and Path(trace_path(d)).exists()
