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
from app.event_emitter import snapshot
from app.llm import llm_configured
from app.main import run_pipeline
from app.tools.assistant_tools import run_assistant, stream_assistant
from app.tools.models import all_backends, default_key, get_backend, list_models, validate

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
    # 模型参数已按模型隔离：旧的全局变量不再生效，提示改用 {模型KEY}_{参数名}
    for k, v in config.LEGACY_MODEL_ENV.items():
        print(f"[warn] 环境变量 {k}={v} 已废弃且不再生效（参数已按模型隔离）；"
              f"请改用 SCLINFORMER_EPOCHS / SCVI_MAX_EPOCHS / SCVI_BATCH_SIZE 等 模型KEY_参数名 形式")
    try:
        reconcile_tasks()
    except Exception as e:
        print(f"[warn] 任务状态对账失败：{e}")


def reconcile_tasks():
    """启动时对账：DB 里是 running、但内存里没有对应运行线程的任务 = 上次进程残留的僵尸。

    结果写回数据库的动作只在流水线返回后执行一次，若服务中途被重启（或那次写库异常），
    任务就会永远停在 running。这里按磁盘产物判断真实结果并修正状态。
    """
    for t in db.list_tasks():
        proc = _parse_process(t.get("process"))
        if (proc.get("status") or "") != "running":
            continue
        task_id = t["id"]
        with _running_lock:
            if task_id in _running:      # 真在跑，跳过
                continue

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
        db.update_task_process(task_id, json.dumps(proc, ensure_ascii=False, default=str))
        print(f"[warn] 任务状态对账：#{task_id} running -> {proc['status']}")


# ---------- 模型（注册表驱动，新增模型自动出现在这里） ----------
@app.get("/api/models")
def api_models():
    """返回可选模型列表 + 参数 Schema（前端据此动态渲染参数表单）。"""
    return {
        "models": list_models(with_schema=True),
        "default": default_key(),
        "warnings": validate(),
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


@app.post("/api/datasets")
def create_dataset(body: DatasetIn):
    dataset_id = db.create_dataset(body.name, body.path)
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
        args=(task_id, body.dataset_id, ds["path"], output_dir, backend.key, model_params),
        daemon=True,
    ).start()
    return {"task_id": task_id, "model": backend.key}


def _run_in_thread(task_id: int, dataset_id: int, dataset_path: str, output_dir: str,
                   model: str = None, model_params: dict = None):
    try:
        outcome = run_pipeline(task_id, dataset_id, dataset_path=dataset_path, output_dir=output_dir,
                               model=model, model_params=model_params)
    finally:
        with _running_lock:
            _running.discard(task_id)

    process = {
        "status": outcome["status"],
        "result": outcome.get("result"),
        "error": outcome.get("error"),
        "events": snapshot(task_id),
    }
    db.update_task_process(task_id, json.dumps(process, ensure_ascii=False, default=str))


@app.get("/api/tasks/{task_id}/state")
def task_state(task_id: int):
    t = db.get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="任务不存在")

    process = _parse_process(t["process"])
    # 运行中的任务：process 只在流水线结束时才落库，中途 DB 里的 events 是空的。
    # 这里补上内存缓冲里的实时事件，否则「切到别的任务再切回来」日志会消失。
    mem = snapshot(task_id)
    if len(mem) > len(process.get("events") or []):
        process["events"] = mem

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
