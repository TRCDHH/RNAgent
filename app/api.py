"""Python FastAPI 后端（第二阶段，本地 MySQL 持久化 + DeepSeek LLM）。

职责（纯业务门户，无 AI 逻辑，AI 全在 LangGraph 引擎）：
- 数据集管理（CRUD）
- 任务管理（触发 pipeline、查状态、SSE 推送、过程持久化）
- AI 聊天（DeepSeek）
- 输出目录静态托管（/output/{task_id}/report.html）
"""

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import db
from app.event_emitter import snapshot
from app.llm import llm_configured
from app.main import run_pipeline
from app.tools.assistant_tools import run_assistant, stream_assistant

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


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


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
        tasks.append({
            "id": t["id"],
            "name": t["name"],
            "path": t["path"],
            "process": _parse_process(t["process"]),
            "create_time": str(t["create_time"]),
        })
    return {"tasks": tasks}


class RunIn(BaseModel):
    dataset_id: int


@app.post("/api/tasks/run")
def run_task(body: RunIn):
    ds = db.get_dataset(body.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="数据集不存在")

    task_id = db.create_task(name=ds["name"], path="")
    task_name = f"{ds['name']}_{task_id}"
    output_dir = str(RUNTIME_TASK_DIR / str(task_id))
    db.update_task_meta(task_id, task_name, output_dir)
    db.update_task_process(task_id, json.dumps({"status": "running", "events": []}, ensure_ascii=False))

    with _running_lock:
        _running.add(task_id)

    threading.Thread(
        target=_run_in_thread,
        args=(task_id, body.dataset_id, ds["path"], output_dir),
        daemon=True,
    ).start()
    return {"task_id": task_id}


def _run_in_thread(task_id: int, dataset_id: int, dataset_path: str, output_dir: str):
    try:
        outcome = run_pipeline(task_id, dataset_id, dataset_path=dataset_path, output_dir=output_dir)
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
    return _parse_process(t["process"])


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
