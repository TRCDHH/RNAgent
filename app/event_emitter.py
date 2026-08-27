"""事件流发射器。

第一阶段用「stdout 打印 + 追加写 runtime/events.jsonl + 内存缓冲」：
- 打印 / jsonl：留档，出 bug 可追踪；
- 内存缓冲（BUFFERS）：供后端 SSE 实时推送给前端，实现「中间过程用户可见」。

第二阶段可无缝替换为 `redis.xadd(f"task:{id}:events", ...)`。
"""

import collections
import json
import threading
import time
from pathlib import Path

_RUNTIME_DIR = Path(__file__).resolve().parent.parent / "runtime"
_EVENTS_FILE = _RUNTIME_DIR / "events.jsonl"

# 线程局部上下文：每个后台任务线程独立记录自己的 task_id / trace_id
_local = threading.local()

# 内存事件缓冲：task_id -> deque（SSE 实时订阅用）
BUFFERS: dict = collections.defaultdict(lambda: collections.deque(maxlen=2000))
_BUFFERS_LOCK = threading.Lock()


def set_context(task_id, trace_id):
    _local.task_id = task_id
    _local.trace_id = trace_id


def emit(event: str, payload: dict | None = None):
    payload = payload or {}
    record = {
        "ts": round(time.time(), 3),
        "task_id": getattr(_local, "task_id", None),
        "trace_id": getattr(_local, "trace_id", None),
        "event": event,
        **payload,
    }
    line = json.dumps(record, ensure_ascii=False, default=str)
    print(f"[event] {line}")

    task_id = record["task_id"]
    if task_id is not None:
        with _BUFFERS_LOCK:
            BUFFERS[task_id].append(record)

    try:
        _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        with open(_EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def snapshot(task_id) -> list:
    """返回某任务当前累积的事件列表（供 SSE 增量读取）。"""
    with _BUFFERS_LOCK:
        return list(BUFFERS.get(task_id, []))
