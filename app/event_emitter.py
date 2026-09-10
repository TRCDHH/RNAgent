"""事件流发射器。

第一阶段用「stdout 打印 + 追加写 runtime/events.jsonl + 内存缓冲」：
- 打印 / jsonl：留档，出 bug 可追踪；
- 内存缓冲（BUFFERS）：供后端 SSE 实时推送给前端，实现「中间过程用户可见」。

**events.jsonl 是日志唯一的持久化来源**：内存 BUFFERS 与 stdout 都随进程消失，
而 task 表的 process.events 只在流水线收尾时写一次（进程中途重启就永远为空）。
因此恢复历史日志请用 `load_events(task_id)`，它直接从该文件按增量索引读取。

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


# 文件增量索引：task_id -> [event]；避免每次调用都全量扫描 events.jsonl
_FILE_INDEX: dict = {}
_FILE_OFFSET = 0
_INDEX_LOCK = threading.Lock()


def load_events(task_id, limit: int = None) -> list:
    """从 runtime/events.jsonl 读回某任务的完整事件（日志的持久化来源）。

    进程中途重启时内存 BUFFERS 会丢、DB 的 process.events 可能为空，
    但该文件是追加写的，因此仍能完整恢复历史日志。
    采用增量索引：只解析新增字节，重复调用几乎零成本。
    """
    global _FILE_OFFSET
    task_id = int(task_id)
    with _INDEX_LOCK:
        try:
            size = _EVENTS_FILE.stat().st_size
        except OSError:
            return []
        if size < _FILE_OFFSET:            # 文件被截断/轮转 → 重建索引
            _FILE_INDEX.clear()
            _FILE_OFFSET = 0
        if size > _FILE_OFFSET:
            with open(_EVENTS_FILE, "r", encoding="utf-8", errors="replace") as f:
                f.seek(_FILE_OFFSET)
                pos = _FILE_OFFSET
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):   # 半行（正在写）→ 留到下次解析
                        break
                    pos = f.tell()
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    tid = rec.get("task_id")
                    if tid is not None:
                        _FILE_INDEX.setdefault(tid, []).append(rec)
                _FILE_OFFSET = pos
        events = _FILE_INDEX.get(task_id, [])
        return list(events[-limit:]) if limit else list(events)
