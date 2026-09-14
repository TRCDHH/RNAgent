"""预处理会话（PreprocessSession）：ReAct 决策层与原子工具之间共享的运行时上下文。

为什么需要它：
- AnnData 是几十 MB 的对象，**不能放进 LangGraph State**（State 要参与 checkpoint 序列化，
  现有设计也刻意只放路径与结构化摘要）；
- ReAct 是多轮循环，所有工具必须操作**同一份** adata，而不是每轮各读一份；
- 因此用「线程局部会话」持有唯一真身，工具通过 get_session() 取用，签名里不再出现 adata。

线程局部（而非全局）与 app/event_emitter.py 的 _local 保持一致的取舍：
预处理跑在任务自己的后台线程里，线程隔离天然满足「每任务一份会话」。
"""

import threading
import time
from datetime import datetime


class PreprocessSession:
    """一次预处理决策的运行时上下文（adata 唯一真身 + 审计 + 轨迹）。"""

    def __init__(self, task_id, dataset_id, dataset_path, output_dir, backend, params):
        self.task_id = int(task_id)
        self.dataset_id = dataset_id
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.backend = backend
        self.params = params or {}

        # ---- 数据（唯一真身：所有工具共同操作它）----
        self.adata = None
        self.metadata = {}
        self.verdicts = {}           # 最近一次 judge 结果
        self.judge_initial = {}      # 首次 judge（报告与轨迹用）

        # ---- 记录 ----
        self.audit_log = []          # 与 pipeline 模式同格式，写入 03_操作审计.md
        self.trace = []              # 工具调用轨迹（可复现 / 固化重放）
        self.judge_history = []      # 每次写操作后的 FAIL 集合（无进展检测）
        self.n_cells_0 = None        # 行数守恒基线
        self.steps = 0               # 已经过的 LLM 决策步数
        self.start_ts = time.time()

        # ---- 终止状态 ----
        self.finished = False
        self.finish_reason = ""
        self.finish_notes = ""


def now_iso() -> str:
    """统一的审计时间戳格式（与 preprocess_tools 中的 datetime.now().isoformat() 一致）。"""
    return datetime.now().isoformat()


_local = threading.local()


def set_session(session: PreprocessSession):
    _local.current = session


def get_session():
    """取当前线程的会话；未初始化返回 None（工具层负责转成可读错误）。"""
    return getattr(_local, "current", None)


def clear_session():
    _local.current = None
