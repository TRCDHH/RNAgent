"""训练会话（TrainingSession）：训练阶段 ReAct 恢复循环的运行时上下文。

与 PreprocessSession 的取舍一致：用线程局部持有"唯一真身"（这里不是 AnnData 而是
训练上下文与尝试历史），工具通过 get_training_session() 取用，签名里不出现大数据对象。

与预处理会话的关键差异：
- 这里的"真身"是**参数与尝试历史**（attempts），不是数据 —— 数据在训练子进程里读；
- 每多一次尝试意味着真实启动一次训练子进程（昂贵），因此本会话带**尝试次数硬上限**。
"""

import threading
import time
from datetime import datetime


class TrainingSession:
    """一次训练阶段的运行时上下文（参数、尝试历史、护栏计数）。"""

    def __init__(self, task_id, output_dir, ctx: dict, max_attempts: int = 3):
        self.task_id = int(task_id)
        self.output_dir = output_dir

        # 训练上下文（来自 model_tools.prepare_training_context）
        self.backend = ctx["backend"]
        self.rna_path = ctx["rna_path"]
        self.env = ctx.get("env") or {}
        self.n_cells = ctx.get("n_cells")
        self.n_genes = ctx.get("n_genes")
        self.obs_columns = ctx.get("obs_columns") or []
        self.base_config = dict(ctx.get("base_config") or {})   # 规则默认参数
        self.model_params = ctx.get("model_params") or {}       # 用户显式覆盖

        self.current_params = dict(self.base_config)            # 下次尝试将使用的参数
        self.attempts = []                                      # [{n, params, ok, error, error_kind, elapsed_s}]
        self.max_attempts = max(1, int(max_attempts))

        self.best_result = None                                 # 成功那次的结果
        self.audit_log = []
        self.trace = []                                         # 恢复循环的工具调用轨迹

        self.start_ts = time.time()
        self.recover_start_ts = None                            # 进入恢复循环的时刻
        self.finished = False
        self.finish_reason = ""
        self.finish_notes = ""

    # ---------------- 尝试记录 ----------------
    def record_attempt(self, params: dict, outcome: dict, elapsed_s: float = None):
        """记录一次训练尝试（成功或失败）。"""
        self.attempts.append({
            "n": len(self.attempts) + 1,
            "params": dict(params or {}),
            "ok": bool(outcome.get("ok")),
            "error_kind": outcome.get("error_kind"),
            "error": (outcome.get("error") or "")[:2000] or None,
            "elapsed_s": round(elapsed_s, 1) if elapsed_s is not None else None,
        })
        if outcome.get("ok") and outcome.get("result") is not None:
            self.best_result = outcome["result"]

    def attempts_left(self) -> int:
        return max(0, self.max_attempts - len(self.attempts))

    def last_failure(self) -> dict:
        for a in reversed(self.attempts):
            if not a["ok"]:
                return a
        return {}

    def recover_budget_exceeded(self) -> bool:
        if self.recover_start_ts is None:
            return False
        from app import config
        return (time.time() - self.recover_start_ts) > config.TRAINING_RECOVER_BUDGET


def now_iso() -> str:
    return datetime.now().isoformat()


_local = threading.local()


def set_training_session(session: TrainingSession):
    _local.current = session


def get_training_session():
    return getattr(_local, "current", None)


def clear_training_session():
    _local.current = None
