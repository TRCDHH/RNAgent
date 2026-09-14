"""预处理 Agent 轨迹归档与重放。

目的（技术方案第 7 节）：把「灵活探索」与「科研可复现」分离开 ——
- 探索阶段由 LLM 决策，全部工具调用与结果落盘为 04_Agent轨迹.json；
- 一次成功的轨迹可用 run_replay() **脱离 LLM** 重放，得到可复现的修复过程。

三种用法：
1. 库调用：run_replay(目录) —— 建会话、加载数据、重放、判定等价性校验一次做完；
2. 命令行：python -m app.tools.preprocess_replay <任务目录> [--dataset ...] [--json]
3. 低层：replay_trace(目录, 执行器) —— 只做重放，不碰会话与数据加载（便于单测）

轨迹与报告同目录，随任务产物一起归档；格式稳定，便于后续做数据集回归评测。
"""

import json
import os
from datetime import datetime

TRACE_FILE = "04_Agent轨迹.json"


def trace_path(output_dir: str) -> str:
    return os.path.join(output_dir, TRACE_FILE)


def save_trace(output_dir: str, session) -> str:
    """把会话的决策轨迹与审计日志写入 04_Agent轨迹.json，返回文件路径。"""
    payload = {
        "task_id": session.task_id,
        "dataset_id": session.dataset_id,
        "dataset_path": session.dataset_path,
        "model": getattr(session.backend, "key", None),
        "model_name": getattr(session.backend, "name", None),
        "created_at": datetime.now().isoformat(),
        "steps": session.steps,
        "finish_reason": session.finish_reason,
        "finish_notes": session.finish_notes,
        "n_cells_0": session.n_cells_0,
        "judge_initial": session.judge_initial,
        "judge_final": session.verdicts,
        "trace": session.trace,
        "audit_log": session.audit_log,
    }
    os.makedirs(output_dir, exist_ok=True)
    path = trace_path(output_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path


def load_trace(output_dir: str):
    """读回轨迹；文件不存在或损坏返回 None（调用方自行降级）。"""
    path = trace_path(output_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def replay_trace(output_dir: str, execute_fn) -> dict:
    """按轨迹重放工具调用（不经过 LLM），返回执行统计。

    execute_fn(name, args) -> dict：由调用方注入的执行器（通常是 ReAct 工具层的
    分发函数）。用注入而非直接 import，避免本模块反向依赖具体工具实现（也便于测试）。

    只重放**写操作**；观察类调用（只读）跳过 —— 它们不改变数据状态。
    本函数不负责准备会话与数据，直接调用前请先 set_session()（高层入口见 run_replay）。
    """
    data = load_trace(output_dir)
    if not data:
        return {"status": "error", "error": f"未找到轨迹文件：{trace_path(output_dir)}"}

    writes = [t for t in (data.get("trace") or []) if t.get("write")]
    results = []
    for step in writes:
        r = execute_fn(step.get("tool"), step.get("args") or {})
        ok = not (isinstance(r, dict) and r.get("error"))
        results.append({
            "tool": step.get("tool"),
            "ok": ok,
            "error": (r.get("error") if isinstance(r, dict) else None),
        })
        if not ok:   # 一步失败即停止，避免在错误状态上继续改数据
            break
    return {"status": "ok", "replayed": len(results), "results": results}


# ---------------------------------------------------------------------------
# 高层封装：会话准备 + 数据加载 + 重放 + 等价性校验
# ---------------------------------------------------------------------------
def _fail_set(verdicts) -> list:
    """取判定结果里的 FAIL 项（等价性用它比对，比只看 PASS/FAIL/WARNING 更细）。"""
    return sorted(
        v.get("item") for v in ((verdicts or {}).get("verdicts") or [])
        if v.get("status") == "FAIL"
    )


def run_replay(trace_dir: str, dataset_path: str = None, model: str = None,
               model_params: dict = None, output_dir: str = None,
               judge_check: bool = True, finalize: bool = True) -> dict:
    """按轨迹完整重放一次预处理（**全程不调用 LLM**）。

    参数：
      trace_dir     含 04_Agent轨迹.json 的任务输出目录（只读，不会被改写）
      dataset_path  原始数据路径；缺省用轨迹里记录的那份
      model         模型 key；缺省用轨迹里记录的那个
      output_dir    产物输出目录；缺省 <trace_dir>/replay
                    —— **刻意不写回原目录**，避免覆盖原任务的报告与产物
      judge_check   是否做「重放后判定 vs 轨迹记录判定」的等价性校验
      finalize      是否继续跑确定性收尾（QC/报告/processed_rna.h5ad）

    执行顺序与原始运行严格一致：重放写操作 -> judge -> （校验）-> finalize。
    这样 judge 比较的是同一时点（收尾前），不会被 prepare_data 的基因筛选影响。
    """
    # 惰性导入：本模块被 preprocess_agent 导入，模块级 import 工具层会形成依赖环
    from app.tools import preprocess_tools as pt
    from app.tools.models import get_backend
    from app.tools.preprocess_agent_tools import execute_by_name
    from app.tools.preprocess_session import (
        PreprocessSession, clear_session, set_session,
    )

    data = load_trace(trace_dir)
    if not data:
        return {"status": "error", "error": f"未找到轨迹文件：{trace_path(trace_dir)}"}

    src = dataset_path or data.get("dataset_path")
    if not src or not os.path.exists(src):
        return {"status": "error", "error": f"原始数据不存在：{src}"}

    backend = get_backend(model or data.get("model"))
    params = {**backend.default_params(), **backend.merge_overrides(model_params)}
    out_dir = output_dir or os.path.join(trace_dir, "replay")

    session = PreprocessSession(data.get("task_id") or 0, data.get("dataset_id"),
                                src, out_dir, backend, params)
    set_session(session)
    try:
        session.adata = pt._read_data(src)
        session.n_cells_0 = int(session.adata.n_obs)
        session.metadata = pt._describe(session.adata, src)

        before = pt.judge(session.adata, backend)
        replay = replay_trace(trace_dir, execute_by_name)
        after = pt.judge(session.adata, backend)

        out = dict(replay)
        out.update({
            "dataset": src,
            "output_dir": out_dir,
            "model": backend.key,
            "model_name": backend.name,
            "n_cells": int(session.adata.n_obs),
            "judge_before": before.get("status"),
            "judge_after": after.get("status"),
            "expected_judge": (data.get("judge_final") or {}).get("status"),
            "expected_fail_items": _fail_set(data.get("judge_final")),
            "actual_fail_items": _fail_set(after),
        })

        if judge_check:
            out["reproduced"] = (
                out["actual_fail_items"] == out["expected_fail_items"]
                and out["judge_after"] == out["expected_judge"]
            )
            if not out["reproduced"]:
                out["note"] = (
                    f"重放判定（{out['judge_after']}, FAIL={out['actual_fail_items']}）"
                    f"与原始轨迹（{out['expected_judge']}, FAIL={out['expected_fail_items']}）"
                    f"不一致：原始数据可能已变更，或运行环境不同"
                )

        if finalize and out.get("status") == "ok":
            result = pt.finalize_preprocess(
                adata=session.adata, verdicts=after, output_dir=out_dir, backend=backend,
                params=params, audit_log=session.audit_log, dataset_path=src)
            out["processed_path"] = result.get("processed_path")
            out["analysis_dir"] = (result.get("report") or {}).get("analysis_dir")
            # 产物目录若落在任务目录之内，给出可直接打开的静态地址
            try:
                base = os.path.abspath(trace_dir)
                target = os.path.abspath(out_dir)
                if target == base or target.startswith(base + os.sep):
                    rel = os.path.relpath(target, base).replace(os.sep, "/")
                    out["report_url"] = f"/output/{data.get('task_id')}/{rel}/report.html"
            except Exception:
                pass
        return out
    finally:
        clear_session()


# ---------------------------------------------------------------------------
# 命令行入口：python -m app.tools.preprocess_replay <任务目录>
# ---------------------------------------------------------------------------
def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m app.tools.preprocess_replay",
        description="按 Agent 轨迹重放一次预处理（不使用 LLM）")
    ap.add_argument("trace_dir", help="任务输出目录（含 04_Agent轨迹.json）")
    ap.add_argument("--dataset", default=None, help="原始数据路径；缺省用轨迹里记录的那份")
    ap.add_argument("--model", default=None, help="模型 key；缺省用轨迹里记录的那个")
    ap.add_argument("--out", default=None, help="输出目录；缺省 <任务目录>/replay（不覆盖原始产物）")
    ap.add_argument("--no-finalize", action="store_true", help="只重放修复，不生成报告等交付物")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    args = ap.parse_args(argv)

    result = run_replay(args.trace_dir, dataset_path=args.dataset, model=args.model,
                        output_dir=args.out, finalize=not args.no_finalize)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    elif result.get("status") != "ok":
        print(f"[x] 重放失败：{result.get('error')}")
    else:
        print(f"[√] 重放完成：{result.get('replayed')} 个写操作")
        for r in result.get("results") or []:
            print(f"    - {r['tool']}：{'成功' if r['ok'] else r['error']}")
        print(f"    判定：{result.get('judge_before')} -> {result.get('judge_after')}"
              f"（原始轨迹 {result.get('expected_judge')}）")
        if "reproduced" in result:
            print(f"    等价性：{'一致' if result['reproduced'] else '不一致'}")
            if result.get("note"):
                print(f"    说明：{result['note']}")
        if result.get("processed_path"):
            print(f"    产物目录：{result.get('output_dir')}")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
