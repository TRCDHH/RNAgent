"""跨模块通用小工具（放在最底层，避免 preprocess_tools <-> models 循环导入）。"""

import math


def json_safe(obj):
    """把 numpy 标量/数组等转成可 JSON 序列化的原生类型（鸭子类型，避免顶层 import numpy）。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(i) for i in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item", None)):
        try:
            return json_safe(obj.item())
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist", None)):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict", None)):
        try:
            return json_safe(obj.to_dict())
        except Exception:
            pass
    return str(obj)


def read_summary_metrics(csv_path: str) -> dict:
    """读取评测指标 summary_metrics.csv，NaN 转 None，返回可 JSON 的 dict。"""
    if not csv_path or not __import__("os").path.exists(csv_path):
        return {}
    import pandas as pd  # noqa: E402 惰性导入

    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    out = {}
    for k in df.columns:
        try:
            v = float(df.iloc[0][k])
        except Exception:
            v = None
        if v is None or math.isnan(v):
            v = None
        out[str(k)] = v
    return out
