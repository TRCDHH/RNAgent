"""模型注册表：自动发现 + 获取 + 一致性自检。

扩展方式（新增模型只需两步，其余文件零改动）：
    1. 新建 app/tools/models/backends/<key>.py，定义 ModelBackend 子类并加 @register_model
    2. 新建 app/skills/registry/models/<key>/SKILL.md，frontmatter 声明参数默认值与决策表

启动自检（validate）：代码声明的参数必须都在 SKILL 中声明，且每个模型都必须有 SKILL，
避免「代码与文档漂移」，问题以 warning 形式暴露在 /api/models 与启动日志中。
"""

import os
import threading

from app import config
from app.tools.models.base import ModelBackend

REGISTRY: dict = {}
_LOCK = threading.Lock()
_DISCOVERED = False
_BACKENDS_PKG = "app.tools.models.backends"


def register_model(cls):
    """装饰器：把模型后端注册进全局注册表。"""
    instance = cls()
    if not instance.key:
        raise ValueError(f"模型后端 {cls.__name__} 缺少 key")
    REGISTRY[instance.key] = instance
    return cls


def discover(force: bool = False):
    """扫描 backends/ 目录下所有模块，import 即注册（无需修改任何 __init__）。"""
    global _DISCOVERED
    if _DISCOVERED and not force:
        return REGISTRY
    with _LOCK:
        if _DISCOVERED and not force:
            return REGISTRY
        import importlib
        import pkgutil

        pkg_dir = os.path.join(os.path.dirname(__file__), "backends")
        for mod in pkgutil.iter_modules([pkg_dir]):
            if mod.name.startswith("_"):
                continue
            importlib.import_module(f"{_BACKENDS_PKG}.{mod.name}")
        _DISCOVERED = True
    return REGISTRY


def all_backends() -> dict:
    discover()
    return REGISTRY


def default_key() -> str:
    backends = all_backends()
    if config.DEFAULT_MODEL in backends:
        return config.DEFAULT_MODEL
    return next(iter(backends), "sclinformer")


def get_backend(key: str = None) -> ModelBackend:
    """按 key 取后端；key 为空/未知时回退默认模型（保证老调用方行为不变）。"""
    backends = all_backends()
    if not backends:
        raise RuntimeError("未注册任何模型后端（app/tools/models/backends/ 为空？）")
    if key and key in backends:
        return backends[key]
    return backends[default_key()]


def list_models(with_schema: bool = True) -> list:
    discover()
    return [b.to_dict(with_schema=with_schema) for b in REGISTRY.values()]


def validate() -> list:
    """一致性自检，返回问题列表（空列表表示全部通过）。"""
    from app.skills.library import get_library

    library = get_library()
    problems = []
    for key, backend in all_backends().items():
        skill = library.load_model(key)
        if skill is None:
            problems.append(f"模型 {key} 缺少 SKILL.md（应在 app/skills/registry/models/{key}/SKILL.md）")
            continue
        for sp in backend.param_specs:
            if sp.name not in skill.param_defaults():
                problems.append(f"模型 {key} 的参数 {sp.name} 未在 SKILL.md 的 params 中声明")
    for key in library.model_keys():
        if key not in all_backends():
            problems.append(f"SKILL.md 中的模型 {key} 没有对应的后端实现")
    return problems
