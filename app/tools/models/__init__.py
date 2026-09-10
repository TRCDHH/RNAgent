"""模型插件层：能力声明 + 参数 Schema + 注册表。

对外常用接口：
    from app.tools.models import get_backend, list_models, default_key
"""

from app.tools.models.base import Capabilities, ModelBackend, ParamSpec
from app.tools.models.registry import (
    all_backends,
    default_key,
    discover,
    get_backend,
    list_models,
    register_model,
    validate,
)

__all__ = [
    "Capabilities",
    "ModelBackend",
    "ParamSpec",
    "all_backends",
    "default_key",
    "discover",
    "get_backend",
    "list_models",
    "register_model",
    "validate",
]
