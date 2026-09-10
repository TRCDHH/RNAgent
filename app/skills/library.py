"""Skill 库：加载 SKILL.md，模拟渐进式披露（progressive disclosure）。

两类技能：
- **阶段技能**（registry/<name>/SKILL.md）：与模型无关的流程知识。
  - disclose(): 只返回 name + description（常驻 system prompt）
  - load(name): 返回完整内容
- **模型技能**（registry/models/<model_key>/SKILL.md）：一个模型一份，是「机器可读的模型契约」：
  frontmatter 声明该模型的**参数默认值**与**决策表**（batch_size 查表规则），
  代码（app/tools/models/backends/*.py）只声明参数的类型/取值范围/文案。
  二者通过 model key 关联，并由 registry.validate() 做一致性自检。

frontmatter 支持一层嵌套块（用于 params 等）：
    params:
      epochs: 1
      batch_size: auto
    batch_size_table: cpu=32; 0=32; 8000=128
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

REGISTRY_DIR = Path(__file__).resolve().parent / "registry"
MODEL_REGISTRY_DIR = REGISTRY_DIR / "models"

# frontmatter 中标记该技能为「模型技能」的 type 值
MODEL_TYPE = "model"


def _parse_kv_table(text: str) -> dict:
    """解析 "cpu=32; 0=32; 8000=128; 24000=512" 形式的阈值表。

    返回 {"cpu": 32, 0: 32, 8000: 128, 24000: 512}（"cpu" 为 CPU 环境专用键）。
    """
    out = {}
    for part in (text or "").split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            out["cpu" if k.lower() == "cpu" else int(k)] = int(v)
        except ValueError:
            continue
    return out


@dataclass
class ModelSkill:
    """模型技能：机器可读的模型契约（参数默认值 + 决策表）+ 人读的 SOP 文档。"""

    key: str
    meta: dict = field(default_factory=dict)
    body: str = ""

    @property
    def name(self) -> str:
        return self.meta.get("display_name") or self.key

    @property
    def description(self) -> str:
        return self.meta.get("description", "")

    def param_defaults(self) -> dict:
        """frontmatter params 块：{参数名: 默认值(字符串)}。"""
        raw = self.meta.get("params")
        return dict(raw) if isinstance(raw, dict) else {}

    def batch_size_table(self) -> dict:
        """batch_size 查表规则：{显存下限MB: batch_size}，另含 'cpu' 键。"""
        return _parse_kv_table(self.meta.get("batch_size_table", ""))

    def batch_size_caps(self) -> dict:
        """数据量收窄规则：{n_cells 上限: batch_size 上限}。"""
        return _parse_kv_table(self.meta.get("batch_size_caps", ""))


class SkillLibrary:
    def __init__(self):
        self._skills = {}
        self._models = {}
        self._load_dir(REGISTRY_DIR, is_model=False)
        self._load_dir(MODEL_REGISTRY_DIR, is_model=True)

    # ---------------- 加载 ----------------
    def _load_dir(self, base: Path, is_model: bool):
        if not base or not base.exists():
            return
        for skill_dir in sorted(base.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            meta, body = self._parse(skill_md.read_text(encoding="utf-8"))
            if is_model or meta.get("type") == MODEL_TYPE:
                self._models[skill_dir.name] = ModelSkill(key=skill_dir.name, meta=meta, body=body)
            else:
                self._skills[skill_dir.name] = {
                    "description": meta.get("description", ""),
                    "body": body,
                }

    @staticmethod
    def _parse(text: str):
        """解析 YAML-like frontmatter，支持一层缩进块；返回 (meta, body)。"""
        meta, body = {}, text
        m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
        if not m:
            return meta, body

        body = m.group(2).strip()
        current_block = None
        for line in m.group(1).splitlines():
            if not line.strip():
                continue
            indented = line[:1].isspace()
            if indented:
                if current_block is None or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                meta[current_block][k.strip()] = v.strip()
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if v == "":
                # 开启一个嵌套块（如 params:）
                current_block = k
                meta[k] = {}
            else:
                current_block = None
                meta[k] = v
        return meta, body

    # ---------------- 阶段技能 ----------------
    def disclose(self) -> List[dict]:
        return [
            {"name": name, "description": s["description"]}
            for name, s in self._skills.items()
        ]

    def load(self, name: str) -> str:
        s = self._skills.get(name)
        if not s:
            return ""
        return f"# {name}\n{s['body']}"

    def names(self) -> List[str]:
        return list(self._skills.keys())

    def load_asset(self, skill_name: str, filename: str) -> str:
        """读取技能目录下的附加资源（如报告模板 report_template.html）；不存在返回空串。"""
        path = REGISTRY_DIR / skill_name / filename
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    # ---------------- 模型技能 ----------------
    def load_model(self, key: str):
        """返回 ModelSkill（含机器可读的默认值/决策表与人读 SOP）；不存在返回 None。"""
        return self._models.get(key)

    def list_models(self) -> List[ModelSkill]:
        return list(self._models.values())

    def model_keys(self) -> List[str]:
        return list(self._models.keys())

    def model_param_defaults(self, key: str) -> Dict[str, Any]:
        skill = self.load_model(key)
        return skill.param_defaults() if skill else {}

    def model_batch_size(self, key: str, n_cells: int, vram_mb: Any, gpu: bool) -> int:
        """按模型 SKILL 中声明的查表规则决定 batch_size（代码不内置任何模型专属规则）。"""
        skill = self.load_model(key)
        table = skill.batch_size_table() if skill else {}
        if not table:
            table = {0: 64}
        thresholds = [(k, v) for k, v in table.items() if k != "cpu"]
        if not gpu:
            base = table.get("cpu") or (min(v for _, v in thresholds) if thresholds else 32)
        else:
            vram = int(vram_mb or 0)
            cands = [v for k, v in thresholds if vram >= k]
            base = max(cands) if cands else (min(v for _, v in thresholds) if thresholds else 32)

        caps = skill.batch_size_caps() if skill else {}
        for thr in sorted(caps):
            if n_cells < thr:
                base = min(base, caps[thr])
                break
        return max(1, min(int(base), max(1, int(n_cells))))


_LIBRARY = None


def get_library() -> SkillLibrary:
    """全局单例（目录内容很少，缓存一次即可）。"""
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = SkillLibrary()
    return _LIBRARY
