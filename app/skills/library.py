"""Skill 库：加载 SKILL.md，模拟渐进式披露（progressive disclosure）。

- disclose(): 只返回每个 skill 的 name + description（常驻 system prompt）
- load(name): 返回该 skill 的完整内容（确定要用时才加载，省 token）
"""

import re
from pathlib import Path

REGISTRY_DIR = Path(__file__).resolve().parent / "registry"


class SkillLibrary:
    def __init__(self):
        self._skills = {}
        if not REGISTRY_DIR.exists():
            return
        for skill_dir in sorted(REGISTRY_DIR.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                meta, body = self._parse(skill_md.read_text(encoding="utf-8"))
                self._skills[skill_dir.name] = {
                    "description": meta.get("description", ""),
                    "body": body,
                }

    @staticmethod
    def _parse(text: str):
        meta, body = {}, text
        m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = m.group(2).strip()
        return meta, body

    def disclose(self) -> list[dict]:
        return [
            {"name": name, "description": s["description"]}
            for name, s in self._skills.items()
        ]

    def load(self, name: str) -> str:
        s = self._skills.get(name)
        if not s:
            return ""
        return f"# {name}\n{s['body']}"

    def names(self) -> list[str]:
        return list(self._skills.keys())
