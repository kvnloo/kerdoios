from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def _parse_frontmatter(text: str) -> dict[str, str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip().strip('"').strip("'")
    return fields


class SkillFrontmatterTests(unittest.TestCase):
    def test_every_skill_has_discoverable_frontmatter(self) -> None:
        skill_files = sorted(SKILLS.glob("*/SKILL.md"))
        self.assertTrue(skill_files, "no skills found under skills/")
        for path in skill_files:
            with self.subTest(skill=path.parent.name):
                fields = _parse_frontmatter(path.read_text(encoding="utf-8"))
                self.assertTrue(
                    fields.get("name"),
                    f"{path}: missing YAML frontmatter 'name' (agent harnesses cannot discover this skill)",
                )
                self.assertTrue(
                    fields.get("description"),
                    f"{path}: missing YAML frontmatter 'description' (agent harnesses cannot trigger this skill)",
                )


if __name__ == "__main__":
    unittest.main()
