from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ContributeDxTests(unittest.TestCase):
    def test_vol_config_exists_with_proof_commands(self) -> None:
        path = ROOT / ".verified-oss-loop" / "config.yml"
        self.assertTrue(path.is_file(), "VOL re-onboard leftover: missing .verified-oss-loop/config.yml")
        text = path.read_text(encoding="utf-8")
        self.assertIn("schema: verified-oss-loop.config.v1", text)
        self.assertIn("python3 -m unittest discover -s tests", text)
        self.assertIn("mutation: n/a", text)
        self.assertIn("never_merge", text)
        self.assertIn("main", text)
        self.assertIn("dispatcher: false", text)
        self.assertIn("gateway_chat_completions: false", text)
        self.assertIn("litellm_zero_is_free: false", text)

    def test_contribute_skill_is_stupidly_easy(self) -> None:
        path = ROOT / "skills" / "contribute" / "SKILL.md"
        self.assertTrue(path.is_file(), "fork DX leftover: missing skills/contribute/SKILL.md")
        text = path.read_text(encoding="utf-8")
        self.assertIn("# Contribute", text)
        self.assertIn("unittest discover -s tests", text)
        self.assertIn("Never merge", text)
        self.assertIn("/chat/completions", text)
        self.assertIn("input_cost_per_token", text)
        self.assertIn("Kerdoios plans", text)
        self.assertIn("POWER WAS NEVER SCARCE", text)
        self.assertIn("dispatcher", text.lower())
        self.assertNotIn("herdr", text.lower())
        self.assertNotIn("oh-my-pi", text.lower())

    def test_agents_points_forkers_at_contribute_skill(self) -> None:
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("skills/contribute/SKILL.md", text)
        self.assertIn("Fork", text)
        self.assertIn("Never merge", text)

    def test_inventory_marks_fork_dx_files_local(self) -> None:
        text = (ROOT / ".verified-oss-loop" / "inventory.yml").read_text(encoding="utf-8")
        for path in ("skills/contribute/SKILL.md", ".verified-oss-loop/config.yml"):
            needle = f"path: {path}"
            idx = text.find(needle)
            self.assertNotEqual(idx, -1, f"missing inventory entry for {path}")
            snippet = text[idx : idx + 180]
            self.assertIn("source: local", snippet, snippet)


if __name__ == "__main__":
    unittest.main()
