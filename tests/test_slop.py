from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules"}
_TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".txt", ".json"}
_SUPPRESSION = re.compile(
    r"#\s*(?:type:\s*ignore(?:\[[^\]]+\])?|noqa(?::\s*[A-Z0-9,\s]+)?)\s*$"
)
_COAUTHOR = re.compile(
    r"Co-authored-by:.*\b(?:Cursor|Copilot)\b",
    re.IGNORECASE,
)
_TODO = re.compile(r"\b(?:TODO|FIXME)\b")


def _tracked_text_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES and path.name not in {
            "AGENTS.md",
            "WORKERS.md",
            "CONTRIBUTING.md",
        }:
            continue
        files.append(path)
    return files


class AntiSlopTests(unittest.TestCase):
    def test_agents_has_anti_slop_section(self) -> None:
        text = (ROOT / "AGENTS.md").read_text()
        self.assertIn("## Anti-slop", text)
        self.assertIn("never `mutmut run`", text)
        self.assertIn("missing/None/default-0 price is unknown, not free", text)

    def test_pr_template_mutation_is_na(self) -> None:
        text = (ROOT / ".github/PULL_REQUEST_TEMPLATE.md").read_text()
        self.assertNotIn("mutmut run", text)
        self.assertRegex(text, r"(?m)^mutation: n/a$")

    def test_no_cursor_or_copilot_coauthor(self) -> None:
        hits: list[str] = []
        for path in _tracked_text_files():
            rel = str(path.relative_to(ROOT))
            if rel == "tests/test_slop.py":
                continue
            for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if _COAUTHOR.search(line):
                    hits.append(f"{rel}:{i}:{line.strip()}")
        self.assertEqual(hits, [])

    def test_no_todo_or_fixme(self) -> None:
        hits: list[str] = []
        for path in _tracked_text_files():
            rel = str(path.relative_to(ROOT))
            if rel == "tests/test_slop.py":
                continue
            for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if _TODO.search(line):
                    hits.append(f"{rel}:{i}:{line.strip()}")
        self.assertEqual(hits, [])

    def test_suppressions_have_a_same_line_why(self) -> None:
        hits: list[str] = []
        for path in _tracked_text_files():
            if path.suffix != ".py":
                continue
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if _SUPPRESSION.search(line.rstrip()):
                    hits.append(f"{path.relative_to(ROOT)}:{i}:{line.strip()}")
        self.assertEqual(hits, [])

    def test_optimize_does_not_restate_free_mode_guard(self) -> None:
        text = (ROOT / "optimize.py").read_text()
        self.assertNotIn("# FREE mode refuses paid overflow", text)

    def test_plan_explain_do_not_advertise_unread_fixture_flag(self) -> None:
        text = (ROOT / "__main__.py").read_text()
        self.assertNotIn('add_argument("--fixture"', text)

    def test_plugin_yaml_has_no_unread_config_schema(self) -> None:
        text = (ROOT / "plugin.yaml").read_text()
        self.assertNotIn("config_schema:", text)
        self.assertNotIn("default_mode:", text)
        self.assertNotIn("maximum_paid_usd:", text)

    def test_explain_schema_does_not_claim_last_inventory(self) -> None:
        text = (ROOT / "__init__.py").read_text()
        self.assertNotIn("last inventory", text.lower())

    def test_workers_md_does_not_retell_claim_loop(self) -> None:
        path = ROOT / "WORKERS.md"
        if not path.exists():
            return
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        self.assertLessEqual(len(lines), 1, msg="WORKERS.md must be absent or a one-line pointer")
        joined = "\n".join(lines).lower()
        self.assertNotIn("claimable", joined)
        self.assertIn("agents.md", joined)

    def test_tests_do_not_bind_unused_naive_paid(self) -> None:
        text = (ROOT / "tests/test_optimize.py").read_text()
        self.assertNotIn("naive_paid", text)
        self.assertNotIn("_ = ", text)

    def test_capability_score_for_is_not_unread(self) -> None:
        text = (ROOT / "types.py").read_text()
        self.assertNotIn("def score_for(", text)

    def test_one_line_provider_wrappers_are_not_unread(self) -> None:
        fixture = (ROOT / "providers/fixture.py").read_text()
        openrouter = (ROOT / "providers/openrouter.py").read_text()
        self.assertNotIn("class FixtureProvider", fixture)
        self.assertNotIn("class OpenRouterProvider", openrouter)


if __name__ == "__main__":
    unittest.main()
