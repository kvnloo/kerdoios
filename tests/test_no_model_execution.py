from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "kerdoios"

_LITELLM_ATTR = re.compile(r"\blitellm\.(?:completion|acompletion|completion_with_retries)\b")
_CHAT_HTTP = re.compile(
    r"/chat/completions|chat\.completions(?:\.create)?|openai\.ChatCompletion",
    re.IGNORECASE,
)
_POST_HELPER = re.compile(r"\bdef\s+post_json\b|\bmethod\s*=\s*[\"']POST[\"']")
_EXECUTE_PARSER = re.compile(r"add_parser\(\s*[\"']execute[\"']")


def _package_py_files() -> list[Path]:
    files = sorted(PKG.rglob("*.py"))
    shim = ROOT / "__init__.py"
    if shim.is_file():
        files.append(shim)
    return files


def _litellm_imports(path: Path) -> list[str]:
    hits: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "litellm" or alias.name.startswith("litellm."):
                    hits.append(f"{path.relative_to(ROOT)}:{node.lineno}:import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "litellm" or node.module.startswith("litellm."):
                hits.append(f"{path.relative_to(ROOT)}:{node.lineno}:from {node.module}")
    return hits


class NoModelExecutionTests(unittest.TestCase):
    def test_package_sources_do_not_import_litellm(self) -> None:
        hits: list[str] = []
        for path in _package_py_files():
            hits.extend(_litellm_imports(path))
        self.assertEqual(hits, [])

    def test_package_sources_do_not_call_chat_completions_http(self) -> None:
        hits: list[str] = []
        for path in _package_py_files():
            rel = str(path.relative_to(ROOT))
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if _LITELLM_ATTR.search(line) or _CHAT_HTTP.search(line) or _POST_HELPER.search(line):
                    hits.append(f"{rel}:{i}:{line.strip()}")
        self.assertEqual(hits, [])

    def test_catalog_http_helper_is_get_only(self) -> None:
        text = (PKG / "providers" / "http.py").read_text(encoding="utf-8")
        self.assertIn("def get_json", text)
        self.assertIn("def get_json_response", text)
        self.assertNotIn("def post_json", text)
        self.assertNotRegex(text, r"Request\([^)]*\bdata\s*=")
        self.assertNotRegex(text, r"method\s*=\s*[\"']POST[\"']")

    def test_cli_has_no_execute_command(self) -> None:
        main = (PKG / "__main__.py").read_text(encoding="utf-8")
        shim = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIsNone(_EXECUTE_PARSER.search(main))
        self.assertIsNone(_EXECUTE_PARSER.search(shim))
        self.assertNotIn("add_parser(\"execute\"", main)
        self.assertNotIn("add_parser(\"execute\"", shim)

    def test_litellm_is_not_vendored(self) -> None:
        vendor = ROOT / "vendor" / "litellm"
        self.assertFalse(vendor.exists(), "do not vendor LiteLLM in this plugin")
        self.assertFalse((PKG / "litellm").exists())

    def test_docs_say_kerdoios_returns_plan_hermes_executes(self) -> None:
        upstream = (ROOT / "UPSTREAM.md").read_text(encoding="utf-8")
        roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
        for name, text in (("UPSTREAM.md", upstream), ("ROADMAP.md", roadmap)):
            self.assertRegex(
                text,
                r"Kerdoios returns [`']?ExecutionPlan",
                msg=f"{name} must say Kerdoios returns ExecutionPlan",
            )
            self.assertRegex(
                text,
                r"Hermes executes",
                msg=f"{name} must say Hermes executes",
            )
            self.assertIn(
                "LiteLLM",
                text,
                msg=f"{name} must leave commodity routing with LiteLLM",
            )
        self.assertIn(
            "31823",
            upstream,
            msg="UPSTREAM.md must keep the existing LiteLLM quota-pools wedge",
        )
        not_done = roadmap.split("**Not done:**", 1)[1].split("**Trap:**", 1)[0]
        self.assertNotIn(
            "execute-through-LiteLLM",
            not_done,
            msg="execute-through-LiteLLM is Hermes/LiteLLM work, not a Kerdoios leftover",
        )


if __name__ == "__main__":
    unittest.main()
