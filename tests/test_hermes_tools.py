from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios import APPLY_SCHEMA, EXPLAIN_SCHEMA, INVENTORY_SCHEMA, PLAN_SCHEMA, register
from kerdoios.observed import Observation, load_observations


class _Ctx:
    def __init__(self) -> None:
        self.tools: dict[str, dict] = {}

    def register_tool(self, *, name: str, toolset: str, schema: dict, handler) -> None:
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler}


class PluginManifestTests(unittest.TestCase):
    def test_plugin_yaml_lists_record_tool(self) -> None:
        text = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^  - kerdoios_record$")
        self.assertIn("- kerdoios_plan", text)
        self.assertIn("- kerdoios_explain", text)
        self.assertIn("- kerdoios_inventory", text)
        self.assertIn("- kerdoios_apply", text)


class RegisterTests(unittest.TestCase):
    def test_register_exposes_record_and_existing_tools(self) -> None:
        ctx = _Ctx()
        register(ctx)
        self.assertEqual(
            set(ctx.tools),
            {"kerdoios_plan", "kerdoios_explain", "kerdoios_inventory", "kerdoios_record", "kerdoios_apply"},
        )
        self.assertEqual(ctx.tools["kerdoios_record"]["toolset"], "kerdoios")
        self.assertEqual(ctx.tools["kerdoios_record"]["schema"]["name"], "kerdoios_record")
        self.assertTrue(callable(ctx.tools["kerdoios_record"]["handler"]))

    def test_plan_and_explain_schemas_expose_observed(self) -> None:
        for schema in (PLAN_SCHEMA, EXPLAIN_SCHEMA):
            observed = schema["parameters"]["properties"]["observed"]
            self.assertEqual(observed["type"], "boolean")
        self.assertNotIn("observed", INVENTORY_SCHEMA["parameters"]["properties"])

    def test_record_schema_matches_cli_fields(self) -> None:
        ctx = _Ctx()
        register(ctx)
        props = ctx.tools["kerdoios_record"]["schema"]["parameters"]["properties"]
        self.assertEqual(
            set(props),
            {"provider", "model", "task_type", "completed", "cost", "retried"},
        )
        required = ctx.tools["kerdoios_record"]["schema"]["parameters"]["required"]
        self.assertEqual(set(required), {"provider", "model"})
        self.assertNotIn("remaining_free_quota", json.dumps(ctx.tools["kerdoios_record"]["schema"]))

    def test_plan_handler_passes_observed_to_plan(self) -> None:
        ctx = _Ctx()
        register(ctx)
        captured: dict[str, bool] = {}

        class _Plan:
            def to_dict(self) -> dict:
                return {"placements": []}

        def fake_plan(offers, req, *, use_observed: bool = False):
            captured["use_observed"] = use_observed
            return _Plan()

        handler = ctx.tools["kerdoios_plan"]["handler"]
        with patch("kerdoios.plan", fake_plan):
            handler({"workers": 1, "mode": "cheap", "observed": True})
            handler({"workers": 1, "mode": "cheap"})
        self.assertEqual(captured["use_observed"], False)
        with patch("kerdoios.plan", fake_plan):
            handler({"workers": 1, "mode": "cheap", "observed": True})
        self.assertTrue(captured["use_observed"])

    def test_explain_handler_passes_observed(self) -> None:
        ctx = _Ctx()
        register(ctx)
        captured: dict[str, bool] = {}

        def fake_explain(offers, req, built=None, *, use_observed: bool = False) -> str:
            captured["use_observed"] = use_observed
            return "ok\n"

        handler = ctx.tools["kerdoios_explain"]["handler"]
        with patch("kerdoios.explain", fake_explain):
            handler({"workers": 1, "mode": "cheap", "observed": True})
        self.assertTrue(captured["use_observed"])

    def test_record_handler_writes_observation(self) -> None:
        ctx = _Ctx()
        register(ctx)
        handler = ctx.tools["kerdoios_record"]["handler"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observed.jsonl"
            with patch("kerdoios.observed.DEFAULT_LOG_PATH", path):
                payload = json.loads(
                    handler(
                        {
                            "provider": "groq",
                            "model": "llama-3.3-70b",
                            "task_type": "coding",
                            "completed": True,
                            "cost": 0.01,
                            "retried": False,
                        }
                    )
                )
            rows = load_observations(path=path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0],
                Observation("groq", "llama-3.3-70b", "coding", True, 0.01, False),
            )
            self.assertEqual(payload["provider"], "groq")
            self.assertEqual(payload["model"], "llama-3.3-70b")
            self.assertTrue(payload["completed"])
            self.assertEqual(payload["actual_cost"], 0.01)

    def test_record_handler_requires_provider_and_model(self) -> None:
        ctx = _Ctx()
        register(ctx)
        handler = ctx.tools["kerdoios_record"]["handler"]
        with self.assertRaises(ValueError):
            handler({"provider": "", "model": "llama"})
        with self.assertRaises(ValueError):
            handler({"model": "llama"})


if __name__ == "__main__":
    unittest.main()
