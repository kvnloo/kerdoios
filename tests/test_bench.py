from __future__ import annotations

import io
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios import register
from kerdoios.bench import (
    DEFAULT_FIXTURE_DIR,
    QUALITY_RATIO_THRESHOLD,
    live_post,
    load_catalog,
    load_golden,
    report,
    regressions,
)
from kerdoios.observed import MIN_OBSERVATIONS
from kerdoios.optimize import hard_filter, plan
from kerdoios.score import capability_fit
from kerdoios.types import Mode, WorkRequirement


class _Ctx:
    def __init__(self) -> None:
        self.tools: dict[str, dict] = {}

    def register_tool(self, *, name: str, toolset: str, schema: dict, handler) -> None:
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler}


class FixtureBenchTests(unittest.TestCase):
    def test_fixtures_are_frozen_json_not_live_llm(self) -> None:
        for name in ("catalog.json", "suite.json", "golden.json"):
            text = (DEFAULT_FIXTURE_DIR / name).read_text(encoding="utf-8")
            payload = json.loads(text)
            self.assertIn("live LLM", payload["comment"])

    def test_offline_report_does_not_regress_vs_golden(self) -> None:
        golden = load_golden(DEFAULT_FIXTURE_DIR / "golden.json")
        payload = report()
        self.assertIs(payload["openrouter_bucket"], False)
        self.assertIsNone(payload["ex_post"])
        self.assertEqual(payload["astra_offer_id"], "frontier/paid")
        self.assertEqual(regressions(payload, golden), [])
        cheap = next(head for head in payload["heads"] if head["id"] == "cheap-100")
        self.assertEqual(cheap["estimated_tokens"], 480000)
        self.assertGreater(cheap["estimated_tokens"], 0)
        self.assertLess(cheap["frontier_token_share"], 1.0)

    def test_token_spend_regression_fails(self) -> None:
        golden = load_golden(DEFAULT_FIXTURE_DIR / "golden.json")
        actual = deepcopy(report())
        actual["heads"][0]["estimated_tokens"] += 1
        reasons = regressions(actual, golden)
        self.assertTrue(any("estimated_tokens" in item for item in reasons))

    def test_quality_floor_regression_fails(self) -> None:
        golden = load_golden(DEFAULT_FIXTURE_DIR / "golden.json")
        actual = deepcopy(report())
        actual["heads"][0]["quality_floor"] = golden["heads"][0]["quality_floor"] - 0.01
        reasons = regressions(actual, golden)
        self.assertTrue(any("quality_floor" in item for item in reasons))

    def test_cli_bench_is_offline(self) -> None:
        from kerdoios.__main__ import main

        buf = io.StringIO()
        with patch("sys.stdout", buf), patch(
            "kerdoios.__main__.discover_all",
            side_effect=AssertionError("bench must not discover live inventory"),
        ), patch("urllib.request.urlopen", side_effect=AssertionError("bench must not POST or GET")):
            rc = main(["bench"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertIs(payload["openrouter_bucket"], False)
        self.assertIsNone(payload["ex_post"])
        self.assertEqual(regressions(payload, load_golden(DEFAULT_FIXTURE_DIR / "golden.json")), [])


class LiveReceiptTests(unittest.TestCase):
    def test_missing_receipts_pass_flags_are_null(self) -> None:
        result = live_post([])
        self.assertIsNone(result["pass_quality"])
        self.assertIsNone(result["pass_tokens"])
        self.assertNotEqual(result["pass_quality"], QUALITY_RATIO_THRESHOLD)
        self.assertEqual(result["baseline_n"], 0)
        payload = report(live=True, observations=[])
        self.assertIsNone(payload["ex_post"]["pass_quality"])
        self.assertIsNone(payload["ex_post"]["pass_tokens"])

    def test_enough_baseline_receipts_can_pass_quality(self) -> None:
        portfolio = [
            SimpleNamespace(task_type="coding", completed=True, model="qwen-coder")
            for _ in range(MIN_OBSERVATIONS)
        ]
        baseline = [
            SimpleNamespace(task_type="baseline", completed=True, model="gpt-6-astra")
            for _ in range(MIN_OBSERVATIONS)
        ]
        result = live_post(portfolio + baseline)
        self.assertTrue(result["pass_quality"])
        self.assertGreaterEqual(result["quality_ratio"], QUALITY_RATIO_THRESHOLD)
        self.assertIsNone(result["pass_tokens"])

    def test_token_gate_uses_receipt_tokens_not_dollars(self) -> None:
        portfolio = [
            SimpleNamespace(
                task_type="coding",
                completed=True,
                model="qwen-coder",
                input_tokens=4000,
                output_tokens=800,
            )
            for _ in range(MIN_OBSERVATIONS)
        ]
        baseline = [
            SimpleNamespace(
                task_type="baseline",
                completed=True,
                model="gpt-6-astra",
                input_tokens=8000,
                output_tokens=1600,
            )
            for _ in range(MIN_OBSERVATIONS)
        ]
        result = live_post(portfolio + baseline)
        self.assertTrue(result["pass_quality"])
        self.assertTrue(result["pass_tokens"])
        self.assertEqual(result["portfolio_tokens"], MIN_OBSERVATIONS * 4800)
        self.assertEqual(result["baseline_tokens"], MIN_OBSERVATIONS * 9600)
        self.assertEqual(result["portfolio_frontier_tokens"], 0)


class OffloadTests(unittest.TestCase):
    def test_astra_level_requirement_rejects_cheap_below_bar(self) -> None:
        offers, _astra = load_catalog(DEFAULT_FIXTURE_DIR / "catalog.json")
        req = WorkRequirement(
            coding=0.94,
            reasoning=0.93,
            tool_use=True,
            tools=("github",),
            context=128_000,
            parallelism=8,
            mode=Mode.CHEAP,
        )
        cheap = next(offer for offer in offers if offer.id == "groq/free-70b")
        self.assertLessEqual(capability_fit(cheap, req), 0.0)
        _kept, rejected = hard_filter(offers, req)
        self.assertIn(cheap.id, {item.offer_id for item in rejected})

    def test_cheaper_offer_that_passes_capability_fit_is_placed(self) -> None:
        offers, _astra = load_catalog(DEFAULT_FIXTURE_DIR / "catalog.json")
        req = WorkRequirement(
            coding=0.75,
            reasoning=0.75,
            tool_use=True,
            tools=("github",),
            context=128_000,
            parallelism=4,
            mode=Mode.CHEAP,
        )
        cheap = next(offer for offer in offers if offer.id == "groq/free-70b")
        cheaper_ok = next(offer for offer in offers if offer.id == "nous/hermes-70b")
        self.assertLessEqual(capability_fit(cheap, req), 0.0)
        self.assertGreater(capability_fit(cheaper_ok, req), 0.0)
        built = plan(offers, req)
        ids = {item.offer_id for item in built.placements}
        self.assertNotIn("groq/free-70b", ids)
        self.assertIn("nous/hermes-70b", ids)


class SurfaceTests(unittest.TestCase):
    def test_plugin_yaml_and_register_expose_kerdoios_bench(self) -> None:
        text = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^  - kerdoios_bench$")
        ctx = _Ctx()
        register(ctx)
        self.assertIn("kerdoios_bench", ctx.tools)
        self.assertEqual(ctx.tools["kerdoios_bench"]["schema"]["name"], "kerdoios_bench")
        self.assertTrue(callable(ctx.tools["kerdoios_bench"]["handler"]))
        schema = json.dumps(ctx.tools["kerdoios_bench"]["schema"])
        self.assertIn("live", schema)
        self.assertNotIn("execute", schema.lower())

    def test_docs_point_at_bench_fixtures(self) -> None:
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        for text in (agents, readme, contributing):
            self.assertIn("python3 -m kerdoios bench", text)
        self.assertIn("tests/fixtures/bench/", readme)


if __name__ == "__main__":
    unittest.main()
