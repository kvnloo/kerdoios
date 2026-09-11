from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios import register
from kerdoios.observed import MIN_OBSERVATIONS, Observation
from kerdoios.optimize import plan
from kerdoios.providers.fixture import fixture_offers
from kerdoios.score import effective_unit_cost
from kerdoios.tokenomics import (
    ante,
    is_paid_baseline,
    naive_paid_cost,
    naive_paid_offer,
    post,
    report,
)
from kerdoios.types import Mode, WorkRequirement


class _Ctx:
    def __init__(self) -> None:
        self.tools: dict[str, dict] = {}

    def register_tool(self, *, name: str, toolset: str, schema: dict, handler) -> None:
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler}


def _cheap_100() -> WorkRequirement:
    return WorkRequirement(
        coding=0.7,
        reasoning=0.6,
        tool_use=True,
        tools=("github",),
        context=128_000,
        parallelism=100,
        maximum_cost=0.50,
        mode=Mode.CHEAP,
        estimated_input_tokens=4000,
        estimated_output_tokens=800,
    )


class PaidBaselineTests(unittest.TestCase):
    def test_unknown_prices_are_not_paid_baseline(self) -> None:
        offers = {offer.id: offer for offer in fixture_offers()}
        self.assertFalse(is_paid_baseline(offers["groq/free-70b"]))
        self.assertFalse(is_paid_baseline(offers["tiny/8k-no-tools"]))
        self.assertFalse(is_paid_baseline(offers["local/qwen-32b"]))
        self.assertFalse(is_paid_baseline(offers["nous/hermes-70b"]))
        frontier = offers["frontier/paid"]
        self.assertTrue(is_paid_baseline(frontier))
        half = replace(
            frontier,
            economics=replace(frontier.economics, output_token_price=None),
        )
        self.assertFalse(is_paid_baseline(half))
        zeroed = replace(
            frontier,
            economics=replace(
                frontier.economics,
                input_token_price=0.0,
                output_token_price=0.0,
            ),
        )
        self.assertFalse(is_paid_baseline(zeroed))

    def test_same_budget_paid_truncate_is_not_naive(self) -> None:
        req = _cheap_100()
        offers = fixture_offers()
        built = plan(offers, req)
        placed = sum(item.workers for item in built.placements)
        paid_only = [offer for offer in offers if is_paid_baseline(offer)]
        truncated = plan(paid_only, req)
        truncated_workers = sum(item.workers for item in truncated.placements)
        self.assertEqual(truncated_workers, 22)
        self.assertAlmostEqual(truncated.estimated_cost, 0.479952, places=6)
        offer = naive_paid_offer(offers, req)
        self.assertIsNotNone(offer)
        naive = naive_paid_cost(offer, req, placed)
        self.assertAlmostEqual(naive, 2.1816, places=6)
        self.assertNotAlmostEqual(naive, truncated.estimated_cost)
        self.assertEqual(placed, 100)
        self.assertGreater(placed, truncated_workers)


class AnteTests(unittest.TestCase):
    def test_fixture_hundred_workers_pass_cost_cell(self) -> None:
        req = _cheap_100()
        result = ante(fixture_offers(), req)
        self.assertEqual(result["placed_workers"], 100)
        self.assertAlmostEqual(result["kerdoios_cost"], 0.392688, places=6)
        self.assertAlmostEqual(result["naive_cost"], 2.1816, places=6)
        self.assertTrue(result["pass_cost"])
        self.assertGreaterEqual(result["cost_reduction"], 0.50)
        self.assertEqual(result["naive_offer_id"], "frontier/paid")

    def test_missing_paid_offer_pass_cost_is_none(self) -> None:
        req = _cheap_100()
        unpaid = [offer for offer in fixture_offers() if not is_paid_baseline(offer)]
        result = ante(unpaid, req)
        self.assertIsNone(result["pass_cost"])
        self.assertIsNone(result["naive_cost"])
        self.assertIsNone(result["naive_offer_id"])


class PostTests(unittest.TestCase):
    def test_missing_baseline_receipts_pass_success_is_none(self) -> None:
        rows = [
            Observation("groq", "llama-3.3-70b", "coding", True, 0.0)
            for _ in range(MIN_OBSERVATIONS)
        ]
        result = post(rows, naive_offer=None, req=_cheap_100())
        self.assertIsNone(result["pass_success"])
        self.assertNotEqual(result["pass_success"], 0.95)
        self.assertIsNone(result["success_ratio"])
        self.assertEqual(result["baseline_n"], 0)
        self.assertEqual(result["portfolio_n"], MIN_OBSERVATIONS)

    def test_enough_baseline_can_pass_success(self) -> None:
        portfolio = [
            Observation("groq", "llama-3.3-70b", "coding", True, 0.0)
            for _ in range(MIN_OBSERVATIONS)
        ]
        baseline = [
            Observation("paid-api", "frontier-codex", "baseline", True, 0.02)
            for _ in range(MIN_OBSERVATIONS)
        ]
        result = post(portfolio + baseline, naive_offer=None, req=_cheap_100())
        self.assertGreaterEqual(result["portfolio_n"], MIN_OBSERVATIONS)
        self.assertGreaterEqual(result["baseline_n"], MIN_OBSERVATIONS)
        self.assertTrue(result["pass_success"])
        self.assertGreaterEqual(result["success_ratio"], 0.95)

    def test_tokens_present_naive_is_prices_times_tokens(self) -> None:
        req = _cheap_100()
        offer = naive_paid_offer(fixture_offers(), req)
        self.assertIsNotNone(offer)
        with_tokens = [
            SimpleNamespace(
                task_type="coding",
                completed=True,
                actual_cost=0.0,
                input_tokens=4000,
                output_tokens=800,
            )
        ]
        priced = post(with_tokens, naive_offer=offer, req=req)
        expected_tokens = (
            offer.economics.input_token_price * 4000
            + offer.economics.output_token_price * 800
        )
        self.assertAlmostEqual(priced["naive_cost"], expected_tokens)

    def test_without_tokens_naive_is_one_unit_cost(self) -> None:
        req = _cheap_100()
        offer = naive_paid_offer(fixture_offers(), req)
        self.assertIsNotNone(offer)
        unit = effective_unit_cost(offer, req)
        rows = [Observation("paid-api", "frontier-codex", "coding", True, 0.01)]
        unpriced = post(rows, naive_offer=offer, req=req)
        self.assertAlmostEqual(unpriced["naive_cost"], unit)


class ReportAndSurfaceTests(unittest.TestCase):
    def test_report_openrouter_bucket_is_false(self) -> None:
        req = _cheap_100()
        payload = report(fixture_offers(), req)
        self.assertIs(payload["openrouter_bucket"], False)
        self.assertIn("thresholds", payload)
        self.assertIn("ex_ante", payload)
        self.assertIn("ex_post", payload)
        self.assertTrue(payload["ex_ante"]["pass_cost"])
        self.assertIsNone(payload["ex_post"]["pass_success"])
        self.assertEqual(payload["thresholds"]["cost_reduction"], 0.50)
        self.assertEqual(payload["thresholds"]["min_observations"], MIN_OBSERVATIONS)

    def test_cli_validate_hundred_workers_cheap(self) -> None:
        from kerdoios.__main__ import main

        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            empty = str(Path(tmp) / "observed.jsonl")
            with patch("sys.stdout", buf):
                rc = main(
                    [
                        "validate",
                        "--workers",
                        "100",
                        "--budget",
                        "0.50",
                        "--mode",
                        "cheap",
                        "--observed-log",
                        empty,
                    ]
                )
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["ex_ante"]["pass_cost"])
        self.assertAlmostEqual(payload["ex_ante"]["kerdoios_cost"], 0.392688, places=6)
        self.assertAlmostEqual(payload["ex_ante"]["naive_cost"], 2.1816, places=6)
        self.assertEqual(payload["ex_ante"]["placed_workers"], 100)
        self.assertIs(payload["openrouter_bucket"], False)
        self.assertIsNone(payload["ex_post"]["pass_success"])

    def test_plugin_yaml_and_register_expose_kerdoios_validate(self) -> None:
        text = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^  - kerdoios_validate$")
        ctx = _Ctx()
        register(ctx)
        self.assertIn("kerdoios_validate", ctx.tools)
        self.assertEqual(ctx.tools["kerdoios_validate"]["toolset"], "kerdoios")
        self.assertEqual(ctx.tools["kerdoios_validate"]["schema"]["name"], "kerdoios_validate")
        self.assertTrue(callable(ctx.tools["kerdoios_validate"]["handler"]))


if __name__ == "__main__":
    unittest.main()
