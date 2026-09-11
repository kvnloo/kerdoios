from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios.explain import explain
from kerdoios.optimize import hard_filter, plan
from kerdoios.pareto import nondominated
from kerdoios.providers.fixture import fixture_offers
from kerdoios.types import Mode, WorkRequirement


class ParetoTests(unittest.TestCase):
    def test_drops_dominated_point(self) -> None:
        points = [(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)]
        front = nondominated(
            points,
            dimensions=[lambda p: p[0], lambda p: p[1]],
            minimize=[True, False],
        )
        self.assertIn((1.0, 1.0), front)
        self.assertNotIn((2.0, 1.0), front)
        self.assertNotIn((1.0, 0.5), front)


class FilterTests(unittest.TestCase):
    def test_context_and_tools_reject_tiny(self) -> None:
        req = WorkRequirement(context=128_000, tool_use=True, tools=("github",), coding=0.6)
        kept, rejected = hard_filter(fixture_offers(), req)
        ids = {offer.id for offer in kept}
        reasons = {item.offer_id: item.reason for item in rejected}
        self.assertNotIn("tiny/8k-no-tools", ids)
        self.assertIn("tiny/8k-no-tools", reasons)

    def test_local_only_keeps_only_local(self) -> None:
        req = WorkRequirement(privacy="local_only", context=128_000, tool_use=True, tools=("github",), coding=0.7)
        kept, rejected = hard_filter(fixture_offers(), req)
        self.assertEqual({offer.id for offer in kept}, {"local/qwen-32b"})
        self.assertTrue(any("local_only" in item.reason for item in rejected))


class PlanTests(unittest.TestCase):
    def test_portfolio_splits_hundred_workers_under_budget(self) -> None:
        req = WorkRequirement(
            coding=0.6,
            reasoning=0.6,
            tool_use=True,
            tools=("github",),
            context=128_000,
            parallelism=100,
            maximum_cost=0.50,
            mode=Mode.CHEAP,
        )
        built = plan(fixture_offers(), req)
        providers = {p.provider for p in built.placements}
        total_workers = sum(p.workers for p in built.placements)
        self.assertGreaterEqual(total_workers, 80)
        self.assertGreaterEqual(len(built.placements), 3)
        self.assertTrue({"groq", "openrouter", "local"} & providers)
        self.assertLessEqual(built.estimated_cost, 0.50 + 1e-9)
        self.assertNotEqual(len(providers), 1, msg=json.dumps(built.to_dict(), indent=2))
        self.assertLess(built.unplaced_workers, 25)

    def test_free_mode_skips_paid_api(self) -> None:
        req = WorkRequirement(
            coding=0.6,
            reasoning=0.6,
            tool_use=True,
            tools=("github",),
            context=128_000,
            parallelism=40,
            mode=Mode.FREE,
        )
        built = plan(fixture_offers(), req)
        self.assertFalse(any(p.provider == "paid-api" for p in built.placements))
        self.assertEqual(built.estimated_cost, 0.0)

    def test_explain_mentions_rejections(self) -> None:
        req = WorkRequirement(
            coding=0.7,
            tool_use=True,
            tools=("github",),
            context=128_000,
            parallelism=10,
            mode=Mode.CHEAP,
        )
        text = explain(fixture_offers(), req)
        self.assertIn("Selected", text)
        self.assertIn("tiny/8k-no-tools", text)


if __name__ == "__main__":
    unittest.main()
