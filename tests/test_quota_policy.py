from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios.explain import explain
from kerdoios.optimize import plan
from kerdoios.providers.fixture import fixture_offers
from kerdoios.quota import PlanningPolicy, QuotaDimension, QuotaState, free_only_rejection
from kerdoios.types import Economics, Mode, ResourceOffer, WorkRequirement

NO_PAID_SPILL = PlanningPolicy(free_only=True, no_paid_spill=True)
EXPLICIT_PAID_SPILL = PlanningPolicy(free_only=True, no_paid_spill=False)


def _requirement() -> WorkRequirement:
    return WorkRequirement(
        coding=0.5,
        reasoning=0.5,
        tool_use=True,
        tools=("github",),
        context=128_000,
        parallelism=1,
        mode=Mode.CHEAP,
    )


def _free_quota(*, remaining: float) -> QuotaState:
    """Groq day window reconstructed to ``remaining`` with a live minute window."""
    return QuotaState(
        provider="groq",
        model="free-coder:free",
        dimensions={
            "tpm": QuotaDimension(limit=8000.0, remaining=500.0, source="provider_header"),
            "tpd": QuotaDimension(limit=1000.0, remaining=remaining, source="locally_reconstructed"),
        },
        observed_at=1000.0,
    )


def _free_offer(*, remaining: float) -> ResourceOffer:
    base = next(offer for offer in fixture_offers() if offer.id == "groq/free-70b")
    # The legacy scalar still says "free"; only the structured day window knows
    # the account is spent. That is exactly the conflation this issue fixes.
    return replace(
        base,
        id="groq/free-coder",
        model="free-coder:free",
        economics=Economics(remaining_free_quota=500.0, quota=_free_quota(remaining=remaining)),
    )


def _paid_offer() -> ResourceOffer:
    return next(offer for offer in fixture_offers() if offer.id == "frontier/paid")


class FreeOnlyPolicyTests(unittest.TestCase):
    def test_exhausted_free_never_falls_into_paid(self) -> None:
        offers = [_free_offer(remaining=0.0), _paid_offer()]
        built = plan(offers, _requirement(), policy=NO_PAID_SPILL)
        self.assertEqual(built.placements, [])
        self.assertTrue(built.no_free_capacity)
        self.assertTrue(built.free_only)
        self.assertTrue(built.no_paid_spill)
        self.assertNotIn("paid-api", {placement.provider for placement in built.placements})
        reasons = {rejection.offer_id: rejection.reason for rejection in built.rejections}
        self.assertIn("quota window exhausted", reasons["groq/free-coder"])
        self.assertIn("no_paid_spill", reasons["frontier/paid"])
        self.assertTrue(built.to_dict()["no_free_capacity"])

    def test_paid_spill_only_with_explicit_opt_in(self) -> None:
        offers = [_free_offer(remaining=0.0), _paid_offer()]
        built = plan(offers, _requirement(), policy=EXPLICIT_PAID_SPILL)
        self.assertTrue(any(placement.provider == "paid-api" for placement in built.placements))
        self.assertFalse(built.no_free_capacity)
        self.assertFalse(built.no_paid_spill)

    def test_free_candidate_wins_and_paid_is_excluded(self) -> None:
        offers = [_free_offer(remaining=900.0), _paid_offer()]
        built = plan(offers, _requirement(), policy=NO_PAID_SPILL)
        providers = {placement.provider for placement in built.placements}
        self.assertEqual(providers, {"groq"})
        self.assertFalse(built.no_free_capacity)

    def test_default_policy_leaves_paid_plans_unchanged(self) -> None:
        built = plan([_paid_offer()], _requirement())
        self.assertTrue(any(placement.provider == "paid-api" for placement in built.placements))
        self.assertFalse(built.free_only)
        self.assertFalse(built.no_free_capacity)

    def test_explain_reports_no_free_capacity(self) -> None:
        offers = [_free_offer(remaining=0.0), _paid_offer()]
        built = plan(offers, _requirement(), policy=NO_PAID_SPILL)
        text = explain(offers, _requirement(), built)
        self.assertIn("No free capacity", text)

    def test_free_only_rejection_prefers_structured_quota_over_legacy_scalar(self) -> None:
        exhausted = _free_offer(remaining=0.0)
        self.assertEqual(
            free_only_rejection(exhausted, NO_PAID_SPILL), "free_only: quota window exhausted"
        )
        # Exhausted stays excluded even when paid spill is explicitly allowed.
        self.assertEqual(
            free_only_rejection(exhausted, EXPLICIT_PAID_SPILL), "free_only: quota window exhausted"
        )
        paid = _paid_offer()
        self.assertIn("no_paid_spill", free_only_rejection(paid, NO_PAID_SPILL) or "")
        self.assertIsNone(free_only_rejection(paid, EXPLICIT_PAID_SPILL))
        self.assertIsNone(free_only_rejection(paid, PlanningPolicy()))


if __name__ == "__main__":
    unittest.main()
