"""Subscription plan windows: burn what perishes, offload what is over pace, fail open."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios.__main__ import main
from kerdoios.bind import hermes_fallback_entries
from kerdoios.explain import explain
from kerdoios.optimize import plan
from kerdoios.presets import PRESETS
from kerdoios.providers import subscription
from kerdoios.providers.fixture import fixture_offers
from kerdoios.score import score
from kerdoios.types import Mode, WorkRequirement

NOW = datetime(2026, 9, 30, 18, 0, tzinfo=timezone.utc)
WEEK = 10080


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _window(used: float, resets_in_h: float, minutes: int = WEEK) -> dict:
    return {"usedPercent": used, "resetsAt": _iso(NOW + timedelta(hours=resets_in_h)), "windowMinutes": minutes}


def _row(provider: str, *windows: dict, age_h: float = 0.1) -> dict:
    usage: dict = {"updatedAt": _iso(NOW - timedelta(hours=age_h))}
    for slot, w in zip(("primary", "secondary", "tertiary"), windows):
        usage[slot] = w
    # Identity fields CodexBar carries; they must never reach kerdoios output.
    usage.update(accountEmail="someone@example.com", accountOrganization="Org Inc", identity={"accountId": "acct-123"},
                 loginMethod="oauth")
    return {"provider": provider, "source": "oauth", "usage": usage}


# Burn: 21% used, reset in 13h -> 79% of the weekly window perishes.
BURN_CACHE = [_row("claude", _window(21, 13))]
# Offload: cursor exhausted; codex 90% used two days into the week (over pace).
OFFLOAD_CACHE = [_row("cursor", _window(100, 200, 43200)), _row("codex", _window(90, 120))]
# Stale: the snapshot is 8h old.
STALE_CACHE = [_row("claude", _window(21, 13), age_h=8)]

REQ = WorkRequirement(coding=0.7, reasoning=0.6, tool_use=True, context=128_000, parallelism=8,
                      mode=Mode.BALANCED, tools=("github",))


def _report(rows) -> subscription.WindowReport:
    windows, skipped = subscription.windows_from_codexbar(rows, NOW)
    return subscription._finish("codexbar", windows, skipped)


def _write(tmp: str, rows) -> Path:
    path = Path(tmp) / "last.json"
    path.write_text(json.dumps(rows))
    return path


class WindowEvaluationTests(unittest.TestCase):
    def test_burn_case_perishing_weekly_window(self) -> None:
        report = _report(BURN_CACHE)
        self.assertEqual(report.status, "ok")
        (w,) = report.windows
        self.assertEqual((w.group, w.window, w.posture), ("claude", "weekly", "BURN"))
        self.assertAlmostEqual(w.remaining_fraction, 0.79)
        self.assertAlmostEqual(w.seconds_until_reset, 13 * 3600, delta=1)
        self.assertEqual(w.window_minutes, WEEK)
        self.assertGreater(w.surplus_fraction_at_reset, 0.7)
        self.assertIn("perishes at reset in 13.0h", w.arithmetic)

    def test_offload_case_exhausted_and_over_pace(self) -> None:
        by_group = {w.group: w for w in _report(OFFLOAD_CACHE).windows}
        self.assertEqual((by_group["cursor"].posture, by_group["cursor"].reason), ("OFFLOAD", "exhausted"))
        self.assertTrue(by_group["cursor"].exhausted)
        self.assertEqual((by_group["codex"].posture, by_group["codex"].reason), ("OFFLOAD", "runs_out_before_reset"))
        self.assertFalse(by_group["codex"].exhausted)

    def test_stale_snapshot_yields_no_windows(self) -> None:
        report = _report(STALE_CACHE)
        self.assertEqual(report.status, "stale")
        self.assertEqual(report.windows, [])
        self.assertIn("stale snapshot", report.skipped[0])
        self.assertEqual(subscription.offers_from_report(report), [])

    def test_snapshot_predating_reset_is_skipped(self) -> None:
        report = _report([_row("grok", _window(68, -5))])
        self.assertEqual(report.windows, [])
        self.assertIn("predates reset", report.skipped[0])

    def test_short_observation_does_not_assert_burn(self) -> None:
        # Window started 10 minutes ago: one window-average rate is not evidence.
        report = _report([_row("claude", _window(1, 5 - 10 / 60, 300))])
        (w,) = report.windows
        self.assertEqual(w.posture, "BALANCED")
        self.assertTrue(w.reason.endswith("_unconfirmed"))

    def test_identity_fields_are_dropped(self) -> None:
        report = _report(BURN_CACHE + OFFLOAD_CACHE)
        blob = json.dumps(report.to_dict()) + repr(subscription.offers_from_report(report))
        for secret in ("someone@example.com", "Org Inc", "acct-123", "loginMethod", "accountEmail"):
            self.assertNotIn(secret, blob)


class PlanningTests(unittest.TestCase):
    def test_burn_pool_ranks_higher_than_same_pool_on_pace(self) -> None:
        burn = subscription.offers_from_report(_report(BURN_CACHE))[0]
        on_pace = subscription.offers_from_report(_report([_row("claude", _window(40, 72))]))[0]
        self.assertEqual(on_pace.economics.subscription.posture, "BALANCED")
        weights = PRESETS[Mode.BALANCED]
        self.assertGreater(score(burn, REQ, weights).fitness, score(on_pace, REQ, weights).fitness * 1.3)

    def test_burn_pool_is_placed_first_with_visible_reason(self) -> None:
        offers = subscription.merge(fixture_offers(), _report(BURN_CACHE))
        built = plan(offers, REQ)
        first = built.placements[0]
        self.assertEqual(first.offer_id, "subscription/claude")
        self.assertEqual(first.estimated_cost, 0.0)
        self.assertTrue(any(r.startswith("BURN claude:weekly") and "perishes" in r for r in first.reasons))
        baseline = plan(fixture_offers(), REQ)
        self.assertNotEqual(baseline.placements[0].offer_id, "subscription/claude")

    def test_over_pace_pool_ranks_lower_than_same_pool_on_pace(self) -> None:
        over = next(o for o in subscription.offers_from_report(_report(OFFLOAD_CACHE)) if o.provider == "codex")
        on_pace = subscription.offers_from_report(_report([_row("codex", _window(40, 72))]))[0]
        weights = PRESETS[Mode.BALANCED]
        self.assertLess(score(over, REQ, weights).fitness, score(on_pace, REQ, weights).fitness * 0.6)
        self.assertIn("over pace", score(over, REQ, weights).reasons[0])

    def test_exhausted_pool_is_hard_rejected(self) -> None:
        offers = subscription.merge(fixture_offers(), _report(OFFLOAD_CACHE))
        built = plan(offers, REQ)
        reasons = {r.offer_id: r.reason for r in built.rejections}
        self.assertIn("exhausted", reasons["subscription/cursor"])
        self.assertNotIn("subscription/cursor", {p.offer_id for p in built.placements})

    def test_stale_or_missing_snapshot_leaves_plan_unchanged(self) -> None:
        baseline = plan(fixture_offers(), REQ).to_dict()
        for report in (_report(STALE_CACHE), subscription.collect(NOW, source="codexbar",
                                                                   codexbar=Path(os.devnull) / "missing.json")):
            self.assertFalse(report.usable)
            self.assertIn("plan unchanged", report.reason)
            self.assertEqual(plan(subscription.merge(fixture_offers(), report), REQ).to_dict(), baseline)

    def test_explain_shows_windows_and_effect(self) -> None:
        report = _report(BURN_CACHE + OFFLOAD_CACHE)
        offers = subscription.merge(fixture_offers(), report)
        built = plan(offers, REQ)
        built.subscription_windows = report.to_dict()
        text = explain(offers, REQ, built)
        self.assertIn("Subscription windows (codexbar): ok", text)
        self.assertIn("BURN     claude:weekly", text)
        self.assertIn("-> subscription/claude: placed", text)
        self.assertIn("-> subscription/cursor: rejected: subscription cursor:30d exhausted", text)

    def test_subscription_offers_are_never_bound_to_hermes(self) -> None:
        offers = subscription.merge(fixture_offers(), _report(BURN_CACHE))
        built = plan(offers, REQ)
        self.assertEqual(built.placements[0].offer_id, "subscription/claude")
        providers = {e["provider"] + "/" + e["model"] for e in hermes_fallback_entries(built, offers)}
        self.assertNotIn("openrouter/subscription", providers)
        self.assertFalse(any(e["model"] == "subscription" for e in hermes_fallback_entries(built, offers)))


class SourceTests(unittest.TestCase):
    def test_z0int_document_maps_and_drops_stale(self) -> None:
        doc = {
            "factory": {"posture": "BURN"},
            "pools": [
                {"id": "claude:weekly", "group": "claude", "kind": "frontier", "remaining": 79.0,
                 "resets_at": _iso(NOW + timedelta(hours=13)), "window_hours": 168.0, "posture": "BURN",
                 "reason": "surplus_perishes_at_reset", "arithmetic": "79% perishes", "confidence": "ok",
                 "projected_surplus_at_reset": 77.0},
                {"id": "grok:primary", "group": "grok", "kind": "frontier", "remaining": 32.0,
                 "resets_at": _iso(NOW - timedelta(days=10)), "posture": "BALANCED", "confidence": "stale",
                 "reason": "observation_predates_reset"},
                {"id": "route:local", "group": "route:local", "kind": "local", "remaining": None},
            ],
        }
        windows, skipped, factory = subscription.windows_from_z0int(doc, NOW)
        self.assertEqual(factory, "BURN")
        self.assertEqual([(w.group, w.window, w.posture, w.source) for w in windows],
                         [("claude", "weekly", "BURN", "z0int")])
        self.assertAlmostEqual(windows[0].surplus_fraction_at_reset, 0.77)
        self.assertEqual(windows[0].window_minutes, WEEK)
        self.assertIn("grok:primary", skipped[0])

    def test_auto_falls_back_to_codexbar_without_z0int(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(subscription, "_z0int_command", return_value=None):
            report = subscription.collect(NOW, source="auto", codexbar=_write(tmp, BURN_CACHE))
        self.assertEqual((report.status, report.source), ("ok", "codexbar"))
        self.assertIn("z0int: FileNotFoundError", report.reason)

    def test_disabled_source_is_a_no_op(self) -> None:
        report = subscription.collect(NOW, source="off")
        self.assertEqual(report.status, "disabled")
        self.assertEqual(subscription.merge(fixture_offers(), report), fixture_offers())

    def test_cli_explain_reads_fixture_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # CLI evaluates at wall-clock time, so build the fixture relative to it.
            now = datetime.now(timezone.utc)
            rows = [_row("claude", {"usedPercent": 21, "resetsAt": _iso(now + timedelta(hours=13)),
                                    "windowMinutes": WEEK})]
            rows[0]["usage"]["updatedAt"] = _iso(now - timedelta(minutes=5))
            env = {subscription.SOURCE_ENV: "codexbar", subscription.CODEXBAR_ENV: str(_write(tmp, rows))}
            with patch.dict(os.environ, env):
                out = io.StringIO()
                with redirect_stdout(out):
                    self.assertEqual(main(["explain", "--mode", "balanced"]), 0)
                text = out.getvalue()
                self.assertIn("Selected claude/subscription", text)
                self.assertIn("BURN claude:weekly", text)
                out = io.StringIO()
                with redirect_stdout(out):
                    self.assertEqual(main(["explain", "--mode", "balanced", "--no-subscriptions"]), 0)
                self.assertNotIn("claude/subscription", out.getvalue())
                self.assertIn("Subscription windows (none): disabled", out.getvalue())


if __name__ == "__main__":
    unittest.main()
