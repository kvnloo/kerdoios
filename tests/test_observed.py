from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios.blend import MAX_BLEND_WEIGHT, _blend_weight, apply_observed, blend_offer
from kerdoios.observed import (
    MIN_OBSERVATIONS,
    Observation,
    aggregate,
    load_observations,
    record,
    tokens_per_verified_task,
)
from kerdoios.optimize import plan
from kerdoios.providers.fixture import fixture_offers
from kerdoios.types import Mode, WorkRequirement


class ObservedLogTests(unittest.TestCase):
    def test_record_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observed.jsonl"
            record(Observation("groq", "llama-3.3-70b", "coding", True, 0.001), path=path)
            record(Observation("groq", "llama-3.3-70b", "coding", False, 0.002), path=path)
            rows = load_observations(path=path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].provider, "groq")

    def test_corrupt_line_does_not_lose_prior_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observed.jsonl"
            record(Observation("groq", "llama-3.3-70b", "coding", True, 0.001), path=path)
            with path.open("a") as fh:
                fh.write("{not json\n")
            record(Observation("groq", "llama-3.3-70b", "coding", True, 0.001), path=path)
            rows = load_observations(path=path)
            self.assertEqual(len(rows), 2)

    def test_missing_log_returns_empty(self) -> None:
        rows = load_observations(path=Path("/tmp/definitely-does-not-exist-kerdoios.jsonl"))
        self.assertEqual(rows, [])


class AggregateTests(unittest.TestCase):
    def test_aggregate_computes_rates(self) -> None:
        rows = [
            Observation("groq", "llama-3.3-70b", "coding", True, 1.0),
            Observation("groq", "llama-3.3-70b", "coding", True, 1.0),
            Observation("groq", "llama-3.3-70b", "coding", False, 1.0, retried=True),
            Observation("groq", "llama-3.3-70b", "coding", True, 1.0),
        ]
        stats = aggregate(rows)
        key = ("groq", "llama-3.3-70b")
        self.assertIn(key, stats)
        s = stats[key]
        self.assertEqual(s.n, 4)
        self.assertAlmostEqual(s.completion_rate, 0.75)
        self.assertAlmostEqual(s.retry_rate, 0.25)

    def test_below_floor_is_not_trusted(self) -> None:
        rows = [Observation("groq", "llama-3.3-70b", "coding", True, 1.0) for _ in range(MIN_OBSERVATIONS - 1)]
        stats = aggregate(rows)
        s = stats[("groq", "llama-3.3-70b")]
        self.assertFalse(s.trusted)

    def test_at_floor_is_trusted(self) -> None:
        rows = [Observation("groq", "llama-3.3-70b", "coding", True, 1.0) for _ in range(MIN_OBSERVATIONS)]
        stats = aggregate(rows)
        s = stats[("groq", "llama-3.3-70b")]
        self.assertTrue(s.trusted)


class CapabilityAggregateTests(unittest.TestCase):
    def test_token_fields_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observed.jsonl"
            record(
                Observation(
                    "astra",
                    "gpt",
                    "coding",
                    True,
                    0.0,
                    capability_id="blender.scene_reasoning",
                    input_tokens=4200,
                    output_tokens=800,
                    latency_ms=1800.0,
                ),
                path=path,
            )
            rows = load_observations(path=path)
            self.assertEqual(rows[0].capability_id, "blender.scene_reasoning")
            self.assertEqual(rows[0].input_tokens, 4200)

    def test_measurement_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observed.jsonl"
            record(
                Observation(
                    "astra",
                    "gpt",
                    "coding",
                    True,
                    0.0,
                    input_tokens=10,
                    output_tokens=2,
                    measurement_state="complete",
                    state_reason="provider_response_usage",
                ),
                path=path,
            )
            row = load_observations(path=path)[0]
            self.assertEqual(row.measurement_state, "complete")
            self.assertEqual(row.state_reason, "provider_response_usage")

    def test_lookup_prefers_exact_capability_when_trusted(self) -> None:
        from kerdoios.observed import lookup_stats

        rows = [
            Observation("astra", "gpt", "coding", True, 0.0, capability_id="blender.scene_reasoning", input_tokens=1000)
            for _ in range(MIN_OBSERVATIONS)
        ] + [
            Observation("astra", "gpt", "coding", False, 0.0, capability_id="coding.edit", input_tokens=5000)
            for _ in range(MIN_OBSERVATIONS)
        ]
        # global completion rate diluted; exact blender should be perfect
        stats = lookup_stats(provider="astra", model="gpt", capability_id="blender.scene_reasoning", observations=rows)
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertTrue(stats.trusted)
        self.assertAlmostEqual(stats.completion_rate, 1.0)
        self.assertEqual(stats.capability_id, "blender.scene_reasoning")

    def test_lookup_falls_back_to_family(self) -> None:
        from kerdoios.observed import lookup_stats

        rows = [
            Observation("astra", "gpt", "coding", True, 0.0, capability_id="blender.need_render")
            for _ in range(MIN_OBSERVATIONS)
        ]
        stats = lookup_stats(
            provider="astra",
            model="gpt",
            capability_id="blender.scene_reasoning",  # no exact rows
            observations=rows,
        )
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.capability_id, "blender")
        self.assertTrue(stats.trusted)


class TokensPerVerifiedTests(unittest.TestCase):
    def test_tokens_per_verified(self) -> None:
        rows = [
            Observation(
                "openrouter",
                "x",
                "coding",
                True,
                0.02,
                capability_id="recovery_action",
                input_tokens=1000,
                output_tokens=200,
                execution_completed=True,
                verified_success=True,
            ),
            Observation(
                "openrouter",
                "x",
                "coding",
                True,
                0.01,
                capability_id="recovery_action",
                input_tokens=500,
                output_tokens=100,
                execution_completed=True,
                verified_success=True,
            ),
            Observation(
                "openrouter",
                "x",
                "coding",
                False,
                0.0,
                capability_id="recovery_action",
                input_tokens=100,
                output_tokens=10,
                execution_completed=False,
                verified_success=False,
            ),
            # ambient close: completed but not verified — must not count
            Observation(
                "openrouter",
                "x",
                "coding",
                True,
                0.0,
                capability_id="recovery_action",
                input_tokens=999,
                output_tokens=999,
                execution_completed=True,
                verified_success=None,
            ),
        ]
        s = tokens_per_verified_task(rows, capability_id="recovery_action")
        self.assertEqual(s["n_verified"], 2)
        # 1000+200 + 500+100 = 1800 / 2 = 900
        self.assertAlmostEqual(s["tokens_per_verified_task"], 900.0)
        self.assertAlmostEqual(s["mean_cost_per_verified"], 0.015)


    def test_partial_verified_tokens_are_observed_not_authoritative(self) -> None:
        rows = [
            Observation(
                "openrouter",
                "x",
                "coding",
                True,
                0.02,
                input_tokens=1000,
                output_tokens=200,
                execution_completed=True,
                verified_success=True,
                measurement_state="partial",
                state_reason="missing_subagent_usage",
            ),
            Observation(
                "openrouter",
                "x",
                "coding",
                True,
                0.01,
                input_tokens=500,
                output_tokens=100,
                execution_completed=True,
                verified_success=True,
                measurement_state="complete",
            ),
        ]
        stats = tokens_per_verified_task(rows)
        self.assertEqual(stats["tokens_per_verified_task"], 900.0)
        self.assertFalse(stats["authoritative"])
        self.assertIsNone(stats["authoritative_tokens_per_verified_task"])
        self.assertEqual(stats["n_measurement_explicit_incomplete"], 1)

    def test_complete_verified_tokens_are_authoritative(self) -> None:
        rows = [
            Observation(
                "openrouter",
                "x",
                "coding",
                True,
                0.02,
                input_tokens=1000,
                output_tokens=200,
                execution_completed=True,
                verified_success=True,
                measurement_state="complete",
            ),
        ]
        stats = tokens_per_verified_task(rows)
        self.assertTrue(stats["authoritative"])
        self.assertEqual(stats["authoritative_tokens_per_verified_task"], 1200.0)


class RecordCliMeasurementStateTests(unittest.TestCase):
    def test_record_cli_forwards_measurement_state(self) -> None:
        from unittest.mock import patch

        from kerdoios.__main__ import main

        with patch("kerdoios.__main__.record") as write:
            rc = main(
                [
                    "record",
                    "--provider",
                    "groq",
                    "--model",
                    "llama",
                    "--completed",
                    "--input-tokens",
                    "10",
                    "--output-tokens",
                    "2",
                    "--measurement-state",
                    "partial",
                    "--state-reason",
                    "last_message_only",
                ]
            )
        self.assertEqual(rc, 0)
        obs = write.call_args.args[0]
        self.assertEqual(obs.measurement_state, "partial")
        self.assertEqual(obs.state_reason, "last_message_only")


class BlendTests(unittest.TestCase):
    def _bad_offer(self):
        return next(o for o in fixture_offers() if o.id == "frontier/paid")

    def test_untrusted_stats_leave_offer_unchanged(self) -> None:
        offer = self._bad_offer()
        rows = [Observation(offer.provider, offer.model, "coding", True, 1.0) for _ in range(MIN_OBSERVATIONS - 1)]
        stats = aggregate(rows)[(offer.provider, offer.model)]
        blended = blend_offer(offer, stats)
        self.assertEqual(blended, offer)

    def test_poor_completion_rate_drags_capability_and_raises_effective_cost(self) -> None:
        offer = self._bad_offer()
        rows = [Observation(offer.provider, offer.model, "coding", i % 5 == 0, offer.economics.input_token_price) for i in range(20)]
        stats = aggregate(rows)[(offer.provider, offer.model)]
        self.assertTrue(stats.trusted)
        blended = blend_offer(offer, stats)
        self.assertLess(blended.capabilities.coding, offer.capabilities.coding)
        self.assertGreater(blended.economics.input_token_price, offer.economics.input_token_price)
        self.assertEqual(blended.capabilities.provenance, "observed_execution")

    def test_blend_weight_never_exceeds_cap(self) -> None:
        offer = self._bad_offer()
        rows = [Observation(offer.provider, offer.model, "coding", True, 0.0) for _ in range(10_000)]
        stats = aggregate(rows)[(offer.provider, offer.model)]
        blended = blend_offer(offer, stats)
        self.assertLessEqual(_blend_weight(10_000), MAX_BLEND_WEIGHT)
        self.assertLessEqual(_blend_weight(stats.n), MAX_BLEND_WEIGHT)
        self.assertEqual(blended.capabilities.provenance, "observed_execution")

    def test_apply_observed_is_noop_with_empty_log(self) -> None:
        offers = fixture_offers()
        blended = apply_observed(offers, path=Path("/tmp/definitely-does-not-exist-kerdoios-2.jsonl"))
        self.assertEqual(blended, offers)


class PlanIntegrationTests(unittest.TestCase):
    def test_plan_default_ignores_observed_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observed.jsonl"
            offer = next(o for o in fixture_offers() if o.id == "frontier/paid")
            for i in range(20):
                record(Observation(offer.provider, offer.model, "coding", i % 5 == 0, 0.01), path=path)
            req = WorkRequirement(coding=0.6, reasoning=0.6, tool_use=True, tools=("github",), context=128_000, parallelism=10, mode=Mode.CHEAP)
            # use_observed defaults False: plan() must not read the module-level
            # DEFAULT_LOG_PATH, so this per-test log has zero effect either way.
            built_default = plan(fixture_offers(), req)
            built_again = plan(fixture_offers(), req)
            self.assertEqual(built_default.to_dict(), built_again.to_dict())

    def test_plan_with_observed_changes_ranking_when_offer_is_unreliable(self) -> None:
        import kerdoios.blend as blend_mod

        offers = fixture_offers()
        offer = next(o for o in offers if o.id == "frontier/paid")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observed.jsonl"
            for i in range(30):
                record(Observation(offer.provider, offer.model, "coding", i % 10 == 0, 0.01), path=path)

            original = blend_mod.apply_observed

            def patched(offers_in, *, path=None, capability_id=None):
                return original(
                    offers_in,
                    path=path or Path(tmp) / "observed.jsonl",
                    capability_id=capability_id,
                )

            blend_mod.apply_observed = patched
            try:
                import kerdoios.optimize as optimize_mod

                optimize_mod.apply_observed = patched
                req = WorkRequirement(
                    coding=0.6,
                    reasoning=0.6,
                    tool_use=True,
                    tools=("github",),
                    context=128_000,
                    parallelism=50,
                    maximum_cost=1.0,
                    mode=Mode.BALANCED,
                )
                built_plain = plan(offers, req, use_observed=False)
                built_observed = plan(offers, req, use_observed=True)
                plain_paid_workers = sum(p.workers for p in built_plain.placements if p.provider == "paid-api")
                observed_paid_workers = sum(
                    p.workers for p in built_observed.placements if p.provider == "paid-api"
                )
                self.assertLessEqual(observed_paid_workers, plain_paid_workers)
            finally:
                blend_mod.apply_observed = original
                optimize_mod.apply_observed = original



class FailureTaxonomyTests(unittest.TestCase):
    """`reason` plus time-decayed trust: stale failures forgive themselves."""

    def test_reason_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observed.jsonl"
            record(Observation("openrouter", "m1", "research", False, 0.0, reason="rate_limited"), path=path)
            rows = load_observations(path=path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].reason, "rate_limited")

    def test_legacy_lines_default_to_no_reason(self) -> None:
        # A pre-taxonomy line has no `reason` and no `recorded_at`; it must load
        # rather than be dropped, and its age must not be assumed to be zero.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observed.jsonl"
            path.write_text(json.dumps({
                "provider": "openrouter", "model": "m1", "task_type": "research",
                "completed": True, "actual_cost": 0.0, "retried": False,
            }) + "\n", encoding="utf-8")
            rows = load_observations(path=path)
            self.assertIsNone(rows[0].reason)
            self.assertGreater(rows[0].recorded_at, 0.0)

    def test_old_failures_decay_below_trust_floor(self) -> None:
        # 10 half-lives old weighs 2**-10 ~= 0.001 each, so five of them cannot
        # clear the floor: a model that 429'd long ago is forgiven, not blacklisted.
        from kerdoios.observed import HALF_LIFE_S

        now = 1_800_000_000.0
        old = now - 10 * HALF_LIFE_S
        rows = [
            Observation("openrouter", "m1", "research", False, 0.0, reason="rate_limited", recorded_at=old)
            for _ in range(MIN_OBSERVATIONS)
        ]
        stats = aggregate(rows, now=now)[("openrouter", "m1")]
        self.assertFalse(stats.trusted)
        self.assertLess(stats.effective_n, MIN_OBSERVATIONS)

    def test_recent_failures_still_penalize(self) -> None:
        # The floor must not have been bought by ignoring the raw count: fresh
        # failures are trusted (so they act) and drag the capability down.
        offer = next(o for o in fixture_offers() if o.id == "frontier/paid")
        rows = [
            Observation(offer.provider, offer.model, "coding", False, 0.01, reason="rate_limited")
            for _ in range(MIN_OBSERVATIONS)
        ]
        stats = aggregate(rows)[(offer.provider, offer.model)]
        self.assertTrue(stats.trusted)
        self.assertAlmostEqual(stats.decayed_completion_rate, 0.0)
        blended = blend_offer(offer, stats)
        self.assertLess(blended.capabilities.coding, offer.capabilities.coding)


if __name__ == "__main__":
    unittest.main()
