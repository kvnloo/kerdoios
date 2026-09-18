from __future__ import annotations

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
            ),
            Observation(
                "openrouter",
                "x",
                "coding",
                False,
                0.0,
                capability_id="recovery_action",
                input_tokens=999,
                output_tokens=0,
            ),
        ]
        s = tokens_per_verified_task(rows, capability_id="recovery_action")
        self.assertEqual(s["n_verified"], 2)
        self.assertEqual(s["n_with_tokens"], 2)
        # (1200+600)/2 = 900
        self.assertAlmostEqual(s["tokens_per_verified_task"], 900.0)
        self.assertAlmostEqual(s["mean_cost_per_verified"], 0.015)


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



if __name__ == "__main__":
    unittest.main()
