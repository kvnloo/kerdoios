from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kerdoios.bind import (
    apply_hermes_config,
    apply_omp_profile,
    hermes_fallback_entries,
    omp_model_roles,
)
from kerdoios.optimize import plan
from kerdoios.providers.fixture import fixture_offers
from kerdoios.types import (
    CapabilityProfile,
    Capacity,
    Economics,
    ExecutionPlan,
    Mode,
    Placement,
    ResourceOffer,
    Telemetry,
    WorkRequirement,
)


def _nvidia_free() -> ResourceOffer:
    return ResourceOffer(
        id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        provider="nvidia",
        resource_type="llm",
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        local=False,
        capabilities=CapabilityProfile(reasoning=0.74, coding=0.74, tool_use=0.85),
        capacity=Capacity(concurrency=4, context_window=128_000),
        economics=Economics(remaining_free_quota=1.0),
        telemetry=Telemetry(),
        tools=("*",),
        source="openrouter:/api/v1/models",
    )


def _ling_free() -> ResourceOffer:
    return ResourceOffer(
        id="inclusionai/ling-3.0-flash-vl:free",
        provider="inclusionai",
        resource_type="llm",
        model="inclusionai/ling-3.0-flash-vl:free",
        local=False,
        capabilities=CapabilityProfile(reasoning=0.74, coding=0.74, tool_use=0.85, vision=0.7),
        capacity=Capacity(concurrency=4, context_window=128_000),
        economics=Economics(remaining_free_quota=1.0),
        telemetry=Telemetry(),
        tools=("*",),
        source="openrouter:/api/v1/models",
    )


class BindPlanTests(unittest.TestCase):
    def test_hermes_entries_strip_free_suffix_on_native_nvidia(self) -> None:
        plan_out = ExecutionPlan(
            estimated_cost=0.0,
            estimated_duration_seconds=0.9,
            confidence=0.7,
            placements=[
                Placement(
                    offer_id=_nvidia_free().id,
                    provider="nvidia",
                    model=_nvidia_free().model,
                    workers=1,
                    estimated_cost=0.0,
                )
            ],
            fallbacks=[_ling_free().id],
            rejections=[],
            mode="cheap",
        )
        entries = hermes_fallback_entries(plan_out, [_nvidia_free(), _ling_free()])
        self.assertEqual(
            entries[0],
            {
                "provider": "nvidia",
                "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            },
        )
        self.assertEqual(
            entries[1],
            {
                "provider": "openrouter",
                "model": "inclusionai/ling-3.0-flash-vl:free",
            },
        )

    def test_omp_roles_use_provider_slash_model(self) -> None:
        built = plan(
            fixture_offers(),
            WorkRequirement(coding=0.6, reasoning=0.6, tool_use=True, tools=("github",), context=128_000, parallelism=1, mode=Mode.CHEAP),
        )
        roles = omp_model_roles(built, fixture_offers())
        self.assertIn("default", roles)
        self.assertIn("smol", roles)
        self.assertIn("/", roles["default"])
        self.assertNotEqual(roles["default"], roles["smol"])

    def test_apply_hermes_replaces_fallback_providers_keeps_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "model:\n  default: grok-4.6\n  provider: xai-oauth\nfallback_providers:\n  - provider: nvidia\n    model: nvidia/llama-3.3-70b-instruct\n",
                encoding="utf-8",
            )
            built = plan(
                fixture_offers(),
                WorkRequirement(coding=0.6, reasoning=0.6, tool_use=True, tools=("github",), context=128_000, parallelism=1, mode=Mode.CHEAP),
            )
            apply_hermes_config(path, built, fixture_offers())
            text = path.read_text(encoding="utf-8")
            self.assertIn("default: grok-4.6", text)
            self.assertIn("provider: xai-oauth", text)
            self.assertNotIn("nvidia/llama-3.3-70b-instruct", text)
            self.assertIn("fallback_providers:", text)
            self.assertIn("provider: groq", text)

    def test_apply_omp_writes_model_roles_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kerdoios.yml"
            built = plan(
                fixture_offers(),
                WorkRequirement(coding=0.6, reasoning=0.6, tool_use=True, tools=("github",), context=128_000, parallelism=1, mode=Mode.CHEAP),
            )
            apply_omp_profile(path, built, fixture_offers())
            text = path.read_text(encoding="utf-8")
            self.assertIn("modelRoles:", text)
            self.assertIn("default:", text)
            self.assertIn("smol:", text)


if __name__ == "__main__":
    unittest.main()
