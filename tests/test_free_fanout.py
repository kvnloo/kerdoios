from __future__ import annotations

import unittest
from pathlib import Path

from kerdoios.free_fanout import blocked, build_fanout
from kerdoios.types import CapabilityProfile, Capacity, Economics, ResourceOffer, Telemetry


def offer(provider: str, model: str, *, quota: float = 1.0) -> ResourceOffer:
    return ResourceOffer(
        id=f"{provider}/{model}",
        provider=provider,
        resource_type="llm",
        model=model,
        local=False,
        capabilities=CapabilityProfile(),
        capacity=Capacity(),
        economics=Economics(remaining_free_quota=quota),
        telemetry=Telemetry(),
    )


class FreeFanoutTests(unittest.TestCase):
    def test_five_slots_skip_cursor_and_paid(self) -> None:
        offers = [
            offer("cursor", "gpt-5.3-codex", quota=0),
            offer("paid-api", "frontier-codex", quota=0),
            offer("nous", "inclusionai/ling-3.0-flash-sante:free"),
            offer("vercel", "openai/gpt-oss-20b"),
            offer("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
            offer("groq", "openai/gpt-oss-20b"),
            offer("nvidia", "openai/gpt-oss-20b"),
        ]
        plan = build_fanout(offers, workers=5)
        providers = [slot["provider"] for slot in plan["slots"]]
        self.assertEqual(providers, ["nous", "vercel", "openrouter", "groq", "nvidia"])
        self.assertNotIn("cursor", providers)
        self.assertEqual(plan["sidestep_http"], [402, 429])
        self.assertFalse(plan["cursor_allowed"])
        self.assertTrue(blocked("cursor"))

    def test_static_table_when_inventory_empty(self) -> None:
        plan = build_fanout([], workers=5)
        self.assertEqual(plan["workers"], 5)
        self.assertTrue(all(slot["source"] == "static_sidestep_table" for slot in plan["slots"]))

    def test_module_does_not_call_a_model(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("kerdoios/free_fanout.py").read_text(encoding="utf-8")
        self.assertNotIn("chat/completions", text)
        self.assertNotIn('method="POST"', text)


if __name__ == "__main__":
    unittest.main()
