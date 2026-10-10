from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from kerdoios.free_fanout import blocked, build_fanout
from kerdoios.types import CapabilityProfile, Capacity, Economics, ResourceOffer, Telemetry


def offer(
    provider: str, model: str, *, quota: float = 1.0, credits: float = 0.0, id: str | None = None
) -> ResourceOffer:
    return ResourceOffer(
        id=id or f"{provider}/{model}",
        provider=provider,
        resource_type="llm",
        model=model,
        local=False,
        capabilities=CapabilityProfile(),
        capacity=Capacity(),
        economics=Economics(remaining_free_quota=quota, remaining_credits=credits),
        telemetry=Telemetry(),
    )


def chains(plan: dict) -> list[list[str]]:
    """Every provider a slot can reach: the primary, then its sidestep chain."""
    return [[slot["provider"]] + [step["provider"] for step in slot["sidestep"]] for slot in plan["slots"]]


CREDIT_ONLY = ("anthropic", "claude-opus-5")


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

    def test_credit_balance_or_no_quota_is_not_free(self) -> None:
        plan = build_fanout(
            [
                offer(*CREDIT_ONLY, quota=0, credits=25.0),
                offer("mistral", "mistral-small", quota=0),
                offer("groq", "openai/gpt-oss-20b"),
                offer("nous", "inclusionai/ling-3.0-flash-sante:free", quota=0, credits=3.0),
            ]
        )
        self.assertEqual(chains(plan), [["nous", "groq"], ["groq", "nous"]])
        self.assertNotIn("reason", plan)

    def test_cursor_and_denylist_names_are_blocked(self) -> None:
        for name in (
            "Cursor",
            " cursor",
            "\tcursor",
            "cursor-agent",
            "anysphere-cursor",
            "hermes/cursor",
            "cursor_cli",
            " xai ",
            "XAI",
            "OpenAI-Codex ",
            "paid-api",
        ):
            with self.subTest(blocked=name):
                self.assertTrue(blocked(name))
        for name in ("precursor-labs", "nous", "openrouter", "", None):
            with self.subTest(allowed=name):
                self.assertFalse(blocked(name))

    def test_blocked_offers_reach_no_slot_and_no_sidestep(self) -> None:
        leaks = [
            offer("anysphere-cursor", "composer-2:free"),
            offer("OpenAI-Codex ", "gpt-5.3-codex"),
            offer("openrouter-eu", "cursor/composer-2:free", id="route-1"),
            offer("litellm", "vendor/ Cursor-agent/composer-2", id="route-2"),
            offer("relay", "composer-2:free", id="relay/cursor/composer-2"),
        ]
        plan = build_fanout(leaks + [offer("nous", "a:free"), offer("groq", "b")])
        self.assertEqual(chains(plan), [["nous", "groq"], ["groq", "nous"]])
        empty = build_fanout(leaks)
        self.assertEqual(
            (empty["slots"], empty["workers"], empty.get("reason")),
            ([], 0, "no_free_offer_in_inventory"),
        )

    def test_cli_exits_nonzero_when_inventory_has_no_free_offer(self) -> None:
        from kerdoios import __main__ as cli

        for offers, want in (
            ([offer(*CREDIT_ONLY, quota=0, credits=25.0)], (1, 0, [], "no_free_offer_in_inventory")),
            ([offer("groq", "b")], (0, 1, ["groq"], None)),
        ):
            out = io.StringIO()
            with mock.patch.object(cli, "discover_all", return_value=offers), contextlib.redirect_stdout(out):
                rc = cli.main(["free-fanout", "--live"])
            plan = json.loads(out.getvalue())
            got = (rc, plan["workers"], [slot["provider"] for slot in plan["slots"]], plan.get("reason"))
            with self.subTest(offers=offers[0].provider):
                self.assertEqual(got, want)

    def test_module_does_not_call_a_model(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("kerdoios/free_fanout.py").read_text(encoding="utf-8")
        self.assertNotIn("chat/completions", text)
        self.assertNotIn('method="POST"', text)


if __name__ == "__main__":
    unittest.main()
