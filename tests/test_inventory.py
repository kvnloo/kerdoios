from __future__ import annotations

import unittest
from unittest.mock import patch

from kerdoios.inventory import discover_all
from kerdoios.providers.openrouter import _offers_from_payload, discover as or_discover
from kerdoios.providers.openai_compat import discover_groq
from kerdoios.types import Mode, WorkRequirement
from kerdoios.optimize import plan
from kerdoios.providers.fixture import fixture_offers


class OpenRouterMapTests(unittest.TestCase):
    def test_maps_free_and_paid_rows(self) -> None:
        payload = {
            "data": [
                {
                    "id": "meta/llama:free",
                    "context_length": 131072,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "text"},
                },
                {
                    "id": "openai/gpt-x",
                    "context_length": 200000,
                    "pricing": {"prompt": "0.000003", "completion": "0.000012"},
                    "architecture": {"modality": "text+image"},
                },
            ]
        }
        offers = _offers_from_payload(payload)
        self.assertEqual(len(offers), 2)
        free = next(o for o in offers if o.model == "meta/llama:free")
        paid = next(o for o in offers if o.model == "openai/gpt-x")
        self.assertEqual(free.economics.remaining_free_quota, 50_000.0)
        self.assertEqual(free.economics.input_token_price, 0.0)
        self.assertGreater(paid.economics.input_token_price, 0.0)
        self.assertGreaterEqual(paid.capabilities.vision, 0.5)
        self.assertEqual(paid.capacity.context_window, 200000)

    def test_discover_returns_empty_on_network_failure(self) -> None:
        with patch("kerdoios.providers.openrouter.get_json", return_value=None):
            self.assertEqual(or_discover(api_key=None), [])

    def test_groq_skips_without_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(discover_groq(api_key=None), [])


class LiveMergeTests(unittest.TestCase):
    def test_live_openrouter_replaces_fixture_clone(self) -> None:
        live = _offers_from_payload(
            {
                "data": [
                    {
                        "id": "free-pool",
                        "context_length": 128000,
                        "pricing": {"prompt": "0", "completion": "0"},
                        "architecture": {"modality": "text"},
                    }
                ]
            }
        )
        with (
            patch("kerdoios.inventory.openrouter.discover", return_value=live),
            patch("kerdoios.inventory.discover_groq", return_value=[]),
            patch("kerdoios.inventory.discover_cerebras", return_value=[]),
            patch("kerdoios.inventory.local.discover", return_value=[]),
        ):
            merged = discover_all(include_fixture=True, live=True)
        or_free = [o for o in merged if o.provider == "openrouter" and o.model == "free-pool"]
        self.assertEqual(len(or_free), 1)
        self.assertEqual(or_free[0].source, "openrouter:/api/v1/models")


class PrivateModeTests(unittest.TestCase):
    def test_private_mode_still_places_local(self) -> None:
        req = WorkRequirement(
            coding=0.7,
            tool_use=True,
            tools=("github",),
            context=128_000,
            parallelism=4,
            mode=Mode.PRIVATE,
        )
        built = plan(fixture_offers(), req)
        self.assertTrue(any(p.provider == "local" for p in built.placements))


if __name__ == "__main__":
    unittest.main()
