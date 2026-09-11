from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from kerdoios.inventory import discover_all
from kerdoios.providers.free import is_free
from kerdoios.providers.openrouter import _offers_from_payload, discover as or_discover
from kerdoios.providers.openai_compat import discover_groq
from kerdoios.types import Economics, Mode, WorkRequirement
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
                    "supported_parameters": ["tools", "tool_choice"],
                },
                {
                    "id": "openai/gpt-x",
                    "context_length": 200000,
                    "pricing": {"prompt": "0.000003", "completion": "0.000012"},
                    "architecture": {"modality": "text+image"},
                    "supported_parameters": ["tools", "tool_choice"],
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
        self.assertEqual(free.tools, ("*",))
        self.assertEqual(paid.tools, ("*",))
        self.assertEqual(free.provider, "meta")
        self.assertEqual(paid.provider, "openai")
        self.assertEqual(free.source, "openrouter:/api/v1/models")
        self.assertEqual(paid.source, "openrouter:/api/v1/models")
        self.assertEqual(free.model, "meta/llama:free")
        self.assertEqual(paid.model, "openai/gpt-x")
        self.assertEqual(free.id, "meta/meta/llama:free")
        self.assertEqual(paid.id, "openai/openai/gpt-x")

    def test_origin_providers_from_nvidia_and_google_free_ids(self) -> None:
        payload = {
            "data": [
                {
                    "id": "nvidia/nemotron-3-nano:free",
                    "context_length": 128000,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "text"},
                    "supported_parameters": ["tools", "tool_choice"],
                },
                {
                    "id": "google/gemma-4-31b-it:free",
                    "context_length": 131072,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "text"},
                    "supported_parameters": ["tools", "tool_choice"],
                },
            ]
        }
        offers = _offers_from_payload(payload)
        self.assertEqual(len(offers), 2)
        providers = {o.provider for o in offers}
        self.assertEqual(providers, {"nvidia", "google"})
        self.assertNotIn("openrouter", providers)
        nvidia = next(o for o in offers if o.provider == "nvidia")
        google = next(o for o in offers if o.provider == "google")
        self.assertEqual(nvidia.model, "nvidia/nemotron-3-nano:free")
        self.assertEqual(google.model, "google/gemma-4-31b-it:free")
        self.assertEqual(nvidia.source, "openrouter:/api/v1/models")
        self.assertEqual(google.source, "openrouter:/api/v1/models")
        self.assertEqual(nvidia.id, "nvidia/nvidia/nemotron-3-nano:free")
        self.assertEqual(google.id, "google/google/gemma-4-31b-it:free")

        with (
            patch("kerdoios.inventory.openrouter.discover", return_value=offers),
            patch("kerdoios.inventory.discover_groq", return_value=[]),
            patch("kerdoios.inventory.discover_cerebras", return_value=[]),
            patch("kerdoios.inventory.local.discover", return_value=[]),
        ):
            inventory = discover_all(include_fixture=True, live=True)
            discovered = discover_all(include_fixture=False, live=True, free_only=True)

        inv_origins = {o.provider for o in inventory if o.source == "openrouter:/api/v1/models"}
        disc_origins = {o.provider for o in discovered if o.source == "openrouter:/api/v1/models"}
        self.assertEqual(inv_origins, {"nvidia", "google"})
        self.assertEqual(disc_origins, {"nvidia", "google"})
        self.assertTrue(any(o.id == "openrouter/free" and o.provider == "openrouter" for o in inventory))

    def test_discover_returns_empty_on_network_failure(self) -> None:
        with patch("kerdoios.providers.openrouter.get_json", return_value=None):
            self.assertEqual(or_discover(api_key=None), [])

    def test_groq_skips_without_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(discover_groq(api_key=None), [])

    def test_openrouter_tools_follow_supported_parameters(self) -> None:
        payload = {
            "data": [
                {
                    "id": "meta/llama:free",
                    "context_length": 8192,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "text"},
                    "supported_parameters": ["temperature", "max_tokens"],
                }
            ]
        }
        offer = _offers_from_payload(payload)[0]
        self.assertEqual(offer.tools, ())
        self.assertLess(offer.capabilities.tool_use, 0.5)


class FreeFilterTests(unittest.TestCase):
    def test_is_free_accepts_colon_free_and_quota(self) -> None:
        paid = next(o for o in fixture_offers() if o.id == "frontier/paid")
        self.assertFalse(is_free(paid))
        self.assertTrue(is_free(replace(paid, model="meta/llama:free")))
        self.assertTrue(
            is_free(
                replace(
                    paid,
                    economics=Economics(
                        input_token_price=3e-6,
                        output_token_price=1.2e-5,
                        remaining_free_quota=10_000,
                    ),
                )
            )
        )
        self.assertTrue(
            is_free(
                replace(
                    paid,
                    economics=Economics(
                        input_token_price=3e-6,
                        output_token_price=1.2e-5,
                        remaining_credits=1.0,
                    ),
                )
            )
        )
        self.assertTrue(is_free(replace(paid, local=True)))
        self.assertTrue(is_free(replace(paid, economics=Economics())))


class LiveMergeTests(unittest.TestCase):
    def test_live_origin_rows_join_unchanged_fixture_catalog(self) -> None:
        live = _offers_from_payload(
            {
                "data": [
                    {
                        "id": "free-pool",
                        "context_length": 128000,
                        "pricing": {"prompt": "0", "completion": "0"},
                        "architecture": {"modality": "text"},
                        "supported_parameters": ["tools", "tool_choice"],
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
        live_rows = [o for o in merged if o.model == "free-pool" and o.source == "openrouter:/api/v1/models"]
        self.assertEqual(len(live_rows), 1)
        self.assertEqual(live_rows[0].provider, "free-pool")
        self.assertEqual(live_rows[0].source, "openrouter:/api/v1/models")
        fixture_pool = [o for o in merged if o.id == "openrouter/free"]
        self.assertEqual(len(fixture_pool), 1)
        self.assertEqual(fixture_pool[0].provider, "openrouter")
        self.assertEqual(fixture_pool[0].source, "fixture")

    def test_free_only_uses_snapshot_when_live_empty(self) -> None:
        with (
            patch("kerdoios.inventory.openrouter.discover", return_value=[]),
            patch("kerdoios.inventory.discover_groq", return_value=[]),
            patch("kerdoios.inventory.discover_cerebras", return_value=[]),
            patch("kerdoios.inventory.local.discover", return_value=[]),
        ):
            rows = discover_all(include_fixture=False, live=True, free_only=True)
        snap_rows = [o for o in rows if o.source == "openrouter:snapshot"]
        self.assertGreaterEqual(len(snap_rows), 10)
        self.assertTrue(all(o.source == "openrouter:snapshot" for o in snap_rows))
        origins = {o.provider for o in snap_rows}
        self.assertIn("google", origins)
        self.assertIn("nvidia", origins)
        self.assertFalse(any(o.source == "fixture" for o in rows))


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
