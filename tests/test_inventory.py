from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from kerdoios.cache import (
    CACHE_TTL,
    cache_path,
    load_inventory_cache,
    save_inventory_cache,
)
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
    def test_live_openrouter_replaces_fixture_clone(self) -> None:
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
        or_free = [o for o in merged if o.provider == "openrouter" and o.model == "free-pool"]
        self.assertEqual(len(or_free), 1)
        self.assertEqual(or_free[0].source, "openrouter:/api/v1/models")

    def test_free_only_uses_snapshot_when_live_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "KERDOIOS_CACHE": tmp}
            with (
                patch.dict("os.environ", env, clear=True),
                patch("kerdoios.inventory.openrouter.discover", return_value=[]),
                patch("kerdoios.inventory.discover_groq", return_value=[]),
                patch("kerdoios.inventory.discover_cerebras", return_value=[]),
                patch("kerdoios.inventory.local.discover", return_value=[]),
            ):
                rows = discover_all(include_fixture=False, live=True, free_only=True)
        or_rows = [o for o in rows if o.provider == "openrouter"]
        self.assertGreaterEqual(len(or_rows), 10)
        self.assertTrue(all(o.source == "openrouter:snapshot" for o in or_rows))
        self.assertFalse(any(o.source == "fixture" for o in rows))


class FreeInventoryCacheTests(unittest.TestCase):
    NOW = datetime(2026, 9, 11, 16, 0, 0, tzinfo=timezone.utc)

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict("os.environ", {"KERDOIOS_CACHE": self.tmp.name})
        self.env.start()
        self.addCleanup(self.env.stop)

    def _live_payload(self, model: str = "meta/llama:free") -> list:
        return _offers_from_payload(
            {
                "data": [
                    {
                        "id": model,
                        "context_length": 8192,
                        "pricing": {"prompt": "0", "completion": "0"},
                        "architecture": {"modality": "text"},
                        "supported_parameters": ["tools", "tool_choice"],
                    }
                ]
            }
        )

    def _live_stack(self, rows) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch("kerdoios.inventory.openrouter.discover", return_value=rows))
        stack.enter_context(patch("kerdoios.inventory.discover_groq", return_value=[]))
        stack.enter_context(patch("kerdoios.inventory.discover_cerebras", return_value=[]))
        stack.enter_context(patch("kerdoios.inventory.local.discover", return_value=[]))
        return stack

    def test_cache_path_uses_kerdoios_cache_override(self) -> None:
        self.assertEqual(cache_path(), Path(self.tmp.name) / "inventory.json")

    def test_save_then_load_roundtrip(self) -> None:
        offers = self._live_payload()
        save_inventory_cache(offers, now=self.NOW)
        loaded = load_inventory_cache(now=self.NOW)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].model, offers[0].model)
        self.assertEqual(loaded[0].provider, offers[0].provider)
        self.assertEqual(loaded[0].economics.input_token_price, 0.0)

    def test_expired_cache_is_not_fresh(self) -> None:
        offers = self._live_payload()
        save_inventory_cache(offers, now=self.NOW)
        expired = load_inventory_cache(now=self.NOW + CACHE_TTL + timedelta(seconds=1))
        self.assertIsNone(expired)

    def test_fresh_cache_skips_network(self) -> None:
        save_inventory_cache(self._live_payload("cached/model:free"), now=self.NOW)
        live = self._live_payload("live/model:free")
        with patch("kerdoios.cache.utcnow", return_value=self.NOW), self._live_stack(live):
            rows = discover_all(include_fixture=False, live=True, free_only=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].model, "cached/model:free")

    def test_refresh_forces_live_openrouter(self) -> None:
        save_inventory_cache(self._live_payload("cached/model:free"), now=self.NOW)
        live = self._live_payload("live/model:free")
        with patch("kerdoios.cache.utcnow", return_value=self.NOW), self._live_stack(live):
            rows = discover_all(include_fixture=False, live=True, free_only=True, refresh=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].model, "live/model:free")
        reloaded = load_inventory_cache(now=self.NOW)
        assert reloaded is not None
        self.assertEqual(reloaded[0].model, "live/model:free")

    def test_snapshot_when_cache_and_network_empty(self) -> None:
        with self._live_stack([]):
            rows = discover_all(include_fixture=False, live=True, free_only=True)
        self.assertTrue(all(o.source == "openrouter:snapshot" for o in rows if o.provider == "openrouter"))
        self.assertGreaterEqual(len(rows), 10)
        self.assertFalse(cache_path().exists())

    def test_stale_cache_used_when_network_empty(self) -> None:
        save_inventory_cache(self._live_payload("stale/model:free"), now=self.NOW)
        later = self.NOW + CACHE_TTL + timedelta(hours=1)
        with patch("kerdoios.cache.utcnow", return_value=later), self._live_stack([]):
            rows = discover_all(include_fixture=False, live=True, free_only=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].model, "stale/model:free")

    def test_cache_file_contains_no_secrets(self) -> None:
        offers = self._live_payload()
        save_inventory_cache(offers, now=self.NOW)
        text = cache_path().read_text()
        payload = json.loads(text)
        blob = json.dumps(payload).lower()
        for needle in ("api_key", "authorization", "bearer ", "openrouter_api_key", "sk-"):
            self.assertNotIn(needle, blob)
        self.assertEqual(set(payload.keys()), {"saved_at", "offers"})


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
