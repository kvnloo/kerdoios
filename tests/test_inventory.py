from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios.cache import (
    CACHE_TTL,
    cache_path,
    load_inventory_cache,
    save_inventory_cache,
)
from kerdoios.inventory import discover_all
from kerdoios.providers.free import filter_free, is_free
from kerdoios.providers.http import JsonResponse
from kerdoios.providers.openrouter import _offers_from_payload, discover as or_discover
from kerdoios.providers.openai_compat import discover_cerebras, discover_groq
from kerdoios.types import Economics, Mode, WorkRequirement
from kerdoios.optimize import plan
from kerdoios.providers.fixture import fixture_offers


# Catalog-shaped mocks only. CI must not need live Groq/Cerebras keys.
_GROQ_MIXED = {
    "data": [
        {"id": "llama-3.3-70b-versatile", "context_window": 131072},
        {"id": "vendor/custom:free", "context_window": 8192},
        {
            "id": "preview-zero",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_window": 8192,
        },
        {"id": "whisper-large-v3", "context_window": 448},
        {
            "id": "enterprise-only",
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            "context_window": 128000,
        },
    ]
}
_CEREBRAS_MIXED = {
    "data": [
        {"id": "gpt-oss-120b", "owned_by": "Cerebras", "context_window": 65536},
        {
            "id": "dedicated-secret",
            "pricing": {"prompt": "0.000002", "completion": "0.000004"},
            "context_window": 131072,
        },
        {"id": "trial-flagged", "tier": "free_trial", "context_window": 65536},
    ]
}
_OR_FREE = {
    "data": [
        {
            "id": "meta/llama:free",
            "context_length": 131072,
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"modality": "text"},
            "supported_parameters": ["tools", "tool_choice"],
        }
    ]
}


def _catalog_json(url: str, api_key: str | None = None, **kwargs):
    if "openrouter" in url:
        return _OR_FREE
    if "groq.com" in url:
        return _GROQ_MIXED
    if "cerebras" in url:
        return _CEREBRAS_MIXED
    return None


def _catalog_response(url: str, api_key: str | None = None, **kwargs):
    body = _catalog_json(url, api_key=api_key, **kwargs)
    if body is None:
        return None
    return JsonResponse(body=body, headers={})


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
        # Catalog mapping has no rate-limit headers; do not invent a sticker quota.
        self.assertEqual(free.economics.remaining_free_quota, 0.0)
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
        self.assertEqual(free.id, "meta/llama:free")
        self.assertEqual(paid.id, "openai/gpt-x")

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
        self.assertEqual(nvidia.id, "nvidia/nemotron-3-nano:free")
        self.assertEqual(google.id, "google/gemma-4-31b-it:free")

        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "KERDOIOS_CACHE": tmp}
            with (
                patch.dict("os.environ", env, clear=True),
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
        with patch("kerdoios.providers.openrouter.get_json_response", return_value=None):
            self.assertEqual(or_discover(api_key=None), [])

    def test_groq_skips_without_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(discover_groq(api_key=None), [])

    def test_cerebras_skips_without_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(discover_cerebras(api_key=None), [])

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
        # Economics() default 0/0 is unset, not a trusted free chat tier.
        self.assertFalse(is_free(replace(paid, economics=Economics())))

    def test_litellm_zero_input_cost_is_not_free_chat(self) -> None:
        """LiteLLM's input_cost_per_token==0 rows are missing/rerank/embedding, not free chat."""
        base = next(o for o in fixture_offers() if o.id == "frontier/paid")
        dump = [
            {
                "model_name": "cohere/rerank-english-v3.0",
                "mode": "rerank",
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            },
            {
                "model_name": "openai/text-embedding-3-small",
                "mode": "embedding",
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            },
            {
                "model_name": "mystery-provider/unpriced-chat",
                "input_cost_per_token": 0,
            },
        ]
        offers = []
        for row in dump:
            offer = replace(
                base,
                id=f"litellm/{row['model_name']}",
                provider="litellm",
                model=row["model_name"],
                local=False,
                economics=Economics(
                    input_token_price=float(row["input_cost_per_token"]),
                    output_token_price=float(row["output_cost_per_token"])
                    if "output_cost_per_token" in row
                    else None,
                ),
            )
            offers.append(offer)
            self.assertFalse(is_free(offer), msg=row["model_name"])
        self.assertEqual(filter_free(offers), [])

    def test_openrouter_missing_prices_are_unknown_not_free(self) -> None:
        payload = {
            "data": [
                {
                    "id": "vendor/unpriced-embed",
                    "context_length": 8192,
                    "pricing": {},
                    "architecture": {"modality": "text"},
                    "supported_parameters": [],
                }
            ]
        }
        offer = _offers_from_payload(payload)[0]
        self.assertIsNone(offer.economics.input_token_price)
        self.assertIsNone(offer.economics.output_token_price)
        self.assertFalse(is_free(offer))
        self.assertEqual(filter_free([offer]), [])

    def test_colon_free_id_stays_free_without_catalog_prices(self) -> None:
        payload = {
            "data": [
                {
                    "id": "meta/llama:free",
                    "context_length": 8192,
                    "pricing": {},
                    "architecture": {"modality": "text"},
                    "supported_parameters": ["tools"],
                }
            ]
        }
        offer = _offers_from_payload(payload)[0]
        self.assertTrue(is_free(offer))


class OpenaiCompatMissingPriceTests(unittest.TestCase):
    """Groq/Cerebras catalog ingest must not treat missing prices as $0."""

    def _discover(self, body: dict, *, groq: bool = True):
        resp = JsonResponse(body=body, headers={})
        with patch("kerdoios.providers.openai_compat.get_json_response", return_value=resp):
            if groq:
                return discover_groq(api_key="g-test")
            return discover_cerebras(api_key="c-test")

    def test_missing_and_none_prices_stay_unknown_not_free(self) -> None:
        offers = self._discover(
            {
                "data": [
                    {"id": "whisper-large-v3", "context_window": 448},
                    {
                        "id": "empty-pricing",
                        "pricing": {},
                        "context_window": 8192,
                    },
                    {
                        "id": "null-prices",
                        "pricing": {"prompt": None, "completion": None},
                        "context_window": 8192,
                    },
                    {
                        "id": "output-only",
                        "pricing": {"completion": "0.000002"},
                        "context_window": 8192,
                    },
                    {
                        "id": "enterprise-only",
                        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                        "context_window": 128000,
                    },
                ]
            }
        )
        by_model = {o.model: o for o in offers}

        missing = by_model["whisper-large-v3"]
        self.assertIsNone(missing.economics.input_token_price)
        self.assertIsNone(missing.economics.output_token_price)
        self.assertFalse(is_free(missing))
        self.assertEqual(filter_free([missing]), [])

        empty = by_model["empty-pricing"]
        self.assertIsNone(empty.economics.input_token_price)
        self.assertIsNone(empty.economics.output_token_price)
        self.assertFalse(is_free(empty))

        nulls = by_model["null-prices"]
        self.assertIsNone(nulls.economics.input_token_price)
        self.assertIsNone(nulls.economics.output_token_price)
        self.assertFalse(is_free(nulls))

        one_side = by_model["output-only"]
        self.assertIsNone(one_side.economics.input_token_price)
        self.assertEqual(one_side.economics.output_token_price, 0.000002)
        self.assertFalse(is_free(one_side))

        paid = by_model["enterprise-only"]
        self.assertEqual(paid.economics.input_token_price, 1e-6)
        self.assertEqual(paid.economics.output_token_price, 2e-6)
        self.assertFalse(is_free(paid))

    def test_advertised_zero_prices_stay_zero(self) -> None:
        offers = self._discover(
            {
                "data": [
                    {
                        "id": "preview-zero",
                        "pricing": {"prompt": "0", "completion": "0"},
                        "context_window": 8192,
                    }
                ]
            }
        )
        offer = offers[0]
        self.assertEqual(offer.economics.input_token_price, 0.0)
        self.assertEqual(offer.economics.output_token_price, 0.0)

    def test_cerebras_missing_prices_stay_unknown_not_free(self) -> None:
        offers = self._discover(
            {"data": [{"id": "dedicated-secret", "context_window": 131072}]},
            groq=False,
        )
        self.assertEqual(len(offers), 1)
        self.assertIsNone(offers[0].economics.input_token_price)
        self.assertIsNone(offers[0].economics.output_token_price)
        self.assertFalse(is_free(offers[0]))
        self.assertEqual(filter_free(offers), [])


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
        snap_rows = [o for o in rows if o.source == "openrouter:snapshot"]
        self.assertGreaterEqual(len(snap_rows), 10)
        self.assertTrue(all(o.source == "openrouter:snapshot" for o in snap_rows))
        origins = {o.provider for o in snap_rows}
        self.assertIn("google", origins)
        self.assertIn("nvidia", origins)
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
        snap_rows = [o for o in rows if o.source == "openrouter:snapshot"]
        self.assertTrue(all(o.source == "openrouter:snapshot" for o in snap_rows))
        self.assertGreaterEqual(len(snap_rows), 10)
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


class KeyedFreeOverlayTests(unittest.TestCase):
    """Keyed Groq/Cerebras free-tier overlays into --free; paid catalog rows do not."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict("os.environ", {"KERDOIOS_CACHE": self.tmp.name}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)

    def _discover_free(self, **env: str):
        merged = {"KERDOIOS_CACHE": self.tmp.name, **env}
        with (
            patch.dict("os.environ", merged, clear=True),
            patch("kerdoios.providers.openrouter.get_json_response", side_effect=_catalog_response),
            patch("kerdoios.providers.openai_compat.get_json_response", side_effect=_catalog_response),
            patch("kerdoios.inventory.local.discover", return_value=[]),
        ):
            return discover_all(include_fixture=False, live=True, free_only=True)

    def test_without_keys_free_has_no_groq_or_cerebras(self) -> None:
        rows = self._discover_free()
        origins = {o.provider for o in rows}
        self.assertNotIn("groq", origins)
        self.assertNotIn("cerebras", origins)
        self.assertTrue(any(str(o.source).startswith("openrouter:") for o in rows))

    def test_keyed_free_tier_overlays_and_skips_paid_catalog_rows(self) -> None:
        rows = self._discover_free(GROQ_API_KEY="g-test", CEREBRAS_API_KEY="c-test")
        groq_models = {o.model for o in rows if o.provider == "groq"}
        cerebras_models = {o.model for o in rows if o.provider == "cerebras"}
        self.assertEqual(
            groq_models,
            {"llama-3.3-70b-versatile", "vendor/custom:free", "preview-zero"},
        )
        self.assertEqual(cerebras_models, {"gpt-oss-120b", "trial-flagged"})
        self.assertTrue(
            any(o.model == "meta/llama:free" and str(o.source).startswith("openrouter:") for o in rows)
        )
        keyed = [o for o in rows if o.provider in {"groq", "cerebras"}]
        self.assertTrue(keyed)
        self.assertTrue(all(o.economics.remaining_free_quota == 0 for o in keyed))
        self.assertFalse(any(o.economics.remaining_free_quota in {80_000.0, 50_000.0} for o in keyed))

    def test_snapshot_still_used_when_openrouter_live_empty_but_keys_set(self) -> None:
        def _no_openrouter(url: str, api_key: str | None = None, **kwargs):
            if "openrouter" in url:
                return None
            return _catalog_response(url, api_key=api_key, **kwargs)

        with (
            patch.dict(
                "os.environ",
                {"GROQ_API_KEY": "g-test", "KERDOIOS_CACHE": self.tmp.name},
                clear=True,
            ),
            patch("kerdoios.providers.openrouter.get_json_response", side_effect=_no_openrouter),
            patch("kerdoios.providers.openai_compat.get_json_response", side_effect=_no_openrouter),
            patch("kerdoios.inventory.local.discover", return_value=[]),
        ):
            rows = discover_all(include_fixture=False, live=True, free_only=True)
        self.assertTrue(any(o.source == "openrouter:snapshot" for o in rows))
        self.assertTrue(any(o.provider == "groq" and o.model == "llama-3.3-70b-versatile" for o in rows))
        self.assertFalse(any(o.provider == "groq" and o.model == "whisper-large-v3" for o in rows))
        self.assertFalse(any(o.source == "fixture" for o in rows))

    def test_live_keeps_non_free_keyed_rows(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"GROQ_API_KEY": "g-test", "KERDOIOS_CACHE": self.tmp.name},
                clear=True,
            ),
            patch("kerdoios.providers.openrouter.get_json_response", side_effect=_catalog_response),
            patch("kerdoios.providers.openai_compat.get_json_response", side_effect=_catalog_response),
            patch("kerdoios.inventory.local.discover", return_value=[]),
        ):
            rows = discover_all(include_fixture=False, live=True, free_only=False)
        groq_models = {o.model for o in rows if o.provider == "groq"}
        self.assertIn("whisper-large-v3", groq_models)
        self.assertIn("llama-3.3-70b-versatile", groq_models)


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
