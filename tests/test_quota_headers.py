from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios.providers.http import quota_from_headers
from kerdoios.providers.openai_compat import discover_cerebras, discover_groq
from kerdoios.providers.openrouter import discover as or_discover
from kerdoios.quota import (
    REQUIRED_DIMENSIONS,
    QuotaState,
    legacy_projection,
    parse_reset_seconds,
    quota_state_from_headers,
)


# Captured header maps only. No live keys, tokens, or Authorization values.
GROQ_HEADERS = {
    "x-ratelimit-limit-requests": "14400",
    "x-ratelimit-limit-tokens": "18000",
    "x-ratelimit-remaining-requests": "14370",
    "x-ratelimit-remaining-tokens": "17997",
    "x-ratelimit-reset-requests": "2m59.56s",
    "x-ratelimit-reset-tokens": "7.66s",
}

# The live Groq shape: minute windows only, no day dimension at all.
GROQ_MINUTE_ONLY = {
    "x-ratelimit-limit-requests": "1000",
    "x-ratelimit-limit-tokens": "8000",
    "x-ratelimit-remaining-requests": "999",
    "x-ratelimit-remaining-tokens": "7890",
    "x-ratelimit-reset-requests": "1m26.4s",
    "x-ratelimit-reset-tokens": "825ms",
}

CEREBRAS_HEADERS = {
    "x-ratelimit-limit-requests-day": "14400",
    "x-ratelimit-limit-tokens-minute": "60000",
    "x-ratelimit-remaining-requests-day": "13693",
    "x-ratelimit-remaining-tokens-minute": "60000",
    "x-ratelimit-reset-requests-day": "11726.2840924263",
    "x-ratelimit-reset-tokens-minute": "26.28409242630005",
}

# The live Cerebras shape: minute, hour and day for both requests and tokens.
CEREBRAS_FULL = {
    "x-ratelimit-limit-requests-minute": "1000",
    "x-ratelimit-limit-requests-hour": "60000",
    "x-ratelimit-limit-requests-day": "1440000",
    "x-ratelimit-limit-tokens-minute": "500000",
    "x-ratelimit-limit-tokens-hour": "30000000",
    "x-ratelimit-limit-tokens-day": "720000000",
    "x-ratelimit-remaining-requests-minute": "990",
    "x-ratelimit-remaining-requests-hour": "59000",
    "x-ratelimit-remaining-requests-day": "1439000",
    "x-ratelimit-remaining-tokens-minute": "499000",
    "x-ratelimit-remaining-tokens-hour": "29990000",
    "x-ratelimit-remaining-tokens-day": "719900000",
    "x-ratelimit-reset-requests-minute": "1m26.4s",
    "x-ratelimit-reset-requests-hour": "30m",
    "x-ratelimit-reset-requests-day": "5h",
    "x-ratelimit-reset-tokens-minute": "825ms",
    "x-ratelimit-reset-tokens-hour": "10m",
    "x-ratelimit-reset-tokens-day": "2h",
}

OPENROUTER_HEADERS = {
    "x-ratelimit-limit": "20",
    "x-ratelimit-remaining": "12",
    "x-ratelimit-reset": "1785658140000",
}



class _FakeHTTP:
    def __init__(self, body: dict, headers: dict[str, str]) -> None:
        self._body = json.dumps(body).encode()
        self.headers = headers

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHTTP:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _groq_body() -> dict:
    return {"data": [{"id": "llama-3.3-70b", "context_window": 128000}]}


def _cerebras_body() -> dict:
    return {"data": [{"id": "qwen-3-32b", "context_length": 128000}]}


def _openrouter_body() -> dict:
    return {
        "data": [
            {
                "id": "meta/llama:free",
                "context_length": 8192,
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {"modality": "text"},
                "supported_parameters": ["tools"],
            },
            {
                "id": "openai/gpt-x",
                "context_length": 200000,
                "pricing": {"prompt": "0.000003", "completion": "0.000012"},
                "architecture": {"modality": "text"},
                "supported_parameters": ["tools"],
            },
        ]
    }


class QuotaFromHeadersTests(unittest.TestCase):
    def test_groq_remaining_tokens_beats_request_count(self) -> None:
        remaining, reset = quota_from_headers(GROQ_HEADERS)
        self.assertEqual(remaining, 17997.0)
        self.assertAlmostEqual(reset or 0.0, 7.66, places=2)

    def test_cerebras_remaining_tokens_minute(self) -> None:
        remaining, reset = quota_from_headers(CEREBRAS_HEADERS)
        self.assertEqual(remaining, 60000.0)
        self.assertAlmostEqual(reset or 0.0, 26.28409242630005, places=5)

    def test_openrouter_remaining_and_unix_ms_reset(self) -> None:
        remaining, reset = quota_from_headers(OPENROUTER_HEADERS, now=1_785_658_080.0)
        self.assertEqual(remaining, 12.0)
        self.assertAlmostEqual(reset or 0.0, 60.0, places=3)

    def test_explicit_remaining_quota_header(self) -> None:
        remaining, reset = quota_from_headers(
            {"remaining-quota": "4321", "retry-after": "9"}
        )
        self.assertEqual(remaining, 4321.0)
        self.assertEqual(reset, 9.0)

    def test_missing_headers_are_unknown_not_invented(self) -> None:
        remaining, reset = quota_from_headers({})
        self.assertIsNone(remaining)
        self.assertIsNone(reset)

    def test_mixed_case_keys(self) -> None:
        remaining, reset = quota_from_headers(
            {
                "X-RateLimit-Remaining-Tokens": "100",
                "X-RateLimit-Reset-Tokens": "1.5s",
            }
        )
        self.assertEqual(remaining, 100.0)
        self.assertAlmostEqual(reset or 0.0, 1.5, places=2)

    def test_negative_remaining_is_not_advertised(self) -> None:
        remaining, reset = quota_from_headers({"x-ratelimit-remaining-tokens": "-1"})
        self.assertIsNone(remaining)
        self.assertIsNone(reset)


class AdapterHeaderTests(unittest.TestCase):
    def test_groq_offer_uses_header_not_hardcoded_80000(self) -> None:
        fake = _FakeHTTP(_groq_body(), GROQ_HEADERS)
        with patch("kerdoios.providers.http.urllib.request.urlopen", return_value=fake):
            offers = discover_groq(api_key="test")
        self.assertEqual(len(offers), 1)
        self.assertEqual(offers[0].economics.remaining_free_quota, 17997.0)
        self.assertAlmostEqual(offers[0].economics.seconds_until_quota_reset or 0.0, 7.66, places=2)

    def test_groq_without_headers_does_not_invent_quota(self) -> None:
        fake = _FakeHTTP(_groq_body(), {})
        with patch("kerdoios.providers.http.urllib.request.urlopen", return_value=fake):
            offers = discover_groq(api_key="test")
        self.assertEqual(offers[0].economics.remaining_free_quota, 0.0)
        self.assertIsNone(offers[0].economics.seconds_until_quota_reset)

    def test_cerebras_offer_uses_header_not_hardcoded_50000(self) -> None:
        fake = _FakeHTTP(_cerebras_body(), CEREBRAS_HEADERS)
        with patch("kerdoios.providers.http.urllib.request.urlopen", return_value=fake):
            offers = discover_cerebras(api_key="test")
        self.assertEqual(offers[0].economics.remaining_free_quota, 60000.0)
        self.assertAlmostEqual(
            offers[0].economics.seconds_until_quota_reset or 0.0, 26.28409242630005, places=5
        )

    def test_openrouter_keyed_applies_headers_to_free_only(self) -> None:
        fake = _FakeHTTP(_openrouter_body(), OPENROUTER_HEADERS)
        with (
            patch("kerdoios.providers.http.urllib.request.urlopen", return_value=fake),
            patch("kerdoios.providers.http.time.time", return_value=1_785_658_080.0),
        ):
            offers = or_discover(api_key="test")
        free = next(o for o in offers if o.model == "meta/llama:free")
        paid = next(o for o in offers if o.model == "openai/gpt-x")
        self.assertEqual(free.economics.remaining_free_quota, 12.0)
        self.assertAlmostEqual(free.economics.seconds_until_quota_reset or 0.0, 60.0, places=0)
        self.assertEqual(paid.economics.remaining_free_quota, 0.0)
        self.assertIsNone(paid.economics.seconds_until_quota_reset)

    def test_openrouter_without_key_ignores_headers(self) -> None:
        fake = _FakeHTTP(_openrouter_body(), OPENROUTER_HEADERS)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("kerdoios.providers.http.urllib.request.urlopen", return_value=fake),
        ):
            offers = or_discover(api_key=None)
        free = next(o for o in offers if o.model == "meta/llama:free")
        self.assertEqual(free.economics.remaining_free_quota, 0.0)

    def test_groq_skips_without_key(self) -> None:
        fake = _FakeHTTP(_groq_body(), GROQ_HEADERS)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("kerdoios.providers.http.urllib.request.urlopen", return_value=fake),
        ):
            self.assertEqual(discover_groq(api_key=None), [])


class StructuredQuotaTests(unittest.TestCase):
    def test_groq_minute_only_never_aliases_tokens_to_the_day_window(self) -> None:
        state = quota_state_from_headers(
            GROQ_MINUTE_ONLY, provider="groq", model="llama-3.3-70b", now=1000.0
        )
        rpm = state.dim("rpm")
        self.assertEqual((rpm.limit, rpm.remaining), (1000.0, 999.0))
        self.assertEqual(rpm.source, "provider_header")
        self.assertAlmostEqual(rpm.seconds_until_reset(1000.0) or 0.0, 86.4, places=5)
        tpm = state.dim("tpm")
        self.assertEqual((tpm.limit, tpm.remaining), (8000.0, 7890.0))
        self.assertEqual(tpm.source, "provider_header")
        self.assertAlmostEqual(tpm.seconds_until_reset(1000.0) or 0.0, 0.825, places=5)
        # Groq exposes no day window: unknown, and emphatically not 7890.
        for dimension in ("rpd", "tpd"):
            entry = state.dim(dimension)
            self.assertIsNone(entry.limit)
            self.assertIsNone(entry.remaining)
            self.assertEqual(entry.source, "unknown")
        self.assertNotEqual(state.dim("tpd").remaining, state.dim("tpm").remaining)
        self.assertTrue(state.has_free_capacity())

    def test_cerebras_minute_hour_day_parsed_with_distinct_resets(self) -> None:
        state = quota_state_from_headers(
            CEREBRAS_FULL, provider="cerebras", model="qwen-3-32b", now=5000.0
        )
        expected = {
            "rpm": (1000.0, 990.0, 86.4),
            "rph": (60000.0, 59000.0, 1800.0),
            "rpd": (1440000.0, 1439000.0, 18000.0),
            "tpm": (500000.0, 499000.0, 0.825),
            "tph": (30000000.0, 29990000.0, 600.0),
            "tpd": (720000000.0, 719900000.0, 7200.0),
        }
        for dimension, (limit, remaining, reset) in expected.items():
            entry = state.dim(dimension)
            self.assertEqual(entry.limit, limit, dimension)
            self.assertEqual(entry.remaining, remaining, dimension)
            self.assertEqual(entry.source, "provider_header", dimension)
            self.assertAlmostEqual(entry.seconds_until_reset(5000.0) or 0.0, reset, places=4, msg=dimension)
        self.assertNotEqual(state.dim("rpm").reset_at, state.dim("rpd").reset_at)
        self.assertNotEqual(state.dim("tpm").reset_at, state.dim("tpd").reset_at)

    def test_reset_forms_normalize_to_seconds_until(self) -> None:
        cases = {
            "1m26.4s": (0.0, 86.4),
            "825ms": (0.0, 0.825),
            "42": (0.0, 42.0),
            "1785658140000": (1_785_658_080.0, 60.0),
            "1785658140": (1_785_658_080.0, 60.0),
        }
        for raw, (now, seconds) in cases.items():
            with self.subTest(raw=raw):
                self.assertAlmostEqual(parse_reset_seconds(raw, now) or 0.0, seconds, places=4)

    def test_bare_float_seconds_reset_keeps_its_dimension(self) -> None:
        state = quota_state_from_headers(
            {
                "x-ratelimit-remaining-requests-day": "100",
                "x-ratelimit-reset-requests-day": "900.5",
                "x-ratelimit-remaining-tokens-minute": "5",
                "x-ratelimit-reset-tokens-minute": "2",
            },
            provider="cerebras",
            model="m",
            now=100.0,
        )
        self.assertAlmostEqual(state.dim("rpd").seconds_until_reset(100.0) or 0.0, 900.5, places=3)
        self.assertAlmostEqual(state.dim("tpm").seconds_until_reset(100.0) or 0.0, 2.0, places=3)

    def test_unix_ms_reset_is_absolute(self) -> None:
        state = quota_state_from_headers(
            OPENROUTER_HEADERS, provider="openrouter", model="meta/llama:free", now=1_785_658_080.0
        )
        self.assertEqual(state.dim("rpm").remaining, 12.0)
        self.assertAlmostEqual(state.dim("rpm").reset_at or 0.0, 1_785_658_140.0, places=3)
        self.assertAlmostEqual(state.dim("rpm").seconds_until_reset(1_785_658_080.0) or 0.0, 60.0, places=3)

    def test_missing_headers_leave_every_dimension_unknown(self) -> None:
        state = quota_state_from_headers({}, provider="groq", model="m")
        self.assertEqual(set(REQUIRED_DIMENSIONS) - set(state.dimensions), set())
        for dimension in REQUIRED_DIMENSIONS:
            entry = state.dim(dimension)
            self.assertIsNone(entry.limit, dimension)
            self.assertIsNone(entry.remaining, dimension)
            self.assertEqual(entry.source, "unknown", dimension)
        self.assertEqual(state.known_dimensions(), {})
        self.assertFalse(state.has_free_capacity())
        self.assertEqual(legacy_projection(state), (None, None))

    def test_negative_remaining_is_not_advertised(self) -> None:
        state = quota_state_from_headers(
            {"x-ratelimit-remaining-tokens": "-1", "x-ratelimit-reset-tokens": "3s"},
            provider="groq",
            model="m",
        )
        tpm = state.dim("tpm")
        self.assertIsNone(tpm.remaining)
        self.assertEqual(tpm.source, "unknown")
        self.assertFalse(state.has_free_capacity())
        self.assertEqual(legacy_projection(state, now=0.0), (None, None))

    def test_reconstruction_is_ceiling_minus_confirmed_usage(self) -> None:
        state = quota_state_from_headers(
            GROQ_MINUTE_ONLY,
            provider="groq",
            model="llama-3.3-70b",
            now=1000.0,
            confirmed_usage_today={"tpd": 4000.0, "rpd": 12.0},
            account_observed_ceiling={"tpd": 200_000.0, "rpd": 1000.0},
        )
        tpd = state.dim("tpd")
        self.assertEqual(tpd.limit, 200_000.0)
        self.assertEqual(tpd.remaining, 196_000.0)
        self.assertEqual(tpd.source, "locally_reconstructed")
        rpd = state.dim("rpd")
        self.assertEqual(rpd.remaining, 988.0)
        self.assertEqual(rpd.source, "locally_reconstructed")
        # The header-observed minute window is untouched by reconstruction.
        self.assertEqual(state.dim("tpm").remaining, 7890.0)
        self.assertEqual(state.dim("tpm").source, "provider_header")

    def test_account_observed_ceiling_beats_documentation_default(self) -> None:
        defaulted = quota_state_from_headers(
            GROQ_MINUTE_ONLY,
            provider="groq",
            model="m",
            doc_default_ceiling={"tpd": 100_000.0},
            confirmed_usage_today={"tpd": 4000.0},
        )
        self.assertEqual(defaulted.dim("tpd").remaining, 96_000.0)
        observed = quota_state_from_headers(
            GROQ_MINUTE_ONLY,
            provider="groq",
            model="m",
            doc_default_ceiling={"tpd": 100_000.0},
            account_observed_ceiling={"tpd": 500_000.0},
            confirmed_usage_today={"tpd": 4000.0},
        )
        self.assertEqual(observed.dim("tpd").limit, 500_000.0)
        self.assertEqual(observed.dim("tpd").remaining, 496_000.0)
        self.assertEqual(observed.dim("tpd").source, "locally_reconstructed")

    def test_account_observed_dimension_beats_documentation_default(self) -> None:
        state = quota_state_from_headers(
            {},
            provider="groq",
            model="m",
            doc_default_ceiling={"tpd": 100_000.0},
            account_observed={"tpd": {"limit": 900.0, "remaining": 700.0}},
        )
        self.assertEqual(state.dim("tpd").remaining, 700.0)
        self.assertEqual(state.dim("tpd").source, "account_observed")
        self.assertEqual(state.dim("tpd").limit, 900.0)

    def test_account_observed_fills_header_gaps_without_clobbering(self) -> None:
        state = quota_state_from_headers(
            GROQ_MINUTE_ONLY,
            provider="groq",
            model="m",
            now=1000.0,
            account_observed={
                "tpm": {"limit": 1.0, "remaining": 1.0},
                "tpd": {"limit": 5000.0, "remaining": 4321.0},
            },
        )
        # The header already reported the minute window; it is not overwritten.
        self.assertEqual(state.dim("tpm").remaining, 7890.0)
        self.assertEqual(state.dim("tpm").source, "provider_header")
        # The day window only exists because the account reported it.
        self.assertEqual(state.dim("tpd").remaining, 4321.0)
        self.assertEqual(state.dim("tpd").source, "account_observed")

    def test_state_serializes_round_trip(self) -> None:
        state = quota_state_from_headers(GROQ_HEADERS, provider="groq", model="m", now=1000.0)
        restored = QuotaState.from_dict(state.to_dict())
        self.assertEqual(restored.dim("tpm"), state.dim("tpm"))
        self.assertEqual(restored.dim("rpd"), state.dim("rpd"))
        self.assertEqual(restored.observed_at, 1000.0)


class AdapterStructuredQuotaTests(unittest.TestCase):
    def test_groq_offer_carries_structured_quota(self) -> None:
        fake = _FakeHTTP(_groq_body(), GROQ_MINUTE_ONLY)
        with patch("kerdoios.providers.http.urllib.request.urlopen", return_value=fake):
            offers = discover_groq(api_key="test")
        quota = offers[0].economics.quota
        self.assertIsInstance(quota, QuotaState)
        self.assertEqual(quota.dim("tpm").remaining, 7890.0)
        self.assertEqual(quota.dim("rpm").remaining, 999.0)
        self.assertEqual(quota.dim("tpd").source, "unknown")
        self.assertIsNone(quota.dim("tpd").remaining)
        # Legacy scalar still reflects the token-minute projection.
        self.assertEqual(offers[0].economics.remaining_free_quota, 7890.0)

    def test_cerebras_offer_carries_day_dimension(self) -> None:
        fake = _FakeHTTP(_cerebras_body(), CEREBRAS_FULL)
        with patch("kerdoios.providers.http.urllib.request.urlopen", return_value=fake):
            offers = discover_cerebras(api_key="test")
        quota = offers[0].economics.quota
        self.assertIsInstance(quota, QuotaState)
        self.assertEqual(quota.dim("tpd").remaining, 719_900_000.0)
        self.assertEqual(quota.dim("rpd").remaining, 1_439_000.0)
        for dimension in REQUIRED_DIMENSIONS:
            self.assertEqual(quota.dim(dimension).source, "provider_header", dimension)

    def test_groq_offer_reconstructs_day_from_injected_ceiling(self) -> None:
        fake = _FakeHTTP(_groq_body(), GROQ_MINUTE_ONLY)
        with patch("kerdoios.providers.http.urllib.request.urlopen", return_value=fake):
            offers = discover_groq(
                api_key="test",
                account_observed_ceiling={"tpd": 10_000.0, "rpd": 100.0},
                confirmed_usage_today={"tpd": 2500.0, "rpd": 4.0},
            )
        quota = offers[0].economics.quota
        self.assertEqual(quota.dim("tpd").remaining, 7500.0)
        self.assertEqual(quota.dim("tpd").source, "locally_reconstructed")
        self.assertEqual(quota.dim("rpd").remaining, 96.0)


if __name__ == "__main__":
    unittest.main()

