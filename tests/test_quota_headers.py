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
from kerdoios.vault import UnconfiguredResolver, override_resolver


# Captured header maps only. No live keys, tokens, or Authorization values.
GROQ_HEADERS = {
    "x-ratelimit-limit-requests": "14400",
    "x-ratelimit-limit-tokens": "18000",
    "x-ratelimit-remaining-requests": "14370",
    "x-ratelimit-remaining-tokens": "17997",
    "x-ratelimit-reset-requests": "2m59.56s",
    "x-ratelimit-reset-tokens": "7.66s",
}

CEREBRAS_HEADERS = {
    "x-ratelimit-limit-requests-day": "14400",
    "x-ratelimit-limit-tokens-minute": "60000",
    "x-ratelimit-remaining-requests-day": "13693",
    "x-ratelimit-remaining-tokens-minute": "60000",
    "x-ratelimit-reset-requests-day": "11726.2840924263",
    "x-ratelimit-reset-tokens-minute": "26.28409242630005",
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
            override_resolver(UnconfiguredResolver()),
            patch("kerdoios.providers.http.urllib.request.urlopen", return_value=fake),
        ):
            offers = or_discover()
        free = next(o for o in offers if o.model == "meta/llama:free")
        self.assertEqual(free.economics.remaining_free_quota, 0.0)

    def test_groq_skips_without_key(self) -> None:
        fake = _FakeHTTP(_groq_body(), GROQ_HEADERS)
        with (
            override_resolver(UnconfiguredResolver()),
            patch.dict("os.environ", {"GROQ_API_KEY": "g-ignored"}, clear=False),
            patch("kerdoios.providers.http.urllib.request.urlopen", return_value=fake),
        ):
            self.assertEqual(discover_groq(), [])


if __name__ == "__main__":
    unittest.main()
