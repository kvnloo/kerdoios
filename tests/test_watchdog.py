from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios.observed import Observation, classify_http, load_observations, record
from kerdoios.watchdog import (
    burn_by_origin,
    origin_of,
    probe_openrouter_free,
    remaining_for_burn,
    report,
)


class _FakeHTTP:
    def __init__(self, body: dict, headers: dict[str, str], status: int = 200) -> None:
        self._body = json.dumps(body).encode()
        self.headers = headers
        self.status = status

    def read(self) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> _FakeHTTP:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _free_catalog_body() -> dict:
    return {
        "data": [
            {"id": "nvidia/nemotron-nano-9b-v2:free"},
            {"id": "google/gemma-3-4b-it:free"},
            {"id": "openai/gpt-4o"},
        ]
    }


class ClassifyHttpTests(unittest.TestCase):
    def test_401_402_404_429_never_success(self) -> None:
        self.assertEqual(classify_http(401), "failed")
        self.assertEqual(classify_http(402), "exhausted")
        self.assertEqual(classify_http(404), "failed")
        self.assertEqual(classify_http(429), "exhausted")
        for status in (401, 402, 404, 429):
            self.assertNotEqual(classify_http(status), "success")
        self.assertEqual(classify_http(500), "failed")
        self.assertNotEqual(classify_http(402), "success")


class RemainingForBurnTests(unittest.TestCase):
    def test_catalog_remaining_zero_is_unknown_not_exhausted(self) -> None:
        obs = Observation(
            provider="openrouter",
            model="nvidia/nemotron-nano-9b-v2:free",
            task_type="coding",
            completed=True,
            actual_cost=0.0,
            remaining_quota=0.0,
            remaining_source="catalog",
        )
        self.assertIsNone(remaining_for_burn(obs))
        self.assertEqual(classify_http(None), "success")
        self.assertEqual(classify_http(200), "success")


class BurnByOriginTests(unittest.TestCase):
    def test_burn_per_origin_not_openrouter_bucket(self) -> None:
        rows = [
            Observation(
                "openrouter",
                "nvidia/nemotron-nano-9b-v2:free",
                "coding",
                True,
                0.01,
                origin_provider="nvidia",
                input_tokens=10,
                output_tokens=20,
                remaining_quota=50.0,
                remaining_source="chat",
            ),
            Observation(
                "openrouter",
                "google/gemma-3-4b-it:free",
                "coding",
                True,
                0.02,
                origin_provider="google",
                input_tokens=5,
                output_tokens=7,
                remaining_quota=40.0,
                remaining_source="chat",
            ),
            Observation(
                "openrouter",
                "groq/llama-3.3-70b",
                "coding",
                True,
                0.0,
                origin_provider="groq",
                http_status=429,
                remaining_quota=0.0,
                remaining_source="chat",
            ),
        ]
        burns = burn_by_origin(rows)
        self.assertEqual(set(burns), {"nvidia", "google", "groq"})
        self.assertNotIn("openrouter", burns)
        self.assertEqual(origin_of(rows[0]), "nvidia")
        self.assertEqual(burns["nvidia"].tokens, 30)
        self.assertAlmostEqual(burns["nvidia"].cost, 0.01)
        self.assertEqual(burns["google"].tokens, 12)
        self.assertEqual(burns["groq"].exhausted, 1)
        self.assertEqual(burns["groq"].success, 0)

    def test_unknown_remaining_among_mixed_rows_is_none(self) -> None:
        rows = [
            Observation(
                "nvidia",
                "nemotron",
                "coding",
                True,
                0.01,
                origin_provider="nvidia",
                remaining_quota=10.0,
                remaining_source="chat",
            ),
            Observation(
                "nvidia",
                "nemotron",
                "coding",
                True,
                0.02,
                origin_provider="nvidia",
            ),
        ]
        burn = burn_by_origin(rows)["nvidia"]
        self.assertIsNone(burn.remaining_quota)

    def test_old_jsonl_without_token_fields_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observed.jsonl"
            path.write_text(
                '{"provider":"groq","model":"llama-3.3-70b","task_type":"coding",'
                '"completed":true,"actual_cost":0.001,"retried":false}\n',
                encoding="utf-8",
            )
            rows = load_observations(path=path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].provider, "groq")
            self.assertEqual(rows[0].model, "llama-3.3-70b")
            self.assertTrue(rows[0].completed)
            self.assertIsNone(rows[0].origin_provider)
            self.assertIsNone(rows[0].input_tokens)
            self.assertIsNone(rows[0].output_tokens)
            self.assertIsNone(rows[0].http_status)


class ProbeOpenRouterFreeTests(unittest.TestCase):
    def test_mocked_catalog_remaining_zero_is_unknown_success(self) -> None:
        fake = _FakeHTTP(_free_catalog_body(), {"x-ratelimit-remaining": "0"}, status=200)
        captured: dict[str, object] = {}

        def fake_urlopen(req, timeout=12.0):
            captured["url"] = req.full_url
            captured["headers"] = {str(k).lower(): v for k, v in req.header_items()}
            return fake

        with patch("kerdoios.watchdog.urllib.request.urlopen", fake_urlopen):
            result = probe_openrouter_free()
        self.assertIsNone(result["remaining"])
        self.assertEqual(result["error_class"], "success")
        self.assertEqual(result["count"], 2)
        self.assertEqual(
            result["ids"],
            ["nvidia/nemotron-nano-9b-v2:free", "google/gemma-3-4b-it:free"],
        )
        self.assertTrue(result["set"])
        self.assertNotIn("authorization", captured["headers"])
        self.assertNotIn("Authorization", json.dumps(result))

    def test_mocked_429_http_error_is_exhausted(self) -> None:
        hdrs = Message()
        err = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/models",
            429,
            "Too Many Requests",
            hdrs,
            io.BytesIO(b""),
        )
        with patch("kerdoios.watchdog.urllib.request.urlopen", side_effect=err):
            result = probe_openrouter_free()
        self.assertEqual(result["http_status"], 429)
        self.assertEqual(result["error_class"], "exhausted")
        self.assertFalse(result["set"])
        self.assertNotEqual(result["error_class"], "success")
        self.assertNotIn("Authorization", json.dumps(result))


class ReportTests(unittest.TestCase):
    def test_report_openrouter_bucket_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observed.jsonl"
            record(
                Observation(
                    "openrouter",
                    "nvidia/nemotron-nano-9b-v2:free",
                    "coding",
                    True,
                    0.01,
                    origin_provider="nvidia",
                    input_tokens=8,
                    output_tokens=2,
                ),
                path=path,
            )
            payload = report(path=path, live_free=False)
        self.assertIs(payload["openrouter_bucket"], False)
        self.assertIn("nvidia", payload["origin_burn"])
        self.assertNotIn("openrouter", payload["origin_burn"])
        self.assertNotIn("live_openrouter_free", payload)


if __name__ == "__main__":
    unittest.main()
