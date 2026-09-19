"""P0 free-capacity convergence: transport≠creator, access evidence, HTTP diagnostics."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios.providers.http import (  # noqa: E402 — intentional
    FetchFailure,
    FailureClass,
    get_json_result,
)
from kerdoios.providers.openrouter import _offers_from_payload  # noqa: E402 — intentional
from kerdoios.inventory import _merge  # noqa: E402 — intentional
from kerdoios.types import (  # noqa: E402 — intentional
    Access,
    AccessClass,
    AccessEvidence,
    CapabilityProfile,
    Capacity,
    Economics,
    ResourceOffer,
    Telemetry,
)
from kerdoios.providers.free import is_free  # noqa: E402 — intentional
from kerdoios.doctor import classify_cerebras_status, doctor_cerebras  # noqa: E402 — intentional
from kerdoios.providers.openai_compat import is_catalog_free_tier  # noqa: E402 — intentional


def _offer(
    *,
    oid: str,
    provider: str,
    model: str,
    transport: str | None = None,
    creator: str | None = None,
    access: Access | None = None,
    local: bool = False,
) -> ResourceOffer:
    return ResourceOffer(
        id=oid,
        provider=provider,
        resource_type="llm",
        model=model,
        local=local,
        capabilities=CapabilityProfile(),
        capacity=Capacity(concurrency=1, context_window=8192),
        economics=Economics(),
        telemetry=Telemetry(),
        transport=transport,
        creator=creator,
        access=access,
    )


class TransportCreatorTests(unittest.TestCase):
    def test_openrouter_creator_is_not_transport(self) -> None:
        offers = _offers_from_payload(
            {
                "data": [
                    {
                        "id": "nvidia/nemotron-X:free",
                        "context_length": 1000000,
                        "pricing": {"prompt": "0", "completion": "0"},
                        "architecture": {"modality": "text"},
                        "supported_parameters": ["tools"],
                    }
                ]
            }
        )
        self.assertEqual(len(offers), 1)
        o = offers[0]
        self.assertEqual(o.transport, "openrouter")
        self.assertEqual(o.provider, "openrouter")
        self.assertEqual(o.creator, "nvidia")
        self.assertEqual(o.model, "nvidia/nemotron-X:free")
        self.assertEqual(o.id, "openrouter/nvidia/nemotron-X:free")
        self.assertEqual(o.access.access_class, AccessClass.ZERO_PRICE)
        self.assertFalse(o.access.remaining_known)

    def test_same_model_on_four_transports_remain_four_offers(self) -> None:
        model = "openai/gpt-oss-120b"
        base = [
            _offer(oid=f"openrouter/{model}", provider="openrouter", model=model, transport="openrouter", creator="openai"),
            _offer(oid=f"groq/{model}", provider="groq", model=model, transport="groq", creator="openai"),
            _offer(oid=f"cerebras/{model}", provider="cerebras", model=model, transport="cerebras", creator="openai"),
            _offer(oid=f"vercel/{model}", provider="vercel", model=model, transport="vercel", creator="openai"),
        ]
        # Merge must not collapse distinct transports.
        merged = _merge(base[:1], base[1:])
        keys = {(o.transport, o.model) for o in merged}
        self.assertEqual(len(merged), 4)
        self.assertEqual(len(keys), 4)

    def test_unknown_quota_stays_unknown(self) -> None:
        access = Access(
            access_class=AccessClass.FREE_TIER,
            evidence=AccessEvidence.PROVIDER_PLAN,
            remaining_known=False,
        )
        o = _offer(
            oid="groq/x",
            provider="groq",
            model="openai/gpt-oss-120b",
            access=access,
        )
        self.assertFalse(o.access.remaining_known)
        self.assertEqual(o.economics.remaining_free_quota, 0.0)
        self.assertTrue(is_free(o))


class HttpDiagnosticsTests(unittest.TestCase):
    def test_http_statuses_remain_distinguishable(self) -> None:
        for status, expected in (
            (401, FailureClass.AUTH_INVALID),
            (402, FailureClass.TEAM_BALANCE_REQUIRED),
            (403, FailureClass.AUTH_FORBIDDEN),
            (429, FailureClass.RATE_LIMITED),
            (500, FailureClass.PROVIDER_ERROR),
        ):
            body = b'{"error":{"message":"boom","code":"x"}}'

            def _make_raiser(st: int = status, bd: bytes = body):
                def _raiser(*_a, **_k):
                    raise HTTPError(
                        "https://example.test/v1/models",
                        st,
                        "err",
                        hdrs={},
                        fp=BytesIO(bd),
                    )

                return _raiser

            with patch("urllib.request.urlopen", side_effect=_make_raiser()):
                result = get_json_result("https://example.test/v1/models", provider="cerebras", endpoint_class="auth_catalog")
            self.assertIsNotNone(result.failure)
            assert result.failure is not None
            self.assertEqual(result.failure.status, status)
            self.assertEqual(result.failure.failure_class, expected)
            self.assertEqual(result.failure.provider, "cerebras")
            blob = json.dumps(result.failure.to_dict())
            self.assertNotIn("Authorization", blob)
            self.assertNotIn("api_key", blob.lower())

    def test_cerebras_402_maps_to_team_balance_required(self) -> None:
        failure = FetchFailure(
            provider="cerebras",
            endpoint_class="auth_catalog",
            status=402,
            failure_class=FailureClass.TEAM_BALANCE_REQUIRED,
            message="Payment Required",
            sanitized_code="payment_required",
        )
        diagnosis = classify_cerebras_status(failure)
        self.assertEqual(diagnosis, "TEAM_BALANCE_REQUIRED")


class DoctorCerebrasTests(unittest.TestCase):
    def test_doctor_cerebras_detects_key_without_printing_it(self) -> None:
        secret = "csk_SUPER_SECRET_VALUE_DO_NOT_LEAK"

        def fake_result(url, **kwargs):
            from kerdoios.providers.http import FetchResult, JsonResponse

            if "public" in url:
                return FetchResult(ok=JsonResponse(body={"data": [{"id": "gpt-oss-120b"}]}, headers={}))
            return FetchResult(
                ok=None,
                failure=FetchFailure(
                    provider="cerebras",
                    endpoint_class="auth_catalog",
                    status=402,
                    failure_class=FailureClass.TEAM_BALANCE_REQUIRED,
                    message="Team requires balance",
                ),
            )

        with (
            patch.dict("os.environ", {"CEREBRAS_API_KEY": secret}, clear=False),
            patch("kerdoios.doctor.get_json_result", side_effect=fake_result),
        ):
            report = doctor_cerebras(probe=False)
        text = json.dumps(report)
        self.assertNotIn(secret, text)
        self.assertTrue(report["configured"])
        self.assertEqual(report["classification"], "TEAM_BALANCE_REQUIRED")
        self.assertIn("Personal", report["guidance"])
        self.assertEqual(report["public_catalog"]["status"], "ok")
        self.assertEqual(report["authenticated_catalog"]["status"], 402)

    def test_retired_groq_ids_not_treated_as_live_free(self) -> None:
        retired = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ]
        for mid in retired:
            self.assertFalse(is_catalog_free_tier({}, mid, "groq"))


class ExcludedFakeFreeTests(unittest.TestCase):
    def test_together_sambanova_github_absent_from_provider_modules(self) -> None:
        providers = ROOT / "kerdoios" / "providers"
        names = {p.stem for p in providers.glob("*.py")}
        self.assertNotIn("together", names)
        self.assertNotIn("sambanova", names)
        self.assertNotIn("github_models", names)
        self.assertNotIn("fireworks", names)


if __name__ == "__main__":
    unittest.main()
