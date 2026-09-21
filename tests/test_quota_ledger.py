from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios.cache import load_inventory_cache, save_inventory_cache
from kerdoios.providers.fixture import fixture_offers
from kerdoios.quota import (
    QuotaLedger,
    QuotaState,
    UsageReceipt,
    compare_remaining,
    ingest_receipts,
    load_receipts,
    project_from_receipts,
    quota_state_from_headers,
    reconcile_projection,
    summarize_receipts,
)
from kerdoios.types import Economics

GROQ_MINUTE_ONLY = {
    "x-ratelimit-limit-requests": "1000",
    "x-ratelimit-limit-tokens": "8000",
    "x-ratelimit-remaining-requests": "999",
    "x-ratelimit-remaining-tokens": "7890",
    "x-ratelimit-reset-requests": "1m26.4s",
    "x-ratelimit-reset-tokens": "825ms",
}

CEREBRAS_DAY = {
    "x-ratelimit-limit-requests-day": "1440000",
    "x-ratelimit-remaining-requests-day": "1439000",
    "x-ratelimit-limit-tokens-day": "720000000",
    "x-ratelimit-remaining-tokens-day": "719900000",
    "x-ratelimit-reset-tokens-day": "2h",
}


def _receipts() -> list[UsageReceipt]:
    return [
        UsageReceipt(
            provider="groq",
            model="llama-3.3-70b",
            request_id="req-1",
            trace_id="trace-1",
            prompt_tokens=1000,
            completion_tokens=500,
            cache_tokens=200,
            start=100.0,
            end=101.5,
            latency=1500.0,
            success=True,
            verified_outcome=True,
        ),
        UsageReceipt(
            provider="groq",
            model="llama-3.3-70b",
            request_id="req-2",
            trace_id="trace-2",
            prompt_tokens=800,
            completion_tokens=200,
            start=200.0,
            end=201.0,
            latency=1000.0,
            success=True,
            verified_outcome=False,
            retry_state="retried_once",
        ),
    ]


class LedgerPersistenceTests(unittest.TestCase):
    def test_ledger_survives_a_fresh_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quota_ledger.json"
            state = quota_state_from_headers(
                GROQ_MINUTE_ONLY,
                provider="groq",
                model="llama-3.3-70b",
                now=1000.0,
                account_observed_ceiling={"tpd": 10_000.0, "rpd": 100.0},
                confirmed_usage_today={"tpd": 2500.0, "rpd": 4.0},
            )
            QuotaLedger(path).record(state)

            reloaded = QuotaLedger(path).load()
            restored = reloaded.get("groq", "llama-3.3-70b")
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.dim("tpd").remaining, 7500.0)
            self.assertEqual(restored.dim("tpd").source, "locally_reconstructed")
            self.assertEqual(restored.dim("rpd").remaining, 96.0)
            self.assertEqual(restored.dim("tpm").remaining, 7890.0)
            self.assertEqual(restored.dim("tpm").source, "provider_header")
            self.assertEqual(restored.observed_at, 1000.0)

    def test_ledger_file_is_json_and_never_stores_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quota_ledger.json"
            QuotaLedger(path).record(
                quota_state_from_headers(GROQ_MINUTE_ONLY, provider="groq", model="m", now=1.0)
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "kerdoios.quota_ledger.v1")
            blob = json.dumps(payload).lower()
            for needle in ("api_key", "authorization", "bearer", "openrouter_api_key", "sk-"):
                self.assertNotIn(needle, blob)

    def test_ledger_reconstructs_from_receipts_after_loss(self) -> None:
        receipts = _receipts()
        ceilings = {"tpd": 10_000.0, "rpd": 100.0}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quota_ledger.json"
            ingest_receipts(receipts, ledger_path=path, account_observed_ceiling=ceilings, now=500.0)
            first = QuotaLedger(path).load().get("groq", "llama-3.3-70b")
            self.assertIsNotNone(first)
            assert first is not None
            # 1000+500+800+200 tokens confirmed; cache tokens are a subset, not extra.
            self.assertEqual(first.dim("tpd").remaining, 7500.0)
            self.assertEqual(first.dim("rpd").remaining, 98.0)

            # Receipts are the source of truth: rebuilding gives the same state.
            rebuilt = project_from_receipts(
                receipts,
                provider="groq",
                model="llama-3.3-70b",
                account_observed_ceiling=ceilings,
                now=500.0,
            )
            self.assertEqual(rebuilt.dim("tpd").remaining, first.dim("tpd").remaining)
            self.assertEqual(rebuilt.dim("rpd").remaining, first.dim("rpd").remaining)

    def test_ingest_receipts_reconciles_existing_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quota_ledger.json"
            ledger = QuotaLedger(path)
            ledger.record(
                quota_state_from_headers(
                    GROQ_MINUTE_ONLY,
                    provider="groq",
                    model="llama-3.3-70b",
                    now=1000.0,
                    account_observed_ceiling={"tpd": 10_000.0, "rpd": 100.0},
                )
            )
            self.assertEqual(ledger.get("groq", "llama-3.3-70b").dim("tpd").remaining, 10_000.0)

            report = ingest_receipts(_receipts(), ledger=ledger, now=1100.0)
            self.assertEqual(report["receipts"], 2)
            self.assertEqual(report["groups"], 1)
            reconciled = ledger.get("groq", "llama-3.3-70b")
            self.assertEqual(reconciled.dim("tpd").remaining, 7500.0)
            self.assertEqual(reconciled.dim("tpd").source, "locally_reconstructed")
            # Header-observed minute windows are not recomputed from day receipts.
            self.assertEqual(reconciled.dim("tpm").remaining, 7890.0)
            self.assertEqual(reconciled.dim("tpm").source, "provider_header")


class ReceiptReconciliationTests(unittest.TestCase):
    def test_receipt_fields_are_consumed(self) -> None:
        receipt = UsageReceipt.from_dict(
            {
                "schema": "tokenomics.receipt.v1",
                "provider": "cerebras",
                "model": "qwen-3-32b",
                "request_id": "r-9",
                "trace_id": "t-9",
                "prompt_tokens": 300,
                "completion_tokens": 120,
                "cache_tokens": 50,
                "start": 10.0,
                "end": 12.0,
                "latency": 2000.0,
                "success": True,
                "verified_outcome": True,
                "retry_state": "fresh",
            }
        )
        usage = summarize_receipts([receipt])
        self.assertEqual(usage.requests, 1)
        self.assertEqual(usage.prompt_tokens, 300)
        self.assertEqual(usage.completion_tokens, 120)
        self.assertEqual(usage.cache_tokens, 50)
        self.assertEqual(usage.total_tokens, 420)
        self.assertEqual(usage.verified, 1)
        self.assertEqual(usage.retried, 0)

    def test_receipt_jsonl_loads_and_filters_by_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipts.jsonl"
            rows = [receipt.to_dict() for receipt in _receipts()]
            rows.append({"not": "a receipt"})
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            receipts = load_receipts(path)
        self.assertEqual(len(receipts), 2)
        only_first = summarize_receipts(receipts, since=50.0, until=150.0)
        self.assertEqual(only_first.requests, 1)
        self.assertEqual(only_first.total_tokens, 1500)

    def test_local_reconstruction_compared_to_provider_reported(self) -> None:
        reported = quota_state_from_headers(
            CEREBRAS_DAY, provider="cerebras", model="qwen-3-32b", now=1000.0
        )
        self.assertEqual(reported.dim("tpd").source, "provider_header")
        projected = project_from_receipts(
            [UsageReceipt(provider="cerebras", model="qwen-3-32b", prompt_tokens=1000, completion_tokens=0)],
            provider="cerebras",
            model="qwen-3-32b",
            account_observed_ceiling={"tpd": 10_000.0},
            now=1000.0,
        )
        rows = compare_remaining(projected, reported)
        self.assertEqual(rows["tpd"]["local_remaining"], 9000.0)
        self.assertEqual(rows["tpd"]["provider_reported"], 719_900_000.0)
        self.assertEqual(rows["tpd"]["delta"], 9000.0 - 719_900_000.0)
        self.assertEqual(rows["tpd"]["local_source"], "locally_reconstructed")
        self.assertNotIn("tpm", rows)

    def test_reconcile_keeps_provider_reported_window(self) -> None:
        reported = quota_state_from_headers(
            CEREBRAS_DAY, provider="cerebras", model="qwen-3-32b", now=1000.0
        )
        reconciled = reconcile_projection(
            reported,
            [UsageReceipt(provider="cerebras", model="qwen-3-32b", prompt_tokens=1000, completion_tokens=0)],
            now=1100.0,
        )
        # The provider's own day statement is not overwritten by a local estimate.
        self.assertEqual(reconciled.dim("tpd").remaining, 719_900_000.0)
        self.assertEqual(reconciled.dim("tpd").source, "provider_header")

    def test_projection_exposes_comparison_provenance(self) -> None:
        state = quota_state_from_headers(
            GROQ_MINUTE_ONLY,
            provider="groq",
            model="m",
            now=1000.0,
            account_observed_ceiling={"tpd": 5000.0},
            confirmed_usage_today={"tpd": 1000.0},
        )
        comparison = state.comparison()
        self.assertEqual(comparison["tpd"]["locally_reconstructed"], 4000.0)
        self.assertIsNone(comparison["tpd"]["provider_reported"])
        self.assertEqual(comparison["tpm"]["provider_reported"], 7890.0)
        self.assertIsNone(comparison["tpm"]["locally_reconstructed"])


class CacheQuotaTests(unittest.TestCase):
    def test_inventory_cache_round_trip_preserves_structured_quota(self) -> None:
        quota = quota_state_from_headers(
            GROQ_MINUTE_ONLY,
            provider="groq",
            model="llama-3.3-70b",
            now=1000.0,
            account_observed_ceiling={"tpd": 10_000.0},
            confirmed_usage_today={"tpd": 2500.0},
        )
        offer = replace(
            fixture_offers()[0],
            economics=Economics(remaining_free_quota=7890.0, quota=quota),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"KERDOIOS_CACHE": tmp}, clear=False):
                save_inventory_cache([offer])
                loaded = load_inventory_cache(ignore_ttl=True)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        restored = loaded[0].economics.quota
        self.assertIsInstance(restored, QuotaState)
        assert isinstance(restored, QuotaState)
        self.assertEqual(restored.dim("tpd").remaining, 7500.0)
        self.assertEqual(restored.dim("tpd").source, "locally_reconstructed")
        self.assertEqual(restored.dim("tpm").remaining, 7890.0)


if __name__ == "__main__":
    unittest.main()
