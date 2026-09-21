"""Dimensional provider quota: model, header parsing, ledger and receipt reconciliation.

Nothing here executes a model. The package only records what a provider
advertised, what an account observed, and what receipts confirm.
"""

from __future__ import annotations

from .ledger import (
    DEFAULT_LEDGER_PATH,
    LEDGER_SCHEMA,
    LedgerEntry,
    QuotaLedger,
    default_ledger_path,
    ledger_key,
)
from .model import (
    ALL_DIMENSIONS,
    DAY_DIMENSIONS,
    DIMENSION_UNITS,
    OPTIONAL_DIMENSIONS,
    QUOTA_SOURCES,
    REQUIRED_DIMENSIONS,
    QuotaDimension,
    QuotaSource,
    QuotaState,
)
from .parse import (
    header_map,
    legacy_projection,
    parse_duration_seconds,
    parse_remaining,
    parse_reset_seconds,
    quota_state_from_headers,
    reset_at_from_raw,
)
from .policy import PlanningPolicy, free_only_rejection, structured_free_capacity
from .receipts import (
    RECEIPT_SCHEMA,
    ConfirmedUsage,
    UsageReceipt,
    coerce_receipt,
    compare_remaining,
    ingest_receipts,
    load_receipts,
    project_from_receipts,
    reconcile_projection,
    summarize_receipts,
)

__all__ = [
    "ALL_DIMENSIONS",
    "ConfirmedUsage",
    "DAY_DIMENSIONS",
    "DEFAULT_LEDGER_PATH",
    "DIMENSION_UNITS",
    "LEDGER_SCHEMA",
    "LedgerEntry",
    "OPTIONAL_DIMENSIONS",
    "PlanningPolicy",
    "QUOTA_SOURCES",
    "QuotaDimension",
    "QuotaLedger",
    "QuotaSource",
    "QuotaState",
    "RECEIPT_SCHEMA",
    "REQUIRED_DIMENSIONS",
    "UsageReceipt",
    "coerce_receipt",
    "compare_remaining",
    "default_ledger_path",
    "free_only_rejection",
    "header_map",
    "ingest_receipts",
    "ledger_key",
    "legacy_projection",
    "load_receipts",
    "parse_duration_seconds",
    "parse_remaining",
    "parse_reset_seconds",
    "project_from_receipts",
    "quota_state_from_headers",
    "reconcile_projection",
    "reset_at_from_raw",
    "structured_free_capacity",
    "summarize_receipts",
]
