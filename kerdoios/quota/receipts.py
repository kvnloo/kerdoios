"""Reconcile the local quota projection with Tokenomics usage receipts.

The receipt is the accounting source of truth; the ledger is only a cache. A
receipt is attributable to ``provider``/``model`` and carries the measured
tokens, timing, outcome and retry state of one request. Summing receipts gives
"confirmed usage", which turns an injected account ceiling into a
``locally_reconstructed`` daily remaining for providers whose headers never
expose a day window (Groq).

``cache_tokens`` is tracked but not added to ``total_tokens``: when a provider
reports it, cached input is a subset of the prompt tokens and counting it again
would overstate quota consumption.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .ledger import QuotaLedger
from .model import ALL_DIMENSIONS, DAY_DIMENSIONS, QuotaDimension, QuotaState

RECEIPT_SCHEMA = "tokenomics.receipt.v1"

# retry_state values that mean "this request was the first attempt".
_NO_RETRY_STATES = frozenset(
    {"", "none", "no", "false", "0", "fresh", "initial", "first", "first_attempt"}
)


def _is_retry(retry_state: str | None) -> bool:
    if not retry_state:
        return False
    normalized = retry_state.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized not in _NO_RETRY_STATES


def _opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


@dataclass(frozen=True)
class UsageReceipt:
    """One measured request, as emitted by Tokenomics."""

    provider: str
    model: str
    request_id: str | None = None
    trace_id: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_tokens: int | None = None
    start: float | None = None
    end: float | None = None
    latency: float | None = None
    success: bool | None = None
    verified_outcome: bool | None = None
    retry_state: str | None = None
    quota_group: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def verified(self) -> bool:
        return self.verified_outcome is True

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
        for key in ("request_id", "trace_id", "retry_state", "quota_group"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.cache_tokens is not None:
            payload["cache_tokens"] = self.cache_tokens
        for key in ("start", "end", "latency"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.success is not None:
            payload["success"] = self.success
        if self.verified_outcome is not None:
            payload["verified_outcome"] = self.verified_outcome
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "UsageReceipt":
        return cls(
            provider=str(_first(raw, "provider") or ""),
            model=str(_first(raw, "model") or ""),
            request_id=_opt_str(_first(raw, "request_id", "requestId", "id")),
            trace_id=_opt_str(_first(raw, "trace_id", "traceId")),
            prompt_tokens=_opt_int(_first(raw, "prompt_tokens", "input_tokens")) or 0,
            completion_tokens=_opt_int(_first(raw, "completion_tokens", "output_tokens")) or 0,
            cache_tokens=_opt_int(_first(raw, "cache_tokens", "cached_input_tokens")),
            start=_opt_float(_first(raw, "start", "started_at")),
            end=_opt_float(_first(raw, "end", "ended_at", "finished_at")),
            latency=_opt_float(_first(raw, "latency", "latency_ms")),
            success=_opt_bool(_first(raw, "success", "completed")),
            verified_outcome=_opt_bool(_first(raw, "verified_outcome", "verified_success")),
            retry_state=_opt_str(_first(raw, "retry_state", "retry")),
            quota_group=_opt_str(_first(raw, "quota_group")),
        )


def _opt_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


@dataclass(frozen=True)
class ConfirmedUsage:
    """Measured usage for one provider/model, summed from receipts."""

    provider: str
    model: str
    requests: int
    prompt_tokens: int
    completion_tokens: int
    cache_tokens: int
    verified: int
    retried: int
    first_start: float | None = None
    last_end: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def for_dimension(self, dimension: str) -> float:
        if dimension in ("rpm", "rph", "rpd", "quota"):
            return float(self.requests)
        if dimension in ("tpm", "tph", "tpd"):
            return float(self.total_tokens)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_tokens": self.cache_tokens,
            "total_tokens": self.total_tokens,
            "verified": self.verified,
            "retried": self.retried,
            "first_start": self.first_start,
            "last_end": self.last_end,
        }


def coerce_receipt(row: UsageReceipt | Mapping[str, Any]) -> UsageReceipt:
    if isinstance(row, UsageReceipt):
        return row
    return UsageReceipt.from_dict(row)


def load_receipts(path: Path | str) -> list[UsageReceipt]:
    """Read JSONL receipts. A malformed line is skipped, not fatal."""
    rows: list[UsageReceipt] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            receipt = UsageReceipt.from_dict(payload)
            if receipt.provider and receipt.model:
                rows.append(receipt)
    return rows


def summarize_receipts(
    receipts: Iterable[UsageReceipt | Mapping[str, Any]],
    *,
    provider: str | None = None,
    model: str | None = None,
    since: float | None = None,
    until: float | None = None,
) -> ConfirmedUsage:
    """Sum confirmed usage, optionally within a time window.

    A receipt without timestamps is counted even inside a window: over-counting
    usage under-estimates remaining quota, which is the safe direction.
    """
    requests = 0
    prompt = 0
    completion = 0
    cache = 0
    verified = 0
    retried = 0
    starts: list[float] = []
    ends: list[float] = []
    for raw in receipts:
        receipt = coerce_receipt(raw)
        if provider is not None and receipt.provider != provider:
            continue
        if model is not None and receipt.model != model:
            continue
        moment = receipt.start if receipt.start is not None else receipt.end
        if moment is not None:
            if since is not None and moment < since:
                continue
            if until is not None and moment > until:
                continue
        requests += 1
        prompt += receipt.prompt_tokens
        completion += receipt.completion_tokens
        cache += receipt.cache_tokens or 0
        if receipt.verified:
            verified += 1
        if _is_retry(receipt.retry_state):
            retried += 1
        if receipt.start is not None:
            starts.append(receipt.start)
        if receipt.end is not None:
            ends.append(receipt.end)
    return ConfirmedUsage(
        provider=provider or "",
        model=model or "",
        requests=requests,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cache_tokens=cache,
        verified=verified,
        retried=retried,
        first_start=min(starts) if starts else None,
        last_end=max(ends) if ends else None,
    )


def project_from_receipts(
    receipts: Iterable[UsageReceipt | Mapping[str, Any]],
    *,
    provider: str,
    model: str,
    account_observed_ceiling: Mapping[str, float] | None = None,
    doc_default_ceiling: Mapping[str, float] | None = None,
    now: float | None = None,
) -> QuotaState:
    """Rebuild a provider/model quota state from receipts plus ceilings.

    Only day-scoped windows are reconstructable this way; every other window
    stays ``unknown`` rather than inheriting a day total.
    """
    clock = time.time() if now is None else now
    usage = summarize_receipts(receipts, provider=provider, model=model)
    dimensions: dict[str, QuotaDimension] = {}
    for dimension in DAY_DIMENSIONS:
        ceiling = _ceiling(dimension, account_observed_ceiling, doc_default_ceiling)
        if ceiling is None:
            continue
        dimensions[dimension] = QuotaDimension(
            limit=ceiling,
            remaining=max(0.0, ceiling - usage.for_dimension(dimension)),
            source="locally_reconstructed",
        )
    return QuotaState(provider=provider, model=model, dimensions=dimensions, observed_at=clock)


def reconcile_projection(
    projection: QuotaState,
    receipts: Iterable[UsageReceipt | Mapping[str, Any]],
    *,
    account_observed_ceiling: Mapping[str, float] | None = None,
    doc_default_ceiling: Mapping[str, float] | None = None,
    now: float | None = None,
) -> QuotaState:
    """Recompute reconstructed day windows from measured usage.

    Provider-reported windows are left untouched (they are the provider's own
    statement); reconstructed or unknown day windows are rebuilt from the
    receipts and the known ceiling.
    """
    clock = time.time() if now is None else now
    usage = summarize_receipts(receipts, provider=projection.provider, model=projection.model)
    dimensions = dict(projection.dimensions)
    for dimension in DAY_DIMENSIONS:
        current = dimensions.get(dimension, QuotaDimension())
        if current.source == "provider_header":
            continue
        ceiling = _ceiling(dimension, account_observed_ceiling, doc_default_ceiling)
        if ceiling is None:
            ceiling = current.limit
        if ceiling is None:
            continue
        dimensions[dimension] = QuotaDimension(
            limit=ceiling,
            remaining=max(0.0, ceiling - usage.for_dimension(dimension)),
            reset_at=current.reset_at,
            source="locally_reconstructed",
        )
    return QuotaState(
        provider=projection.provider,
        model=projection.model,
        dimensions=dimensions,
        observed_at=clock,
    )


def compare_remaining(
    local: QuotaState,
    provider: QuotaState,
) -> dict[str, dict[str, Any]]:
    """Per-dimension locally-reconstructed vs provider-reported remaining."""
    rows: dict[str, dict[str, Any]] = {}
    for dimension in ALL_DIMENSIONS:
        local_value = local.remaining(dimension)
        provider_value = provider.remaining(dimension)
        if local_value is None and provider_value is None:
            continue
        rows[dimension] = {
            "local_remaining": local_value,
            "local_source": local.dim(dimension).source,
            "provider_reported": provider_value,
            "provider_source": provider.dim(dimension).source,
            "delta": (
                local_value - provider_value
                if local_value is not None and provider_value is not None
                else None
            ),
        }
    return rows


def ingest_receipts(
    receipts: Sequence[UsageReceipt | Mapping[str, Any]],
    *,
    ledger: QuotaLedger | None = None,
    ledger_path: Path | str | None = None,
    account_observed_ceiling: Mapping[str, float] | None = None,
    doc_default_ceiling: Mapping[str, float] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Project states from receipts and persist them.

    Returns the reconciled states plus a provider-reported/local comparison
    wherever the caller supplied a reported state through ``ledger``.
    """
    clock = time.time() if now is None else now
    rows = [coerce_receipt(row) for row in receipts]
    groups: dict[tuple[str, str], list[UsageReceipt]] = {}
    for receipt in rows:
        groups.setdefault((receipt.provider, receipt.model), []).append(receipt)

    store = ledger
    if store is None and ledger_path is not None:
        store = QuotaLedger(ledger_path).load()

    states: list[QuotaState] = []
    for (provider, model), group in sorted(groups.items()):
        state = project_from_receipts(
            group,
            provider=provider,
            model=model,
            account_observed_ceiling=account_observed_ceiling,
            doc_default_ceiling=doc_default_ceiling,
            now=clock,
        )
        if store is not None:
            existing = store.get(provider, model)
            if existing is not None:
                state = reconcile_projection(
                    existing,
                    group,
                    account_observed_ceiling=account_observed_ceiling,
                    doc_default_ceiling=doc_default_ceiling,
                    now=clock,
                )
            store.record(state, now=clock)
        states.append(state)

    return {
        "schema": "kerdoios.quota_reconciliation.v1",
        "receipts": len(rows),
        "groups": len(states),
        "states": [state.to_dict() for state in states],
        "usage": [
            summarize_receipts(group, provider=provider, model=model).to_dict()
            for (provider, model), group in sorted(groups.items())
        ],
    }


def _ceiling(
    dimension: str,
    account_observed_ceiling: Mapping[str, float] | None,
    doc_default_ceiling: Mapping[str, float] | None,
) -> float | None:
    for source in (account_observed_ceiling, doc_default_ceiling):
        if not source:
            continue
        value = _opt_float(source.get(dimension))
        if value is not None:
            return value
    return None
