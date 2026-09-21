"""Parse provider rate-limit headers into a :class:`QuotaState`.

Header shapes we actually see:

* Groq exposes only MINUTE windows (``x-ratelimit-*-requests`` and
  ``x-ratelimit-*-tokens``); there is no day dimension in its headers.
* Cerebras exposes minute, hour and day windows for both requests and tokens
  (``x-ratelimit-limit-tokens-day: ...``).
* OpenRouter sends a window-ambiguous ``x-ratelimit-limit/remaining`` pair
  documented as requests per minute, with an absolute Unix-millisecond reset.

Reset values arrive as durations (``1m26.4s``, ``825ms``), bare float seconds,
absolute Unix seconds, or absolute Unix milliseconds. All of them normalize to
an absolute ``reset_at`` in epoch seconds so a persisted ledger stays meaningful
after a restart.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Mapping

from .model import (
    ALL_DIMENSIONS,
    LEGACY_PRECEDENCE,
    QuotaDimension,
    QuotaState,
)

# Trailing ms is matched first so "120ms" is not eaten as minutes + "s".
_MS_DURATION = re.compile(r"\A(\d+(?:\.\d+)?)ms\Z", re.IGNORECASE)
_HMS_DURATION = re.compile(
    r"\A(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?\Z",
    re.IGNORECASE,
)

_RETRY_AFTER = "retry-after"


def header_map(raw: Mapping[str, Any] | object) -> dict[str, str]:
    items = getattr(raw, "items", None)
    if items is None:
        return {}
    return {str(key).lower(): str(value).strip() for key, value in items()}


def parse_remaining(raw: str) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    # Providers send -1 for "not applicable"; that is missing, not a budget.
    if not math.isfinite(value) or value < 0:
        return None
    return value


def parse_duration_seconds(raw: str) -> float | None:
    ms_match = _MS_DURATION.fullmatch(raw)
    if ms_match:
        return float(ms_match.group(1)) / 1000.0
    match = _HMS_DURATION.fullmatch(raw.strip())
    if match is None or not any(match.groups()):
        return None
    hours, minutes, seconds = (float(part) if part else 0.0 for part in match.groups())
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_reset_seconds(raw: str, now: float) -> float | None:
    """Seconds from ``now`` until the window resets, in any observed form."""
    duration = parse_duration_seconds(raw)
    if duration is not None:
        return duration
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    if value >= 1e12:
        return max(0.0, value / 1000.0 - now)
    if value >= 1e9:
        return max(0.0, value - now)
    return value


def reset_at_from_raw(raw: str, now: float) -> float | None:
    seconds = parse_reset_seconds(raw, now)
    if seconds is None:
        return None
    return now + seconds


def _build_header_fields() -> dict[str, tuple[str, str]]:
    fields: dict[str, tuple[str, str]] = {}
    # Groq: the unsuffixed request/token headers describe the MINUTE window.
    for plural, prefix in (("requests", "rp"), ("tokens", "tp")):
        for name in ("limit", "remaining", "reset"):
            fields[f"x-ratelimit-{name}-{plural}"] = (f"{prefix}m", name)
    # Cerebras: explicit minute/hour/day windows for both units.
    for suffix, window in (("minute", "m"), ("hour", "h"), ("day", "d")):
        for plural, prefix in (("requests", "rp"), ("tokens", "tp")):
            for name in ("limit", "remaining", "reset"):
                fields[f"x-ratelimit-{name}-{plural}-{suffix}"] = (f"{prefix}{window}", name)
    # OpenRouter's window-ambiguous pair is documented as requests per minute.
    for name in ("limit", "remaining", "reset"):
        fields[f"x-ratelimit-{name}"] = ("rpm", name)
    # A named free-quota budget with its own reset; the unit is not stated, so
    # it gets its own dimension rather than being aliased onto rpm/tpd.
    fields["remaining-quota"] = ("quota", "remaining")
    fields["x-remaining-quota"] = ("quota", "remaining")
    fields["seconds-until-quota-reset"] = ("quota", "reset")
    return fields


_HEADER_FIELDS = _build_header_fields()


def _header_dimensions(headers: Mapping[str, str] | object, now: float) -> dict[str, QuotaDimension]:
    mapped = header_map(headers)
    slots: dict[str, dict[str, float]] = {}
    for key, (dimension, field) in _HEADER_FIELDS.items():
        raw = mapped.get(key)
        if raw is None or raw == "":
            continue
        slot = slots.setdefault(dimension, {})
        if field == "reset":
            if slot.get("reset_at") is None:
                value = reset_at_from_raw(raw, now)
                if value is not None:
                    slot["reset_at"] = value
            continue
        if slot.get(field) is not None:
            continue
        value = parse_remaining(raw)
        if value is not None:
            slot[field] = value

    retry_raw = mapped.get(_RETRY_AFTER)
    if retry_raw:
        fallback = reset_at_from_raw(retry_raw, now)
        if fallback is not None:
            # retry-after is a reset for whichever dimension is present without one.
            for slot in slots.values():
                if slot.get("remaining") is not None and slot.get("reset_at") is None:
                    slot["reset_at"] = fallback

    dimensions: dict[str, QuotaDimension] = {}
    for dimension, slot in slots.items():
        # A reset alone describes no budget; keep the window unknown rather
        # than claiming a limit or remaining we were never told.
        if slot.get("limit") is None and slot.get("remaining") is None:
            continue
        dimensions[dimension] = QuotaDimension(
            limit=slot.get("limit"),
            remaining=slot.get("remaining"),
            reset_at=slot.get("reset_at"),
            source="provider_header",
        )
    return dimensions


def quota_state_from_headers(
    headers: Mapping[str, str] | object,
    *,
    provider: str = "",
    model: str = "",
    now: float | None = None,
    confirmed_usage_today: Mapping[str, float] | None = None,
    account_observed_ceiling: Mapping[str, float] | None = None,
    doc_default_ceiling: Mapping[str, float] | None = None,
    account_observed: Mapping[str, Mapping[str, Any]] | None = None,
) -> QuotaState:
    """Build a dimensional quota state from one response's headers.

    Precedence per dimension, highest first:

    1. ``provider_header`` — this response carried the window.
    2. ``account_observed`` — the account ceiling/usage measured outside headers.
    3. ``locally_reconstructed`` — ``ceiling - confirmed_usage_today`` for a
       window the provider never exposes (Groq's day dimension).

    ``account_observed_ceiling`` beats ``doc_default_ceiling``: a measured
    account ceiling overrides a generic documented sticker. Missing windows
    stay ``unknown``; a minute token count is never reused as a day count.
    """
    clock = time.time() if now is None else now
    dimensions = _header_dimensions(headers, clock)

    account_ceilings: dict[str, float] = {}
    for dimension, value in (account_observed_ceiling or {}).items():
        parsed = _opt_float(value)
        if dimension in ALL_DIMENSIONS and parsed is not None:
            account_ceilings[dimension] = parsed
    doc_ceilings: dict[str, float] = {}
    for dimension, value in (doc_default_ceiling or {}).items():
        parsed = _opt_float(value)
        if dimension in ALL_DIMENSIONS and parsed is not None:
            doc_ceilings[dimension] = parsed

    for dimension, values in (account_observed or {}).items():
        if dimension not in ALL_DIMENSIONS or not isinstance(values, Mapping):
            continue
        existing = dimensions.get(dimension)
        if existing is not None and existing.remaining is not None:
            # The response header is this account's freshest statement; an
            # out-of-band observation only fills a window the header omitted.
            continue
        limit = _opt_float(values.get("limit"))
        if limit is None and existing is not None:
            limit = existing.limit
        remaining = _opt_float(values.get("remaining"))
        reset_at = _opt_float(values.get("reset_at"))
        if remaining is None and limit is not None:
            # A ceiling without a measured remaining is reconstruction input.
            account_ceilings[dimension] = limit
            if reset_at is None:
                continue
        dimensions[dimension] = QuotaDimension(
            limit=limit,
            remaining=remaining,
            reset_at=reset_at,
            source="account_observed",
        )

    usage = confirmed_usage_today or {}
    for dimension in ALL_DIMENSIONS:
        current = dimensions.get(dimension)
        if current is not None and current.remaining is not None:
            continue
        ceiling = account_ceilings.get(dimension)
        if ceiling is None:
            ceiling = doc_ceilings.get(dimension)
        if ceiling is None:
            continue
        used = _opt_float(usage.get(dimension)) or 0.0
        dimensions[dimension] = QuotaDimension(
            limit=ceiling,
            remaining=max(0.0, ceiling - used),
            reset_at=current.reset_at if current is not None else None,
            source="locally_reconstructed",
        )

    return QuotaState(provider=provider, model=model, dimensions=dimensions, observed_at=clock)


def legacy_projection(state: QuotaState, *, now: float | None = None) -> tuple[float | None, float | None]:
    """One scalar pair for callers that predate the dimensional model.

    Picks the most day-broad known token window first (tpd > tph > tpm), then
    the named quota budget, then requests. It never mixes a token count into a
    request slot and returns ``(None, None)`` when nothing is known.
    """
    clock = time.time() if now is None else now
    for dimension in LEGACY_PRECEDENCE:
        entry = state.dim(dimension)
        if entry.remaining is None:
            continue
        return entry.remaining, entry.seconds_until_reset(clock)
    return None, None


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed
