"""Tiny JSON GET helper. Adapters must tolerate missing keys and network failure."""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

# Named remaining-quota first, then token remaining. Request-count headers are the wrong unit.
_REMAINING_RESET: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("remaining-quota", ("seconds-until-quota-reset", "x-ratelimit-reset", "retry-after")),
    ("x-remaining-quota", ("x-ratelimit-reset", "retry-after")),
    ("x-ratelimit-remaining-tokens-day", ("x-ratelimit-reset-tokens-day",)),
    ("x-ratelimit-remaining-tokens-hour", ("x-ratelimit-reset-tokens-hour",)),
    ("x-ratelimit-remaining-tokens-minute", ("x-ratelimit-reset-tokens-minute",)),
    ("x-ratelimit-remaining-tokens", ("x-ratelimit-reset-tokens",)),
    ("x-ratelimit-remaining", ("x-ratelimit-reset", "retry-after")),
)

# Trailing ms is matched first so "120ms" is not eaten as minutes+"s".
_MS_DURATION = re.compile(r"\A(\d+(?:\.\d+)?)ms\Z", re.IGNORECASE)
_HMS_DURATION = re.compile(
    r"\A(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?\Z",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class JsonResponse:
    body: dict[str, Any]
    headers: dict[str, str]


def _header_map(raw: Mapping[str, Any] | object) -> dict[str, str]:
    items = getattr(raw, "items", None)
    if items is None:
        return {}
    return {str(key).lower(): str(value).strip() for key, value in items()}


def _parse_remaining(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    # Providers send -1 for "not applicable"; that is missing, not a budget.
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _parse_duration_seconds(raw: str) -> float | None:
    ms_match = _MS_DURATION.fullmatch(raw)
    if ms_match:
        return float(ms_match.group(1)) / 1000.0
    match = _HMS_DURATION.fullmatch(raw.strip())
    if match is None or not any(match.groups()):
        return None
    hours, minutes, seconds = (float(part) if part else 0.0 for part in match.groups())
    return hours * 3600.0 + minutes * 60.0 + seconds


def _parse_reset_seconds(raw: str, now: float) -> float | None:
    duration = _parse_duration_seconds(raw)
    if duration is not None:
        return duration
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    if value >= 1e12:
        return max(0.0, value / 1000.0 - now)
    if value >= 1e9:
        return max(0.0, value - now)
    return value


def quota_from_headers(
    headers: Mapping[str, str] | object,
    *,
    now: float | None = None,
) -> tuple[float | None, float | None]:
    """Parse advertised remaining quota. Missing stays None — never invent a sticker budget."""
    mapped = _header_map(headers)
    clock = time.time() if now is None else now
    for remaining_key, reset_keys in _REMAINING_RESET:
        raw_remaining = mapped.get(remaining_key)
        if raw_remaining is None or raw_remaining == "":
            continue
        remaining = _parse_remaining(raw_remaining)
        if remaining is None:
            continue
        reset: float | None = None
        for reset_key in reset_keys:
            raw_reset = mapped.get(reset_key)
            if not raw_reset:
                continue
            reset = _parse_reset_seconds(raw_reset, clock)
            if reset is not None:
                break
        if reset is None:
            raw_retry = mapped.get("retry-after")
            if raw_retry:
                reset = _parse_reset_seconds(raw_retry, clock)
        return remaining, reset
    return None, None


def get_json_response(
    url: str, *, api_key: str | None = None, timeout: float = 12.0
) -> JsonResponse | None:
    headers = {"User-Agent": "kerdoios/0.1", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_headers = _header_map(resp.headers)
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(body, dict):
        return None
    return JsonResponse(body=body, headers=raw_headers)


def get_json(url: str, *, api_key: str | None = None, timeout: float = 12.0) -> dict[str, Any] | None:
    fetched = get_json_response(url, api_key=api_key, timeout=timeout)
    return None if fetched is None else fetched.body
