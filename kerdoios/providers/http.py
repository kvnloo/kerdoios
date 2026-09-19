"""Tiny JSON GET helper. Adapters must tolerate missing keys and network failure."""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
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


class FailureClass(str, Enum):
    AUTH_INVALID = "AUTH_INVALID"
    AUTH_FORBIDDEN = "AUTH_FORBIDDEN"
    TEAM_BALANCE_REQUIRED = "TEAM_BALANCE_REQUIRED"
    RATE_LIMITED = "RATE_LIMITED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    NETWORK = "NETWORK"
    DECODE = "DECODE"
    UNKNOWN = "UNKNOWN"
    WORKING = "WORKING"


@dataclass(frozen=True)
class JsonResponse:
    body: dict[str, Any]
    headers: dict[str, str]


@dataclass(frozen=True)
class FetchFailure:
    provider: str
    endpoint_class: str
    status: int | None
    failure_class: FailureClass
    message: str = ""
    sanitized_code: str | None = None
    retry_after: str | None = None
    rate_limit_headers: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "endpoint_class": self.endpoint_class,
            "status": self.status,
            "failure_class": self.failure_class.value,
            "message": self.message,
            "sanitized_code": self.sanitized_code,
            "retry_after": self.retry_after,
            "rate_limit_headers": dict(self.rate_limit_headers or {}),
        }


@dataclass(frozen=True)
class FetchResult:
    ok: JsonResponse | None = None
    failure: FetchFailure | None = None


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


def classify_http_status(status: int | None, *, provider: str = "") -> FailureClass:
    if status is None:
        return FailureClass.UNKNOWN
    if status == 401:
        return FailureClass.AUTH_INVALID
    if status == 402:
        # Cerebras documents Team keys without balance as 402.
        if provider == "cerebras":
            return FailureClass.TEAM_BALANCE_REQUIRED
        return FailureClass.TEAM_BALANCE_REQUIRED
    if status == 403:
        return FailureClass.AUTH_FORBIDDEN
    if status == 404:
        return FailureClass.MODEL_UNAVAILABLE
    if status == 429:
        return FailureClass.RATE_LIMITED
    if 500 <= status <= 599:
        return FailureClass.PROVIDER_ERROR
    return FailureClass.UNKNOWN


def _sanitize_error_body(raw: bytes | str | None) -> tuple[str, str | None]:
    if raw is None:
        return "", None
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    text = text.strip()
    if not text:
        return "", None
    code: str | None = None
    message = text[:240]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return message, None
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            message = str(err.get("message") or err.get("type") or message)[:240]
            code_raw = err.get("code") or err.get("type")
            if code_raw is not None:
                code = str(code_raw)[:80]
        elif isinstance(err, str):
            message = err[:240]
        else:
            message = str(payload.get("message") or message)[:240]
            if payload.get("code") is not None:
                code = str(payload.get("code"))[:80]
    # Never keep secret-looking substrings in diagnostic text.
    lowered = message.lower()
    for needle in ("bearer ", "authorization:", "api_key", "api-key", "sk-", "csk_"):
        if needle in lowered:
            message = "[redacted]"
            break
    return message, code


_RATE_HEADER_PREFIXES = ("x-ratelimit", "ratelimit", "retry-after", "remaining-quota", "x-remaining")


def _rate_headers(mapped: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in mapped.items():
        if key == "retry-after" or any(key.startswith(p) for p in _RATE_HEADER_PREFIXES):
            out[key] = value
    return out


def get_json_result(
    url: str,
    *,
    api_key: str | None = None,
    timeout: float = 12.0,
    provider: str = "",
    endpoint_class: str = "catalog",
) -> FetchResult:
    """Typed fetch. Never persists Authorization / API keys."""
    headers = {"User-Agent": "kerdoios/0.1", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_headers = _header_map(resp.headers)
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        err_headers = _header_map(getattr(exc, "headers", {}) or {})
        try:
            payload = exc.read()
        except Exception:  # noqa: BLE001 — best-effort body
            payload = b""
        message, code = _sanitize_error_body(payload)
        if not message:
            message = str(exc.reason or exc)[:240]
        return FetchResult(
            failure=FetchFailure(
                provider=provider,
                endpoint_class=endpoint_class,
                status=int(exc.code) if exc.code is not None else None,
                failure_class=classify_http_status(
                    int(exc.code) if exc.code is not None else None,
                    provider=provider,
                ),
                message=message,
                sanitized_code=code,
                retry_after=err_headers.get("retry-after"),
                rate_limit_headers=_rate_headers(err_headers) or None,
            )
        )
    except urllib.error.URLError as exc:
        return FetchResult(
            failure=FetchFailure(
                provider=provider,
                endpoint_class=endpoint_class,
                status=None,
                failure_class=FailureClass.NETWORK,
                message=str(getattr(exc, "reason", exc))[:240],
            )
        )
    except TimeoutError as exc:
        return FetchResult(
            failure=FetchFailure(
                provider=provider,
                endpoint_class=endpoint_class,
                status=None,
                failure_class=FailureClass.NETWORK,
                message=str(exc)[:240] or "timeout",
            )
        )
    except (json.JSONDecodeError, OSError) as exc:
        return FetchResult(
            failure=FetchFailure(
                provider=provider,
                endpoint_class=endpoint_class,
                status=None,
                failure_class=FailureClass.DECODE if isinstance(exc, json.JSONDecodeError) else FailureClass.NETWORK,
                message=str(exc)[:240],
            )
        )
    if not isinstance(body, dict):
        return FetchResult(
            failure=FetchFailure(
                provider=provider,
                endpoint_class=endpoint_class,
                status=None,
                failure_class=FailureClass.DECODE,
                message="response body was not a JSON object",
            )
        )
    return FetchResult(ok=JsonResponse(body=body, headers=raw_headers))


def get_json_response(
    url: str, *, api_key: str | None = None, timeout: float = 12.0
) -> JsonResponse | None:
    """Quiet fail-open wrapper for ordinary inventory adapters."""
    result = get_json_result(url, api_key=api_key, timeout=timeout)
    return result.ok


def get_json(url: str, *, api_key: str | None = None, timeout: float = 12.0) -> dict[str, Any] | None:
    fetched = get_json_response(url, api_key=api_key, timeout=timeout)
    return None if fetched is None else fetched.body
