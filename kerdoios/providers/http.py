"""Tiny JSON GET helper. Adapters must tolerate missing keys and network failure."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from ..quota import (
    header_map,
    legacy_projection,
    quota_state_from_headers,
)


@dataclass(frozen=True)
class JsonResponse:
    body: dict[str, Any]
    headers: dict[str, str]


def quota_from_headers(
    headers: Mapping[str, str] | object,
    *,
    now: float | None = None,
) -> tuple[float | None, float | None]:
    """Legacy scalar projection of the dimensional quota.

    Prefer :func:`kerdoios.quota.quota_state_from_headers`, which keeps every
    window and its provenance. This wrapper only exists so older callers get
    the same ``(remaining, reset_seconds)`` pair they always did, chosen by
    ``legacy_projection``'s day-broad-first order. Missing stays None.
    """
    clock = time.time() if now is None else now
    state = quota_state_from_headers(headers, now=clock)
    return legacy_projection(state, now=clock)


def get_json_response(
    url: str, *, api_key: str | None = None, timeout: float = 12.0
) -> JsonResponse | None:
    headers = {"User-Agent": "kerdoios/0.1", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_headers = header_map(resp.headers)
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(body, dict):
        return None
    return JsonResponse(body=body, headers=raw_headers)


def get_json(url: str, *, api_key: str | None = None, timeout: float = 12.0) -> dict[str, Any] | None:
    fetched = get_json_response(url, api_key=api_key, timeout=timeout)
    return None if fetched is None else fetched.body
