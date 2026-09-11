"""Tiny JSON GET helper. Adapters must tolerate missing keys and network failure."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def get_json(url: str, *, api_key: str | None = None, timeout: float = 12.0) -> dict[str, Any] | None:
    headers = {"User-Agent": "kerdoios/0.1", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
