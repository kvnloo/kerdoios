"""Tokenomics watchdog: burn rate per origin provider.

Reuses observed JSONL. Catalog GET remaining=0 is unknown, not exhausted.
401/402/404/429 are never success. Listing is GET only; this module does
not call models.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .observed import Observation, classify_http, load_observations
from .providers.http import quota_from_headers
from .providers.openrouter import OPENROUTER_MODELS


def origin_of(obs: Observation) -> str:
    return obs.origin_provider or obs.provider


def remaining_for_burn(obs: Observation) -> float | None:
    """Catalog remaining is not burn remaining. None stays unknown, not 0."""
    if obs.remaining_source == "catalog":
        return None
    return obs.remaining_quota


@dataclass(frozen=True)
class OriginBurn:
    origin: str
    tokens: int
    cost: float
    remaining_quota: float | None
    exhausted: int = 0
    failed: int = 0
    success: int = 0


def _row_class(obs: Observation) -> str:
    if obs.http_status is not None:
        return classify_http(obs.http_status)
    if obs.error_class:
        return obs.error_class
    return "success" if obs.completed else "failed"


def burn_by_origin(observations: list[Observation]) -> dict[str, OriginBurn]:
    buckets: dict[str, list[Observation]] = defaultdict(list)
    for obs in observations:
        buckets[origin_of(obs)].append(obs)
    out: dict[str, OriginBurn] = {}
    for origin, rows in buckets.items():
        tokens = 0
        cost = 0.0
        known: list[float] = []
        unknown = False
        exhausted = 0
        failed = 0
        success = 0
        for obs in rows:
            tokens += (obs.input_tokens or 0) + (obs.output_tokens or 0)
            cost += obs.actual_cost
            remaining = remaining_for_burn(obs)
            if remaining is None:
                unknown = True
            else:
                known.append(remaining)
            cls = _row_class(obs)
            if cls == "exhausted":
                exhausted += 1
            elif cls == "failed":
                failed += 1
            else:
                success += 1
        out[origin] = OriginBurn(
            origin=origin,
            tokens=tokens,
            cost=cost,
            remaining_quota=None if unknown else (min(known) if known else None),
            exhausted=exhausted,
            failed=failed,
            success=success,
        )
    return out


def probe_openrouter_free(*, timeout: float = 12.0) -> dict:
    """Public catalog GET. No key. Catalog remaining=0 is unknown, not exhausted."""
    headers = {"User-Agent": "kerdoios/0.1", "Accept": "application/json"}
    req = urllib.request.Request(OPENROUTER_MODELS, headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            http_status = int(getattr(resp, "status", None) or resp.getcode())
            raw_headers = resp.headers
            body = json.loads(resp.read().decode())
        latency_ms = (time.perf_counter() - started) * 1000.0
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        status = int(exc.code)
        return {
            "ids": [],
            "count": 0,
            "latency_ms": latency_ms,
            "http_status": status,
            "remaining": None,
            "remaining_source": "catalog",
            "error_class": classify_http(status),
            "set": False,
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "ids": [],
            "count": 0,
            "latency_ms": latency_ms,
            "http_status": None,
            "remaining": None,
            "remaining_source": "catalog",
            "error_class": "failed",
            "set": False,
        }

    remaining, _reset = quota_from_headers(raw_headers)
    if remaining == 0.0:
        remaining = None
    ids: list[str] = []
    if isinstance(body, dict):
        for item in body.get("data") or []:
            mid = str(item.get("id") or "")
            if ":free" in mid.lower():
                ids.append(mid)
    error_class = classify_http(http_status)
    return {
        "ids": ids,
        "count": len(ids),
        "latency_ms": latency_ms,
        "http_status": http_status,
        "remaining": remaining,
        "remaining_source": "catalog",
        "error_class": error_class,
        "set": error_class == "success",
    }


def report(*, path: Path | None = None, live_free: bool = False) -> dict:
    observations = load_observations(path=path)
    burns = burn_by_origin(observations)
    payload = {
        "origin_burn": {origin: asdict(burn) for origin, burn in burns.items()},
        "openrouter_bucket": False,
    }
    if live_free:
        payload["live_openrouter_free"] = probe_openrouter_free()
    return payload
