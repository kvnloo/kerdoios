"""Read-only capacity projection for z0intelligence and operator HUDs.

Kerdoios remains the owner of provider/resource capacity truth. This module
serializes existing ResourceOffer/QuotaState values; it does not perform
semantic routing or execute models.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from .cache import cache_dir
from .inventory import discover_all
from .types import ResourceOffer

SCHEMA = "kerdoios.capacity.v1"


def _quota(offer: ResourceOffer) -> dict[str, Any]:
    state = offer.economics.quota
    if state is not None:
        raw = state.to_dict()
        return {
            "dimensions": raw.get("dimensions") or {},
            "observed_at": raw.get("observed_at"),
        }

    # The legacy scalar is only projected when it is positively known useful.
    # A default 0.0 does not mean "exhausted"; without typed provenance it is
    # unknown and must stay unknown.
    legacy = offer.economics.remaining_free_quota
    if legacy > 0:
        return {"remaining_free_quota": legacy}
    return {}


def _reset_at(offer: ResourceOffer, *, now: float) -> float | None:
    state = offer.economics.quota
    if state is not None:
        resets = [
            dim.reset_at
            for dim in state.dimensions.values()
            if dim.reset_at is not None
        ]
        if resets:
            return min(resets)
    seconds = offer.economics.seconds_until_quota_reset
    if seconds is None:
        return None
    return now + max(0.0, float(seconds))


def _entry(offer: ResourceOffer, *, now: float) -> dict[str, Any]:
    price = {
        "input_token": offer.economics.input_token_price,
        "output_token": offer.economics.output_token_price,
        "hourly": offer.economics.hourly_price,
    }
    return {
        "offer_id": offer.id,
        "provider": offer.provider,
        "model": offer.model,
        "resource_type": offer.resource_type,
        "local": offer.local,
        "source": offer.source,
        "health": "unknown",
        "quota": _quota(offer),
        "price": price,
        "remaining_credits": (
            offer.economics.remaining_credits
            if offer.economics.remaining_credits > 0
            else None
        ),
        "reset_at": _reset_at(offer, now=now),
        "latency_p50_ms": offer.telemetry.latency_p50_ms,
        "latency_p95_ms": offer.telemetry.latency_p95_ms,
        "availability": offer.telemetry.availability,
        "observed_at": (
            offer.economics.quota.observed_at
            if offer.economics.quota is not None
            else None
        ),
    }


def build_projection(
    offers: Iterable[ResourceOffer],
    *,
    now: float | None = None,
) -> dict[str, Any]:
    clock = time.time() if now is None else float(now)
    entries = [_entry(offer, now=clock) for offer in offers]
    entries.sort(key=lambda row: (str(row.get("provider")), str(row.get("model"))))
    return {
        "schema": SCHEMA,
        "generated_at": clock,
        "entries": entries,
        "summary": {
            "offers": len(entries),
            "providers": len({str(row.get("provider")) for row in entries}),
        },
    }


def default_path() -> Path:
    return cache_dir() / "capacity.json"


def write_projection(payload: dict[str, Any], path: Path | None = None) -> Path:
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA}")
    dest = path or default_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="refresh provider discovery instead of preferring the existing cache",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the complete projection instead of only its path",
    )
    args = parser.parse_args(argv)

    # Free capacity is the first governed provider lane. discover_all() owns
    # provider discovery/cache semantics; this projection adds no second ledger.
    offers = discover_all(
        include_fixture=False,
        live=True,
        free_only=True,
        refresh=bool(args.refresh),
    )
    payload = build_projection(offers)
    path = write_projection(payload, Path(args.output).expanduser() if args.output else None)
    if args.json:
        print(json.dumps({**payload, "path": str(path)}, indent=2, sort_keys=True))
    else:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
