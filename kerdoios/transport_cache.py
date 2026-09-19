"""Per-transport inventory cache. Failed refresh must not delete a valid stale snapshot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .cache import CACHE_TTL, _offer_from_dict, _strip_secrets, cache_dir, utcnow
from .types import ResourceOffer


@dataclass(frozen=True)
class TransportCacheMeta:
    transport: str
    fetched_at: str
    expires_at: str
    source: str
    success: bool
    catalog_fingerprint: str
    fresh: bool
    stale: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "fetched_at": self.fetched_at,
            "expires_at": self.expires_at,
            "source": self.source,
            "success": self.success,
            "catalog_fingerprint": self.catalog_fingerprint,
            "fresh": self.fresh,
            "stale": self.stale,
        }


def transport_cache_path(transport: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in transport)
    return cache_dir() / "transports" / f"{safe}.json"


def fingerprint_offers(offers: list[ResourceOffer]) -> str:
    ids = sorted(o.id for o in offers)
    return hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]


def save_transport_cache(
    transport: str,
    offers: list[ResourceOffer],
    *,
    source: str,
    success: bool = True,
    now: datetime | None = None,
    ttl: timedelta | None = None,
) -> Path:
    moment = (now or utcnow()).astimezone(timezone.utc)
    horizon = moment + (ttl or CACHE_TTL)
    payload = _strip_secrets(
        {
            "transport": transport,
            "fetched_at": moment.isoformat(),
            "expires_at": horizon.isoformat(),
            "source": source,
            "success": success,
            "catalog_fingerprint": fingerprint_offers(offers),
            "offers": [o.to_dict() for o in offers],
        }
    )
    path = transport_cache_path(transport)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a successful snapshot with a failed empty refresh.
    if not success and path.exists():
        return path
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_transport_cache(
    transport: str,
    *,
    now: datetime | None = None,
    ignore_ttl: bool = False,
) -> tuple[list[ResourceOffer], TransportCacheMeta] | None:
    path = transport_cache_path(transport)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    fetched_at = str(payload.get("fetched_at") or "")
    expires_at = str(payload.get("expires_at") or "")
    try:
        datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    moment = now or utcnow()
    fresh = moment <= expires
    if not ignore_ttl and not fresh:
        # Return labeled stale rather than deleting.
        pass
    rows = payload.get("offers") or []
    offers: list[ResourceOffer] = []
    for item in rows:
        if isinstance(item, dict):
            offer = _offer_from_dict(item)
            if offer is not None:
                offers.append(offer)
    if not offers and not payload.get("success"):
        return None
    meta = TransportCacheMeta(
        transport=str(payload.get("transport") or transport),
        fetched_at=fetched_at,
        expires_at=expires_at,
        source=str(payload.get("source") or "cache"),
        success=bool(payload.get("success")),
        catalog_fingerprint=str(payload.get("catalog_fingerprint") or ""),
        fresh=fresh,
        stale=not fresh,
    )
    if not ignore_ttl and not fresh and not offers:
        return None
    return offers, meta
