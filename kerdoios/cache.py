"""On-disk inventory cache. No secrets. Directory from KERDOIOS_CACHE."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .types import CapabilityProfile, Capacity, Economics, ResourceOffer, Telemetry

CACHE_TTL = timedelta(hours=6)
_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "secret",
    "password",
    "bearer",
    "access_token",
    "auth_token",
    "openrouter_api_key",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def cache_dir() -> Path:
    override = os.environ.get("KERDOIOS_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "kerdoios"


def cache_path() -> Path:
    return cache_dir() / "inventory.json"


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return lowered in _SECRET_KEYS or lowered.endswith("_api_key")


def _strip_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_secrets(v) for k, v in value.items() if not _is_secret_key(str(k))}
    if isinstance(value, list):
        return [_strip_secrets(item) for item in value]
    return value


def _offer_from_dict(raw: dict[str, Any]) -> ResourceOffer | None:
    try:
        caps_raw = raw.get("capabilities") or {}
        if not isinstance(caps_raw, dict):
            return None
        tools = raw.get("tools") or ()
        privacy = raw.get("privacy_ok") or ("public", "confidential")
        model = raw.get("model")
        return ResourceOffer(
            id=str(raw["id"]),
            provider=str(raw["provider"]),
            resource_type=raw.get("resource_type") or "llm",  # type: ignore[arg-type]  # catalog json is untyped; default to llm
            model=str(model) if model is not None else None,
            local=bool(raw.get("local")),
            capabilities=CapabilityProfile(**caps_raw),
            capacity=Capacity(**(raw.get("capacity") or {})),
            economics=Economics(**(raw.get("economics") or {})),
            telemetry=Telemetry(**(raw.get("telemetry") or {})),
            tools=tuple(tools),
            privacy_ok=tuple(privacy),
            source=str(raw.get("source") or "cache"),
            confidence=float(raw.get("confidence") or 0.5),
        )
    except (TypeError, ValueError, KeyError):
        return None


def save_inventory_cache(offers: list[ResourceOffer], *, now: datetime | None = None) -> Path:
    stamp = (now or utcnow()).astimezone(timezone.utc).isoformat()
    payload = _strip_secrets(
        {
            "saved_at": stamp,
            "offers": [offer.to_dict() for offer in offers],
        }
    )
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _parse_saved_at(raw: str) -> datetime | None:
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_inventory_cache(
    *,
    now: datetime | None = None,
    ignore_ttl: bool = False,
) -> list[ResourceOffer] | None:
    path = cache_path()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    saved_at = _parse_saved_at(str(payload.get("saved_at") or ""))
    if saved_at is None:
        return None
    moment = now or utcnow()
    if not ignore_ttl and moment - saved_at > CACHE_TTL:
        return None
    rows = payload.get("offers")
    if not isinstance(rows, list):
        return None
    offers: list[ResourceOffer] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        offer = _offer_from_dict(item)
        if offer is not None:
            offers.append(offer)
    if not offers:
        return None
    return offers
