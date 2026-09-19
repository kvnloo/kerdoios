"""Nous Portal catalog via Hermes credential resolver — no OAuth copy."""

from __future__ import annotations

from typing import Any

from ..types import AccessClass, AccessEvidence, ResourceOffer
from .compat_transport import discover_openai_compat


def _hermes_runtime() -> dict[str, Any] | None:
    try:
        from hermes_cli.auth_nous import resolve_nous_runtime_credentials  # type: ignore — optional import
    except Exception:  # noqa: BLE001 — fail-open inventory/doctor
        return None
    try:
        return resolve_nous_runtime_credentials()
    except Exception:  # noqa: BLE001 — Kerdoios must not own OAuth failures
        return None


def discover(*, free_only: bool = False) -> list[ResourceOffer]:
    runtime = _hermes_runtime()
    if not runtime:
        return []
    api_key = runtime.get("api_key") or runtime.get("access_token")
    base = str(runtime.get("base_url") or "https://inference-api.nousresearch.com/v1").rstrip("/")
    if not api_key:
        return []
    return discover_openai_compat(
        transport="nous",
        url=f"{base}/models",
        api_key=str(api_key),
        access_class=AccessClass.FREE_TIER,
        evidence=AccessEvidence.ACCOUNT_ENTITLEMENT,
        free_only=free_only,
        concurrency=8,
        latency_ms=600.0,
    )
