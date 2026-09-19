"""Cloudflare Workers AI — Neurons/day free allotment."""

from __future__ import annotations

import os

from ..types import (
    Access,
    AccessClass,
    AccessEvidence,
    CapabilityProfile,
    Capacity,
    Economics,
    ResourceOffer,
    Telemetry,
)
from .compat_transport import env_key
from .http import get_json_response


def discover(*, free_only: bool = False) -> list[ResourceOffer]:
    token = env_key("CLOUDFLARE_API_TOKEN", "CF_API_TOKEN")
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    if not token or not account:
        return []
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/models/search"
    fetched = get_json_response(url, api_key=token)
    if not fetched:
        # Fallback OpenAI-compat style path used by some Workers AI setups.
        compat = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1/models"
        fetched = get_json_response(compat, api_key=token)
        if not fetched:
            return []
    rows = fetched.body.get("result") or fetched.body.get("data") or []
    offers: list[ResourceOffer] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("name") or item.get("id") or "")
        if not mid:
            continue
        offers.append(
            ResourceOffer(
                id=f"cloudflare/{mid}",
                provider="cloudflare",
                resource_type="llm",
                model=mid,
                local=False,
                capabilities=CapabilityProfile(reasoning=0.65, coding=0.65, tool_use=0.5),
                capacity=Capacity(concurrency=5, context_window=int(item.get("context_window") or 8192)),
                economics=Economics(),
                telemetry=Telemetry(latency_p50_ms=900.0),
                tools=(),
                source=url,
                confidence=0.5,
                transport="cloudflare",
                creator=mid.split("/")[0] if "/" in mid else "cloudflare",
                access=Access(
                    access_class=AccessClass.FREE_TIER,
                    evidence=AccessEvidence.PROVIDER_PLAN,
                    remaining_known=False,
                ),
                freshness="fresh",
            )
        )
    if free_only:
        return offers
    return offers
