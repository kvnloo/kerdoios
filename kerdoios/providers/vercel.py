"""Vercel AI Gateway — zero-price rows vs recurring free credit are distinct."""

from __future__ import annotations

from ..types import AccessClass, AccessEvidence, ResourceOffer
from .compat_transport import discover_openai_compat, env_key


def discover(*, free_only: bool = False) -> list[ResourceOffer]:
    key = env_key("AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_API_KEY")
    offers = discover_openai_compat(
        transport="vercel",
        url="https://ai-gateway.vercel.sh/v1/models",
        api_key=key,
        access_class=AccessClass.RECURRING_CREDIT,
        evidence=AccessEvidence.PROVIDER_PLAN,
        free_only=False,
        concurrency=20,
        latency_ms=700.0,
        zero_price_as_zero=True,
    )
    if not free_only:
        return offers
    # Free inventory: keep literal zero-price rows and recurring-credit rows.
    # Paid-priced rows stay out until remaining credit is known from headers.
    kept: list[ResourceOffer] = []
    for offer in offers:
        access = offer.access
        if access is None:
            continue
        if access.access_class == AccessClass.ZERO_PRICE:
            kept.append(offer)
            continue
        if access.access_class == AccessClass.RECURRING_CREDIT:
            # Recurring credit is perishable capacity even when remaining unknown.
            kept.append(offer)
    return kept
