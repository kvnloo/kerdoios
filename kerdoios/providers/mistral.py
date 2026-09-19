"""Mistral free/credits plan via OpenAI-compatible API."""

from __future__ import annotations

from ..types import AccessClass, AccessEvidence, ResourceOffer
from .compat_transport import discover_openai_compat, env_key


def discover(*, free_only: bool = False) -> list[ResourceOffer]:
    key = env_key("MISTRAL_API_KEY")
    return discover_openai_compat(
        transport="mistral",
        url="https://api.mistral.ai/v1/models",
        api_key=key,
        access_class=AccessClass.RECURRING_CREDIT,
        evidence=AccessEvidence.PROVIDER_PLAN,
        free_only=free_only,
        concurrency=10,
        latency_ms=700.0,
    )
