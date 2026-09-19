"""Cohere trial/evaluation key — not production."""

from __future__ import annotations

from ..types import AccessClass, AccessEvidence, ResourceOffer
from .compat_transport import discover_openai_compat, env_key


def discover(*, free_only: bool = False) -> list[ResourceOffer]:
    key = env_key("COHERE_API_KEY")
    return discover_openai_compat(
        transport="cohere",
        url="https://api.cohere.com/compatibility/v1/models",
        api_key=key,
        access_class=AccessClass.TRIAL_CREDIT,
        evidence=AccessEvidence.PROVIDER_PLAN,
        free_only=free_only,
        concurrency=5,
        latency_ms=900.0,
        restriction="evaluation_only",
    )
