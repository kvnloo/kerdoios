"""Google Gemini API free tier (OpenAI-compatible endpoint when keyed)."""

from __future__ import annotations

from ..types import AccessClass, AccessEvidence, ResourceOffer
from .compat_transport import discover_openai_compat, env_key


def discover(*, free_only: bool = False) -> list[ResourceOffer]:
    key = env_key("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY")
    return discover_openai_compat(
        transport="gemini",
        url="https://generativelanguage.googleapis.com/v1beta/openai/models",
        api_key=key,
        access_class=AccessClass.FREE_TIER,
        evidence=AccessEvidence.PROVIDER_PLAN,
        free_only=free_only,
        concurrency=10,
        latency_ms=800.0,
    )
