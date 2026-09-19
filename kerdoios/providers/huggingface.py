"""Hugging Face Inference Providers — small recurring monthly credit."""

from __future__ import annotations

from ..types import AccessClass, AccessEvidence, ResourceOffer
from .compat_transport import discover_openai_compat, env_key


def discover(*, free_only: bool = False) -> list[ResourceOffer]:
    key = env_key("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN")
    return discover_openai_compat(
        transport="huggingface",
        url="https://router.huggingface.co/v1/models",
        api_key=key,
        access_class=AccessClass.RECURRING_CREDIT,
        evidence=AccessEvidence.PROVIDER_PLAN,
        free_only=free_only,
        concurrency=5,
        latency_ms=1200.0,
    )
