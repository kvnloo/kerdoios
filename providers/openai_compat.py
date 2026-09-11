"""Env-gated OpenAI-compatible catalogs (Groq, Cerebras). Return [] without a key."""

from __future__ import annotations

import os

from ..types import CapabilityProfile, Capacity, Economics, ResourceOffer, Telemetry
from .base import ResourceProvider
from .http import get_json_response, quota_from_headers

GROQ_MODELS = "https://api.groq.com/openai/v1/models"
CEREBRAS_MODELS = "https://api.cerebras.ai/v1/models"


def _catalog(
    *,
    provider: str,
    url: str,
    api_key: str | None,
    concurrency: int,
    latency_ms: float,
) -> list[ResourceOffer]:
    if not api_key:
        return []
    fetched = get_json_response(url, api_key=api_key)
    if not fetched:
        return []
    remaining, reset = quota_from_headers(fetched.headers)
    # Absent headers mean unknown remaining, not a hardcoded free-tier sticker.
    free_quota = remaining if remaining is not None else 0.0
    offers: list[ResourceOffer] = []
    for item in fetched.body.get("data") or []:
        mid = str(item.get("id") or "")
        if not mid:
            continue
        ctx = item.get("context_window") or item.get("context_length") or 128_000
        try:
            ctx_i = int(ctx)
        except (TypeError, ValueError):
            ctx_i = 128_000
        offers.append(
            ResourceOffer(
                id=f"{provider}/{mid}",
                provider=provider,
                resource_type="llm",
                model=mid,
                local=False,
                capabilities=CapabilityProfile(
                    reasoning=0.72,
                    coding=0.74,
                    tool_use=0.88,
                    provenance="provider_claim",
                ),
                capacity=Capacity(concurrency=concurrency, context_window=ctx_i),
                economics=Economics(
                    remaining_free_quota=free_quota,
                    seconds_until_quota_reset=reset,
                ),
                telemetry=Telemetry(latency_p50_ms=latency_ms),
                tools=("github", "*"),
                source=url,
                confidence=0.6,
            )
        )
    return offers


def discover_groq(*, api_key: str | None = None) -> list[ResourceOffer]:
    key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
    return _catalog(
        provider="groq",
        url=GROQ_MODELS,
        api_key=key,
        concurrency=20,
        latency_ms=280.0,
    )


def discover_cerebras(*, api_key: str | None = None) -> list[ResourceOffer]:
    key = api_key if api_key is not None else os.environ.get("CEREBRAS_API_KEY")
    return _catalog(
        provider="cerebras",
        url=CEREBRAS_MODELS,
        api_key=key,
        concurrency=20,
        latency_ms=190.0,
    )


class GroqProvider(ResourceProvider):
    name = "groq"

    def discover(self) -> list[ResourceOffer]:
        return discover_groq()


class CerebrasProvider(ResourceProvider):
    name = "cerebras"

    def discover(self) -> list[ResourceOffer]:
        return discover_cerebras()
