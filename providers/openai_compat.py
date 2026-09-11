"""Env-gated OpenAI-compatible catalogs (Groq, Cerebras). Return [] without a key."""

from __future__ import annotations

import os
from typing import Any

from ..types import CapabilityProfile, Capacity, Economics, ResourceOffer, Telemetry
from .base import ResourceProvider
from .http import get_json

GROQ_MODELS = "https://api.groq.com/openai/v1/models"
CEREBRAS_MODELS = "https://api.cerebras.ai/v1/models"

# Provider-documented free/trial *chat* ids. Groq/Cerebras /models rows usually
# omit :free and pricing, so a stamped remaining_free_quota would make whisper,
# guards, and enterprise rows look free. Do not copy these into the snapshot.
_FREE_TIER_CHAT_IDS: dict[str, frozenset[str]] = {
    "groq": frozenset(
        {
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "llama-3.3-70b-specdec",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3-32b",
            "qwen/qwen3.6-27b",
            "qwen/qwen3.8-27b",
            "groq/compound",
            "groq/compound-mini",
        }
    ),
    "cerebras": frozenset(
        {
            "gpt-oss-120b",
            "qwen-3.8-27b",
            "qwen-3-32b",
            "gemma-4-31b",
            "zai-glm-4.7",
            "llama-3.3-70b",
            "llama3.1-8b",
        }
    ),
}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _catalog_prices(item: dict[str, Any]) -> tuple[float | None, float | None]:
    pricing = item.get("pricing")
    if not isinstance(pricing, dict):
        return None, None
    inp = _num(pricing.get("prompt"))
    if inp is None:
        inp = _num(pricing.get("input"))
    out = _num(pricing.get("completion"))
    if out is None:
        out = _num(pricing.get("output"))
    return inp, out


def _advertised_zero_price(item: dict[str, Any]) -> bool:
    """True only when the catalog itself advertised both sides as 0, not missing prices."""
    pricing = item.get("pricing")
    if not isinstance(pricing, dict):
        return False
    has_in = "prompt" in pricing or "input" in pricing
    has_out = "completion" in pricing or "output" in pricing
    if not (has_in and has_out):
        return False
    inp, out = _catalog_prices(item)
    return inp == 0.0 and out == 0.0


def is_catalog_free_tier(item: dict[str, Any], mid: str, provider: str) -> bool:
    """Identify free-tier rows from catalog evidence. Never infer it from a quota we stamped."""
    if ":free" in mid.lower():
        return True
    if item.get("free") is True or item.get("free_tier") is True:
        return True
    tier = str(item.get("tier") or "").strip().lower().replace("-", "_")
    if tier in {"free", "free_tier", "free_trial"}:
        return True
    if _advertised_zero_price(item):
        return True
    return mid.lower() in _FREE_TIER_CHAT_IDS.get(provider, frozenset())


def _catalog(
    *,
    provider: str,
    url: str,
    api_key: str | None,
    concurrency: int,
    latency_ms: float,
    free_only: bool = False,
) -> list[ResourceOffer]:
    if not api_key:
        return []
    payload = get_json(url, api_key=api_key)
    if not payload:
        return []
    offers: list[ResourceOffer] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "")
        if not mid:
            continue
        free_tier = is_catalog_free_tier(item, mid, provider)
        if free_only and not free_tier:
            continue
        ctx = item.get("context_window") or item.get("context_length") or 128_000
        try:
            ctx_i = int(ctx)
        except (TypeError, ValueError):
            ctx_i = 128_000
        inp, out = _catalog_prices(item)
        if free_tier:
            # $0/$0 is the free signal; live remaining budget is issue #9 (headers).
            economics = Economics(
                input_token_price=inp if inp is not None else 0.0,
                output_token_price=out if out is not None else 0.0,
            )
        else:
            economics = Economics(
                input_token_price=inp or 0.0,
                output_token_price=out or 0.0,
            )
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
                economics=economics,
                telemetry=Telemetry(latency_p50_ms=latency_ms),
                tools=("github", "*"),
                source=url,
                confidence=0.6,
            )
        )
    return offers


def discover_groq(*, api_key: str | None = None, free_only: bool = False) -> list[ResourceOffer]:
    key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
    return _catalog(
        provider="groq",
        url=GROQ_MODELS,
        api_key=key,
        concurrency=20,
        latency_ms=280.0,
        free_only=free_only,
    )


def discover_cerebras(*, api_key: str | None = None, free_only: bool = False) -> list[ResourceOffer]:
    key = api_key if api_key is not None else os.environ.get("CEREBRAS_API_KEY")
    return _catalog(
        provider="cerebras",
        url=CEREBRAS_MODELS,
        api_key=key,
        concurrency=20,
        latency_ms=190.0,
        free_only=free_only,
    )


class GroqProvider(ResourceProvider):
    name = "groq"

    def discover(self) -> list[ResourceOffer]:
        return discover_groq()


class CerebrasProvider(ResourceProvider):
    name = "cerebras"

    def discover(self) -> list[ResourceOffer]:
        return discover_cerebras()
