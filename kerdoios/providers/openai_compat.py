"""Env-gated OpenAI-compatible catalogs (Groq, Cerebras). Return [] without a key."""

from __future__ import annotations

import os
from typing import Any

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
from .base import ResourceProvider
from .http import get_json_response, quota_from_headers

GROQ_MODELS = "https://api.groq.com/openai/v1/models"
CEREBRAS_MODELS = "https://api.cerebras.ai/v1/models"
CEREBRAS_PUBLIC_MODELS = "https://api.cerebras.ai/public/v1/models"

# Current Groq Free-plan chat ids (dynamic catalog is authority for existence;
# this set is Free-plan evidence only — retired ids must not appear here).
_GROQ_FREE_PLAN_CHAT_IDS = frozenset(
    {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "groq/compound",
        "groq/compound-mini",
    }
)

# Non-chat / non-inference rows that must never look like free chat capacity.
_GROQ_NON_CHAT_HINTS = (
    "whisper",
    "tts",
    "guard",
    "prompt-guard",
    "distil-whisper",
    "playai",
)


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


def _groq_looks_non_chat(mid: str) -> bool:
    low = mid.lower()
    return any(hint in low for hint in _GROQ_NON_CHAT_HINTS)


def is_catalog_free_tier(item: dict[str, Any], mid: str, provider: str) -> bool:
    """Identify free-tier rows from catalog / Free-plan evidence. Never invent quota."""
    if ":free" in mid.lower():
        return True
    if item.get("free") is True or item.get("free_tier") is True:
        return True
    tier = str(item.get("tier") or "").strip().lower().replace("-", "_")
    if tier in {"free", "free_tier", "free_trial"}:
        return True
    if _advertised_zero_price(item):
        return True
    if provider == "groq":
        if _groq_looks_non_chat(mid):
            return False
        return mid.lower() in {m.lower() for m in _GROQ_FREE_PLAN_CHAT_IDS}
    if provider == "cerebras":
        # Missing prices are unknown, not free. Free-only discovery applies
        # Personal free-plan evidence separately.
        return False
    return False


def _access_for(
    *,
    provider: str,
    mid: str,
    item: dict[str, Any],
    free_tier: bool,
    remaining_known: bool,
) -> Access:
    if not free_tier:
        return Access(
            access_class=AccessClass.PAID if _catalog_prices(item) != (None, None) else AccessClass.UNKNOWN,
            evidence=AccessEvidence.PROVIDER_CATALOG,
            remaining_known=False,
        )
    tier = str(item.get("tier") or "").strip().lower().replace("-", "_")
    if tier in {"free_trial", "trial"}:
        cls = AccessClass.TRIAL_CREDIT
    elif ":free" in mid.lower() or _advertised_zero_price(item):
        cls = AccessClass.ZERO_PRICE
    else:
        cls = AccessClass.FREE_TIER
    return Access(
        access_class=cls,
        evidence=AccessEvidence.PROVIDER_PLAN if provider in {"groq", "cerebras"} else AccessEvidence.PROVIDER_CATALOG,
        remaining_known=remaining_known,
        restriction="evaluation_only" if provider == "cohere" else None,
    )


def _creator_from_model(mid: str, transport: str) -> str:
    if "/" in mid:
        return mid.split("/", 1)[0]
    return transport


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
    fetched = get_json_response(url, api_key=api_key)
    if not fetched:
        return []
    remaining, reset = quota_from_headers(fetched.headers)
    # Absent headers mean unknown remaining, not a hardcoded free-tier sticker.
    remaining_known = remaining is not None
    free_quota = remaining if remaining is not None else 0.0
    offers: list[ResourceOffer] = []
    for item in fetched.body.get("data") or []:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "")
        if not mid:
            continue
        free_tier = is_catalog_free_tier(item, mid, provider)
        inp_chk, out_chk = _catalog_prices(item)
        priced = inp_chk is not None and out_chk is not None and ((inp_chk or 0) > 0 or (out_chk or 0) > 0)
        if provider == "cerebras" and free_only and not free_tier and not priced:
            free_tier = True
        # Advertised remaining quota headers are account free-tier evidence for
        # unpriced chat rows (never invent a sticker; only use what headers say).
        if (
            provider in {"groq", "cerebras"}
            and not free_tier
            and remaining is not None
            and not priced
            and not (provider == "groq" and _groq_looks_non_chat(mid))
        ):
            free_tier = True
        if free_only and not free_tier:
            continue
        ctx = item.get("context_window") or item.get("context_length") or 128_000
        try:
            ctx_i = int(ctx)
        except (TypeError, ValueError):
            ctx_i = 128_000
        inp, out = _catalog_prices(item)
        economics = Economics(
            # Missing prompt/completion is unknown. `or 0.0` would
            # turn that into a fake $0 chat tier.
            input_token_price=inp,
            output_token_price=out,
            remaining_free_quota=free_quota if free_tier else 0.0,
            seconds_until_quota_reset=reset if free_tier else None,
        )
        access = _access_for(
            provider=provider,
            mid=mid,
            item=item,
            free_tier=free_tier,
            remaining_known=remaining_known and free_tier,
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
                transport=provider,
                creator=_creator_from_model(mid, provider),
                access=access,
                freshness="fresh",
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
    # Prefer authenticated catalog when keyed; fall back to public catalog for
    # existence evidence only (not usable without a Personal key).
    if key:
        return _catalog(
            provider="cerebras",
            url=CEREBRAS_MODELS,
            api_key=key,
            concurrency=20,
            latency_ms=190.0,
            free_only=free_only,
        )
    if free_only:
        fetched = get_json_response(CEREBRAS_PUBLIC_MODELS)
        if not fetched:
            return []
        # Public catalog proves model existence but is not a usable reservoir.
        offers: list[ResourceOffer] = []
        for item in fetched.body.get("data") or []:
            if not isinstance(item, dict):
                continue
            mid = str(item.get("id") or "")
            if not mid:
                continue
            offers.append(
                ResourceOffer(
                    id=f"cerebras/{mid}",
                    provider="cerebras",
                    resource_type="llm",
                    model=mid,
                    local=False,
                    capabilities=CapabilityProfile(
                        reasoning=0.72,
                        coding=0.74,
                        tool_use=0.88,
                        provenance="provider_claim",
                    ),
                    capacity=Capacity(concurrency=20, context_window=128_000),
                    economics=Economics(),
                    telemetry=Telemetry(latency_p50_ms=190.0),
                    tools=("github", "*"),
                    source=CEREBRAS_PUBLIC_MODELS,
                    confidence=0.4,
                    transport="cerebras",
                    creator=_creator_from_model(mid, "cerebras"),
                    access=Access(
                        access_class=AccessClass.FREE_TIER,
                        evidence=AccessEvidence.PROVIDER_CATALOG,
                        remaining_known=False,
                        restriction="public_catalog_only_unauthenticated",
                    ),
                    freshness="fresh",
                )
            )
        # Unauthenticated public rows are not connected capacity — return empty
        # for free inventory. Doctor exposes the public catalog separately.
        return []
    return []


class GroqProvider(ResourceProvider):
    name = "groq"

    def discover(self) -> list[ResourceOffer]:
        return discover_groq()


class CerebrasProvider(ResourceProvider):
    name = "cerebras"

    def discover(self) -> list[ResourceOffer]:
        return discover_cerebras()
