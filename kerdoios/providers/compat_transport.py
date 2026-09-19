"""Shared helpers for OpenAI-compatible free-capacity transports."""

from __future__ import annotations

import os
from typing import Any, Callable

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
from .http import get_json_response, quota_from_headers


def env_key(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def creator_from_model(mid: str, transport: str) -> str:
    if "/" in mid:
        return mid.split("/", 1)[0]
    return transport


def discover_openai_compat(
    *,
    transport: str,
    url: str,
    api_key: str | None,
    access_class: AccessClass,
    evidence: AccessEvidence = AccessEvidence.PROVIDER_PLAN,
    free_only: bool = False,
    concurrency: int = 10,
    latency_ms: float = 700.0,
    restriction: str | None = None,
    model_filter: Callable[[dict[str, Any], str], bool] | None = None,
    zero_price_as_zero: bool = False,
) -> list[ResourceOffer]:
    if not api_key:
        return []
    fetched = get_json_response(url, api_key=api_key)
    if not fetched:
        return []
    remaining, reset = quota_from_headers(fetched.headers)
    remaining_known = remaining is not None
    offers: list[ResourceOffer] = []
    for item in fetched.body.get("data") or []:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or item.get("name") or "")
        if not mid:
            continue
        if model_filter is not None and not model_filter(item, mid):
            continue
        pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
        inp = pricing.get("prompt") or pricing.get("input")
        out = pricing.get("completion") or pricing.get("output")
        try:
            inp_f = float(inp) if inp is not None else None
        except (TypeError, ValueError):
            inp_f = None
        try:
            out_f = float(out) if out is not None else None
        except (TypeError, ValueError):
            out_f = None
        is_zero = inp_f == 0.0 and out_f == 0.0 and inp_f is not None and out_f is not None
        row_class = AccessClass.ZERO_PRICE if (zero_price_as_zero and is_zero) else access_class
        if free_only and row_class not in {
            AccessClass.LOCAL,
            AccessClass.ZERO_PRICE,
            AccessClass.FREE_TIER,
            AccessClass.RECURRING_CREDIT,
            AccessClass.TRIAL_CREDIT,
            AccessClass.PROMOTIONAL,
        }:
            # Still include when transport is known free-tier and prices unknown.
            if access_class not in {
                AccessClass.FREE_TIER,
                AccessClass.RECURRING_CREDIT,
                AccessClass.TRIAL_CREDIT,
                AccessClass.ZERO_PRICE,
                AccessClass.PROMOTIONAL,
            }:
                continue
            if inp_f is not None and out_f is not None and (inp_f > 0 or out_f > 0) and not is_zero:
                continue
        ctx = item.get("context_window") or item.get("context_length") or item.get("max_tokens") or 128_000
        try:
            ctx_i = int(ctx)
        except (TypeError, ValueError):
            ctx_i = 128_000
        offers.append(
            ResourceOffer(
                id=f"{transport}/{mid}",
                provider=transport,
                resource_type="llm",
                model=mid,
                local=False,
                capabilities=CapabilityProfile(
                    reasoning=0.7,
                    coding=0.7,
                    tool_use=0.7,
                    provenance="provider_claim",
                ),
                capacity=Capacity(concurrency=concurrency, context_window=ctx_i),
                economics=Economics(
                    input_token_price=inp_f,
                    output_token_price=out_f,
                    remaining_free_quota=remaining if remaining is not None else 0.0,
                    remaining_credits=0.0,
                    seconds_until_quota_reset=reset,
                ),
                telemetry=Telemetry(latency_p50_ms=latency_ms),
                tools=("*",),
                source=url,
                confidence=0.55,
                transport=transport,
                creator=creator_from_model(mid, transport),
                access=Access(
                    access_class=row_class,
                    evidence=evidence,
                    remaining_known=remaining_known,
                    restriction=restriction,
                ),
                freshness="fresh",
            )
        )
    return offers
