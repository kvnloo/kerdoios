"""OpenRouter catalog. The /models endpoint is public; a key is optional."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..types import CapabilityProfile, Capacity, Economics, ResourceOffer, Telemetry
from .base import ResourceProvider
from .http import get_json

OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"
_SNAPSHOT = Path(__file__).resolve().parent / "openrouter_free.snapshot.json"


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _supports_tools(item: dict[str, Any]) -> bool:
    params = item.get("supported_parameters") or []
    if isinstance(params, dict):
        keys = params.keys()
    else:
        keys = params
    return "tools" in keys or "tool_choice" in keys


def _offers_from_payload(payload: dict[str, Any]) -> list[ResourceOffer]:
    offers: list[ResourceOffer] = []
    for item in payload.get("data") or []:
        mid = str(item.get("id") or "")
        if not mid:
            continue
        pricing = item.get("pricing") or {}
        inp = _num(pricing.get("prompt"))
        out = _num(pricing.get("completion"))
        ctx = item.get("context_length")
        try:
            ctx_i = int(ctx) if ctx is not None else 8192
        except (TypeError, ValueError):
            ctx_i = 8192
        arch = item.get("architecture") or {}
        modality = str(arch.get("modality") or "")
        vision = 0.7 if "image" in modality.lower() else 0.0
        free = ":free" in mid.lower() or (inp == 0 and out == 0)
        has_tools = _supports_tools(item)
        offers.append(
            ResourceOffer(
                id=f"openrouter/{mid}",
                provider="openrouter",
                resource_type="llm",
                model=mid,
                local=False,
                capabilities=CapabilityProfile(
                    reasoning=0.68 if free else 0.80,
                    coding=0.70 if free else 0.82,
                    tool_use=0.85 if has_tools else 0.2,
                    vision=vision,
                    provenance="provider_claim",
                ),
                capacity=Capacity(concurrency=30 if free else 20, context_window=ctx_i),
                economics=Economics(
                    # Missing prompt/completion is unknown. `or 0.0` would
                    # turn that into a fake $0 chat tier.
                    input_token_price=inp,
                    output_token_price=out,
                    remaining_free_quota=50_000.0 if free else 0.0,
                ),
                telemetry=Telemetry(latency_p50_ms=900.0 if free else 600.0),
                tools=("*",) if has_tools else (),
                source="openrouter:/api/v1/models",
                confidence=0.7,
            )
        )
    return offers


def discover(*, api_key: str | None = None) -> list[ResourceOffer]:
    key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
    payload = get_json(OPENROUTER_MODELS, api_key=key)
    if not payload:
        return []
    return _offers_from_payload(payload)


def discover_snapshot() -> list[ResourceOffer]:
    try:
        payload = json.loads(_SNAPSHOT.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return [replace(o, source="openrouter:snapshot", confidence=0.55) for o in _offers_from_payload(payload)]


class OpenRouterProvider(ResourceProvider):
    name = "openrouter"

    def discover(self) -> list[ResourceOffer]:
        return discover()
