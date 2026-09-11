"""OpenAI-compatible local endpoints (Ollama, LM Studio, llama.cpp)."""

from __future__ import annotations

import os

from ..types import CapabilityProfile, Capacity, Economics, ResourceOffer, Telemetry
from .base import ResourceProvider
from .http import get_json

DEFAULT_BASES = ("http://127.0.0.1:11434/v1", "http://127.0.0.1:1234/v1")


def _bases() -> list[str]:
    env = os.environ.get("KERDOIOS_LOCAL_BASE_URL")
    bases: list[str] = []
    if env:
        bases.append(env.rstrip("/"))
    bases.extend(DEFAULT_BASES)
    return bases


def discover() -> list[ResourceOffer]:
    offers: list[ResourceOffer] = []
    seen: set[str] = set()
    for base in _bases():
        payload = get_json(base + "/models", timeout=3.0)
        if not payload:
            continue
        for item in payload.get("data") or []:
            mid = str(item.get("id") or "")
            if not mid:
                continue
            oid = f"local/{mid}"
            if oid in seen:
                continue
            seen.add(oid)
            offers.append(
                ResourceOffer(
                    id=oid,
                    provider="local",
                    resource_type="llm",
                    model=mid,
                    local=True,
                    capabilities=CapabilityProfile(
                        reasoning=0.70,
                        coding=0.72,
                        tool_use=0.80,
                        provenance="provider_claim",
                    ),
                    capacity=Capacity(concurrency=4, context_window=32768),
                    economics=Economics(),
                    telemetry=Telemetry(latency_p50_ms=800.0, failure_rate=0.01, availability=0.99),
                    tools=("github", "*"),
                    privacy_ok=("public", "confidential", "local_only"),
                    source=base,
                    confidence=0.6,
                )
            )
    return offers


class LocalProvider(ResourceProvider):
    name = "local"

    def discover(self) -> list[ResourceOffer]:
        return discover()
