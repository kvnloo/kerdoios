"""Live Turbo inventory and placement only; execution belongs to the caller.

DEFAULT_ADMISSION is client policy shared per provider, NOT provider quota.
Callers must enforce it across models and supply cooldown providers in excluded.
Unknown quota is preserved in Economics.quota (the legacy scalar is not evidence).
Catalog membership establishes tier eligibility, not guaranteed remaining capacity.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, fields, replace
from typing import Iterable, Mapping

from .optimize import plan
from .providers.openai_compat import (
    CEREBRAS_MODELS,
    GROQ_MODELS,
    discover_cerebras,
    discover_groq,
    is_catalog_free_tier,
)
from .quota import QuotaState
from .types import (
    CapabilityProfile, Capacity, Economics, Mode, ResourceOffer, Telemetry,
    WorkRequirement,
)

OPENCODE_MODEL = "opencode/deepseek-v4.1-flash-free"
OPENCODE_COMMAND = ("/usr/bin/opencode", "models")
ENDPOINTS = {"groq": GROQ_MODELS, "cerebras": CEREBRAS_MODELS}
KEY_NAMES = {"groq": "GROQ_API_KEY", "cerebras": "CEREBRAS_API_KEY"}
DEFAULT_ADMISSION = {
    provider: {"concurrency": 1, "rpm": 6, "tokens_per_minute": None}
    for provider in ("groq", "cerebras", "opencode")
}


@dataclass(frozen=True)
class _CatalogFreeOffer(ResourceOffer):
    # Preserve explicit free_tier flags that the base ResourceOffer cannot carry.
    catalog_free_model: str | None = None


def _admit(offer: ResourceOffer) -> ResourceOffer:
    # Keep quota evidence separate: these numbers are local safety caps only.
    return replace(offer, capacity=replace(
        offer.capacity, concurrency=1, requests_per_minute=6,
        tokens_per_minute=None, tokens_per_day=None,
    ))


def discover_turbo_offers() -> list[ResourceOffer]:
    """GET keyed free-tier catalogs and list OpenCode models; never run inference.

    Missing keys/catalogs/binary produce no offers for that provider. OpenCode
    discovery requires no key and accepts only an exact, standalone model ID.
    """
    offers: list[ResourceOffer] = [
        _CatalogFreeOffer(
            **{field.name: getattr(offer, field.name) for field in fields(ResourceOffer)},
            catalog_free_model=offer.model,
        )
        for offer in (*discover_groq(free_only=True), *discover_cerebras(free_only=True))
    ]
    try:
        result = subprocess.run(
            list(OPENCODE_COMMAND), capture_output=True, text=True,
            timeout=20, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None and OPENCODE_MODEL in {
        line.strip() for line in result.stdout.splitlines()
    }:
        offers.append(ResourceOffer(
            id=OPENCODE_MODEL, provider="opencode", resource_type="llm",
            model=OPENCODE_MODEL.split("/", 1)[1], local=False,
            capabilities=CapabilityProfile(), capacity=Capacity(),
            economics=Economics(
                quota=QuotaState(provider="opencode", model=OPENCODE_MODEL.split("/", 1)[1]),
            ),
            telemetry=Telemetry(), source="/usr/bin/opencode models",
        ))
    # Replace legacy adapter latency claims with the common, unmeasured planner
    # prior. No benchmark or generation-rate observation occurred in discovery.
    return [_admit(replace(offer, telemetry=Telemetry())) for offer in offers]


def _positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _eligible(offer: ResourceOffer, coding: bool) -> bool:
    if offer.resource_type != "llm" or offer.local or offer.source == "fixture":
        return False
    if offer.economics.quota is not None and offer.economics.quota.is_exhausted():
        return False
    if coding:
        return (
            offer.provider == "opencode" and offer.id == OPENCODE_MODEL
            and offer.model == OPENCODE_MODEL.split("/", 1)[1]
            and offer.source == "/usr/bin/opencode models"
        )
    if offer.provider not in ENDPOINTS or offer.source != ENDPOINTS[offer.provider]:
        return False
    if offer.id != f"{offer.provider}/{offer.model}":
        return False
    if isinstance(offer, _CatalogFreeOffer) and offer.catalog_free_model == offer.model:
        return True
    # Also accept base offers whose tier evidence survives serialization.
    evidence = {
        "pricing": {
            "prompt": offer.economics.input_token_price,
            "completion": offer.economics.output_token_price,
        },
    }
    return is_catalog_free_tier(evidence, offer.model or "", offer.provider)


def choose_turbo_offer(
    offers: Iterable[ResourceOffer], *, coding: bool = False,
    excluded: Iterable[str] = (),
    measurements: Mapping[str, Mapping[str, object]] | None = None,
) -> ResourceOffer | None:
    """Return the original offer selected by the FAST single-worker planner.

    Measurements are keyed by offer.id: latency_ms and output_tok_s. Only finite
    positive observations override telemetry. Missing latency retains the planner
    prior (800ms for discovery); missing throughput remains None, with the base
    planner's concurrency heuristic, never a zero-valued observation. Inputs are
    not mutated. Unknown remaining is eligible for a conservative trial; known
    exhausted windows and caller-supplied provider cooldowns are excluded. This
    does not reserve quota, retry, execute, or guarantee a free request.
    """
    blocked = set(excluded)
    original = {
        offer.id: offer for offer in offers
        if offer.provider not in blocked and _eligible(offer, coding)
    }
    projected = []
    for offer in original.values():
        observed = (measurements or {}).get(offer.id, {})
        latency = _positive(observed.get("latency_ms"))
        throughput = _positive(observed.get("output_tok_s"))
        projected.append(_admit(replace(offer, telemetry=replace(
            offer.telemetry,
            latency_p50_ms=latency if latency is not None else offer.telemetry.latency_p50_ms,
            output_tok_s=throughput if throughput is not None else offer.telemetry.output_tok_s,
        ))))
    if not projected:
        return None
    result = plan(projected, WorkRequirement(mode=Mode.FAST))
    return original[result.placements[0].offer_id] if result.placements else None
