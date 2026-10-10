"""Plan a free-model probe fanout. This module does not call a model.

402 and 429 are immediate sidesteps to the next free provider. Cursor and other
paid parents are never a slot.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .providers.free import is_free
from .types import ResourceOffer

SCHEMA = "kerdoios.free_fanout.v1"
SIDESTEP_HTTP = (402, 429)
BLOCKED_PROVIDERS = frozenset(
    {"cursor", "paid-api", "openai-codex", "xai", "xai-oauth"}
)
# Operator sidestep order. Vercel is a gateway credit lane, not a validated $0 claim.
PREFERRED: tuple[tuple[str, str], ...] = (
    ("nous", "inclusionai/ling-3.0-flash-sante:free"),
    ("vercel", "openai/gpt-oss-20b"),
    ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
    ("groq", "openai/gpt-oss-20b"),
    ("nvidia", "openai/gpt-oss-20b"),
)


def blocked(provider: str | None) -> bool:
    name = (provider or "").strip().lower()
    return name in BLOCKED_PROVIDERS or any(part.startswith("cursor") for part in re.split("[^a-z0-9]+", name))


def _cursor_route(route: str | None) -> bool:
    return any(part.strip().startswith("cursor") for part in (route or "").lower().split("/"))


def _free_without_credits(offer: ResourceOffer) -> bool:
    # A credit balance is money already paid: spending it is not a $0 probe.
    return is_free(replace(offer, economics=replace(offer.economics, remaining_credits=0.0)))


def _slot(provider: str, model: str, *, source: str) -> dict[str, Any]:
    return {"provider": provider, "model": model, "source": source}


def _from_offers(offers: list[ResourceOffer]) -> list[dict[str, Any]]:
    by_provider: dict[str, ResourceOffer] = {}
    for offer in offers:
        if offer.resource_type != "llm" or not offer.model or blocked(offer.provider):
            continue
        if _cursor_route(offer.model) or _cursor_route(offer.id) or not _free_without_credits(offer):
            continue
        by_provider.setdefault(offer.provider, offer)
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for provider, model in PREFERRED:
        offer = by_provider.get(provider)
        if offer is None or not offer.model:
            continue
        ordered.append(_slot(provider, offer.model, source="inventory"))
        seen.add(provider)
    for provider, offer in by_provider.items():
        if provider in seen or not offer.model:
            continue
        ordered.append(_slot(provider, offer.model, source="inventory"))
    return ordered


def _static_slots() -> list[dict[str, Any]]:
    return [_slot(provider, model, source="static_sidestep_table") for provider, model in PREFERRED]


def build_fanout(offers: list[ResourceOffer] | None = None, *, workers: int = 5) -> dict[str, Any]:
    if workers < 1 or workers > 8:
        raise ValueError("workers must be an integer from 1 to 8")
    supplied = list(offers or [])
    pool = _from_offers(supplied) if supplied else _static_slots()
    chosen = pool[:workers]
    slots = []
    for index, primary in enumerate(chosen):
        chain = [primary] + [item for item in pool if item["provider"] != primary["provider"]]
        slots.append(
            {
                "slot": index,
                "provider": primary["provider"],
                "model": primary["model"],
                "source": primary["source"],
                "sidestep": chain[1:],
            }
        )
    plan = {
        "schema": SCHEMA,
        "workers": len(slots),
        "requested_workers": workers,
        "sidestep_http": list(SIDESTEP_HTTP),
        "blocked_providers": sorted(BLOCKED_PROVIDERS),
        "cursor_allowed": False,
        "slots": slots,
        "note": "Plan only. z0intelligence executes. 402/429 jump to the next free provider in the same call.",
    }
    if not slots:
        plan["reason"] = "no_free_offer_in_inventory"
    return plan
