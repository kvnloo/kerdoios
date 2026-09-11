"""Merge fixture and live ResourceOffers. Live rows win on matching provider+model."""

from __future__ import annotations

from .providers import local, openrouter
from .providers.fixture import fixture_offers
from .providers.free import filter_free
from .providers.openai_compat import discover_cerebras, discover_groq
from .types import ResourceOffer


def collect_live() -> list[ResourceOffer]:
    live_rows: list[ResourceOffer] = []
    live_rows.extend(openrouter.discover())
    live_rows.extend(discover_groq())
    live_rows.extend(discover_cerebras())
    live_rows.extend(local.discover())
    return live_rows


def _merge(base: list[ResourceOffer], live_rows: list[ResourceOffer]) -> list[ResourceOffer]:
    if not live_rows:
        return base
    keys = {(o.provider, o.model) for o in live_rows}
    kept = [o for o in base if (o.provider, o.model) not in keys]
    return kept + live_rows


def discover_all(
    *,
    include_fixture: bool = True,
    live: bool = False,
    free_only: bool = False,
) -> list[ResourceOffer]:
    live_rows: list[ResourceOffer] = []
    if live or free_only:
        live_rows = collect_live()
    if free_only and not live_rows:
        live_rows = openrouter.discover_snapshot()
    offers = fixture_offers() if include_fixture else []
    offers = _merge(offers, live_rows)
    if free_only:
        return filter_free(offers)
    return offers
