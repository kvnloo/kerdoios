"""Merge fixture and live ResourceOffers. Live rows win on matching provider+model."""

from __future__ import annotations

from .providers import local, openrouter
from .providers.fixture import fixture_offers
from .providers.openai_compat import discover_cerebras, discover_groq
from .types import ResourceOffer


def discover_all(*, include_fixture: bool = True, live: bool = False) -> list[ResourceOffer]:
    offers: list[ResourceOffer] = fixture_offers() if include_fixture else []
    if not live:
        return offers
    live_rows: list[ResourceOffer] = []
    live_rows.extend(openrouter.discover())
    live_rows.extend(discover_groq())
    live_rows.extend(discover_cerebras())
    live_rows.extend(local.discover())
    if not live_rows:
        return offers
    keys = {(o.provider, o.model) for o in live_rows}
    kept = [o for o in offers if (o.provider, o.model) not in keys]
    return kept + live_rows
