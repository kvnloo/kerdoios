"""Merge fixture and live ResourceOffers. Live rows win on matching provider+model.

Free inventory is cached under KERDOIOS_CACHE (default ~/.cache/kerdoios)
for six hours. --refresh skips a fresh cache. Snapshot is last resort.
Keyed Groq/Cerebras free-tier overlays the OpenRouter seed when those keys exist.
"""

from __future__ import annotations

from .cache import load_inventory_cache, save_inventory_cache
from .providers import local, openrouter
from .providers.fixture import fixture_offers
from .providers.free import filter_free
from .providers.openai_compat import discover_cerebras, discover_groq
from .types import ResourceOffer


def collect_live(*, free_only: bool = False) -> list[ResourceOffer]:
    live_rows: list[ResourceOffer] = []
    live_rows.extend(openrouter.discover())
    live_rows.extend(discover_groq(free_only=free_only))
    live_rows.extend(discover_cerebras(free_only=free_only))
    live_rows.extend(local.discover())
    return live_rows


def _has_openrouter_catalog(offers: list[ResourceOffer]) -> bool:
    return any(o.source.startswith("openrouter:") for o in offers)


def _merge(base: list[ResourceOffer], live_rows: list[ResourceOffer]) -> list[ResourceOffer]:
    if not live_rows:
        return base
    keys = {(o.provider, o.model) for o in live_rows}
    kept = [o for o in base if (o.provider, o.model) not in keys]
    return kept + live_rows


def _discover_free(*, refresh: bool = False) -> list[ResourceOffer]:
    if not refresh:
        cached = load_inventory_cache()
        if cached:
            return filter_free(cached)
    live_rows = collect_live(free_only=True)
    if live_rows:
        if not _has_openrouter_catalog(live_rows):
            # Keyed Groq/Cerebras must overlay the OpenRouter seed, not replace it.
            live_rows = openrouter.discover_snapshot() + live_rows
        offers = filter_free(live_rows)
        save_inventory_cache(offers)
        return offers
    stale = load_inventory_cache(ignore_ttl=True)
    if stale:
        return filter_free(stale)
    return filter_free(openrouter.discover_snapshot())


def discover_all(
    *,
    include_fixture: bool = True,
    live: bool = False,
    free_only: bool = False,
    refresh: bool = False,
) -> list[ResourceOffer]:
    if free_only:
        return _discover_free(refresh=refresh)
    live_rows: list[ResourceOffer] = []
    if live:
        live_rows = collect_live(free_only=False)
    offers = fixture_offers() if include_fixture else []
    offers = _merge(offers, live_rows)
    return offers
