"""Merge fixture and live ResourceOffers. Live rows win on matching transport+model.

Free inventory is cached under KERDOIOS_CACHE (default ~/.cache/kerdoios)
for six hours. --refresh skips a fresh cache. Snapshot is last resort.
Keyed Groq/Cerebras free-tier overlays the OpenRouter seed when those keys exist.
"""

from __future__ import annotations

from .cache import load_inventory_cache, save_inventory_cache
from .transport_cache import save_transport_cache
from .providers import local, openrouter
from .providers.fixture import fixture_offers
from .providers.free import filter_free, is_free
from .providers.openai_compat import discover_cerebras, discover_groq
from .types import ResourceOffer


def _offer_key(offer: ResourceOffer) -> tuple[str, str | None]:
    return (offer.transport or offer.provider, offer.model)


def collect_live(*, free_only: bool = False) -> list[ResourceOffer]:
    live_rows: list[ResourceOffer] = []
    live_rows.extend(openrouter.discover())
    live_rows.extend(discover_groq(free_only=free_only))
    live_rows.extend(discover_cerebras(free_only=free_only))
    live_rows.extend(local.discover())
    # Optional transports — fail open when modules/keys absent.
    for mod_name, discover_name in (
        ("kerdoios.providers.nous", "discover"),
        ("kerdoios.providers.vercel", "discover"),
        ("kerdoios.providers.gemini", "discover"),
        ("kerdoios.providers.mistral", "discover"),
        ("kerdoios.providers.cohere", "discover"),
        ("kerdoios.providers.cloudflare", "discover"),
        ("kerdoios.providers.nvidia_nim", "discover"),
        ("kerdoios.providers.huggingface", "discover"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[discover_name])
            fn = getattr(mod, discover_name)
            live_rows.extend(fn(free_only=free_only) if free_only else fn())
        except Exception:  # noqa: BLE001 — inventory stays fail-open
            continue
    return live_rows


def _has_openrouter_catalog(offers: list[ResourceOffer]) -> bool:
    return any((o.transport or o.provider) == "openrouter" or str(o.source).startswith("openrouter:") for o in offers)


def _merge(base: list[ResourceOffer], live_rows: list[ResourceOffer]) -> list[ResourceOffer]:
    if not live_rows:
        return base
    keys = {_offer_key(o) for o in live_rows}
    kept = [o for o in base if _offer_key(o) not in keys]
    return kept + live_rows


def _keep_free_inventory(offers: list[ResourceOffer]) -> list[ResourceOffer]:
    """OpenRouter/local use is_free; keyed Groq/Cerebras overlay is already free-tier filtered."""
    kept: list[ResourceOffer] = []
    for offer in offers:
        transport = offer.transport or offer.provider
        if transport in {"groq", "cerebras"}:
            kept.append(offer)
            continue
        if is_free(offer):
            kept.append(offer)
    return kept


def _discover_free(*, refresh: bool = False) -> list[ResourceOffer]:
    if not refresh:
        cached = load_inventory_cache()
        if cached:
            return _keep_free_inventory(cached)
    live_rows = collect_live(free_only=True)
    if live_rows:
        if not _has_openrouter_catalog(live_rows):
            # Keyed Groq/Cerebras must overlay the OpenRouter seed, not replace it.
            live_rows = openrouter.discover_snapshot() + live_rows
        offers = _keep_free_inventory(live_rows)
        save_inventory_cache(offers)
        by_transport: dict[str, list[ResourceOffer]] = {}
        for offer in offers:
            key = offer.transport or offer.provider
            by_transport.setdefault(key, []).append(offer)
        for transport, rows in by_transport.items():
            save_transport_cache(transport, rows, source="live", success=True)
        return offers
    stale = load_inventory_cache(ignore_ttl=True)
    if stale:
        # Label freshness without inventing new capacity.
        from dataclasses import replace

        return [
            replace(o, freshness="stale") if o.freshness != "stale" else o
            for o in _keep_free_inventory(stale)
        ]
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
