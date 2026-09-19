"""Free-offer predicates. Do not treat missing LiteLLM prices as free."""

from __future__ import annotations

from ..types import FREE_ACCESS_CLASSES, AccessClass, ResourceOffer


def is_free(offer: ResourceOffer) -> bool:
    """Keep :free ids, typed free access, remaining quota/credits, or local stock.

    Missing / None / default-0 catalog prices are unknown, not a free chat
    tier. LiteLLM's ``input_cost_per_token == 0`` rows are mostly missing
    prices, rerankers, or embeddings — do not ingest that dump as ``--free``.
    """
    if offer.local:
        return True
    if offer.access is not None and offer.access.access_class in FREE_ACCESS_CLASSES:
        # Public-catalog-only rows are not connected reservoirs.
        if offer.access.restriction == "public_catalog_only_unauthenticated":
            return False
        return True
    model = (offer.model or "").lower()
    if ":free" in model:
        return True
    econ = offer.economics
    if econ.remaining_free_quota > 0 or econ.remaining_credits > 0:
        return True
    return False


def filter_free(offers: list[ResourceOffer]) -> list[ResourceOffer]:
    return [offer for offer in offers if is_free(offer)]


def access_class_of(offer: ResourceOffer) -> AccessClass:
    if offer.access is not None:
        return offer.access.access_class
    if offer.local:
        return AccessClass.LOCAL
    if ":free" in (offer.model or "").lower():
        return AccessClass.ZERO_PRICE
    if offer.economics.remaining_credits > 0:
        return AccessClass.RECURRING_CREDIT
    if offer.economics.remaining_free_quota > 0:
        return AccessClass.FREE_TIER
    return AccessClass.UNKNOWN
