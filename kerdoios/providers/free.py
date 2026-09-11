"""Free-offer predicates. Do not treat missing LiteLLM prices as free."""

from __future__ import annotations

from ..types import ResourceOffer


def is_free(offer: ResourceOffer) -> bool:
    """Keep :free ids, remaining quota/credits, or local stock.

    Missing / None / default-0 catalog prices are unknown, not a free chat
    tier. LiteLLM's ``input_cost_per_token == 0`` rows are mostly missing
    prices, rerankers, or embeddings — do not ingest that dump as ``--free``.
    """
    if offer.local:
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
