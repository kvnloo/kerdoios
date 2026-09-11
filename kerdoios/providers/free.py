"""Free-offer predicates. Do not treat missing LiteLLM prices as free."""

from __future__ import annotations

from ..types import ResourceOffer


def is_free(offer: ResourceOffer) -> bool:
    """Keep :free ids, $0/$0 prices, remaining quota/credits, or local stock.

    LiteLLM rows with ``input_cost_per_token == 0`` are often missing prices,
    rerankers, or embeddings — not a real free chat model. Do not ingest them
    as free.
    """
    if offer.local:
        return True
    model = (offer.model or "").lower()
    if ":free" in model:
        return True
    econ = offer.economics
    if econ.remaining_free_quota > 0 or econ.remaining_credits > 0:
        return True
    if econ.input_token_price == 0 and econ.output_token_price == 0:
        return True
    return False


def filter_free(offers: list[ResourceOffer]) -> list[ResourceOffer]:
    return [offer for offer in offers if is_free(offer)]
