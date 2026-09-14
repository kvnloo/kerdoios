"""Pull scoring inputs toward observed reality.

score.py's ScoredOffer.fitness is built entirely from claimed/fixture
numbers (CapabilityProfile, Economics, Telemetry). This module produces a
new list of ResourceOffer with those fields shrunk toward what actually
happened, using aggregate() from observed.py. It never mutates the offers
passed in (ResourceOffer is frozen); it returns replacements.

Shrinkage: below MIN_OBSERVATIONS, keep the prior untouched (return the
offer unchanged). At or above the floor, blend linearly with weight
capped at 0.8 so a long observed history still leaves room for the prior
to matter (an offer that was reliable for months should not flip to
worthless from one bad week) and provenance is marked observed_execution
so operators can see the fitness terms did move.

The economics blend divides effective cost by the observed completion
rate: this is the "value/cost" fix — cost per completed task, not cost
per token. A model with a cheap sticker price but a low completion rate
gets a higher effective marginal cost here, which is exactly the AgentWeb
finding (opus-5: high nominal capability, $87/completed task, worst
$/task in the benchmark) that a token-price-only view cannot see.
"""

from __future__ import annotations

from dataclasses import replace

from .observed import MIN_OBSERVATIONS, OutcomeStats, aggregate, load_observations
from .types import ResourceOffer

# Cap how far a single blend can move an offer, even with a huge observed
# sample. Keeps kerdoios from fully trusting a noisy or adversarial log.
MAX_BLEND_WEIGHT = 0.8


def _blend_weight(n: int) -> float:
    if n < MIN_OBSERVATIONS:
        return 0.0
    # Approach MAX_BLEND_WEIGHT asymptotically; n=MIN_OBSERVATIONS starts
    # near 0.3 * MAX_BLEND_WEIGHT, n=4x floor is already close to the cap.
    raw = 1.0 - (MIN_OBSERVATIONS / float(n))
    return min(MAX_BLEND_WEIGHT, max(0.3, raw) * MAX_BLEND_WEIGHT)


def blend_offer(offer: ResourceOffer, stats: OutcomeStats) -> ResourceOffer:
    if not stats.trusted:
        return offer
    weight = _blend_weight(stats.n)
    if weight <= 0.0:
        return offer

    completion_rate = max(stats.decayed_completion_rate, 1e-6)
    cap = offer.capabilities
    # The decayed observed completion rate becomes a proxy for real-world
    # capability fit: a model that claims 0.9 coding but only completes 40%
    # of tasks is not actually a 0.9. Decayed, not raw, so a model that
    # 429'd last month is not permanently dragged down by it. Blend the
    # claimed score toward the observed rate rather than replacing it
    # outright, since completion rate conflates capability with
    # prompt/task mismatch.
    blended_cap = replace(
        cap,
        reasoning=cap.reasoning * (1 - weight) + completion_rate * weight,
        coding=cap.coding * (1 - weight) + completion_rate * weight,
        tool_use=cap.tool_use * (1 - weight) + completion_rate * weight,
        provenance="observed_execution",
    )

    econ = offer.economics
    # Cost per completed task: divide effective spend by completion rate so
    # a cheap-but-unreliable offer stops looking artificially attractive.
    # Only scale the parts of price the blend actually has evidence for.
    cost_inflation = 1.0 / completion_rate
    blended_econ = replace(
        econ,
        input_token_price=econ.input_token_price * (1 - weight) + econ.input_token_price * cost_inflation * weight,
        output_token_price=econ.output_token_price * (1 - weight) + econ.output_token_price * cost_inflation * weight,
    )

    telem = offer.telemetry
    observed_failure = max(0.0, 1.0 - stats.completion_rate)
    blended_telem = replace(
        telem,
        failure_rate=telem.failure_rate * (1 - weight) + observed_failure * weight,
    )

    return replace(
        offer,
        capabilities=blended_cap,
        economics=blended_econ,
        telemetry=blended_telem,
        confidence=min(1.0, offer.confidence + 0.1 * weight),
    )


def apply_observed(offers: list[ResourceOffer], *, path=None) -> list[ResourceOffer]:
    """Return offers with fields blended toward observed outcomes.

    No-op (returns offers unchanged) when there is no log or nothing meets
    MIN_OBSERVATIONS yet — this is opt-in and additive, never a silent
    behavior change for a fresh checkout.
    """
    observations = load_observations(path=path)
    if not observations:
        return offers
    stats = aggregate(observations)
    blended: list[ResourceOffer] = []
    for offer in offers:
        if offer.model is None:
            blended.append(offer)
            continue
        key = (offer.provider, offer.model)
        offer_stats = stats.get(key)
        if offer_stats is None:
            blended.append(offer)
            continue
        blended.append(blend_offer(offer, offer_stats))
    return blended
