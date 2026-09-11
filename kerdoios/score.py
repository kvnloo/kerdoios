from __future__ import annotations

from dataclasses import dataclass

from .types import Mode, ResourceOffer, WorkRequirement


@dataclass(frozen=True)
class ScoredOffer:
    offer: ResourceOffer
    capability: float
    unit_cost: float
    latency: float
    throughput: float
    urgency: float
    fitness: float
    reasons: tuple[str, ...]


def effective_unit_cost(offer: ResourceOffer, req: WorkRequirement) -> float:
    tokens = max(1, req.estimated_input_tokens + req.estimated_output_tokens)
    free = offer.economics.remaining_free_quota + offer.economics.remaining_credits
    inp = offer.economics.input_token_price
    out = offer.economics.output_token_price
    # Unset prices are unknown at the catalog layer; arithmetic needs a float.
    # is_free still refuses None/default-0 as a free chat tier.
    token_cost = (0.0 if inp is None else inp) * req.estimated_input_tokens
    token_cost += (0.0 if out is None else out) * req.estimated_output_tokens
    if free > 0:
        # remaining free units are treated as covering this worker's tokens first
        covered = min(1.0, free / float(tokens))
        token_cost *= 1.0 - covered
    retry = offer.telemetry.failure_rate * token_cost
    return max(0.0, token_cost + retry)


def capability_fit(offer: ResourceOffer, req: WorkRequirement) -> float:
    cap = offer.capabilities
    if req.coding and cap.coding < req.coding:
        return 0.0
    if req.reasoning and cap.reasoning < req.reasoning:
        return 0.0
    if req.tool_use and cap.tool_use < 0.5:
        return 0.0
    if req.vision and cap.vision < 0.5:
        return 0.0
    return (cap.coding + cap.reasoning + cap.tool_use) / 3.0


def score(offer: ResourceOffer, req: WorkRequirement, weights: dict[str, float]) -> ScoredOffer:
    cap = capability_fit(offer, req)
    unit_cost = effective_unit_cost(offer, req)
    latency = offer.telemetry.latency_p50_ms
    throughput = float(offer.capacity.concurrency)
    urgency = offer.economics.expiration_urgency()
    # invert cost/latency into "higher is better" fitness
    cost_term = 1.0 / (1.0 + unit_cost * 100.0)
    lat_term = 1.0 / (1.0 + latency / 1000.0)
    thr_term = throughput / (throughput + 10.0)
    fitness = (
        weights.get("cost", 0.0) * cost_term
        + weights.get("capability", 0.0) * cap
        + weights.get("latency", 0.0) * lat_term
        + weights.get("throughput", 0.0) * thr_term
    ) * (0.85 + 0.15 * min(urgency, 3.0) / 3.0)
    if req.mode == Mode.PRIVATE and offer.local:
        fitness *= 1.4
    reasons = []
    if unit_cost == 0.0:
        reasons.append("$0 marginal cost (quota or credit)")
    if urgency > 1.5:
        reasons.append("expiring free capacity")
    if offer.local:
        reasons.append("local resource")
    if req.mode == Mode.PRIVATE and offer.local:
        reasons.append("private/local preference")
    reasons.append(f"capability {cap:.2f}")
    reasons.append(f"p50 {latency:.0f}ms")
    return ScoredOffer(
        offer=offer,
        capability=cap,
        unit_cost=unit_cost,
        latency=latency,
        throughput=throughput,
        urgency=urgency,
        fitness=fitness,
        reasons=tuple(reasons),
    )
