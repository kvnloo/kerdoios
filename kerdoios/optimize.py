from __future__ import annotations

from .blend import apply_observed
from .pareto import nondominated
from .presets import PRESETS
from .quota import PlanningPolicy, free_only_rejection
from .score import ScoredOffer, capability_fit, score
from .types import (
    ExecutionPlan,
    Mode,
    Placement,
    Rejection,
    ResourceOffer,
    WorkRequirement,
)


def hard_filter(offers: list[ResourceOffer], req: WorkRequirement) -> tuple[list[ResourceOffer], list[Rejection]]:
    kept: list[ResourceOffer] = []
    rejected: list[Rejection] = []
    for offer in offers:
        reason = _hard_reject(offer, req)
        if reason:
            rejected.append(Rejection(offer_id=offer.id, reason=reason))
            continue
        kept.append(offer)
    return kept, rejected


def _hard_reject(offer: ResourceOffer, req: WorkRequirement) -> str | None:
    if req.privacy == "local_only" and not offer.local:
        return "privacy local_only excludes remote providers"
    if req.privacy not in offer.privacy_ok and req.privacy != "public":
        return f"privacy {req.privacy} not in offer allow-list"
    if offer.capacity.context_window < req.context:
        return f"context {offer.capacity.context_window} < {req.context}"
    if req.vision and offer.capabilities.vision < 0.5:
        return "vision required"
    if req.tool_use and offer.capabilities.tool_use < 0.5:
        return "tool use required"
    if req.tools and any(tool not in offer.tools and "*" not in offer.tools for tool in req.tools):
        return f"missing tools {sorted(set(req.tools) - set(offer.tools))}"
    if capability_fit(offer, req) <= 0.0:
        return "capability below requirement"
    if offer.telemetry.availability < req.minimum_reliability:
        return "reliability below requirement"
    if req.maximum_latency_ms is not None and offer.telemetry.latency_p50_ms > req.maximum_latency_ms:
        return "latency above maximum"
    return None


def allocate(offers: list[ScoredOffer], req: WorkRequirement) -> tuple[list[Placement], int]:
    remaining = req.parallelism
    budget = req.maximum_cost
    spent = 0.0
    placements: list[Placement] = []
    ranked = sorted(offers, key=lambda item: item.fitness, reverse=True)
    for item in ranked:
        if remaining <= 0:
            break
        take = min(remaining, item.offer.capacity.concurrency)
        if take <= 0:
            continue
        if req.mode == Mode.FREE and item.unit_cost > 0:
            continue
        while take > 0:
            cost = item.unit_cost * take
            if budget is not None and spent + cost > budget + 1e-12:
                # shrink until it fits or give up this offer
                take -= 1
                continue
            break
        if take <= 0:
            continue
        cost = item.unit_cost * take
        placements.append(
            Placement(
                offer_id=item.offer.id,
                provider=item.offer.provider,
                model=item.offer.model,
                workers=take,
                estimated_cost=round(cost, 6),
                reasons=list(item.reasons),
            )
        )
        remaining -= take
        spent += cost
    return placements, remaining


def plan(
    offers: list[ResourceOffer],
    req: WorkRequirement,
    *,
    use_observed: bool = False,
    policy: PlanningPolicy | None = None,
) -> ExecutionPlan:
    posture = policy or PlanningPolicy()
    if use_observed:
        offers = apply_observed(offers, capability_id=req.capability_id)
    eligible, rejections = hard_filter(offers, req)
    if posture.free_only:
        # Free-only is a candidate filter, not a scoring nudge: a paid tier must
        # never win by scoring higher once free capacity is spent.
        kept: list[ResourceOffer] = []
        for offer in eligible:
            reason = free_only_rejection(offer, posture)
            if reason:
                rejections.append(Rejection(offer_id=offer.id, reason=reason))
                continue
            kept.append(offer)
        eligible = kept
    weights = PRESETS[req.mode]
    scored = [score(offer, req, weights) for offer in eligible]
    scored = [item for item in scored if item.capability > 0]
    # Single-worker jobs can drop dominated endpoints. Portfolios keep extra
    # capacity even when one offer is worse on every quality axis.
    if req.parallelism <= 1 and scored:
        scored = nondominated(
            scored,
            dimensions=[
                lambda item: item.unit_cost,
                lambda item: item.capability,
                lambda item: item.latency,
                lambda item: -item.throughput,
            ],
            minimize=[True, False, True, True],
        )
    placements, unplaced = allocate(scored, req)
    total_cost = sum(item.estimated_cost for item in placements)
    if not placements:
        duration = 0.0
        confidence = 0.0
    else:
        slowest = max(
            next(s.latency for s in scored if s.offer.id == p.offer_id) for p in placements
        )
        duration = slowest / 1000.0
        confidence = min(s.capability for s in scored if any(p.offer_id == s.offer.id for p in placements))
    fallbacks = [item.offer.id for item in sorted(scored, key=lambda s: s.fitness, reverse=True)]
    fallbacks = [fid for fid in fallbacks if fid not in {p.offer_id for p in placements}]
    # Everything free was spent and nothing was eligible: say so explicitly so
    # a caller cannot read the empty plan as "budget happened to be zero".
    no_free_capacity = posture.free_only and not placements and not eligible
    return ExecutionPlan(
        estimated_cost=round(total_cost, 6),
        estimated_duration_seconds=round(duration, 3),
        confidence=round(confidence, 3),
        placements=placements,
        fallbacks=fallbacks[:8],
        rejections=rejections,
        mode=req.mode.value,
        unplaced_workers=unplaced,
        free_only=posture.free_only,
        no_paid_spill=posture.no_paid_spill,
        no_free_capacity=no_free_capacity,
    )
