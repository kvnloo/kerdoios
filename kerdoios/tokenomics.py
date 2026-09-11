"""North-star tokenomics: kerdoios vs naive paid-only.

Fair naive is the cheapest paid offer with known prices, quota/credits
zeroed, and ``unit_cost * kerdoios_placed_workers``. Re-running ``plan()``
under the same ``maximum_cost`` (same-budget truncate) is not that baseline.
Success stays unknown without ``task_type=baseline`` receipts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .observed import MIN_OBSERVATIONS, Observation, load_observations
from .optimize import hard_filter, plan
from .score import effective_unit_cost
from .types import ExecutionPlan, ResourceOffer, WorkRequirement

COST_REDUCTION_THRESHOLD = 0.50
SUCCESS_RATIO_THRESHOLD = 0.95
BASELINE_TASK_TYPE = "baseline"


def is_paid_baseline(offer: ResourceOffer) -> bool:
    """Unknown prices, local stock, and leftover quota/credits are not a baseline."""
    if offer.local:
        return False
    econ = offer.economics
    if econ.remaining_free_quota != 0 or econ.remaining_credits != 0:
        return False
    inp = econ.input_token_price
    out = econ.output_token_price
    if inp is None or out is None:
        return False
    return (inp + out) > 0


def naive_paid_offer(
    offers: list[ResourceOffer], req: WorkRequirement
) -> ResourceOffer | None:
    eligible, _rejected = hard_filter(offers, req)
    paid = [offer for offer in eligible if is_paid_baseline(offer)]
    if not paid:
        return None
    return min(paid, key=lambda offer: effective_unit_cost(offer, req))


def naive_paid_cost(offer: ResourceOffer, req: WorkRequirement, placed_workers: int) -> float:
    return round(effective_unit_cost(offer, req) * placed_workers, 6)


def _obs_naive_cost(obs: Any, offer: ResourceOffer, req: WorkRequirement) -> float:
    inp = getattr(obs, "input_tokens", None)
    out = getattr(obs, "output_tokens", None)
    if inp is None and out is None:
        return effective_unit_cost(offer, req)
    pin = offer.economics.input_token_price
    pout = offer.economics.output_token_price
    return (0.0 if pin is None else pin) * (0 if inp is None else inp) + (
        0.0 if pout is None else pout
    ) * (0 if out is None else out)


def ante(
    offers: list[ResourceOffer],
    req: WorkRequirement,
    built: ExecutionPlan | None = None,
) -> dict[str, Any]:
    if built is None:
        built = plan(offers, req)
    placed = sum(item.workers for item in built.placements)
    offer = naive_paid_offer(offers, req)
    payload: dict[str, Any] = {
        "kerdoios_cost": built.estimated_cost,
        "naive_cost": None,
        "placed_workers": placed,
        "cost_reduction": None,
        "pass_cost": None,
        "naive_offer_id": None,
    }
    if offer is None:
        return payload
    naive = naive_paid_cost(offer, req, placed)
    payload["naive_cost"] = naive
    payload["naive_offer_id"] = offer.id
    if naive <= 0:
        return payload
    reduction = 1.0 - (built.estimated_cost / naive)
    payload["cost_reduction"] = reduction
    payload["pass_cost"] = reduction >= COST_REDUCTION_THRESHOLD
    return payload


def post(
    observations: list[Any],
    *,
    naive_offer: ResourceOffer | None = None,
    req: WorkRequirement,
) -> dict[str, Any]:
    portfolio = [obs for obs in observations if getattr(obs, "task_type", None) != BASELINE_TASK_TYPE]
    baseline = [obs for obs in observations if getattr(obs, "task_type", None) == BASELINE_TASK_TYPE]
    pn = len(portfolio)
    bn = len(baseline)
    payload: dict[str, Any] = {
        "portfolio_n": pn,
        "baseline_n": bn,
        "portfolio_success_rate": None,
        "baseline_success_rate": None,
        "success_ratio": None,
        "pass_success": None,
        "naive_cost": None,
    }
    if pn:
        payload["portfolio_success_rate"] = sum(
            1 for obs in portfolio if getattr(obs, "completed", False)
        ) / pn
    if bn:
        payload["baseline_success_rate"] = sum(
            1 for obs in baseline if getattr(obs, "completed", False)
        ) / bn
    if pn >= MIN_OBSERVATIONS and bn >= MIN_OBSERVATIONS:
        p_rate = payload["portfolio_success_rate"]
        b_rate = payload["baseline_success_rate"]
        if p_rate is not None and b_rate and b_rate > 0:
            ratio = p_rate / b_rate
            payload["success_ratio"] = ratio
            payload["pass_success"] = ratio >= SUCCESS_RATIO_THRESHOLD
    if naive_offer is not None and observations:
        payload["naive_cost"] = sum(_obs_naive_cost(obs, naive_offer, req) for obs in observations)
    return payload


def report(
    offers: list[ResourceOffer],
    req: WorkRequirement,
    built: ExecutionPlan | None = None,
    *,
    path: Path | None = None,
    observations: list[Observation] | None = None,
) -> dict[str, Any]:
    if built is None:
        built = plan(offers, req)
    if observations is None:
        observations = load_observations(path=path)
    offer = naive_paid_offer(offers, req)
    return {
        "thresholds": {
            "cost_reduction": COST_REDUCTION_THRESHOLD,
            "success_ratio": SUCCESS_RATIO_THRESHOLD,
            "min_observations": MIN_OBSERVATIONS,
        },
        "ex_ante": ante(offers, req, built),
        "ex_post": post(observations, naive_offer=offer, req=req),
        "openrouter_bucket": False,
    }
