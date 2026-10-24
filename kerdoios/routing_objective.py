"""Kevin's latency/burn/acceptance objective (OMP 4409; confirmed 2026-10-24).

latency_u = TTFT_u + out_tokens_t / tok/s_u
pace_u = burn_u * tokens_t / (remaining_u / time_to_reset_u)
score_u = latency_u * (1 + λ * pace_u) / p_accept[u, task_class(t)]

Minimize over hard-filtered offers. Unknown inputs remain unscored; this module
never manufactures observations from catalog defaults or wall-clock latency.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .blend import _blend_weight
from .observed import MIN_OBSERVATIONS
from .optimize import hard_filter
from .types import ResourceOffer, WorkRequirement

FORMULA_ID = "kerdoios.latency_burn_acceptance.v1"


@dataclass(frozen=True)
class RoutingScore:
    offer: ResourceOffer
    score: float
    latency: float
    pace: float
    p_accept: float
    inputs: dict


def _finite(value, *, positive=False):
    return (type(value) in (float, int) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def plan_routing(offers: list[ResourceOffer], req: WorkRequirement, metrics: dict,
                 *, lambda_value: float = 1.0, lambda_evidence: dict | None = None):
    """Sole routing optimizer; callers enforce this plan, never re-score it.

    Rate/balance/reset must use the same explicitly documented quota unit and
    hours. Lambda=1 is explicitly unmeasured until a measured config replaces it.
    Acceptance observations must be exact task-class verified acceptances; mere
    HTTP completion is not task acceptance. Existing blend trust floor applies.
    """
    if not _finite(lambda_value):
        raise ValueError("lambda must be finite and nonnegative")
    evidence = lambda_evidence or {"status": "unmeasured", "source": "Kevin confirmation 2026-10-24; default lambda=1"}
    if lambda_value != 1 and evidence.get("status") != "measured":
        raise ValueError("non-default lambda requires measured evidence")
    if evidence.get("status") == "measured" and not all(evidence.get(k) for k in ("source", "timestamp")):
        raise ValueError("measured lambda needs source and timestamp")
    eligible, rejected = hard_filter(offers, req)
    ranked, unscored = [], []
    for offer in eligible:
        values = metrics.get(offer.id, {})
        required = ("ttft_seconds", "tokens_per_second", "burn", "remaining", "time_to_reset_hours", "p_accept")
        if (any(not _finite(values.get(k), positive=k not in ("burn", "ttft_seconds")) for k in required)
                or values.get("p_accept", 0) > 1):
            unscored.append({"offer_id": offer.id, "provider": offer.transport or offer.provider,
                             "model": offer.model, "reason": "unknown_objective_inputs"})
            continue
        p_accept = values["p_accept"]
        observed = values.get("observed_acceptance")
        if observed is not None:
            n, accepted = observed.get("n"), observed.get("accepted")
            if (type(n) is not int or n < 0 or type(accepted) is not int or not 0 <= accepted <= n
                    or observed.get("task_class") != req.capability_id or not observed.get("verifier_source")):
                raise ValueError("invalid task-class acceptance observations")
            if n >= MIN_OBSERVATIONS:
                weight = _blend_weight(n)
                p_accept = p_accept * (1 - weight) + accepted / n * weight
        total_tokens = req.estimated_input_tokens + req.estimated_output_tokens
        latency = values["ttft_seconds"] + req.estimated_output_tokens / values["tokens_per_second"]
        pace = values["burn"] * total_tokens / (values["remaining"] / values["time_to_reset_hours"])
        score = latency * (1 + lambda_value * pace) / p_accept
        if not math.isfinite(score):
            unscored.append({"offer_id": offer.id, "provider": offer.transport or offer.provider,
                             "model": offer.model, "reason": "nonfinite_objective"})
            continue
        inputs = {**values, "input_tokens": req.estimated_input_tokens,
                  "out_tokens": req.estimated_output_tokens, "tokens": total_tokens,
                  "lambda": lambda_value, "lambda_evidence": evidence,
                  "p_accept_used": p_accept, "formula_id": FORMULA_ID}
        ranked.append(RoutingScore(offer, score, latency, pace, p_accept, inputs))
    ranked.sort(key=lambda row: (row.score, row.offer.transport or row.offer.provider, row.offer.model, row.offer.id))
    return {"ranked": ranked, "unscored": unscored, "rejections": rejected,
            "formula_id": FORMULA_ID}
