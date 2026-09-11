from __future__ import annotations

from .optimize import plan as build_plan
from .types import ExecutionPlan, ResourceOffer, WorkRequirement


def explain(offers: list[ResourceOffer], req: WorkRequirement, built: ExecutionPlan | None = None, *, use_observed: bool = False) -> str:
    result = built or build_plan(offers, req, use_observed=use_observed)
    lines = [
        f"Kerdoios plan  mode={result.mode}  cost=${result.estimated_cost:.4f}  "
        f"workers={sum(p.workers for p in result.placements)}/{req.parallelism}  "
        f"confidence={result.confidence:.2f}",
        "",
    ]
    if result.unplaced_workers:
        lines.append(f"Unplaced workers: {result.unplaced_workers} (capacity, budget, or FREE-mode paid skip)")
        lines.append("")
    by_id = {offer.id: offer for offer in offers}
    for placement in result.placements:
        offer = by_id.get(placement.offer_id)
        model = placement.model or "n/a"
        lines.append(f"Selected {placement.provider}/{model}  workers={placement.workers}  cost=${placement.estimated_cost:.4f}")
        for reason in placement.reasons:
            lines.append(f"  + {reason}")
        if offer and offer.capacity.context_window:
            lines.append(f"  + context {offer.capacity.context_window}")
        lines.append("")
    if result.fallbacks:
        lines.append("Fallback order: " + ", ".join(result.fallbacks))
        lines.append("")
    if result.rejections:
        lines.append("Not selected (hard filter):")
        for rejection in result.rejections[:12]:
            lines.append(f"  - {rejection.offer_id}: {rejection.reason}")
    return "\n".join(lines).rstrip() + "\n"
