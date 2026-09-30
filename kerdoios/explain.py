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
    if result.no_free_capacity:
        lines.append(
            "No free capacity: free-only policy blocked paid spill; "
            "the plan places nothing rather than billing."
        )
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
    sub = result.subscription_windows
    if sub is not None:
        lines.extend(_subscription_lines(sub, result))
    if result.rejections:
        lines.append("Not selected (hard filter):")
        for rejection in result.rejections[:12]:
            lines.append(f"  - {rejection.offer_id}: {rejection.reason}")
    return "\n".join(lines).rstrip() + "\n"


def _subscription_lines(sub: dict, result: ExecutionPlan) -> list[str]:
    head = f"Subscription windows ({sub.get('source')}): {sub.get('status')}"
    if sub.get("factory_posture"):
        head += f"  factory={sub['factory_posture']}"
    if sub.get("snapshot_at"):
        head += f"  snapshot={sub['snapshot_at']}"
    lines = [head]
    if sub.get("status") != "ok":
        lines.append(f"  ~ {sub.get('reason')}")
    for w in sub.get("windows") or []:
        lines.append(
            f"  {w['posture']:<8} {w['group']}:{w['window']:<22} "
            f"{w['remaining_fraction'] * 100:5.1f}% left  resets {w['resets_at']}  {w['arithmetic']}"
        )
    for skip in sub.get("skipped") or []:
        lines.append(f"  ~ skipped {skip}")
    # Effect on the plan per plan offer, so an OFFLOAD verdict is visible even
    # when the hard-filter list below is truncated.
    placed = {p.offer_id for p in result.placements}
    groups = sorted({w["group"] for w in sub.get("windows") or []})
    rejected = {r.offer_id: r.reason for r in result.rejections}
    for group in groups:
        oid = f"subscription/{group}"
        if oid in placed:
            effect = "placed"
        elif oid in rejected:
            effect = f"rejected: {rejected[oid]}"
        elif oid in result.fallbacks:
            effect = f"fallback #{result.fallbacks.index(oid) + 1}"
        else:
            effect = "eligible, ranked below the fallback list"
        lines.append(f"  -> {oid}: {effect}")
    lines.append("")
    return lines
