"""Quality-aware token bench vs an Astra-class offer.

Offline default replays ``plan()`` / ``score()`` / ``allocate()`` on frozen
fixtures. Token spend is first-class: free quota and local are $0 and still
burn tokens. Quality is verified success vs Astra ``task_type=baseline``
receipts, not an LLM judge.

``pass_quality`` / ``pass_tokens`` stay null until ``MIN_OBSERVATIONS``.
Do not invent 0.95. $ vs naive paid-only is a sibling cell, not this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .observed import MIN_OBSERVATIONS, Observation, load_observations
from .optimize import plan
from .score import capability_fit
from .types import (
    CapabilityProfile,
    Capacity,
    Economics,
    Mode,
    ResourceOffer,
    Telemetry,
    WorkRequirement,
)

QUALITY_RATIO_THRESHOLD = 0.95
BASELINE_TASK_TYPE = "baseline"
ASTRA_OFFER_ID = "frontier/paid"

DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "bench"


def offer_from_dict(row: dict[str, Any]) -> ResourceOffer:
    return ResourceOffer(
        id=row["id"],
        provider=row["provider"],
        resource_type=row["resource_type"],
        model=row.get("model"),
        local=bool(row["local"]),
        capabilities=CapabilityProfile(**row["capabilities"]),
        capacity=Capacity(**row["capacity"]),
        economics=Economics(**row["economics"]),
        telemetry=Telemetry(**row["telemetry"]),
        tools=tuple(row.get("tools") or ()),
        privacy_ok=tuple(row.get("privacy_ok") or ("public", "confidential")),
        source=row.get("source", "fixture"),
        confidence=float(row.get("confidence", 0.5)),
    )


def req_from_task(task: dict[str, Any]) -> WorkRequirement:
    return WorkRequirement(
        coding=float(task.get("coding") or 0.0),
        reasoning=float(task.get("reasoning") or 0.0),
        tool_use=bool(task.get("tool_use")),
        vision=bool(task.get("vision")),
        context=int(task.get("context") or 8192),
        parallelism=int(task.get("parallelism") or 1),
        estimated_input_tokens=int(task.get("estimated_input_tokens") or 4000),
        estimated_output_tokens=int(task.get("estimated_output_tokens") or 800),
        maximum_cost=task.get("maximum_cost"),
        privacy=task.get("privacy") or "public",
        mode=Mode(str(task.get("mode") or "cheap")),
        tools=tuple(task.get("tools") or ()),
    )


def load_catalog(path: Path) -> tuple[list[ResourceOffer], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    astra_id = str(payload.get("astra_offer_id") or ASTRA_OFFER_ID)
    return [offer_from_dict(row) for row in payload["offers"]], astra_id


def load_suite(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["tasks"])


def load_golden(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_astra_class(offer: ResourceOffer, astra_offer_id: str = ASTRA_OFFER_ID) -> bool:
    if offer.id == astra_offer_id:
        return True
    model = (offer.model or "").lower()
    return "astra" in model


def is_astra_observation(obs: Any, astra_offer_id: str = ASTRA_OFFER_ID) -> bool:
    model = (getattr(obs, "model", None) or "").lower()
    if "astra" in model or model == "frontier-codex":
        return True
    offer_id = (getattr(obs, "offer_id", None) or "").lower()
    return offer_id == astra_offer_id.lower()


def tokens_per_worker(req: WorkRequirement) -> int:
    return req.estimated_input_tokens + req.estimated_output_tokens


def _obs_tokens(obs: Any) -> int | None:
    inp = getattr(obs, "input_tokens", None)
    out = getattr(obs, "output_tokens", None)
    if inp is None and out is None:
        return None
    return int(0 if inp is None else inp) + int(0 if out is None else out)


def head_metrics(
    offers: list[ResourceOffer],
    req: WorkRequirement,
    *,
    task_id: str,
    astra_offer_id: str = ASTRA_OFFER_ID,
) -> dict[str, Any]:
    built = plan(offers, req)
    by_id = {offer.id: offer for offer in offers}
    astra = next((offer for offer in offers if is_astra_class(offer, astra_offer_id)), None)
    astra_cap = capability_fit(astra, req) if astra is not None else 0.0
    tpw = tokens_per_worker(req)
    total_tokens = 0
    frontier_tokens = 0
    min_cap: float | None = None
    for placement in built.placements:
        total_tokens += placement.workers * tpw
        offer = by_id.get(placement.offer_id)
        if offer is None:
            continue
        if is_astra_class(offer, astra_offer_id):
            frontier_tokens += placement.workers * tpw
        cap = capability_fit(offer, req)
        min_cap = cap if min_cap is None else min(min_cap, cap)
    if min_cap is None:
        quality_floor = 0.0
    elif astra_cap > 0:
        quality_floor = round(min_cap / astra_cap, 6)
    else:
        quality_floor = round(min_cap, 6)
    share = round((frontier_tokens / total_tokens) if total_tokens else 0.0, 6)
    return {
        "id": task_id,
        "estimated_tokens": total_tokens,
        "estimated_cost": built.estimated_cost,
        "frontier_tokens": frontier_tokens,
        "frontier_token_share": share,
        "unplaced_workers": built.unplaced_workers,
        "quality_floor": quality_floor,
        "astra_offer_id": astra.id if astra is not None else None,
        "placed_workers": sum(item.workers for item in built.placements),
    }


def _sum_heads(heads: list[dict[str, Any]]) -> dict[str, Any]:
    estimated_tokens = sum(int(head["estimated_tokens"]) for head in heads)
    frontier_tokens = sum(int(head["frontier_tokens"]) for head in heads)
    unplaced = sum(int(head["unplaced_workers"]) for head in heads)
    cost = round(sum(float(head["estimated_cost"]) for head in heads), 6)
    floors = [float(head["quality_floor"]) for head in heads]
    share = round((frontier_tokens / estimated_tokens) if estimated_tokens else 0.0, 6)
    return {
        "estimated_tokens": estimated_tokens,
        "estimated_cost": cost,
        "frontier_tokens": frontier_tokens,
        "frontier_token_share": share,
        "unplaced_workers": unplaced,
        "quality_floor": round(min(floors), 6) if floors else 0.0,
    }


def planner_report(
    offers: list[ResourceOffer],
    tasks: list[dict[str, Any]],
    *,
    astra_offer_id: str = ASTRA_OFFER_ID,
) -> dict[str, Any]:
    heads = [
        head_metrics(offers, req_from_task(task), task_id=str(task["id"]), astra_offer_id=astra_offer_id)
        for task in tasks
    ]
    return {"heads": heads, "totals": _sum_heads(heads)}


def live_post(
    observations: list[Any],
    *,
    astra_offer_id: str = ASTRA_OFFER_ID,
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
        "quality_ratio": None,
        "pass_quality": None,
        "portfolio_tokens": None,
        "baseline_tokens": None,
        "portfolio_frontier_tokens": None,
        "pass_tokens": None,
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
        if p_rate is not None and b_rate:
            ratio = p_rate / b_rate
            payload["quality_ratio"] = ratio
            payload["pass_quality"] = ratio >= QUALITY_RATIO_THRESHOLD
    p_tokens = [_obs_tokens(obs) for obs in portfolio]
    b_tokens = [_obs_tokens(obs) for obs in baseline]
    if pn >= MIN_OBSERVATIONS and bn >= MIN_OBSERVATIONS and p_tokens and b_tokens:
        if None not in p_tokens and None not in b_tokens:
            payload["portfolio_tokens"] = sum(int(tok) for tok in p_tokens)
            payload["baseline_tokens"] = sum(int(tok) for tok in b_tokens)
            payload["portfolio_frontier_tokens"] = sum(
                int(tok)
                for obs, tok in zip(portfolio, p_tokens)
                if is_astra_observation(obs, astra_offer_id)
            )
            payload["pass_tokens"] = (
                payload["portfolio_tokens"] <= payload["baseline_tokens"]
                and payload["portfolio_frontier_tokens"] <= payload["baseline_tokens"]
            )
    return payload


def regressions(actual: dict[str, Any], golden: dict[str, Any]) -> list[str]:
    """Token spend or quality-floor regressions vs the frozen golden."""
    reasons: list[str] = []
    golden_heads = {head["id"]: head for head in golden["heads"]}
    actual_heads = {head["id"]: head for head in actual["heads"]}
    if set(golden_heads) != set(actual_heads):
        reasons.append(f"heads {sorted(actual_heads)} != golden {sorted(golden_heads)}")
        return reasons
    pairs = list(golden_heads.items())
    if "totals" in golden:
        pairs.append(("totals", golden["totals"]))
        actual_heads = {**actual_heads, "totals": actual["totals"]}
    for hid, expected in pairs:
        got = actual_heads[hid]
        if int(got["estimated_tokens"]) > int(expected["estimated_tokens"]):
            reasons.append(
                f"{hid} estimated_tokens {got['estimated_tokens']} > golden {expected['estimated_tokens']}"
            )
        if int(got["frontier_tokens"]) > int(expected["frontier_tokens"]):
            reasons.append(
                f"{hid} frontier_tokens {got['frontier_tokens']} > golden {expected['frontier_tokens']}"
            )
        if int(got["unplaced_workers"]) > int(expected["unplaced_workers"]):
            reasons.append(
                f"{hid} unplaced_workers {got['unplaced_workers']} > golden {expected['unplaced_workers']}"
            )
        got_floor = got["quality_floor"]
        want_floor = expected["quality_floor"]
        if want_floor is not None and (got_floor is None or float(got_floor) + 1e-12 < float(want_floor)):
            reasons.append(f"{hid} quality_floor {got_floor} < golden {want_floor}")
    return reasons


def report(
    *,
    fixtures: Path | None = None,
    live: bool = False,
    path: Path | None = None,
    observations: list[Observation] | None = None,
) -> dict[str, Any]:
    fixture_dir = Path(fixtures) if fixtures is not None else DEFAULT_FIXTURE_DIR
    offers, astra_id = load_catalog(fixture_dir / "catalog.json")
    tasks = load_suite(fixture_dir / "suite.json")
    planned = planner_report(offers, tasks, astra_offer_id=astra_id)
    payload: dict[str, Any] = {
        "thresholds": {
            "quality_ratio": QUALITY_RATIO_THRESHOLD,
            "min_observations": MIN_OBSERVATIONS,
        },
        "fixtures": str(fixture_dir),
        "astra_offer_id": astra_id,
        "heads": planned["heads"],
        "totals": planned["totals"],
        "openrouter_bucket": False,
        "ex_post": None,
    }
    if live:
        if observations is None:
            observations = load_observations(path=path)
        payload["ex_post"] = live_post(observations, astra_offer_id=astra_id)
    return payload
