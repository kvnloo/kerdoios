"""Accept z0int.allocation_request.v1 → WorkRequirement (planner only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import Mode, PrivacyClass, QuotaConstraint, WorkRequirement


SCHEMA = "z0int.allocation_request.v1"


class AllocationRequestError(ValueError):
    pass


def load_allocation_request(path: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path, dict):
        return path
    p = Path(path)
    if str(path) == "-":
        import sys

        return json.load(sys.stdin)
    return json.loads(p.read_text(encoding="utf-8"))


def work_requirement_from_allocation_request(
    doc: dict[str, Any],
    *,
    defaults: WorkRequirement | None = None,
) -> WorkRequirement:
    schema = doc.get("schema") or doc.get("type")
    if schema and schema not in (SCHEMA, "allocation_request.v1"):
        raise AllocationRequestError(f"unsupported schema {schema!r}")
    base = defaults or WorkRequirement()
    req_block = doc.get("requirement") or doc.get("work") or doc
    quotas_raw = doc.get("quotas") or req_block.get("quotas") or ()
    quotas: list[QuotaConstraint] = []
    for q in quotas_raw:
        if not isinstance(q, dict):
            continue
        quotas.append(
            QuotaConstraint(
                name=str(q.get("name") or q.get("id") or "default"),
                limit=float(q.get("limit") or q.get("max") or 0),
                unit=str(q.get("unit") or "requests"),
                group=q.get("group"),
                remaining=_opt_float(q.get("remaining")),
            )
        )
    mode_raw = req_block.get("mode") or base.mode
    if isinstance(mode_raw, str):
        try:
            mode = Mode(mode_raw)
        except ValueError:
            mode = base.mode
    else:
        mode = mode_raw if isinstance(mode_raw, Mode) else base.mode
    return WorkRequirement(
        coding=float(req_block.get("coding", base.coding)),
        reasoning=float(req_block.get("reasoning", base.reasoning)),
        tool_use=bool(req_block.get("tool_use", base.tool_use)),
        vision=bool(req_block.get("vision", base.vision)),
        context=int(req_block.get("context", base.context)),
        parallelism=int(req_block.get("parallelism") or req_block.get("workers") or base.parallelism),
        estimated_input_tokens=int(
            req_block.get("estimated_input_tokens") or base.estimated_input_tokens
        ),
        estimated_output_tokens=int(
            req_block.get("estimated_output_tokens") or base.estimated_output_tokens
        ),
        maximum_cost=_opt_float(req_block.get("maximum_cost") if "maximum_cost" in req_block else base.maximum_cost),
        maximum_latency_ms=_opt_float(
            req_block.get("maximum_latency_ms") if "maximum_latency_ms" in req_block else base.maximum_latency_ms
        ),
        minimum_reliability=float(req_block.get("minimum_reliability", base.minimum_reliability)),
        privacy=_privacy(req_block.get("privacy", base.privacy)),
        mode=mode,
        tools=tuple(req_block.get("tools") or base.tools),
        capability_id=req_block.get("capability_id") or doc.get("capability_id") or base.capability_id,
        quotas=tuple(quotas) if quotas else base.quotas,
        join_policy=doc.get("join_policy") or req_block.get("join_policy") or base.join_policy,
        retry_policy=doc.get("retry_policy") or req_block.get("retry_policy") or base.retry_policy,
    )


def _opt_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)  # value pre-checked


def _privacy(value: object) -> PrivacyClass:
    if value in ("public", "confidential", "local_only"):
        return value
    return "public"
