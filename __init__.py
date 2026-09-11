"""Kerdoios — Hermes plugin for compute allotment.

Hermes orchestrates. This plugin only plans placement.
"""
from __future__ import annotations

import json
from typing import Any

from .explain import explain
from .inventory import discover_all
from .optimize import plan
from .types import Mode, WorkRequirement


def _req_from_args(args: dict[str, Any]) -> WorkRequirement:
    mode_raw = str(args.get("mode") or "balanced").lower()
    try:
        mode = Mode(mode_raw)
    except ValueError:
        mode = Mode.BALANCED
    tools = args.get("tools") or []
    if isinstance(tools, str):
        tools = [part.strip() for part in tools.split(",") if part.strip()]
    budget = args.get("budget")
    return WorkRequirement(
        coding=float(args.get("coding") or 0.7),
        reasoning=float(args.get("reasoning") or 0.6),
        tool_use=bool(args.get("tools_required", True)),
        vision=bool(args.get("vision") or False),
        context=int(args.get("context") or 128_000),
        parallelism=int(args.get("workers") or args.get("parallelism") or 1),
        estimated_input_tokens=int(args.get("input_tokens") or 4000),
        estimated_output_tokens=int(args.get("output_tokens") or 800),
        maximum_cost=float(budget) if budget is not None and budget != "" else None,
        privacy=str(args.get("privacy") or "public"),  # type: ignore[arg-type]
        mode=mode,
        tools=tuple(tools) if tools else ("github",),
    )


PLAN_SCHEMA = {
    "name": "kerdoios_plan",
    "description": (
        "Allot LLM/compute workers across providers under a budget. "
        "Returns a portfolio ExecutionPlan. Does not call models."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "workers": {"type": "integer", "minimum": 1, "description": "Requested parallelism"},
            "budget": {"type": "number", "description": "Maximum monetary cost in USD"},
            "mode": {
                "type": "string",
                "enum": ["free", "cheap", "balanced", "fast", "max", "scale", "private"],
            },
            "context": {"type": "integer", "description": "Minimum context window"},
            "privacy": {"type": "string", "enum": ["public", "confidential", "local_only"]},
            "coding": {"type": "number"},
            "reasoning": {"type": "number"},
            "live": {"type": "boolean", "description": "Query live provider adapters in addition to the fixture catalog"},
        },
    },
}

EXPLAIN_SCHEMA = {
    "name": "kerdoios_explain",
    "description": "Explain a Kerdoios placement against the fixture or last inventory.",
    "parameters": {
        "type": "object",
        "properties": {
            "workers": {"type": "integer", "minimum": 1},
            "budget": {"type": "number"},
            "mode": {"type": "string"},
            "privacy": {"type": "string"},
        },
    },
}


def register(ctx: Any) -> None:
    def _offers(args: dict[str, Any]):
        live = bool(args.get("live"))
        return discover_all(include_fixture=True, live=live)

    def handle_plan(args: dict[str, Any], **kwargs: Any) -> str:
        requirement = _req_from_args(args)
        return json.dumps(plan(_offers(args), requirement).to_dict(), indent=2)

    def handle_explain(args: dict[str, Any], **kwargs: Any) -> str:
        requirement = _req_from_args(args)
        return explain(_offers(args), requirement)

    ctx.register_tool(name="kerdoios_plan", toolset="kerdoios", schema=PLAN_SCHEMA, handler=handle_plan)
    ctx.register_tool(name="kerdoios_explain", toolset="kerdoios", schema=EXPLAIN_SCHEMA, handler=handle_explain)

    def _cli(ns: Any) -> None:
        command = getattr(ns, "kerdoios_command", None) or getattr(ns, "command", None)
        workers = int(getattr(ns, "workers", 8) or 8)
        budget = getattr(ns, "budget", None)
        mode = str(getattr(ns, "mode", "balanced") or "balanced")
        privacy = str(getattr(ns, "privacy", "public") or "public")
        live = bool(getattr(ns, "live", False))
        args = {"workers": workers, "budget": budget, "mode": mode, "privacy": privacy, "live": live}
        if command == "inventory":
            print(json.dumps([offer.to_dict() for offer in _offers(args)], indent=2, default=str))
            return
        if command == "explain":
            print(handle_explain(args))
            return
        print(handle_plan(args))

    def _setup(subparser: Any) -> None:
        subs = subparser.add_subparsers(dest="kerdoios_command")
        for name in ("inventory", "plan", "explain"):
            p = subs.add_parser(name)
            p.add_argument("--live", action="store_true")
            if name != "inventory":
                p.add_argument("--workers", type=int, default=8)
                p.add_argument("--budget", type=float, default=None)
                p.add_argument("--mode", default="balanced")
                p.add_argument("--privacy", default="public")
        subparser.set_defaults(func=_cli)

    if hasattr(ctx, "register_cli_command"):
        ctx.register_cli_command(
            name="kerdoios",
            help="Allot compute across providers (Hermes Kerdoios)",
            setup_fn=_setup,
            handler_fn=_cli,
        )
