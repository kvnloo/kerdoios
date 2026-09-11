"""Kerdoios — Hermes plugin for compute allotment.

Hermes orchestrates. This plugin only plans placement.
"""
from __future__ import annotations

import json
from typing import Any

from .aodl import AodlIngestError, load_aodl, work_requirement_from_aodl
from .doctor import inspect, require_vault
from .explain import explain
from .inventory import discover_all
from .observed import Observation, record
from .optimize import plan
from .providers.free import is_free
from .types import Mode, PrivacyClass, WorkRequirement


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
    privacy_raw = str(args.get("privacy") or "public")
    privacy: PrivacyClass = (
        privacy_raw if privacy_raw in ("public", "confidential", "local_only") else "public"
    )
    flags = WorkRequirement(
        coding=float(args.get("coding") or 0.7),
        reasoning=float(args.get("reasoning") or 0.6),
        tool_use=bool(args.get("tools_required", True)),
        vision=bool(args.get("vision") or False),
        context=int(args.get("context") or 128_000),
        parallelism=int(args.get("workers") or args.get("parallelism") or 1),
        estimated_input_tokens=int(args.get("input_tokens") or 4000),
        estimated_output_tokens=int(args.get("output_tokens") or 800),
        maximum_cost=float(budget) if budget is not None and budget != "" else None,
        privacy=privacy,
        mode=mode,
        tools=tuple(tools) if tools else ("github",),
    )
    aodl = args.get("aodl")
    if aodl is None or aodl == "":
        return flags
    doc = load_aodl(aodl) if isinstance(aodl, str) else aodl
    return work_requirement_from_aodl(doc, defaults=flags)


def _inventory_row(offer: Any) -> dict[str, Any]:
    return {
        "id": offer.id,
        "provider": offer.provider,
        "model": offer.model,
        "concurrency": offer.capacity.concurrency,
        "context": offer.capacity.context_window,
        "source": offer.source,
        "free": is_free(offer),
        "prices": {
            "input": offer.economics.input_token_price,
            "output": offer.economics.output_token_price,
        },
        "tools": list(offer.tools),
    }


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
            "free": {"type": "boolean", "description": "Keep free models as the initial list (OpenRouter public/snapshot plus keyed Groq/Cerebras free-tier; no fixture mix)"},
            "observed": {
                "type": "boolean",
                "description": "Blend in observed execution outcomes (KERDOIOS_OBSERVED_LOG or ~/.hermes/cache/kerdoios/observed.jsonl)",
            },
            "aodl": {"description": "AODL-shaped work spec object or JSON path"},
        },
    },
}

EXPLAIN_SCHEMA = {
    "name": "kerdoios_explain",
    "description": "Explain a Kerdoios placement against the current inventory.",
    "parameters": {
        "type": "object",
        "properties": {
            "workers": {"type": "integer", "minimum": 1},
            "budget": {"type": "number"},
            "mode": {"type": "string"},
            "privacy": {"type": "string"},
            "free": {"type": "boolean", "description": "Keep free models as the initial list (OpenRouter plus keyed Groq/Cerebras free-tier)"},
            "observed": {
                "type": "boolean",
                "description": "Blend in observed execution outcomes (KERDOIOS_OBSERVED_LOG or ~/.hermes/cache/kerdoios/observed.jsonl)",
            },
            "aodl": {"description": "AODL-shaped work spec object or JSON path"},
        },
    },
}

INVENTORY_SCHEMA = {
    "name": "kerdoios_inventory",
    "description": (
        "List ResourceOffers. free=true seeds from public free models "
        "(cache, then OpenRouter /models, then the vendored snapshot if cache and OpenRouter are empty; "
        "keyed Groq/Cerebras free-tier overlays when those vault secrets resolve)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "live": {"type": "boolean", "description": "Query live provider adapters in addition to the fixture catalog"},
            "free": {"type": "boolean", "description": "Keep free models as the initial list (OpenRouter plus keyed Groq/Cerebras free-tier; no fixture mix)"},
            "refresh": {"type": "boolean", "description": "Bypass inventory cache and fetch live OpenRouter"},
        },
    },
}

RECORD_SCHEMA = {
    "name": "kerdoios_record",
    "description": "Log an observed execution outcome.",
    "parameters": {
        "type": "object",
        "properties": {
            "provider": {"type": "string"},
            "model": {"type": "string"},
            "task_type": {"type": "string", "description": "Caller-controlled bucket (default unknown)"},
            "completed": {"type": "boolean"},
            "cost": {"type": "number", "description": "Actual cost in USD"},
            "retried": {"type": "boolean"},
        },
        "required": ["provider", "model"],
    },
}


def register(ctx: Any) -> None:
    # hermes plugins doctor / enable load this path. Fail closed without vault.
    require_vault()
    def _offers(args: dict[str, Any]):
        live = bool(args.get("live"))
        free_only = bool(args.get("free"))
        refresh = bool(args.get("refresh"))
        include_fixture = True
        if free_only:
            live = True
            include_fixture = False
        return discover_all(
            include_fixture=include_fixture,
            live=live,
            free_only=free_only,
            refresh=refresh,
        )

    def handle_plan(args: dict[str, Any], **kwargs: Any) -> str:
        try:
            requirement = _req_from_args(args)
        except AodlIngestError as exc:
            return f"aodl: {exc}"
        return json.dumps(
            plan(_offers(args), requirement, use_observed=bool(args.get("observed"))).to_dict(),
            indent=2,
        )

    def handle_explain(args: dict[str, Any], **kwargs: Any) -> str:
        try:
            requirement = _req_from_args(args)
        except AodlIngestError as exc:
            return f"aodl: {exc}"
        return explain(_offers(args), requirement, use_observed=bool(args.get("observed")))

    def handle_inventory(args: dict[str, Any], **kwargs: Any) -> str:
        return json.dumps([_inventory_row(offer) for offer in _offers(args)], indent=2)

    def handle_record(args: dict[str, Any], **kwargs: Any) -> str:
        provider = str(args.get("provider") or "").strip()
        model = str(args.get("model") or "").strip()
        if not provider or not model:
            raise ValueError("kerdoios_record requires provider and model")
        cost_raw = args.get("cost")
        actual_cost = 0.0 if cost_raw is None or cost_raw == "" else float(cost_raw)
        observation = Observation(
            provider=provider,
            model=model,
            task_type=str(args.get("task_type") or "unknown"),
            completed=bool(args.get("completed")),
            actual_cost=actual_cost,
            retried=bool(args.get("retried")),
        )
        record(observation)
        return json.dumps(observation.to_dict(), indent=2)

    ctx.register_tool(name="kerdoios_plan", toolset="kerdoios", schema=PLAN_SCHEMA, handler=handle_plan)
    ctx.register_tool(name="kerdoios_explain", toolset="kerdoios", schema=EXPLAIN_SCHEMA, handler=handle_explain)
    ctx.register_tool(name="kerdoios_inventory", toolset="kerdoios", schema=INVENTORY_SCHEMA, handler=handle_inventory)
    ctx.register_tool(name="kerdoios_record", toolset="kerdoios", schema=RECORD_SCHEMA, handler=handle_record)

    def _cli(ns: Any) -> None:
        command = getattr(ns, "kerdoios_command", None) or getattr(ns, "command", None)
        if command == "record":
            print(
                handle_record(
                    {
                        "provider": getattr(ns, "provider", None),
                        "model": getattr(ns, "model", None),
                        "task_type": getattr(ns, "task_type", "unknown"),
                        "completed": bool(getattr(ns, "completed", False)),
                        "cost": getattr(ns, "cost", 0.0),
                        "retried": bool(getattr(ns, "retried", False)),
                    }
                )
            )
            return
        if command == "doctor":
            report = inspect()
            print(json.dumps(report.to_dict(), indent=2))
            if not report.ok:
                raise SystemExit(1)
            return
        workers = int(getattr(ns, "workers", 8) or 8)
        budget = getattr(ns, "budget", None)
        mode = str(getattr(ns, "mode", "balanced") or "balanced")
        privacy = str(getattr(ns, "privacy", "public") or "public")
        live = bool(getattr(ns, "live", False))
        free = bool(getattr(ns, "free", False))
        refresh = bool(getattr(ns, "refresh", False))
        aodl = getattr(ns, "aodl", None)
        args = {
            "workers": workers,
            "budget": budget,
            "mode": mode,
            "privacy": privacy,
            "live": live,
            "free": free,
            "refresh": refresh,
            "observed": bool(getattr(ns, "observed", False)),
            "aodl": aodl,
        }
        if command == "inventory":
            print(handle_inventory(args))
            return
        if command == "explain":
            print(handle_explain(args))
            return
        print(handle_plan(args))

    def _setup(subparser: Any) -> None:
        subs = subparser.add_subparsers(dest="kerdoios_command")
        for name in ("inventory", "plan", "explain", "doctor"):
            p = subs.add_parser(name)
            if name == "doctor":
                continue
            p.add_argument("--live", action="store_true")
            p.add_argument("--free", action="store_true")
            if name == "inventory":
                p.add_argument("--refresh", action="store_true")
            if name != "inventory":
                p.add_argument("--workers", type=int, default=8)
                p.add_argument("--budget", type=float, default=None)
                p.add_argument("--mode", default="balanced")
                p.add_argument("--privacy", default="public")
                p.add_argument("--observed", action="store_true")
                p.add_argument("--aodl", default=None)
        record_p = subs.add_parser("record")
        record_p.add_argument("--provider", required=True)
        record_p.add_argument("--model", required=True)
        record_p.add_argument("--task-type", default="unknown")
        record_p.add_argument("--completed", action="store_true")
        record_p.add_argument("--cost", type=float, default=0.0)
        record_p.add_argument("--retried", action="store_true")
        subparser.set_defaults(func=_cli)

    if hasattr(ctx, "register_cli_command"):
        ctx.register_cli_command(
            name="kerdoios",
            help="Allot compute across providers (Hermes Kerdoios)",
            setup_fn=_setup,
            handler_fn=_cli,
        )
