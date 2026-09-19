from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aodl import AodlIngestError, load_aodl, work_requirement_from_aodl
from .allocation_request import AllocationRequestError, load_allocation_request, work_requirement_from_allocation_request
from .explain import explain
from .inventory import discover_all
from .observed import Observation, import_allocation_observations, record, tokens_per_verified_task
from .optimize import plan
from .providers.free import is_free
from .types import Mode, WorkRequirement



def _inventory_table(offers) -> str:
    headers = (
        "TRANSPORT",
        "CREATOR",
        "MODEL",
        "ACCESS_CLASS",
        "REMAINING",
        "RESET/EXPIRY",
        "CONTEXT",
        "TOOLS",
        "SOURCE",
        "FRESHNESS",
    )
    lines = ["\t".join(headers)]
    for offer in offers:
        access = offer.access
        remaining = "unknown"
        if access is not None and access.remaining_known:
            remaining = str(offer.economics.remaining_free_quota or offer.economics.remaining_credits)
        reset = offer.economics.seconds_until_quota_reset or offer.economics.seconds_until_credits_expire
        lines.append(
            "\t".join(
                [
                    str(offer.transport or offer.provider),
                    str(offer.creator or ""),
                    str(offer.model or ""),
                    "" if access is None else access.access_class.value,
                    remaining,
                    "" if reset is None else str(reset),
                    str(offer.capacity.context_window),
                    ",".join(offer.tools) if offer.tools else "-",
                    str(offer.source),
                    str(offer.freshness or ""),
                ]
            )
        )
    return "\n".join(lines)


def _req(ns: argparse.Namespace) -> WorkRequirement:
    flags = WorkRequirement(
        coding=ns.coding,
        reasoning=ns.reasoning,
        tool_use=not ns.no_tools,
        context=ns.context,
        parallelism=ns.workers,
        maximum_cost=ns.budget,
        privacy=ns.privacy,
        mode=Mode(ns.mode),
        tools=("github",) if not ns.no_tools else (),
        capability_id=getattr(ns, "capability_id", None),
    )
    path = getattr(ns, "aodl", None)
    alloc = getattr(ns, "allocation_request", None)
    if alloc:
        return work_requirement_from_allocation_request(load_allocation_request(alloc), defaults=flags)
    if not path:
        return flags
    return work_requirement_from_aodl(load_aodl(path), defaults=flags)



def _inventory_row(offer) -> dict:
    access = offer.access
    remaining = None
    if access is not None and access.remaining_known:
        remaining = {
            "free_quota": offer.economics.remaining_free_quota,
            "credits": offer.economics.remaining_credits,
        }
    reset = offer.economics.seconds_until_quota_reset or offer.economics.seconds_until_credits_expire
    return {
        "id": offer.id,
        "transport": offer.transport or offer.provider,
        "provider": offer.provider,
        "creator": offer.creator,
        "model": offer.model,
        "access_class": None if access is None else access.access_class.value,
        "remaining": remaining,
        "remaining_known": False if access is None else access.remaining_known,
        "reset_or_expiry": reset,
        "concurrency": offer.capacity.concurrency,
        "context": offer.capacity.context_window,
        "source": offer.source,
        "freshness": offer.freshness,
        "free": is_free(offer),
        "prices": {
            "input": offer.economics.input_token_price,
            "output": offer.economics.output_token_price,
        },
        "tools": list(offer.tools),
        "access": None if access is None else access.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kerdoios", description="Hermes Kerdoios — allot scarce compute")
    sub = parser.add_subparsers(dest="command", required=True)

    inv_p = sub.add_parser("inventory", help="Show resource inventory")
    inv_p.add_argument("--live", action="store_true", help="Also query live adapters (OpenRouter public, local, keyed Groq/Cerebras)")
    inv_p.add_argument("--no-fixture", action="store_true", help="Omit the deterministic fixture catalog")
    inv_p.add_argument(
        "--free",
        action="store_true",
        help="Keep free models as the initial list (OpenRouter public/snapshot, plus keyed Groq/Cerebras free-tier; implies live, omits fixture)",
    )
    inv_p.add_argument("--refresh", action="store_true", help="Bypass inventory cache and fetch live OpenRouter")
    inv_p.add_argument("--include-fixture", action="store_true", help="Include deterministic fixture rows in inventory output")
    inv_p.add_argument("--json", action="store_true", help="Emit complete typed JSON (default for inventory)")
    inv_p.add_argument("--table", action="store_true", help="Emit a human-readable free-capacity table")

    doctor_p = sub.add_parser("doctor", help="Diagnose provider auth/catalog without running inventory")
    doctor_p.add_argument("provider", nargs="?", default=None, help="Provider name (default: all registered)")
    doctor_p.add_argument("--probe", action="store_true", help="Opt-in: one minimal chat completion against an accessible model")

    plan_p = sub.add_parser("plan", help="Build an execution portfolio")
    explain_p = sub.add_parser("explain", help="Print why workers were placed")
    for item in (plan_p, explain_p):
        item.add_argument("--live", action="store_true")
        item.add_argument(
            "--free",
            action="store_true",
            help="Keep free models as the initial list (OpenRouter public/snapshot plus keyed Groq/Cerebras free-tier; implies live, omits fixture)",
        )
        item.add_argument("--observed", action="store_true", help="Blend in observed execution outcomes (KERDOIOS_OBSERVED_LOG or ~/.hermes/cache/kerdoios/observed.jsonl)")
        item.add_argument("--workers", type=int, default=8)
        item.add_argument("--budget", type=float, default=None)
        item.add_argument("--mode", default="cheap", choices=[m.value for m in Mode])
        item.add_argument("--context", type=int, default=128_000)
        item.add_argument("--privacy", default="public", choices=["public", "confidential", "local_only"])
        item.add_argument("--coding", type=float, default=0.7)
        item.add_argument("--reasoning", type=float, default=0.6)
        item.add_argument("--no-tools", action="store_true")
        item.add_argument(
            "--aodl",
            default=None,
            help="AODL-shaped work spec JSON path (or - for stdin)",
        )
        item.add_argument(
            "--capability-id",
            dest="capability_id",
            default=None,
            help="z0int capability id for residual allocation (e.g. coding.delegate)",
        )
        item.add_argument(
            "--allocation-request",
            dest="allocation_request",
            default=None,
            help="z0int.allocation_request.v1 JSON path (or - for stdin)",
        )

    record_p = sub.add_parser("record", help="Log an observed execution outcome")
    record_p.add_argument("--provider", required=True)
    record_p.add_argument("--model", required=True)
    record_p.add_argument("--task-type", default="unknown")
    record_p.add_argument("--completed", action="store_true")
    record_p.add_argument("--cost", type=float, default=0.0)
    record_p.add_argument("--retried", action="store_true")
    record_p.add_argument("--capability-id", dest="capability_id", default=None)
    record_p.add_argument("--input-tokens", dest="input_tokens", type=int, default=None)
    record_p.add_argument("--output-tokens", dest="output_tokens", type=int, default=None)
    record_p.add_argument("--cached-input-tokens", dest="cached_input_tokens", type=int, default=None)
    record_p.add_argument("--context-tokens", dest="context_tokens", type=int, default=None)
    record_p.add_argument("--latency-ms", dest="latency_ms", type=float, default=None)
    record_p.add_argument("--fallback-count", dest="fallback_count", type=int, default=0)
    record_p.add_argument("--quota-before", dest="quota_before", type=float, default=None)
    record_p.add_argument("--quota-after", dest="quota_after", type=float, default=None)
    record_p.add_argument("--execution-completed", dest="execution_completed", action="store_true")
    record_p.add_argument("--verified-success", dest="verified_success", action="store_true")
    record_p.add_argument("--verified-null", dest="verified_null", action="store_true", help="explicitly leave verified_success null")
    record_p.add_argument("--harness-id", dest="harness_id", default=None)
    record_p.add_argument("--session-id", dest="session_id", default=None)
    record_p.add_argument("--trace-id", dest="trace_id", default=None)
    record_p.add_argument("--turn-id", dest="turn_id", default=None)
    record_p.add_argument("--quota-group", dest="quota_group", default=None)
    obs_p = sub.add_parser("observations", help="Observation log utilities")
    obs_sub = obs_p.add_subparsers(dest="obs_command", required=True)
    obs_imp = obs_sub.add_parser("import", help="Import z0int.allocation_observation.v1 JSONL")
    obs_imp.add_argument("path", type=Path, help="JSONL path")
    obs_imp.add_argument("--dest", type=Path, default=None, help="observed.jsonl override")
    econ_p = sub.add_parser("economics", help="Tokens/cost per verified task from observed log")
    econ_p.add_argument("--capability-id", dest="capability_id", default=None)
    econ_p.add_argument("--path", type=Path, default=None, help="observed.jsonl override")


    ns = parser.parse_args(argv)
    if ns.command == "observations":
        if ns.obs_command == "import":
            print(json.dumps(import_allocation_observations(ns.path, dest=ns.dest), indent=2))
            return 0
        print("unknown observations subcommand", file=sys.stderr)
        return 2
    if ns.command == "record":
        record(
            Observation(
                provider=ns.provider,
                model=ns.model,
                task_type=ns.task_type,
                completed=ns.completed or bool(getattr(ns, "execution_completed", False)),
                actual_cost=ns.cost,
                retried=ns.retried,
                capability_id=ns.capability_id,
                input_tokens=ns.input_tokens,
                output_tokens=ns.output_tokens,
                cached_input_tokens=ns.cached_input_tokens,
                context_tokens=ns.context_tokens,
                latency_ms=ns.latency_ms,
                fallback_count=ns.fallback_count or 0,
                quota_before=ns.quota_before,
                quota_after=ns.quota_after,
                execution_completed=bool(getattr(ns, "execution_completed", False)) or ns.completed,
                verified_success=(
                    None
                    if getattr(ns, "verified_null", False)
                    else (True if getattr(ns, "verified_success", False) else None)
                ),
                harness_id=getattr(ns, "harness_id", None),
                session_id=getattr(ns, "session_id", None),
                trace_id=getattr(ns, "trace_id", None),
                turn_id=getattr(ns, "turn_id", None),
                quota_group=getattr(ns, "quota_group", None),
            )
        )
        return 0
    if ns.command == "economics":
        print(
            json.dumps(
                tokens_per_verified_task(
                    capability_id=getattr(ns, "capability_id", None),
                    path=getattr(ns, "path", None),
                ),
                indent=2,
            )
        )
        return 0
    if ns.command == "doctor":
        from .doctor import run_doctor

        print(json.dumps(run_doctor(getattr(ns, "provider", None), probe=bool(getattr(ns, "probe", False))), indent=2))
        return 0
    requirement = None
    if ns.command != "inventory":
        try:
            requirement = _req(ns)
        except AodlIngestError as exc:
            print(f"aodl: {exc}", file=sys.stderr)
            return 2
        except AllocationRequestError as exc:
            print(f"allocation_request: {exc}", file=sys.stderr)
            return 2

    include_fixture = True
    live = False
    free_only = bool(getattr(ns, "free", False))
    refresh = bool(getattr(ns, "refresh", False))
    if ns.command == "inventory":
        include_fixture = bool(getattr(ns, "include_fixture", False)) or (not ns.no_fixture and not free_only)
        if free_only and not getattr(ns, "include_fixture", False):
            include_fixture = False
        if getattr(ns, "no_fixture", False):
            include_fixture = False
        live = ns.live
    else:
        live = bool(getattr(ns, "live", False))
    if free_only:
        live = True
        if not getattr(ns, "include_fixture", False):
            include_fixture = False
    offers = discover_all(
        include_fixture=include_fixture,
        live=live,
        free_only=free_only,
        refresh=refresh,
    )
    if ns.command == "inventory":
        rows = [_inventory_row(o) for o in offers]
        if getattr(ns, "table", False):
            print(_inventory_table(offers))
        else:
            print(json.dumps(rows, indent=2))
        return 0
    built = plan(offers, requirement, use_observed=bool(getattr(ns, "observed", False)))
    if ns.command == "explain":
        print(explain(offers, requirement, built))
        return 0
    print(json.dumps(built.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
