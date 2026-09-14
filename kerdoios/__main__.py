from __future__ import annotations

import argparse
import json
import sys

from .aodl import AodlIngestError, load_aodl, work_requirement_from_aodl
from .explain import explain
from .inventory import discover_all
from .observed import Observation, record
from .optimize import plan
from .providers.free import is_free
from .types import Mode, WorkRequirement


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
    )
    path = getattr(ns, "aodl", None)
    if not path:
        return flags
    return work_requirement_from_aodl(load_aodl(path), defaults=flags)


def _inventory_row(offer) -> dict:
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

    record_p = sub.add_parser("record", help="Log an observed execution outcome")
    record_p.add_argument("--provider", required=True)
    record_p.add_argument("--model", required=True)
    record_p.add_argument("--task-type", default="unknown")
    record_p.add_argument("--completed", action="store_true")
    record_p.add_argument("--cost", type=float, default=0.0)
    record_p.add_argument("--retried", action="store_true")
    record_p.add_argument(
        "--reason",
        default=None,
        choices=["rate_limited", "timeout", "upstream_5xx", "empty_response", "worker_crash", "unknown"],
        help="Failure taxonomy; omit on success",
    )

    run_p = sub.add_parser(
        "run",
        help="Run a plan's tasks with self-healing retries and fallback failover",
    )
    run_p.add_argument("--plan", required=True, help="ExecutionPlan JSON file (from `kerdoios plan`)")
    run_p.add_argument("--tasks", required=True, help="JSONL file, one task object per line")
    run_p.add_argument(
        "--worker-cmd",
        required=True,
        help="Worker command template; {model}, {provider}, {task_id} are substituted, "
        "task JSON goes on stdin. See kerdoios/heal.py for the worker contract.",
    )
    run_p.add_argument("--jobs", type=int, default=None, help="Thread-pool size (default: min(32, tasks))")
    run_p.add_argument("--out", default=None, help="Write result JSONL here (default: stdout)")
    run_p.add_argument("--task-type", default="unknown")
    run_p.add_argument("--max-retries", type=int, default=None)
    run_p.add_argument("--backoff-s", type=float, default=None)
    run_p.add_argument("--timeout", type=float, default=None, help="Per-attempt seconds")
    run_p.add_argument("--observed-log", default=None)

    ns = parser.parse_args(argv)
    if ns.command == "record":
        record(
            Observation(
                provider=ns.provider,
                model=ns.model,
                task_type=ns.task_type,
                completed=ns.completed,
                actual_cost=ns.cost,
                retried=ns.retried,
                reason=ns.reason,
            )
        )
        return 0
    if ns.command == "run":
        from .heal import run_batch

        with open(ns.plan, encoding="utf-8") as fh:
            plan_doc = json.load(fh)
        tasks = []
        with open(ns.tasks, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        if not tasks:
            print("run: no tasks in --tasks file", file=sys.stderr)
            return 2
        results = run_batch(
            plan_doc,
            tasks,
            ns.worker_cmd,
            jobs=ns.jobs,
            task_type=ns.task_type,
            observed_log=ns.observed_log,
            max_retries=ns.max_retries,
            backoff_s=ns.backoff_s,
            attempt_timeout_s=ns.timeout,
            out=ns.out,
        )
        if not ns.out:
            for result in results:
                print(json.dumps(result))
        return 0
    requirement = None
    if ns.command != "inventory":
        try:
            requirement = _req(ns)
        except AodlIngestError as exc:
            print(f"aodl: {exc}", file=sys.stderr)
            return 2
    include_fixture = True
    live = False
    free_only = bool(getattr(ns, "free", False))
    refresh = bool(getattr(ns, "refresh", False))
    if ns.command == "inventory":
        include_fixture = not ns.no_fixture
        live = ns.live
    else:
        live = bool(getattr(ns, "live", False))
    if free_only:
        live = True
        include_fixture = False
    offers = discover_all(
        include_fixture=include_fixture,
        live=live,
        free_only=free_only,
        refresh=refresh,
    )
    if ns.command == "inventory":
        print(json.dumps([_inventory_row(o) for o in offers], indent=2))
        return 0
    built = plan(offers, requirement, use_observed=bool(getattr(ns, "observed", False)))
    if ns.command == "explain":
        print(explain(offers, requirement, built))
        return 0
    print(json.dumps(built.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
