from __future__ import annotations

import argparse
import json
import sys

from .explain import explain
from .inventory import discover_all
from .optimize import plan
from .providers.free import is_free
from .types import Mode, WorkRequirement


def _req(ns: argparse.Namespace) -> WorkRequirement:
    return WorkRequirement(
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
        item.add_argument("--fixture", action="store_true", default=True)
        item.add_argument("--live", action="store_true")
        item.add_argument(
            "--free",
            action="store_true",
            help="Keep free models as the initial list (OpenRouter public/snapshot plus keyed Groq/Cerebras free-tier; implies live, omits fixture)",
        )
        item.add_argument("--workers", type=int, default=8)
        item.add_argument("--budget", type=float, default=None)
        item.add_argument("--mode", default="cheap", choices=[m.value for m in Mode])
        item.add_argument("--context", type=int, default=128_000)
        item.add_argument("--privacy", default="public", choices=["public", "confidential", "local_only"])
        item.add_argument("--coding", type=float, default=0.7)
        item.add_argument("--reasoning", type=float, default=0.6)
        item.add_argument("--no-tools", action="store_true")

    ns = parser.parse_args(argv)
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
    requirement = _req(ns)
    built = plan(offers, requirement)
    if ns.command == "explain":
        print(explain(offers, requirement, built))
        return 0
    print(json.dumps(built.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
