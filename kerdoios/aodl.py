"""Map AODL-shaped work specs onto WorkRequirement.

Kerdoios does not vendor AODL. This reads JSON that looks like an AODL
work spec (specVersion plus work / constraints / intentGraph) and copies
the fields the planner already understands.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .types import PrivacyClass, WorkRequirement

_PRIVACY: dict[str, PrivacyClass] = {
    "public": "public",
    "confidential": "confidential",
    "local_only": "local_only",
}


class AodlIngestError(ValueError):
    """AODL-shaped document could not be mapped onto WorkRequirement."""


def load_aodl(path: str) -> Any:
    if path == "-":
        raw = sys.stdin.read()
    else:
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise AodlIngestError(f"cannot read {path}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AodlIngestError(f"invalid JSON: {exc.msg}") from exc


def work_requirement_from_aodl(
    doc: Any,
    *,
    defaults: WorkRequirement | None = None,
) -> WorkRequirement:
    if not isinstance(doc, dict):
        raise AodlIngestError("document must be a JSON object")
    spec = doc.get("specVersion")
    if not isinstance(spec, str) or not spec.strip():
        raise AodlIngestError("missing specVersion")

    work = _optional_object(doc.get("work"), "work") or {}
    constraints = _optional_object(doc.get("constraints"), "constraints") or {}
    policies = _optional_object(doc.get("policies"), "policies") or {}
    base = defaults or WorkRequirement()

    context = _optional_int(
        _first(work.get("context"), constraints.get("context"), doc.get("context")),
        field="context",
    )
    parallelism = _optional_int(
        _first(
            work.get("parallelism"),
            work.get("workers"),
            constraints.get("parallelism"),
            _positive_max_children(policies),
            doc.get("parallelism"),
            doc.get("workers"),
        ),
        field="parallelism",
    )
    privacy = _optional_privacy(
        _first(work.get("privacy"), constraints.get("privacy"), doc.get("privacy"))
    )
    tools = _optional_tools(
        _first(work.get("tools"), doc.get("tools"), _tool_node_ids(doc))
    )
    budget = _optional_budget(
        _first(
            work.get("budget"),
            work.get("maximum_cost"),
            _budget_usd(constraints),
            doc.get("budget"),
            doc.get("maximum_cost"),
        )
    )

    mapped_tools = base.tools
    tool_use = base.tool_use
    if tools is not None:
        mapped_tools = tools
        tool_use = bool(tools)

    return replace(
        base,
        context=context if context is not None else base.context,
        parallelism=parallelism if parallelism is not None else base.parallelism,
        privacy=privacy if privacy is not None else base.privacy,
        tools=mapped_tools,
        tool_use=tool_use,
        maximum_cost=budget if budget is not None else base.maximum_cost,
    )


def _optional_object(value: Any, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise AodlIngestError(f"{field} must be an object")
    return value


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _positive_max_children(policies: dict[str, Any]) -> Any:
    dynamic = policies.get("dynamic")
    if dynamic is None:
        return policies.get("parallelism")
    if not isinstance(dynamic, dict):
        raise AodlIngestError("policies.dynamic must be an object")
    if "maxChildren" not in dynamic:
        return policies.get("parallelism")
    value = dynamic["maxChildren"]
    if value == 0:
        return policies.get("parallelism")
    return value


def _tool_node_ids(doc: dict[str, Any]) -> tuple[str, ...] | None:
    graph = doc.get("intentGraph")
    if graph is None:
        return None
    if not isinstance(graph, dict):
        raise AodlIngestError("intentGraph must be an object")
    nodes = graph.get("nodes")
    if nodes is None:
        return None
    if not isinstance(nodes, list):
        raise AodlIngestError("intentGraph.nodes must be a list")
    tools: list[str] = []
    for node in nodes:
        if not isinstance(node, dict):
            raise AodlIngestError("intentGraph node must be an object")
        if node.get("kind") != "tool":
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise AodlIngestError("tool node id must be a string")
        tools.append(node_id)
    return tuple(tools) if tools else None


def _budget_usd(constraints: dict[str, Any]) -> Any:
    budgets = constraints.get("budgets")
    if budgets is None:
        return None
    if not isinstance(budgets, dict):
        raise AodlIngestError("constraints.budgets must be an object")
    return _first(budgets.get("usd"), budgets.get("money"), budgets.get("cost"))


def _optional_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AodlIngestError(f"{field} must be an integer")
    if isinstance(value, float) and value != int(value):
        raise AodlIngestError(f"{field} must be an integer")
    number = int(value)
    if number < 1:
        raise AodlIngestError(f"{field} must be >= 1")
    return number


def _optional_privacy(value: Any) -> PrivacyClass | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AodlIngestError("privacy must be public, confidential, or local_only")
    mapped = _PRIVACY.get(value)
    if mapped is None:
        raise AodlIngestError("privacy must be public, confidential, or local_only")
    return mapped


def _optional_tools(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        if all(isinstance(item, str) and item for item in value):
            return value
        raise AodlIngestError("tools must be a list of strings")
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise AodlIngestError("tools must be a list of strings")
    return tuple(value)


def _optional_budget(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AodlIngestError("budget must be a number")
    if value < 0:
        raise AodlIngestError("budget must be >= 0")
    return float(value)
