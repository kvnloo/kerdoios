"""Kubernetes capacity as ordinary ResourceOffers, not a side-channel.

Kubernetes used to be reached by an inline ``kubectl`` call inside
``projection.k8s_rows()``, gated behind a ``--k8s`` flag and absent from
``discover_all``. That made it a second discovery path with its own shape
(``RuntimeEntry``) which dropped economics and quota entirely, and it meant a
cluster could not be offered to placement the way every other substrate is.

This module makes the cluster an ordinary :class:`ResourceProvider`. It emits
``ResourceOffer``s on the same ``discover()`` contract as local, Groq and
Cerebras, so Kerdoios placement consumes it identically and nothing outside the
adapter/placement boundary needs an ``if kubernetes``.

Three properties are deliberate:

* **Optional.** Discovery is off unless ``KERDOIOS_K8S=1`` or a caller asks
  explicitly, so Kubernetes never becomes a required substrate.
* **Observed.** Availability comes from the node's ``Ready`` condition and
  ``spec.unschedulable``, never from the fact that a node was listed. A
  ``NotReady`` or cordoned node reports zero availability.
* **Honest about what it is.** A node is not a model. ``model`` is ``None``; the
  node name lives in ``id`` and ``node``. ``resource_type`` is ``gpu`` only when
  ``nvidia.com/gpu`` is actually allocatable.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from ..types import CapabilityProfile, Capacity, Economics, ResourceOffer, Telemetry
from .base import ResourceProvider

NAME = "kubernetes"

# A node with no allocatable GPU is CPU capacity; one with an allocatable
# nvidia.com/gpu is GPU capacity. Nothing here assumes a GPU exists.
GPU_RESOURCE = "nvidia.com/gpu"


def _parse_quantity(value: Any) -> float | None:
    """Parse a Kubernetes quantity ("4", "1500m", "8Gi", "1Ti") to a float.

    Returns None for anything unparseable rather than guessing: an unknown
    capacity must not silently become zero.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    suffixes = {
        "Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4, "Pi": 1024**5,
        "K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4, "P": 1000**5,
        "k": 1000,
    }
    for suffix, factor in sorted(suffixes.items(), key=lambda kv: -len(kv[0])):
        if text.endswith(suffix):
            try:
                return float(text[: -len(suffix)]) * factor
            except ValueError:
                return None
    if text.endswith("m"):  # milli-units, e.g. 1500m CPU cores
        try:
            return float(text[:-1]) / 1000.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _kubectl_json(args: list[str], *, kubeconfig: str | None = None, timeout: float = 10.0) -> dict | None:
    env = dict(os.environ)
    if kubeconfig:
        env["KUBECONFIG"] = kubeconfig
    try:
        proc = subprocess.run(
            ["kubectl", *args], capture_output=True, text=True, timeout=timeout, env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def node_readiness(node: dict) -> tuple[bool, bool, str]:
    """(ready, unschedulable, reason) from a node object, as observed."""
    conditions = {
        str(c.get("type")): c for c in (node.get("status", {}).get("conditions") or [])
    }
    ready_condition = conditions.get("Ready") or {}
    is_ready = str(ready_condition.get("status")) == "True"
    unschedulable = bool(node.get("spec", {}).get("unschedulable"))
    if unschedulable:
        reason = "cordoned (spec.unschedulable)"
    elif is_ready:
        reason = "node Ready"
    else:
        reason = str(ready_condition.get("reason") or "Ready condition not True")
    return is_ready, unschedulable, reason


def node_offers(*, kubeconfig: str | None = None, nodes: list[dict] | None = None) -> list[ResourceOffer]:
    """One offer per node. Pass ``nodes`` to bypass kubectl (used by tests)."""
    if nodes is None:
        payload = _kubectl_json(["get", "nodes", "-o", "json"], kubeconfig=kubeconfig)
        if not payload:
            return []
        nodes = payload.get("items") or []

    offers: list[ResourceOffer] = []
    for node in nodes:
        name = str((node.get("metadata") or {}).get("name") or "")
        if not name:
            continue
        alloc = (node.get("status") or {}).get("allocatable") or {}
        is_ready, unschedulable, reason = node_readiness(node)
        usable = is_ready and not unschedulable
        gpu_quantity = alloc.get(GPU_RESOURCE)
        gpu_count = _parse_quantity(gpu_quantity) or 0.0

        offers.append(
            ResourceOffer(
                id=f"k8s/node/{name}",
                provider=NAME,
                # "gpu" only when a GPU is genuinely allocatable.
                resource_type="gpu" if gpu_count > 0 else "cpu",
                # A node is not a model. The name lives in `id` and `node`.
                model=None,
                local=True,
                capabilities=CapabilityProfile(provenance="observed_execution"),
                capacity=Capacity(
                    concurrency=int(_parse_quantity(alloc.get("pods")) or 1),
                    cpu_cores=_parse_quantity(alloc.get("cpu")),
                    ram_gb=(
                        _parse_quantity(alloc.get("memory")) / 1024**3
                        if _parse_quantity(alloc.get("memory")) is not None
                        else None
                    ),
                    vram_gb=float(gpu_count) if gpu_count else None,
                ),
                economics=Economics(),
                # Availability is observed, not assumed. A NotReady or cordoned
                # node is offered at zero availability so placement can see it
                # and decline it, rather than being told it is runnable.
                telemetry=Telemetry(
                    latency_p50_ms=0.0 if usable else float("inf"),
                    latency_p95_ms=0.0 if usable else float("inf"),
                    failure_rate=0.0 if usable else 1.0,
                    availability=0.99 if usable else 0.0,
                ),
                tools=(),
                privacy_ok=("public", "confidential", "local_only"),
                source=f"k8s:node-readiness:{reason}",
                confidence=0.7 if usable else 0.2,
            )
        )
    return offers


def enabled() -> bool:
    """Kubernetes discovery is opt-in, never a required substrate."""
    return os.environ.get("KERDOIOS_K8S", "").strip().lower() in {"1", "true", "yes", "on"}


def discover(*, kubeconfig: str | None = None, force: bool = False) -> list[ResourceOffer]:
    """Offer Kubernetes nodes like any other provider. Empty unless enabled."""
    if not force and not enabled():
        return []
    return node_offers(kubeconfig=kubeconfig)


class KubernetesProvider(ResourceProvider):
    name = NAME

    def __init__(self, *, kubeconfig: str | None = None, force: bool = False) -> None:
        self._kubeconfig = kubeconfig
        self._force = force

    def discover(self) -> list[ResourceOffer]:
        return discover(kubeconfig=self._kubeconfig, force=self._force)
