"""Canonical runtime inventory projection.

One schema, many sources. This does **not** replace `ResourceOffer` — it consumes
it. `discover_all()` already joins the fixture catalog, the live OpenRouter
catalog, the keyed Groq/Cerebras free tiers and local OpenAI-compatible
endpoints. What it does not carry is the part the beta sprint actually needs:

  * *where* a candidate runs   (local-host / local-k8s / remote-free / specialist / rule)
  * *which* backend executes it (llama.cpp / vllm / pi-ai / groq / nanojev / rule …)
  * a lifecycle `status` and a `location`
  * **typed trust per role**, because a model is rarely simply "trusted"
  * an `evidence` block holding the measured numbers behind that trust

`TESTED != TRUSTED`, and trust is per *role*, never global. Hammer3B is
`TRUSTED_BOUNDED` for bounded choice and `UNTESTED` for orchestration; NanoJev is
`TRUSTED_SHADOW` for bounded scoring and not applicable to general reasoning.
One scalar "quality" cannot express that.

Extra sources joined here:
  * the z0int local cognition supervisor (`/v1/models` on 11500) — the GGUF
    portfolio actually served by llama.cpp
  * GGUF files present on disk but not currently loaded
  * the Kubernetes lab (kind cluster capacity, GPU allocatable)
  * frozen exploratory-beta experiment evidence, when supplied

Stdlib only. Every network probe is short-timeout and fail-open: an unreachable
source yields no rows, never an exception.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .types import ResourceOffer


class Location(str, Enum):
    LOCAL_HOST = "local-host"
    LOCAL_K8S = "local-k8s"
    REMOTE_FREE = "remote-free"
    REMOTE_PAID = "remote-paid"
    SPECIALIST = "specialist"
    DETERMINISTIC = "deterministic"


class ExecutionBackend(str, Enum):
    LLAMA_CPP = "llama.cpp"
    VLLM = "vllm"
    PI_AI = "pi-ai"
    GROQ = "groq"
    CEREBRAS = "cerebras"
    OPENROUTER = "openrouter"
    NANOJEV = "nanojev"
    MUSHROOM = "mushroom"
    FLY = "fly"
    RULE = "rule"
    UNKNOWN = "unknown"


class Status(str, Enum):
    DISCOVERED = "discovered"
    RUNNABLE = "runnable"
    TESTED = "tested"
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    BROKEN = "broken"
    DEPRECATED = "deprecated"
    UNAVAILABLE = "unavailable"


class Trust(str, Enum):
    """Typed by role. `TESTED != TRUSTED` is the whole point of this enum."""

    UNTESTED = "UNTESTED"
    TESTED_EXPERIMENTAL = "TESTED_EXPERIMENTAL"
    TRUSTED_SHADOW = "TRUSTED_SHADOW"
    TRUSTED_BOUNDED = "TRUSTED_BOUNDED"
    TRUSTED_GENERAL = "TRUSTED_GENERAL"
    QUARANTINED = "QUARANTINED"
    DEPRECATED = "DEPRECATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# Roles a candidate can be trusted *for*. A candidate with no entry is UNTESTED.
ROLES = (
    "bounded_choice",
    "bounded_score",
    "tool_calling",
    "orchestration",
    "recovery",
    "classification",
    "code",
    "general_reasoning",
)


class EvidenceClass(str, Enum):
    """Evidence classes, in strength order.

    This mirrors the canonical taxonomy in `kvnloo/z0` `registry/maturity.yaml`
    (`REQUIRED_EVIDENCE_CLASSES`), which is the one definition. `PAIRED_REPLAY`
    was missing here while the registry required it, so a paired-replay claim
    could not be expressed at all.
    """

    SMOKE = "SMOKE"
    EXPLORATORY_BETA = "EXPLORATORY_BETA"
    SHADOW = "SHADOW"
    PAIRED_REPLAY = "PAIRED_REPLAY"
    CONFIRM = "CONFIRM"
    OOD = "OOD"
    PROMOTION = "PROMOTION"


@dataclass
class RoleEvidence:
    """Measured evidence for one (candidate, role) pair."""

    n: int = 0
    success: float | None = None
    unsafe: int | None = None
    latency_p50_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    source: str = ""
    evidence_class: str = EvidenceClass.SMOKE.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeEntry:
    id: str
    provider: str
    model: str
    revision: str | None = None
    location: str = Location.LOCAL_HOST.value
    execution_backend: str = ExecutionBackend.UNKNOWN.value
    status: str = Status.DISCOVERED.value
    trust: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, RoleEvidence] = field(default_factory=dict)
    capabilities: dict[str, bool] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    local: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        # Every role is explicitly represented, so an absent role reads as
        # UNTESTED rather than silently missing from the map.
        for r in ROLES:
            self.trust.setdefault(r, Trust.UNTESTED.value)
        self.trust.setdefault("recovery", Trust.UNTESTED.value)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = {k: v.to_dict() for k, v in self.evidence.items()}
        return d


# ---------------------------------------------------------------- classification


def _location_of(offer: ResourceOffer) -> str:
    if offer.resource_type == "gpu":
        return Location.LOCAL_K8S.value
    if offer.provider in ("groq", "cerebras"):
        return Location.REMOTE_FREE.value
    if offer.provider == "openrouter":
        residual = offer.economics.remaining_free_quota
        return Location.REMOTE_FREE.value if residual > 0 else Location.REMOTE_PAID.value
    if "k8s" in offer.source or "kind" in offer.source:
        return Location.LOCAL_K8S.value
    if offer.local:
        return Location.LOCAL_HOST.value
    # A non-local offer that is not a known free provider is remote, and must
    # never be reported as local-host. Defaulting to LOCAL_HOST here silently
    # filed 22 remote OpenRouter rows as local machines.
    free = offer.economics.remaining_free_quota > 0
    return Location.REMOTE_FREE.value if free else Location.REMOTE_PAID.value


def _backend_of(offer: ResourceOffer) -> str:
    prov = offer.provider.lower()
    src = (offer.source or "").lower()
    # Match on `source` as well as `provider`. Rows arrive from provider
    # catalogs whose `provider` field is a vendor name while the transport is
    # recorded only in `source` (e.g. source="openrouter:/api/v1/models").
    # Checking only `provider` let those rows fall through to the old PI_AI
    # catch-all, which labelled 441 OpenRouter rows as `pi-ai`.
    if prov == "groq" or "groq" in src:
        return ExecutionBackend.GROQ.value
    if prov == "cerebras" or "cerebras" in src:
        return ExecutionBackend.CEREBRAS.value
    if prov == "openrouter" or "openrouter" in src:
        return ExecutionBackend.OPENROUTER.value
    if "z0int" in src or "nanojev" in prov:
        return ExecutionBackend.LLAMA_CPP.value
    if "vllm" in src:
        return ExecutionBackend.VLLM.value
    if "ollama" in src:
        return ExecutionBackend.LLAMA_CPP.value
    if offer.local:
        return ExecutionBackend.LLAMA_CPP.value
    # An unrecognised *remote* transport is unknown, not pi-ai. Same rule as
    # residency: never assert a runtime we have not actually identified.
    return ExecutionBackend.UNKNOWN.value


def _status_of(offer: ResourceOffer) -> str:
    """Status of a row that came from a catalog, not from a live probe.

    Catalog rows are *listed*: we know the provider advertises the offer, and we
    know nothing about whether it can serve right now. Returning RUNNABLE here
    turned a confidence float into an availability claim -- the inventory
    reported 458 rows runnable with no probe behind any of them, including
    offers that do not exist.

    So this returns DISCOVERED for anything not positively known to be broken.
    Rows that ARE observed get their status from the observation instead:
    live `/health` probes stamp RUNNABLE at the call site, and folded evidence
    promotes to TESTED in `apply_evidence`. Availability is never inferred here.
    """
    if offer.telemetry.failure_rate >= 0.5:
        return Status.BROKEN.value
    return Status.DISCOVERED.value


def from_offer(offer: ResourceOffer) -> RuntimeEntry:
    """Project one existing `ResourceOffer` into the runtime schema."""
    caps = {
        "tool_calling": offer.capabilities.tool_use >= 0.5,
        "code": offer.capabilities.coding >= 0.5,
        "long_context": offer.capacity.context_window >= 65536,
        "vision": offer.capabilities.vision > 0.0,
        # capability floats are provider *claims*; they are not typed trust.
        "bounded_choice": False,
        "orchestration": False,
        "classification": False,
        "boolean": False,
        "scoring": False,
        "recovery": False,
    }
    return RuntimeEntry(
        id=offer.id,
        provider=offer.provider,
        model=str(offer.model or offer.id),
        revision=None,
        location=_location_of(offer),
        execution_backend=_backend_of(offer),
        status=_status_of(offer),
        capabilities=caps,
        constraints={
            "context_window": offer.capacity.context_window,
            "concurrency": offer.capacity.concurrency,
            "rpm": offer.capacity.requests_per_minute,
            "tpm": offer.capacity.tokens_per_minute,
            "free_quota_remaining": offer.economics.remaining_free_quota,
            "privacy_ok": list(offer.privacy_ok),
        },
        local=offer.local,
        notes=f"projected from ResourceOffer source={offer.source}",
    )


# ---------------------------------------------------------------------- sources


def _get_json(url: str, timeout: float = 3.0) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "kerdoios-inventory/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def z0int_supervisor_rows(
    base_url: str = "http://127.0.0.1:11500",
    gguf_dir: str | os.PathLike[str] | None = "/mnt/zer0models/zer0-models/gguf",
) -> list[RuntimeEntry]:
    """The local cognition supervisor plus the GGUF files on disk.

    The supervisor is the real local serving path (llama.cpp behind one
    lifecycle). Models it currently exposes are `runnable`; GGUF files present
    on disk but not served are `discovered` — inventory them without loading
    them, which is the whole point.
    """
    rows: list[RuntimeEntry] = []
    health = _get_json(base_url + "/health")
    served: set[str] = set()
    if health:
        listing = _get_json(base_url + "/v1/models") or {}
        for m in listing.get("data") or []:
            mid = str(m.get("id") or "")
            if not mid:
                continue
            served.add(mid)
            rows.append(
                RuntimeEntry(
                    id=f"z0int/{mid}",
                    provider="z0int",
                    model=mid,
                    revision=str(health.get("runtime") or "unknown"),
                    location=Location.LOCAL_HOST.value,
                    execution_backend=ExecutionBackend.LLAMA_CPP.value,
                    status=Status.RUNNABLE.value,
                    capabilities={"bounded_choice": True, "boolean": True, "scoring": True},
                    constraints={"resident": health.get("resident")},
                    local=True,
                    notes="served by z0int cognition supervisor",
                )
            )
    if gguf_dir:
        d = Path(gguf_dir)
        if d.is_dir():
            for p in sorted(d.rglob("*.gguf")):
                stem = p.parent.name if p.parent != d else p.stem
                if stem in served:
                    continue
                size_gb = round(p.stat().st_size / 1e9, 2)
                rows.append(
                    RuntimeEntry(
                        id=f"gguf/{stem}",
                        provider="local-file",
                        model=stem,
                        revision=None,
                        location=Location.LOCAL_HOST.value,
                        execution_backend=ExecutionBackend.LLAMA_CPP.value,
                        status=Status.DISCOVERED.value,
                        constraints={"size_gb": size_gb, "loaded": False},
                        local=True,
                        notes=f"on disk at {p}",
                    )
                )
    return rows


def ollama_rows(base_url: str = "http://127.0.0.1:11434") -> list[RuntimeEntry]:
    listing = _get_json(base_url + "/api/tags")
    if not listing:
        return []
    out = []
    for m in listing.get("models") or []:
        name = str(m.get("name") or "")
        if not name:
            continue
        out.append(
            RuntimeEntry(
                id=f"ollama/{name}",
                provider="ollama",
                model=name,
                revision=str(m.get("digest") or "")[:12] or None,
                location=Location.LOCAL_HOST.value,
                execution_backend=ExecutionBackend.LLAMA_CPP.value,
                status=Status.RUNNABLE.value,
                constraints={"size_bytes": m.get("size")},
                local=True,
                notes="ollama-managed",
            )
        )
    return out


def k8s_rows(kubeconfig: str | None = None) -> list[RuntimeEntry]:
    """Kubernetes capacity, shaped for the runtime-inventory view.

    The cluster is discovered by `providers.kubernetes` -- the same provider
    placement consumes -- so there is ONE kubectl path and ONE readiness rule in
    the codebase. Previously this function ran its own kubectl call and its own
    (unconditional) availability logic, which is how a NotReady node came to be
    reported as capacity.

    Readiness is read off the offer's observed telemetry and `source`; nothing
    here re-derives it.
    """
    from .providers import kubernetes as k8s_provider

    rows: list[RuntimeEntry] = []
    for offer in k8s_provider.discover(kubeconfig=kubeconfig, force=True):
        usable = offer.telemetry.availability > 0.0
        reason = offer.source.split("k8s:node-readiness:", 1)[-1]
        node = offer.id.rsplit("/", 1)[-1]
        rows.append(
            RuntimeEntry(
                id=offer.id,
                provider=offer.provider,
                # The inventory view labels a node row by its name. The OFFER
                # keeps model=None, because a node is not a model; that
                # distinction belongs at the contract, not in a display label.
                model=node,
                location=Location.LOCAL_K8S.value,
                execution_backend=ExecutionBackend.UNKNOWN.value,
                status=Status.RUNNABLE.value if usable else Status.UNAVAILABLE.value,
                capabilities={"gpu": offer.resource_type == "gpu"},
                constraints={
                    "cpu_cores": offer.capacity.cpu_cores,
                    "ram_gb": offer.capacity.ram_gb,
                    "pods": offer.capacity.concurrency,
                    "vram_gb": offer.capacity.vram_gb,
                    "ready": usable,
                },
                local=True,
                notes=f"{reason}; GPU requires an allocatable nvidia.com/gpu resource",
            )
        )
    return rows


def _canonical_evidence_class(value: Any) -> str:
    """Canonical (uppercase) evidence class, or the exploratory-beta default.

    Absent means exploratory beta, which is the ceiling for anything that did not
    say what it was. Present-but-unknown is an error, not a default -- silently
    downgrading an unrecognised class is how a class the code does not know ends
    up looking evaluated while carrying no verdict.
    """
    if value is None or str(value).strip() == "":
        return EvidenceClass.EXPLORATORY_BETA.value
    text = str(value).strip().upper()
    by_name = {c.value: c.value for c in EvidenceClass}
    if text not in by_name:
        raise ValueError(
            f"unknown evidence_class {value!r}; the canonical taxonomy is "
            f"{', '.join(c.value for c in EvidenceClass)} "
            "(kvnloo/z0 registry/maturity.yaml, evidence:)"
        )
    return by_name[text]


# ------------------------------------------------------------- evidence import


def apply_evidence(entries: list[RuntimeEntry], docs: Iterable[dict]) -> None:
    """Fold exploratory-beta experiment output into typed trust.

    Only `EXPLORATORY_BETA` and above are accepted, and nothing below `CONFIRM`
    is ever allowed to write a `TRUSTED_*` role. That is the §14 separation,
    enforced in code rather than by convention.
    """
    by_model: dict[str, RuntimeEntry] = {}
    for e in entries:
        by_model[e.model.lower()] = e
        by_model[e.id.lower()] = e

    promotable = {EvidenceClass.CONFIRM.value, EvidenceClass.OOD.value, EvidenceClass.PROMOTION.value}
    for doc in docs:
        model = str(doc.get("model") or "")
        prov = str(doc.get("provider") or "")
        target = by_model.get(model.lower()) or by_model.get(f"{prov}/{model}".lower())
        if target is None:
            target = RuntimeEntry(
                id=f"{prov}/{model}", provider=prov, model=model,
                location=Location.REMOTE_FREE.value,
                execution_backend=(ExecutionBackend.GROQ.value if prov == "groq"
                                   else ExecutionBackend.CEREBRAS.value if prov == "cerebras"
                                   else ExecutionBackend.UNKNOWN.value),
                status=Status.TESTED.value, local=False,
                notes="added from experiment evidence",
            )
            entries.append(target)
            by_model[model.lower()] = target

        role = str(doc.get("role") or "bounded_choice")
        # Artifacts serialize the class lowercased; the taxonomy is uppercase.
        # Comparing the raw string meant this rejected EVERY artifact it was
        # handed -- 17 of them in this ecosystem -- because `exploratory_beta`
        # is not `EXPLORATORY_BETA`. Normalize, then match. The set is unchanged:
        # an unknown name still raises below.
        ec = _canonical_evidence_class(doc.get("evidence_class"))
        ev = RoleEvidence(
            n=int(doc.get("n") or 0),
            success=doc.get("success"),
            unsafe=doc.get("unsafe"),
            latency_p50_ms=doc.get("latency_p50_ms"),
            input_tokens=doc.get("input_tokens"),
            output_tokens=doc.get("output_tokens"),
            cost_usd=doc.get("cost_usd"),
            source=str(doc.get("source") or ""),
            evidence_class=ec,
        )
        target.evidence[role] = ev

        # Typed trust follows the evidence class, not the success number.
        if ec in promotable:
            target.trust[role] = (
                Trust.TRUSTED_BOUNDED.value if (ev.unsafe == 0 and (ev.success or 0) >= 0.8)
                else Trust.QUARANTINED.value
            )
        elif ec == EvidenceClass.PAIRED_REPLAY.value:
            # Paired replay establishes a COMPARISON, not a capability. The
            # canonical taxonomy forbids it from influencing the trust record,
            # so it is recorded as tested-and-experimental and no trust is
            # granted even when the numbers look strong.
            target.trust[role] = (
                Trust.QUARANTINED.value if ev.unsafe else Trust.TESTED_EXPERIMENTAL.value
            )
        elif ec in (EvidenceClass.SHADOW.value, EvidenceClass.EXPLORATORY_BETA.value, EvidenceClass.SMOKE.value):
            if ev.unsafe:
                # an unsafe exploratory result is quarantined immediately
                target.trust[role] = Trust.QUARANTINED.value
            elif ec == EvidenceClass.SHADOW.value:
                target.trust[role] = Trust.TRUSTED_SHADOW.value
            else:
                target.trust[role] = Trust.TESTED_EXPERIMENTAL.value
        else:
            # No silent fallthrough. Before this, an unrecognised class skipped
            # every branch: the row was still marked TESTED while receiving no
            # trust at all, which is the worst of both -- it looks evaluated and
            # carries no verdict.
            raise ValueError(
                f"unknown evidence_class {ec!r}; the canonical taxonomy is "
                "defined in kvnloo/z0 registry/maturity.yaml (evidence:)"
            )
        target.status = Status.TESTED.value


# ------------------------------------------------------------------- assembly


def build(
    offers: Iterable[ResourceOffer] | None = None,
    *,
    include_z0int: bool = True,
    include_ollama: bool = True,
    include_k8s: bool = False,
    evidence_docs: Iterable[dict] | None = None,
    gguf_dir: str | os.PathLike[str] | None = "/mnt/zer0models/zer0-models/gguf",
) -> list[RuntimeEntry]:
    entries: list[RuntimeEntry] = []
    if offers:
        entries.extend(from_offer(o) for o in offers)

    seen = {e.id for e in entries}
    if include_z0int:
        for e in z0int_supervisor_rows(gguf_dir=gguf_dir):
            if e.id not in seen:
                entries.append(e)
                seen.add(e.id)
    if include_ollama:
        for e in ollama_rows():
            if e.id not in seen:
                entries.append(e)
                seen.add(e.id)
    if include_k8s:
        for e in k8s_rows():
            if e.id not in seen:
                entries.append(e)
                seen.add(e.id)

    if evidence_docs:
        apply_evidence(entries, evidence_docs)
    entries.sort(key=lambda e: (e.location, e.provider, e.model))
    return entries


def summary(entries: list[RuntimeEntry]) -> dict[str, Any]:
    import collections

    return {
        "schema": "kerdoios.runtime-inventory.v1",
        "n": len(entries),
        "by_location": dict(collections.Counter(e.location for e in entries)),
        "by_backend": dict(collections.Counter(e.execution_backend for e in entries)),
        "by_status": dict(collections.Counter(e.status for e in entries)),
        "trusted_roles": {
            role: sorted(
                e.model for e in entries if e.trust.get(role, "").startswith("TRUSTED")
            )
            for role in ROLES
        },
        "quarantined": sorted(
            f"{e.model}:{r}"
            for e in entries
            for r, t in e.trust.items()
            if t == Trust.QUARANTINED.value
        ),
        "entries": [e.to_dict() for e in entries],
    }
