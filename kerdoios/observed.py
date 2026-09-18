"""Observed-execution feedback loop.

Every ResourceOffer capability/cost/telemetry number in fixture.py and the
live adapters is a static guess: ``provider_claim`` or ``fixture``
provenance, never corrected by what actually happened when Hermes ran a
placement. That is the same failure mode measured on AgentWeb — an
unvalidated "quality" signal (there, an LLM judge; here, a vendor spec
sheet) does not track real task outcomes. Judge-vs-completion correlation
was -0.34 and raw cost-vs-completion was -0.43: cheaper models won on
completion, the most-claimed-capable model had the worst $/completed-task.

This module is the minimal fix: log what actually happened per placement,
and let ``blend.py`` pull scoring back toward reality once there is enough
signal to trust it (``MIN_OBSERVATIONS`` shrinkage floor, same idea as the
AgentWeb benchmark's ``MIN_N=15``).

Storage is an append-only JSONL file so a bad write never corrupts prior
history and the log is trivially diffable/greppable. No network, no
dependency beyond stdlib.

Capability bridge (z0int):
  Observations MAY carry ``capability_id`` + token fields. Aggregation stays
  backward-compatible at (provider, model). Prefer
  ``lookup_stats(provider, model, capability_id)`` for residual allocation:
  exact capability → family prefix → global model.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOG_PATH = Path(
    os.environ.get("KERDOIOS_OBSERVED_LOG", str(Path.home() / ".hermes" / "cache" / "kerdoios" / "observed.jsonl"))
)

# Below this many observations for a (provider, model) pair, do not trust
# observed data enough to override the prior. Mirrors the AgentWeb
# benchmark's MIN_N=15 floor: a handful of lucky/unlucky runs must not
# overwrite a fixture or provider_claim capability score.
MIN_OBSERVATIONS = 5


def _opt_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _opt_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


@dataclass(frozen=True)
class Observation:
    """One placement outcome.

    `task_type` remains a free-text coarse bucket (coding/reasoning/…).
    `capability_id` is the cross-repo bridge with z0int (e.g.
    ``coding.delegate``, ``blender.scene_reasoning``). Optional token fields
    power frontier-token economics without a second telemetry store.
    """

    provider: str
    model: str
    task_type: str
    completed: bool
    actual_cost: float
    retried: bool = False
    capability_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    context_tokens: int | None = None
    latency_ms: float | None = None
    fallback_count: int = 0
    quota_before: float | None = None
    quota_after: float | None = None
    # V2: separate process exit from verified outcome
    execution_completed: bool | None = None
    verified_success: bool | None = None
    harness_id: str | None = None
    session_id: str | None = None
    process_id: str | None = None
    trace_id: str | None = None
    turn_id: str | None = None
    bridge_generation: str | None = None
    build_id: str | None = None
    baseline_tokens: int | None = None
    actual_tokens: int | None = None
    quota_group: str | None = None

    def to_dict(self) -> dict:
        payload: dict = {
            "provider": self.provider,
            "model": self.model,
            "task_type": self.task_type,
            "completed": self.completed,
            "actual_cost": self.actual_cost,
            "retried": self.retried,
        }
        if self.capability_id:
            payload["capability_id"] = self.capability_id
        if self.input_tokens is not None:
            payload["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            payload["output_tokens"] = self.output_tokens
        if self.cached_input_tokens is not None:
            payload["cached_input_tokens"] = self.cached_input_tokens
        if self.context_tokens is not None:
            payload["context_tokens"] = self.context_tokens
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        if self.fallback_count:
            payload["fallback_count"] = int(self.fallback_count)
        if self.quota_before is not None:
            payload["quota_before"] = self.quota_before
        if self.quota_after is not None:
            payload["quota_after"] = self.quota_after
        # Always serialize V2 keys so null ≠ missing (legacy logs omit keys).
        payload["execution_completed"] = self.execution_completed
        payload["verified_success"] = self.verified_success
        # ambient close may set completed=True while verified_success is null
        for key in (
            "harness_id",
            "session_id",
            "process_id",
            "trace_id",
            "turn_id",
            "bridge_generation",
            "build_id",
            "quota_group",
        ):
            val = getattr(self, key, None)
            if val:
                payload[key] = val
        if self.baseline_tokens is not None:
            payload["baseline_tokens"] = self.baseline_tokens
        if self.actual_tokens is not None:
            payload["actual_tokens"] = self.actual_tokens
        return payload


def record(observation: Observation, *, path: Path | None = None) -> None:
    """Append one observation. Never raises on a writable filesystem; a
    malformed observation is a caller bug, not something to hide."""
    target = path or DEFAULT_LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(observation.to_dict()) + "\n")


@dataclass(frozen=True)
class OutcomeStats:
    provider: str
    model: str
    n: int
    completion_rate: float
    mean_cost: float
    retry_rate: float
    capability_id: str | None = None
    mean_input_tokens: float | None = None
    mean_output_tokens: float | None = None
    mean_latency_ms: float | None = None

    @property
    def trusted(self) -> bool:
        return self.n >= MIN_OBSERVATIONS


def _from_payload(payload: dict) -> Observation:
    completed = bool(payload.get("completed", False))
    has_v2 = "execution_completed" in payload or "verified_success" in payload
    if "execution_completed" in payload:
        raw_ex = payload.get("execution_completed")
        execution_completed = None if raw_ex is None else bool(raw_ex)
    else:
        execution_completed = None
    if "verified_success" in payload:
        raw_v = payload.get("verified_success")
        verified = None if raw_v is None else bool(raw_v)
    elif has_v2:
        verified = None
    else:
        # pre-V2 log line: completed stood in for verified success
        verified = completed
    return Observation(
        provider=str(payload["provider"]),
        model=str(payload["model"]),
        task_type=str(payload.get("task_type") or "unknown"),
        completed=completed,
        actual_cost=float(payload.get("actual_cost") or 0.0),
        retried=bool(payload.get("retried", False)),
        capability_id=payload.get("capability_id"),
        input_tokens=_opt_int(payload.get("input_tokens")),
        output_tokens=_opt_int(payload.get("output_tokens")),
        cached_input_tokens=_opt_int(payload.get("cached_input_tokens")),
        context_tokens=_opt_int(payload.get("context_tokens")),
        latency_ms=_opt_float(payload.get("latency_ms")),
        fallback_count=int(payload.get("fallback_count") or 0),
        quota_before=_opt_float(payload.get("quota_before")),
        quota_after=_opt_float(payload.get("quota_after")),
        execution_completed=execution_completed,
        verified_success=verified,
        harness_id=payload.get("harness_id"),
        session_id=payload.get("session_id"),
        process_id=payload.get("process_id"),
        trace_id=payload.get("trace_id"),
        turn_id=payload.get("turn_id"),
        bridge_generation=payload.get("bridge_generation"),
        build_id=payload.get("build_id"),
        baseline_tokens=_opt_int(payload.get("baseline_tokens")),
        actual_tokens=_opt_int(payload.get("actual_tokens")),
        quota_group=payload.get("quota_group"),
    )




def load_observations(*, path: Path | None = None) -> list[Observation]:
    target = path or DEFAULT_LOG_PATH
    if not target.exists():
        return []
    rows: list[Observation] = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # One corrupt line must not lose every prior observation.
                continue
            try:
                rows.append(_from_payload(payload))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _stats_for(rows: list[Observation], *, provider: str, model: str, capability_id: str | None) -> OutcomeStats:
    n = len(rows)
    completed = sum(1 for r in rows if r.completed)
    inputs = [r.input_tokens for r in rows if r.input_tokens is not None]
    outputs = [r.output_tokens for r in rows if r.output_tokens is not None]
    lats = [r.latency_ms for r in rows if r.latency_ms is not None]
    return OutcomeStats(
        provider=provider,
        model=model,
        n=n,
        completion_rate=completed / n if n else 0.0,
        mean_cost=sum(r.actual_cost for r in rows) / n if n else 0.0,
        retry_rate=sum(1 for r in rows if r.retried) / n if n else 0.0,
        capability_id=capability_id,
        mean_input_tokens=(sum(inputs) / len(inputs)) if inputs else None,
        mean_output_tokens=(sum(outputs) / len(outputs)) if outputs else None,
        mean_latency_ms=(sum(lats) / len(lats)) if lats else None,
    )


def aggregate(observations: list[Observation]) -> dict[tuple[str, str], OutcomeStats]:
    """Aggregate by (provider, model) across all task types / capabilities.

    Floor used by blend.py. Capability-aware callers should use
    ``aggregate_by_capability`` + ``lookup_stats``.
    """
    buckets: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for obs in observations:
        buckets[(obs.provider, obs.model)].append(obs)
    stats: dict[tuple[str, str], OutcomeStats] = {}
    for key, rows in buckets.items():
        stats[key] = _stats_for(rows, provider=key[0], model=key[1], capability_id=None)
    return stats


def capability_family(capability_id: str | None) -> str | None:
    if not capability_id:
        return None
    if "." not in capability_id:
        return capability_id
    return capability_id.split(".", 1)[0]


def aggregate_by_capability(
    observations: list[Observation],
) -> dict[tuple[str, str, str], OutcomeStats]:
    """Aggregate by (provider, model, capability_key).

    Keys include exact capability_id and family prefix (``coding`` for
    ``coding.delegate``) so lookup can shrink when N is thin.
    """
    exact: dict[tuple[str, str, str], list[Observation]] = defaultdict(list)
    family: dict[tuple[str, str, str], list[Observation]] = defaultdict(list)
    for obs in observations:
        if not obs.capability_id:
            continue
        exact[(obs.provider, obs.model, obs.capability_id)].append(obs)
        fam = capability_family(obs.capability_id)
        if fam and fam != obs.capability_id:
            family[(obs.provider, obs.model, fam)].append(obs)
    out: dict[tuple[str, str, str], OutcomeStats] = {}
    for key, rows in exact.items():
        out[key] = _stats_for(rows, provider=key[0], model=key[1], capability_id=key[2])
    for key, rows in family.items():
        # family rollup always overwrites exact-only collision only when fam==exact id
        out[key] = _stats_for(rows, provider=key[0], model=key[1], capability_id=key[2])
    return out



def lookup_stats(
    *,
    provider: str,
    model: str,
    capability_id: str | None = None,
    by_capability: dict[tuple[str, str, str], OutcomeStats] | None = None,
    by_model: dict[tuple[str, str], OutcomeStats] | None = None,
    observations: list[Observation] | None = None,
) -> OutcomeStats | None:
    """Hierarchical fallback: exact capability → family → (provider, model).

    Prefer trusted stats at the finest grain available.
    """
    if observations is not None:
        by_model = by_model or aggregate(observations)
        by_capability = by_capability or aggregate_by_capability(observations)
    by_capability = by_capability or {}
    by_model = by_model or {}

    candidates: list[OutcomeStats] = []
    if capability_id:
        exact = by_capability.get((provider, model, capability_id))
        if exact is not None:
            candidates.append(exact)
        fam = capability_family(capability_id)
        if fam and fam != capability_id:
            fam_stats = by_capability.get((provider, model, fam))
            if fam_stats is not None:
                candidates.append(fam_stats)
    global_stats = by_model.get((provider, model))
    if global_stats is not None:
        candidates.append(global_stats)

    trusted = [s for s in candidates if s.trusted]
    if trusted:
        return trusted[0]
    return candidates[0] if candidates else None


def tokens_per_verified_task(
    observations: list[Observation] | None = None,
    *,
    path: Path | None = None,
    capability_id: str | None = None,
) -> dict:
    """Premium/frontier tokens per *verified* task.

    Only rows with ``verified_success is True`` count. ``execution_completed``
    alone (or legacy ``completed``) is not verified success. Optional
    ``capability_id`` filters exact id then family prefix.
    """
    rows = observations if observations is not None else load_observations(path=path)
    if capability_id:
        exact = [r for r in rows if r.capability_id == capability_id]
        fam = capability_family(capability_id)
        family_rows = [r for r in rows if r.capability_id and capability_family(r.capability_id) == fam] if fam else []
        rows = exact or family_rows or rows
    completed = [r for r in rows if r.verified_success is True]
    token_rows = [
        r
        for r in completed
        if r.input_tokens is not None or r.output_tokens is not None
    ]
    total_tokens = 0
    for r in token_rows:
        total_tokens += int(r.input_tokens or 0) + int(r.output_tokens or 0)
    n_v = len(completed)
    n_tok = len(token_rows)
    return {
        "schema": "kerdoios.tokens_per_verified.v1",
        "capability_id": capability_id,
        "n_observations": len(rows),
        "n_verified": n_v,
        "n_with_tokens": n_tok,
        "total_tokens_on_verified": total_tokens,
        "tokens_per_verified_task": (total_tokens / n_tok) if n_tok else None,
        "mean_cost_per_verified": (
            sum(r.actual_cost for r in completed) / n_v if n_v else None
        ),
        "completion_rate": (n_v / len(rows)) if rows else None,
    }



def import_allocation_observations(
    path: Path | str,
    *,
    dest: Path | None = None,
) -> dict:
    """Import z0int.allocation_observation.v1 JSONL into the observed log.

    Maps execution_completed / verified_success without collapsing them.
    Does not execute models.
    """
    src = Path(path)
    count = 0
    skipped = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        schema = payload.get("schema") or payload.get("type")
        if schema and schema not in (
            "z0int.allocation_observation.v1",
            "allocation_observation.v1",
            "observation",
        ):
            # still accept plain Observation-shaped rows
            if "provider" not in payload:
                skipped += 1
                continue
        provider = payload.get("provider") or (payload.get("placement") or {}).get("provider")
        model = payload.get("model") or (payload.get("placement") or {}).get("model")
        if not provider or not model:
            skipped += 1
            continue
        execution_completed = payload.get("execution_completed")
        verified_success = payload.get("verified_success", None)
        # ambient process close is not verified
        completed_legacy = bool(payload.get("completed", False))
        if execution_completed is None:
            execution_completed = completed_legacy
        obs = Observation(
            provider=str(provider),
            model=str(model),
            task_type=str(payload.get("task_type") or payload.get("capability_id") or "unknown"),
            completed=bool(execution_completed) if execution_completed is not None else completed_legacy,
            actual_cost=float(payload.get("actual_cost") or payload.get("cost") or 0.0),
            retried=bool(payload.get("retried", False)),
            capability_id=payload.get("capability_id"),
            input_tokens=_opt_int(payload.get("input_tokens") or payload.get("actual_input_tokens")),
            output_tokens=_opt_int(payload.get("output_tokens") or payload.get("actual_output_tokens")),
            cached_input_tokens=_opt_int(payload.get("cached_input_tokens")),
            context_tokens=_opt_int(payload.get("context_tokens")),
            latency_ms=_opt_float(payload.get("latency_ms")),
            fallback_count=int(payload.get("fallback_count") or 0),
            quota_before=_opt_float(payload.get("quota_before")),
            quota_after=_opt_float(payload.get("quota_after")),
            execution_completed=bool(execution_completed) if execution_completed is not None else None,
            verified_success=bool(verified_success) if verified_success is not None else None,
            harness_id=payload.get("harness_id"),
            session_id=payload.get("session_id"),
            process_id=payload.get("process_id"),
            trace_id=payload.get("trace_id"),
            turn_id=payload.get("turn_id"),
            bridge_generation=payload.get("bridge_generation"),
            build_id=payload.get("build_id"),
            baseline_tokens=_opt_int(payload.get("baseline_tokens")),
            actual_tokens=_opt_int(payload.get("actual_tokens")),
            quota_group=payload.get("quota_group"),
        )
        record(obs, path=dest)
        count += 1
    return {"imported": count, "skipped": skipped, "path": str(dest or DEFAULT_LOG_PATH)}
