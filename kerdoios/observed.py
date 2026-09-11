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
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

DEFAULT_LOG_PATH = Path(
    os.environ.get("KERDOIOS_OBSERVED_LOG", str(Path.home() / ".hermes" / "cache" / "kerdoios" / "observed.jsonl"))
)

# Below this many observations for a (provider, model) pair, do not trust
# observed data enough to override the prior. Mirrors the AgentWeb
# benchmark's MIN_N=15 floor: a handful of lucky/unlucky runs must not
# overwrite a fixture or provider_claim capability score.
MIN_OBSERVATIONS = 5


@dataclass(frozen=True)
class Observation:
    """One placement outcome. `task_type` is a free-text bucket the caller
    controls (e.g. "coding", "reasoning", "tool_use") — kerdoios does not
    prescribe a taxonomy, matching the WorkRequirement fields it already has.
    """

    provider: str
    model: str
    task_type: str
    completed: bool
    actual_cost: float
    retried: bool = False
    origin_provider: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    http_status: int | None = None
    remaining_quota: float | None = None
    remaining_source: str | None = None
    error_class: str | None = None

    def to_dict(self) -> dict:
        payload = {
            "provider": self.provider,
            "model": self.model,
            "task_type": self.task_type,
            "completed": self.completed,
            "actual_cost": self.actual_cost,
            "retried": self.retried,
        }
        for key in (
            "origin_provider",
            "input_tokens",
            "output_tokens",
            "http_status",
            "remaining_quota",
            "remaining_source",
            "error_class",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


def classify_http(http_status: int | None) -> str:
    """Classify an HTTP status. Remaining quota is not an input; 0 is not exhausted.

    402/429 are exhausted. 401/404 and any other >=400 are failed. Those codes
    are never success.
    """
    if http_status is None:
        return "success"
    if http_status in (402, 429):
        return "exhausted"
    if http_status >= 400:
        return "failed"
    return "success"


def normalize(observation: Observation) -> Observation:
    """If http_status is set, coerce completed and error_class from it.

    Do not default origin_provider onto old 6-field rows.
    """
    if observation.http_status is None:
        return observation
    error_class = classify_http(observation.http_status)
    completed = error_class == "success"
    if observation.error_class == error_class and observation.completed == completed:
        return observation
    return replace(observation, completed=completed, error_class=error_class)


def record(observation: Observation, *, path: Path | None = None) -> Observation:
    """Append one normalized observation and return the stored row."""
    stored = normalize(observation)
    target = path or DEFAULT_LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(stored.to_dict()) + "\n")
    return stored


@dataclass(frozen=True)
class OutcomeStats:
    provider: str
    model: str
    n: int
    completion_rate: float
    mean_cost: float
    retry_rate: float

    @property
    def trusted(self) -> bool:
        return self.n >= MIN_OBSERVATIONS


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
                rows.append(_observation_from_payload(payload))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def aggregate(observations: list[Observation]) -> dict[tuple[str, str], OutcomeStats]:
    """Aggregate by (provider, model) across all task types.

    Task-type-level stats are a natural follow-up (the AgentWeb benchmark
    scores per model x task type) but need more volume than a single
    project generates on day one; provider+model is the floor that makes
    the MIN_OBSERVATIONS shrinkage meaningful without waiting for a
    task-type taxonomy to stabilize.
    """
    buckets: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for obs in observations:
        buckets[(obs.provider, obs.model)].append(obs)
    stats: dict[tuple[str, str], OutcomeStats] = {}
    for key, rows in buckets.items():
        n = len(rows)
        completed = sum(1 for r in rows if r.completed)
        stats[key] = OutcomeStats(
            provider=key[0],
            model=key[1],
            n=n,
            completion_rate=completed / n,
            mean_cost=sum(r.actual_cost for r in rows) / n,
            retry_rate=sum(1 for r in rows if r.retried) / n,
        )
    return stats


def _opt_str(payload: dict, key: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    text = str(payload[key]).strip()
    return text or None


def _opt_int(payload: dict, key: str) -> int | None:
    if key not in payload or payload[key] is None or payload[key] == "":
        return None
    return int(payload[key])


def _opt_float(payload: dict, key: str) -> float | None:
    if key not in payload or payload[key] is None or payload[key] == "":
        return None
    return float(payload[key])


def _observation_from_payload(payload: dict) -> Observation:
    """Load a row. Old 6-field JSONL stays valid; origin_provider is not inferred."""
    return Observation(
        provider=payload["provider"],
        model=payload["model"],
        task_type=payload.get("task_type", "unknown"),
        completed=bool(payload["completed"]),
        actual_cost=float(payload["actual_cost"]),
        retried=bool(payload.get("retried", False)),
        origin_provider=_opt_str(payload, "origin_provider"),
        input_tokens=_opt_int(payload, "input_tokens"),
        output_tokens=_opt_int(payload, "output_tokens"),
        http_status=_opt_int(payload, "http_status"),
        remaining_quota=_opt_float(payload, "remaining_quota"),
        remaining_source=_opt_str(payload, "remaining_source"),
        error_class=_opt_str(payload, "error_class"),
    )
