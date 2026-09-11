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

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "task_type": self.task_type,
            "completed": self.completed,
            "actual_cost": self.actual_cost,
            "retried": self.retried,
        }


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
                rows.append(
                    Observation(
                        provider=payload["provider"],
                        model=payload["model"],
                        task_type=payload.get("task_type", "unknown"),
                        completed=bool(payload["completed"]),
                        actual_cost=float(payload["actual_cost"]),
                        retried=bool(payload.get("retried", False)),
                    )
                )
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
