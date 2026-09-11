from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Literal


ResourceType = Literal["llm", "gpu", "cpu", "vm", "service"]
PrivacyClass = Literal["public", "confidential", "local_only"]
ProvenanceKind = Literal["provider_claim", "public_benchmark", "hermes_benchmark", "observed_execution", "fixture"]


class Mode(str, Enum):
    FREE = "free"
    CHEAP = "cheap"
    BALANCED = "balanced"
    FAST = "fast"
    MAX = "max"
    SCALE = "scale"
    PRIVATE = "private"


@dataclass(frozen=True)
class CapabilityProfile:
    reasoning: float = 0.5
    coding: float = 0.5
    tool_use: float = 0.5
    vision: float = 0.0
    provenance: ProvenanceKind = "provider_claim"

    def score_for(self, coding: float | None = None, reasoning: float | None = None, tool_use: float | None = None) -> float:
        parts = [self.coding, self.reasoning, self.tool_use]
        weights = [1.0, 1.0, 1.0]
        if coding is not None:
            parts[0] = self.coding if self.coding >= coding else 0.0
        if reasoning is not None:
            parts[1] = self.reasoning if self.reasoning >= reasoning else 0.0
        if tool_use is not None:
            parts[2] = self.tool_use if self.tool_use >= tool_use else 0.0
        return sum(parts) / sum(weights)


@dataclass(frozen=True)
class Capacity:
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_day: int | None = None
    concurrency: int = 1
    context_window: int = 8192
    cpu_cores: float | None = None
    ram_gb: float | None = None
    vram_gb: float | None = None


@dataclass(frozen=True)
class Economics:
    input_token_price: float | None = None
    output_token_price: float | None = None
    hourly_price: float = 0.0
    remaining_free_quota: float = 0.0
    remaining_credits: float = 0.0
    seconds_until_quota_reset: float | None = None
    seconds_until_credits_expire: float | None = None

    def marginal_cost_per_token(self) -> float:
        if self.remaining_free_quota > 0 or self.remaining_credits > 0:
            return 0.0
        inp = self.input_token_price
        out = self.output_token_price
        # Unset prices are unknown; callers must not treat this 0 as a free tier.
        return (0.0 if inp is None else inp) + (0.0 if out is None else out)

    def expiration_urgency(self) -> float:
        """Higher when free capacity is about to vanish. Permanent local stock is 1.0."""
        horizon = self.seconds_until_credits_expire or self.seconds_until_quota_reset
        free = self.remaining_free_quota + self.remaining_credits
        if free <= 0:
            return 1.0
        if horizon is None or horizon <= 0:
            return 1.0
        # remaining / time — invert so soon-to-expire ranks high
        days = max(horizon / 86400.0, 1e-6)
        return min(10.0, 1.0 + (free / days))


@dataclass(frozen=True)
class Telemetry:
    latency_p50_ms: float = 800.0
    latency_p95_ms: float = 2000.0
    failure_rate: float = 0.02
    availability: float = 0.99


@dataclass(frozen=True)
class ResourceOffer:
    id: str
    provider: str
    resource_type: ResourceType
    model: str | None
    local: bool
    capabilities: CapabilityProfile
    capacity: Capacity
    economics: Economics
    telemetry: Telemetry
    tools: tuple[str, ...] = ()
    privacy_ok: tuple[PrivacyClass, ...] = ("public", "confidential")
    source: str = "fixture"
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkRequirement:
    coding: float = 0.0
    reasoning: float = 0.0
    tool_use: bool = False
    vision: bool = False
    context: int = 8192
    parallelism: int = 1
    estimated_input_tokens: int = 4000
    estimated_output_tokens: int = 800
    maximum_cost: float | None = None
    maximum_latency_ms: float | None = None
    minimum_reliability: float = 0.0
    privacy: PrivacyClass = "public"
    mode: Mode = Mode.BALANCED
    tools: tuple[str, ...] = ()


@dataclass
class Placement:
    offer_id: str
    provider: str
    model: str | None
    workers: int
    estimated_cost: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Rejection:
    offer_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPlan:
    estimated_cost: float
    estimated_duration_seconds: float
    confidence: float
    placements: list[Placement]
    fallbacks: list[str]
    rejections: list[Rejection]
    mode: str
    unplaced_workers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_cost": self.estimated_cost,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "confidence": self.confidence,
            "placements": [p.to_dict() for p in self.placements],
            "fallbacks": list(self.fallbacks),
            "rejections": [r.to_dict() for r in self.rejections],
            "mode": self.mode,
            "unplaced_workers": self.unplaced_workers,
        }
