from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .quota import QuotaState


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


PostureKind = Literal["BURN", "BALANCED", "RESERVE", "OFFLOAD"]


@dataclass(frozen=True)
class SubscriptionWindow:
    """One flat-rate plan window (e.g. Claude weekly) as observed locally.

    Subscription quota is sunk cost: the question is not price but whether
    unused capacity perishes at ``resets_at`` (BURN) or the observed pace
    empties the window before then (OFFLOAD). No account identity is kept.
    """

    group: str  # claude | codex | cursor | grok | ...
    window: str  # 5h | weekly | 30d | scoped extra id
    remaining_fraction: float  # 0..1 of the window
    resets_at: str | None
    seconds_until_reset: float | None
    window_minutes: float | None
    posture: PostureKind = "BALANCED"
    reason: str = ""
    arithmetic: str = ""
    surplus_fraction_at_reset: float | None = None  # projected unused share of capacity at reset
    confidence: str = "ok"  # ok | low | stale
    observed_at: str | None = None
    source: str = "codexbar"

    @property
    def exhausted(self) -> bool:
        return self.remaining_fraction <= 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Economics:
    input_token_price: float | None = None
    output_token_price: float | None = None
    hourly_price: float = 0.0
    remaining_free_quota: float = 0.0
    remaining_credits: float = 0.0
    seconds_until_quota_reset: float | None = None
    seconds_until_credits_expire: float | None = None
    # Dimensional quota (rpm/rpd/tpm/tpd + provenance). The scalar fields above
    # stay for existing callers; this is the non-lossy picture.
    quota: "QuotaState | None" = None
    # Binding subscription window when this offer is a flat-rate plan pool.
    subscription: SubscriptionWindow | None = None

    def marginal_cost_per_token(self) -> float:
        if self.remaining_free_quota > 0 or self.remaining_credits > 0:
            return 0.0
        if self.subscription is not None and not self.subscription.exhausted:
            return 0.0  # flat-rate plan: the window is already paid for
        inp = self.input_token_price
        out = self.output_token_price
        # Unset prices are unknown; callers must not treat this 0 as a free tier.
        return (0.0 if inp is None else inp) + (0.0 if out is None else out)

    def expiration_urgency(self) -> float:
        """Higher when free capacity is about to vanish. Permanent local stock is 1.0."""
        horizon = self.seconds_until_credits_expire or self.seconds_until_quota_reset
        free = self.remaining_free_quota + self.remaining_credits
        if self.subscription is not None:
            # Plan windows are metered as a fraction of the window, not tokens.
            free = self.subscription.remaining_fraction
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
class QuotaConstraint:
    """One scarce-resource budget (shared across MoA members when group matches)."""

    name: str
    limit: float
    unit: str = "requests"  # requests | tokens | usd | seconds
    group: str | None = None  # shared group id for multi-model MoA
    remaining: float | None = None

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
    # Cross-repo bridge with z0int CapabilityCards (e.g. coding.delegate).
    # Optional: residual allocator only; does not invent new capability axes.
    capability_id: str | None = None
    # Multi-constraint quotas (shared groups keep MoA from double-booking).
    quotas: tuple[QuotaConstraint, ...] = ()
    # Optional join policy for multi-placement plans (planner metadata only).
    join_policy: str | None = None  # all | any | quorum
    retry_policy: dict[str, Any] | None = None


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
    # Free-only posture actually applied to this plan, plus the explicit
    # "everything free is spent" outcome so callers never infer paid spill.
    free_only: bool = False
    no_paid_spill: bool = True
    no_free_capacity: bool = False
    # V2 planner metadata — never inference / never self-heal execution
    retry_policy: dict[str, Any] | None = None
    quota_reservations: list[dict[str, Any]] = field(default_factory=list)
    join_policy: str | None = None
    schema: str = "kerdoios.execution_plan.v2"
    # Subscription window report (status, reason, windows) when it was consulted.
    subscription_windows: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "estimated_cost": self.estimated_cost,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "confidence": self.confidence,
            "placements": [p.to_dict() for p in self.placements],
            "fallbacks": list(self.fallbacks),
            "rejections": [r.to_dict() for r in self.rejections],
            "mode": self.mode,
            "unplaced_workers": self.unplaced_workers,
            "free_only": self.free_only,
            "no_paid_spill": self.no_paid_spill,
            "no_free_capacity": self.no_free_capacity,
            "retry_policy": self.retry_policy or {"max_retries": 0, "on_failure": "fallback"},
            "quota_reservations": list(self.quota_reservations),
            "join_policy": self.join_policy or "all",
            **({"subscription_windows": self.subscription_windows} if self.subscription_windows is not None else {}),
        }
