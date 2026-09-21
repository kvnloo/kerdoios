"""Multi-dimensional quota model.

A single scalar cannot describe a provider's quota: Groq's
``x-ratelimit-remaining-tokens`` is tokens-per-MINUTE, while Cerebras also
exposes hourly and daily windows. Collapsing those into one number silently
conflates dimensions, so every value here stays attached to the window it
belongs to and to the provenance that produced it.

``source`` is a closed set. A dimension we know nothing about keeps
``source="unknown"`` and ``remaining=None``; it is never defaulted to 0 as if
that were a measurement, and never copied from another window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

QuotaSource = Literal[
    "provider_header",
    "account_observed",
    "locally_reconstructed",
    "unknown",
]

QUOTA_SOURCES: tuple[QuotaSource, ...] = (
    "provider_header",
    "account_observed",
    "locally_reconstructed",
    "unknown",
)

# The required set. rpm/rpd are requests, tpm/tpd are tokens.
REQUIRED_DIMENSIONS: tuple[str, ...] = ("rpm", "rpd", "tpm", "tpd")
# Window granularity extra to the required set, plus the window-ambiguous
# "quota" budget some providers expose. Kept because losing a window would
# force us to alias it onto a required dimension.
OPTIONAL_DIMENSIONS: tuple[str, ...] = ("rph", "tph", "quota")
ALL_DIMENSIONS: tuple[str, ...] = REQUIRED_DIMENSIONS + OPTIONAL_DIMENSIONS

DIMENSION_UNITS: dict[str, str] = {
    "rpm": "requests",
    "rph": "requests",
    "rpd": "requests",
    "tpm": "tokens",
    "tph": "tokens",
    "tpd": "tokens",
    "quota": "quota",
}

# Which dimensions are day-scoped and therefore reconstructable from receipts.
DAY_DIMENSIONS: tuple[str, ...] = ("rpd", "tpd")

# Fixes the order used when a legacy scalar caller needs one number.
LEGACY_PRECEDENCE: tuple[str, ...] = ("tpd", "tph", "tpm", "quota", "rpd", "rph", "rpm")


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _coerce_source(value: Any) -> QuotaSource:
    text = str(value or "unknown")
    if text in QUOTA_SOURCES:
        return text  # type: ignore[return-value]  # narrowed by membership in QUOTA_SOURCES
    return "unknown"


@dataclass(frozen=True)
class QuotaDimension:
    """One window of one unit. ``reset_at`` is absolute Unix epoch seconds."""

    limit: float | None = None
    remaining: float | None = None
    reset_at: float | None = None
    source: QuotaSource = "unknown"

    @property
    def known(self) -> bool:
        return self.source != "unknown" and (self.limit is not None or self.remaining is not None)

    def seconds_until_reset(self, now: float | None = None) -> float | None:
        if self.reset_at is None:
            return None
        clock = time.time() if now is None else now
        return max(0.0, self.reset_at - clock)

    def to_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "QuotaDimension":
        if not isinstance(raw, Mapping):
            return cls()
        return cls(
            limit=_opt_float(raw.get("limit")),
            remaining=_opt_float(raw.get("remaining")),
            reset_at=_opt_float(raw.get("reset_at")),
            source=_coerce_source(raw.get("source")),
        )


@dataclass(frozen=True)
class QuotaState:
    """Per provider/model quota picture across every known window."""

    provider: str = ""
    model: str = ""
    dimensions: Mapping[str, QuotaDimension] = field(default_factory=dict)
    observed_at: float | None = None

    def __post_init__(self) -> None:
        dimensions: dict[str, QuotaDimension] = {}
        for key, value in self.dimensions.items():
            if key in ALL_DIMENSIONS and isinstance(value, QuotaDimension):
                dimensions[key] = value
        for key in REQUIRED_DIMENSIONS:
            dimensions.setdefault(key, QuotaDimension())
        object.__setattr__(self, "dimensions", dimensions)

    def dim(self, key: str) -> QuotaDimension:
        return self.dimensions.get(key, QuotaDimension())

    def remaining(self, key: str) -> float | None:
        return self.dim(key).remaining

    def provider_reported_remaining(self, key: str) -> float | None:
        dimension = self.dim(key)
        return dimension.remaining if dimension.source == "provider_header" else None

    def locally_reconstructed_remaining(self, key: str) -> float | None:
        dimension = self.dim(key)
        return dimension.remaining if dimension.source == "locally_reconstructed" else None

    def account_observed_remaining(self, key: str) -> float | None:
        dimension = self.dim(key)
        return dimension.remaining if dimension.source == "account_observed" else None

    def known_dimensions(self) -> dict[str, QuotaDimension]:
        return {key: dim for key, dim in self.dimensions.items() if dim.known}

    def exhausted_dimensions(self) -> dict[str, QuotaDimension]:
        """Known windows at (or below) zero. One exhausted window binds the account."""
        return {
            key: dim
            for key, dim in self.dimensions.items()
            if dim.source != "unknown" and dim.remaining is not None and dim.remaining <= 0
        }

    def is_exhausted(self) -> bool:
        return bool(self.exhausted_dimensions())

    def has_free_capacity(self) -> bool:
        """True only when a window is known positive and no window is known spent.

        Unknown is not capacity: with no known window this is False, so a
        free-only planner cannot mistake missing headers for free credit.
        """
        if self.is_exhausted():
            return False
        return any(
            dim.source != "unknown" and dim.remaining is not None and dim.remaining > 0
            for dim in self.dimensions.values()
        )

    def comparison(self) -> dict[str, dict[str, Any]]:
        """Expose reconstructed remaining beside provider-reported remaining."""
        rows: dict[str, dict[str, Any]] = {}
        for key in ALL_DIMENSIONS:
            dimension = self.dim(key)
            if not dimension.known:
                continue
            rows[key] = {
                "unit": DIMENSION_UNITS.get(key, "quota"),
                "remaining": dimension.remaining,
                "limit": dimension.limit,
                "source": dimension.source,
                "provider_reported": self.provider_reported_remaining(key),
                "locally_reconstructed": self.locally_reconstructed_remaining(key),
                "account_observed": self.account_observed_remaining(key),
            }
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "observed_at": self.observed_at,
            "dimensions": {key: dim.to_dict() for key, dim in self.dimensions.items()},
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "QuotaState":
        if not isinstance(raw, Mapping):
            return cls()
        dims_raw = raw.get("dimensions")
        dimensions: dict[str, QuotaDimension] = {}
        if isinstance(dims_raw, Mapping):
            for key, value in dims_raw.items():
                if key in ALL_DIMENSIONS:
                    dimensions[str(key)] = QuotaDimension.from_dict(value)
        return cls(
            provider=str(raw.get("provider") or ""),
            model=str(raw.get("model") or ""),
            dimensions=dimensions,
            observed_at=_opt_float(raw.get("observed_at")),
        )
