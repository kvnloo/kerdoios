"""Free-only posture for the planner.

Paid capacity is never entered implicitly. A plan runs free-only, and that
posture is only relaxed by explicitly setting ``no_paid_spill=False``. The
planner consults the dimensional quota first and falls back to the legacy free
predicate, so an exhausted reconstructed day window cannot be papered over by a
still-positive minute window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .model import QuotaState


@dataclass(frozen=True)
class PlanningPolicy:
    """``free_only`` restricts candidates; ``no_paid_spill`` blocks the paid fallback."""

    free_only: bool = False
    no_paid_spill: bool = True

    @property
    def paid_spill_allowed(self) -> bool:
        """Paid capacity is reachable only when a caller opted in explicitly."""
        return not self.free_only or not self.no_paid_spill


def structured_free_capacity(quota: object) -> bool | None:
    """Tri-state free-capacity check from a structured quota.

    ``None`` means there is no structured quota to judge, so the caller must
    fall back to its catalog/legacy predicate.
    """
    if isinstance(quota, QuotaState):
        return quota.has_free_capacity()
    if isinstance(quota, Mapping):
        # A cached offer may carry the plain serialized form.
        return QuotaState.from_dict(quota).has_free_capacity()
    return None


def free_only_rejection(offer: object, policy: PlanningPolicy) -> str | None:
    """Reason to drop ``offer`` under a free-only policy, or ``None`` to keep it.

    Imports the legacy ``is_free`` predicate lazily: the quota package must not
    sit in the import cycle with ``providers``.
    """
    if not policy.free_only:
        return None
    quota = getattr(getattr(offer, "economics", None), "quota", None)
    # An exhausted known window binds regardless of tier: a paid row whose day
    # budget is gone cannot serve either.
    if structured_free_capacity(quota) is False:
        return "free_only: quota window exhausted"
    from ..providers.free import is_free

    if is_free(offer):
        return None
    if policy.paid_spill_allowed:
        return None
    return "free_only: paid tier excluded (no_paid_spill)"
