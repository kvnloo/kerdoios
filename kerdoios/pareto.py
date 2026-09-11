from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def nondominated(
    items: Sequence[T],
    dimensions: Sequence[Callable[[T], float]],
    *,
    minimize: Sequence[bool],
) -> list[T]:
    """Pareto filter. `minimize[i]` True means smaller values of dimensions[i] are better."""
    if len(dimensions) != len(minimize):
        raise ValueError("dimensions and minimize must be the same length")
    kept: list[T] = []
    for candidate in items:
        if any(_dominates(other, candidate, dimensions, minimize) for other in items if other is not candidate):
            continue
        kept.append(candidate)
    return kept


def _dominates(
    a: T,
    b: T,
    dimensions: Sequence[Callable[[T], float]],
    minimize: Sequence[bool],
) -> bool:
    better_or_equal = True
    strictly_better = False
    for axis, shrink in zip(dimensions, minimize):
        av, bv = axis(a), axis(b)
        if shrink:
            if av > bv:
                better_or_equal = False
                break
            if av < bv:
                strictly_better = True
        else:
            if av < bv:
                better_or_equal = False
                break
            if av > bv:
                strictly_better = True
    return better_or_equal and strictly_better
