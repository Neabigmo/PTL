"""Leakage-safe finite-measurement rules for a fixed source shortlist.

The target-audit replay uses this module as a small policy kernel.  The
allocator knows only the current allocation depths and the cost of extending
one candidate plus the shared control.  A caller may supply an ``observe``
function; in that mode the policy receives the caller's current observation,
not the complete (and potentially future) measurement surface.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any, Callable, Iterable, Sequence

import numpy as np


def _interval(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector with at least one item")
    return array


def _indices(selected: Iterable[int] | np.ndarray, n_items: int) -> np.ndarray:
    values = np.asarray(list(selected) if not isinstance(selected, np.ndarray) else selected)
    if values.dtype == bool:
        if values.size != n_items:
            raise ValueError("boolean shortlist mask has the wrong length")
        values = np.flatnonzero(values)
    values = values.reshape(-1)
    if values.size == 0:
        raise ValueError("the shortlist must contain at least one item")
    try:
        integer_values = values.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("selected indices must be integer item indices") from exc
    if np.issubdtype(values.dtype, np.floating) and not np.all(values == integer_values):
        raise ValueError("selected indices must be integer item indices")
    if (
        np.any(integer_values < 0)
        or np.any(integer_values >= n_items)
        or np.unique(integer_values).size != integer_values.size
    ):
        raise ValueError("selected indices must be unique valid item indices")
    return integer_values


def empirical_target_interval(
    replicates: np.ndarray,
    *,
    confidence: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a target-only normal-approximation interval for each item.

    ``replicates`` is ``[independent measurement, item]`` and must contain at
    least two independent measurements.  No reference or evaluation fallback
    is permitted: callers must pass exactly the target data available at the
    current audit depth.  This is an empirical decision diagnostic, not a
    simultaneous confidence guarantee.
    """

    values = np.asarray(replicates, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("replicates must be a finite [replicate, item] array with at least two rows")
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    half = z * sd / np.sqrt(values.shape[0])
    return mean - half, mean + half


def _best_gap(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    if left.size == 0 or right.size == 0:
        return 0.0, 0
    limit = min(left.size, right.size)
    gaps = np.cumsum(np.sort(left)[::-1][:limit]) - np.cumsum(np.sort(right)[:limit])
    index = int(np.argmax(gaps))
    return float(gaps[index]), index + 1


def audit_shortlist(
    lower: np.ndarray,
    upper: np.ndarray,
    selected: Iterable[int] | np.ndarray,
    *,
    epsilon: float,
) -> dict[str, Any]:
    """Classify a fixed source shortlist as ``REUSE``, ``REVISE``, or ``UNRESOLVED``.

    For each feasible exchange count ``m``, the upper regret bound compares
    the ``m`` largest selected upper bounds with the ``m`` smallest outside
    lower bounds.  The lower bound uses the same exchange construction with
    selected lower and outside upper bounds.  Both are normalized by the
    shortlist size, matching the decision-risk scale used by the replay.
    """

    lower = _interval(lower, "lower")
    upper = _interval(upper, "upper")
    if lower.shape != upper.shape or np.any(lower > upper):
        raise ValueError("lower and upper must have equal shape and lower<=upper")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    selected_idx = _indices(selected, lower.size)
    outside_idx = np.setdiff1d(np.arange(lower.size), selected_idx, assume_unique=True)
    ub_gap, ub_m = _best_gap(upper[selected_idx], lower[outside_idx])
    lb_gap, lb_m = _best_gap(lower[selected_idx], upper[outside_idx])
    ub = max(0.0, ub_gap / float(selected_idx.size))
    lb = max(0.0, lb_gap / float(selected_idx.size))
    if ub <= epsilon:
        state = "REUSE"
    elif lb > epsilon:
        state = "REVISE"
    else:
        state = "UNRESOLVED"
    return {
        "state": state,
        "epsilon": epsilon,
        "n_items": int(lower.size),
        "k": int(selected_idx.size),
        "ub_regret": float(ub),
        "lb_regret": float(lb),
        "ub_m": int(ub_m),
        "lb_m": int(lb_m),
        "selected_indices": selected_idx,
        "outside_indices": outside_idx,
    }


def boundary_priority(lower: np.ndarray, upper: np.ndarray, selected: Iterable[int] | np.ndarray) -> np.ndarray:
    """Return a deterministic priority from the currently observed intervals."""

    lower = _interval(lower, "lower")
    upper = _interval(upper, "upper")
    if lower.shape != upper.shape or np.any(lower > upper):
        raise ValueError("lower and upper must have equal shape and lower<=upper")
    selected_idx = _indices(selected, lower.size)
    selected_mask = np.zeros(lower.size, dtype=bool)
    selected_mask[selected_idx] = True
    priority = upper - lower
    if np.any(~selected_mask):
        selected_edge = float(np.max(upper[selected_idx]))
        outside_edge = float(np.min(lower[~selected_mask]))
        priority[selected_mask] += np.maximum(upper[selected_mask] - outside_edge, 0.0)
        priority[~selected_mask] += np.maximum(selected_edge - lower[~selected_mask], 0.0)
    return priority


def _validate_schedule(depths: Sequence[int]) -> np.ndarray:
    schedule = np.asarray(tuple(depths), dtype=int)
    if schedule.ndim != 1 or schedule.size < 2 or np.any(schedule <= 0) or np.any(np.diff(schedule) <= 0):
        raise ValueError("depths must be a strictly increasing positive schedule with at least two depths")
    return schedule


def nested_prefix_upgrade_cost(
    allocated_depths: np.ndarray,
    item: int,
    next_depth: int,
    control_depth: int,
    *,
    n_replicates: int = 2,
) -> int:
    """Return the actual extra cells for one target upgrade.

    Each candidate has ``n_replicates`` target replicates.  The control is
    shared by all candidates, so its prefix is extended only when the new
    candidate depth exceeds the current shared-control depth.
    """

    allocated = np.asarray(allocated_depths, dtype=int)
    if allocated.ndim != 1 or allocated.size < 1 or np.any(allocated <= 0):
        raise ValueError("allocated_depths must be a non-empty positive vector")
    if not 0 <= int(item) < allocated.size:
        raise ValueError("item is outside allocated_depths")
    next_depth = int(next_depth)
    control_depth = int(control_depth)
    n_replicates = int(n_replicates)
    if next_depth <= int(allocated[item]) or control_depth < 1 or n_replicates < 1:
        raise ValueError("next_depth must extend the selected item and control depth must be positive")
    target_extra = next_depth - int(allocated[item])
    control_extra = max(0, next_depth - control_depth)
    return n_replicates * (target_extra + control_extra)


def allocate_nested_prefixes(
    depths: Sequence[int],
    n_items: int,
    additional_cell_budget: int,
    priority: Callable[[Any], np.ndarray],
    *,
    n_replicates: int = 2,
    observe: Callable[[np.ndarray], Any] | None = None,
) -> tuple[np.ndarray, int, int]:
    """Allocate nested prefixes under an actual-cell budget.

    Allocation starts with every candidate at ``depths[0]`` and repeatedly
    extends one candidate to the next scheduled depth.  The shared control is
    extended as needed and is charged once per replicate.  When ``observe``
    is supplied, it is called with a copy of the current per-candidate depths
    and its return value is the only object passed to ``priority``.  This is
    the information firewall used by the target replay: the caller can
    construct an observation containing only already measured prefixes, while
    retaining hidden future depths in its private cache.
    """

    schedule = _validate_schedule(depths)
    n_items = int(n_items)
    additional_cell_budget = int(additional_cell_budget)
    n_replicates = int(n_replicates)
    if n_items < 1 or additional_cell_budget < 0 or n_replicates < 1:
        raise ValueError("n_items and n_replicates must be positive and budget non-negative")

    allocated = np.full(n_items, int(schedule[0]), dtype=int)
    control_depth, spent = int(schedule[0]), 0
    while True:
        current_depths = allocated.copy()
        policy_input = current_depths if observe is None else observe(current_depths.copy())
        scores = np.asarray(priority(policy_input), dtype=float)
        if scores.shape != allocated.shape or not np.isfinite(scores).all():
            raise ValueError("priority must return one finite score per item")

        candidates: list[tuple[float, int, int, int]] = []
        for item in range(allocated.size):
            position = int(np.searchsorted(schedule, allocated[item]))
            if position + 1 == schedule.size:
                continue
            next_depth = int(schedule[position + 1])
            cost = nested_prefix_upgrade_cost(
                allocated,
                item,
                next_depth,
                control_depth,
                n_replicates=n_replicates,
            )
            if spent + cost <= additional_cell_budget:
                # Stable item-index tie breaking makes every replay exactly
                # reproducible even when priorities are tied.
                candidates.append((-float(scores[item]), item, next_depth, cost))
        if not candidates:
            break
        _, item, next_depth, cost = min(candidates)
        allocated[item] = next_depth
        control_depth = max(control_depth, next_depth)
        spent += cost
    return allocated, control_depth, spent


__all__ = [
    "allocate_nested_prefixes",
    "audit_shortlist",
    "boundary_priority",
    "empirical_target_interval",
    "nested_prefix_upgrade_cost",
]
