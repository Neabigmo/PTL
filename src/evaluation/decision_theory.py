"""Decision-level consequences of transporting a reliability ordering.

The functions in this module deliberately operate on item-level risk vectors
and accept optional replicate matrices.  They keep selection, evaluation, and
measurement floors separate so a top-k result cannot be mistaken for a global
ordering estimand.  Ties are handled deterministically by the original item
index; no outcome-dependent tie rule is introduced.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


DEFAULT_BUDGET_FRACTIONS = (0.05, 0.10, 0.20, 0.50)


def _finite_vector(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a finite vector with at least two items")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _finite_replicates(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape [replicate, item]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def top_k_size(n_items: int, budget_fraction: float) -> int:
    """Return the prespecified ``max(1, floor(b*n))`` selection size."""

    if int(n_items) < 2:
        raise ValueError("at least two items are required")
    budget = float(budget_fraction)
    if not 0.0 < budget <= 1.0:
        raise ValueError("budget_fraction must lie in (0, 1]")
    return max(1, int(np.floor(budget * int(n_items))))


def deterministic_top_k(risk: np.ndarray, k: int) -> np.ndarray:
    """Return deterministic lowest-risk indices, breaking ties by index."""

    values = _finite_vector(risk, "risk")
    if not 1 <= int(k) <= values.size:
        raise ValueError("k must lie between one and the number of items")
    order = np.lexsort((np.arange(values.size, dtype=np.int64), values))
    return order[: int(k)]


def _boundary_inversion(source_risk: np.ndarray, target_risk: np.ndarray, selected: np.ndarray) -> tuple[float, int]:
    n_items = source_risk.size
    selected_mask = np.zeros(n_items, dtype=bool)
    selected_mask[selected] = True
    outside = np.flatnonzero(~selected_mask)
    if outside.size == 0:
        return float("nan"), 0
    selected_target = target_risk[selected]
    outside_target = target_risk[outside]
    inversions = int(np.sum(selected_target[:, None] > outside_target[None, :]))
    return float(inversions / float(selected.size * outside.size)), inversions


def top_k_decision_metrics(
    source_risk: np.ndarray,
    target_risk: np.ndarray,
    *,
    budget_fraction: float,
) -> dict[str, Any]:
    """Compute retention, target regret, normalized regret, and boundary inversion."""

    source = _finite_vector(source_risk, "source_risk")
    target = _finite_vector(target_risk, "target_risk")
    if source.shape != target.shape:
        raise ValueError("source_risk and target_risk must have equal shape")
    k = top_k_size(source.size, budget_fraction)
    selected = deterministic_top_k(source, k)
    oracle = deterministic_top_k(target, k)
    selected_mask = np.zeros(source.size, dtype=bool)
    selected_mask[selected] = True
    oracle_mask = np.zeros(source.size, dtype=bool)
    oracle_mask[oracle] = True
    selected_mean = float(np.mean(target[selected]))
    oracle_mean = float(np.mean(target[oracle]))
    regret = selected_mean - oracle_mean
    random_mean = float(np.mean(target))
    denominator = random_mean - oracle_mean
    normalized = float(regret / denominator) if denominator > 0.0 else float("nan")
    boundary, inversion_count = _boundary_inversion(source, target, selected)
    mistakes = int(np.sum(selected_mask & ~oracle_mask))
    bound_denominator = k * (source.size - k)
    lower_bound = float(mistakes * mistakes / bound_denominator) if bound_denominator else float("nan")
    return {
        "budget_fraction": float(budget_fraction),
        "k": int(k),
        "n_items": int(source.size),
        "retention": float(np.mean(selected_mask & oracle_mask) * source.size / k),
        "selected_target_risk": selected_mean,
        "oracle_target_risk": oracle_mean,
        "random_target_risk": random_mean,
        "regret": float(regret),
        "normalized_regret": normalized,
        "normalized_regret_denominator": float(denominator),
        "boundary_inversion": boundary,
        "boundary_inversion_count": int(inversion_count),
        "mis_selection_count": mistakes,
        "mis_selection_fraction": float(mistakes / k),
        "boundary_inversion_lower_bound": lower_bound,
        "boundary_bound_holds": bool(np.isnan(lower_bound) or boundary + 1e-12 >= lower_bound),
        "selected_indices": selected,
        "oracle_indices": oracle,
    }


def decision_curve(
    source_risk: np.ndarray,
    target_risk: np.ndarray,
    *,
    budgets: Iterable[float] = DEFAULT_BUDGET_FRACTIONS,
) -> list[dict[str, Any]]:
    """Return a deterministic top-k decision curve for fixed risk vectors."""

    return [
        top_k_decision_metrics(source_risk, target_risk, budget_fraction=float(budget))
        for budget in budgets
    ]


def replicate_decision_curve(
    source_replicates: np.ndarray,
    target_replicates: np.ndarray,
    *,
    budgets: Iterable[float] = DEFAULT_BUDGET_FRACTIONS,
    independent_only: bool = False,
) -> list[dict[str, Any]]:
    """Average decision metrics over source/target replicate pairs.

    When ``independent_only`` is true and the two matrices have the same
    replicate count, diagonal pairs are omitted.  This is the required
    within-context measurement-floor operation; cross-context curves retain
    all source/target pairs because their measurement draws are independent.
    """

    source = _finite_replicates(source_replicates, "source_replicates")
    target = _finite_replicates(target_replicates, "target_replicates")
    if source.shape[1] != target.shape[1]:
        raise ValueError("replicate contexts must contain the same items")
    outputs: list[dict[str, Any]] = []
    for budget in budgets:
        rows = [
            top_k_decision_metrics(source[i], target[j], budget_fraction=float(budget))
            for i in range(source.shape[0])
            for j in range(target.shape[0])
            if not (independent_only and source.shape == target.shape and i == j)
        ]
        if not rows:
            raise ValueError("independent_only removed every replicate pair")
        numeric_keys = (
            "retention", "selected_target_risk", "oracle_target_risk", "random_target_risk",
            "regret", "normalized_regret", "normalized_regret_denominator", "boundary_inversion",
            "mis_selection_fraction", "boundary_inversion_lower_bound",
        )
        row = {key: float(np.nanmean([item[key] for item in rows])) for key in numeric_keys}
        row.update({"budget_fraction": float(budget), "k": rows[0]["k"], "n_items": rows[0]["n_items"], "n_replicate_pairs": len(rows), "independent_only": bool(independent_only)})
        row["boundary_inversion_count"] = int(round(np.nanmean([item["boundary_inversion_count"] for item in rows])))
        row["mis_selection_count"] = int(round(np.nanmean([item["mis_selection_count"] for item in rows])))
        row["boundary_bound_holds"] = bool(all(item["boundary_bound_holds"] for item in rows))
        outputs.append(row)
    return outputs


def measurement_floor_excess(
    cross_context: dict[str, Any],
    within_target: dict[str, Any],
    *,
    field: str = "regret",
) -> float:
    """Subtract the independent measurement-only decision floor."""

    if field not in cross_context or field not in within_target:
        raise KeyError(field)
    return float(cross_context[field] - within_target[field])
