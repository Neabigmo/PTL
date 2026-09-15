"""Tie/strength/flip decomposition for finite-measurement rank transport.

The decomposition operates on the empirical three-state pair-order law
``pi=(P(-1), P(0), P(+1))``.  It is deliberately separate from the stable
posterior state used by the metric-dependence audit: posterior tail
probabilities classify a direction, whereas the empirical order law carries
the magnitude needed for an exact squared-distance identity.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _state_codes(risk_replicates: np.ndarray, *, tie_tolerance: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return upper-triangular pair indices and {-1, 0, +1} order states."""

    values = np.asarray(risk_replicates, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("risk replicates must have shape [replicate, item]")
    if not np.isfinite(values).all():
        raise ValueError("risk replicates contain non-finite values")
    pairs = np.column_stack(np.triu_indices(values.shape[1], k=1)).astype(np.int64)
    differences = values[:, pairs[:, 0]] - values[:, pairs[:, 1]]
    tolerance = float(tie_tolerance)
    states = np.where(differences < -tolerance, -1, np.where(differences > tolerance, 1, 0)).astype(np.int8)
    return pairs, states


def crossfit_pair_state_components(
    source_states: np.ndarray,
    target_states: np.ndarray,
) -> dict[str, np.ndarray | int]:
    """Estimate tie and directional-bias components with an independent cross-product.

    ``source_states`` and ``target_states`` contain one categorical pair-order
    observation per replicate and pair.  The two replicate halves are kept
    independent: for each pair, the product of the half-A and half-B
    differences estimates the square of the population difference.  This is
    the finite-sample correction needed before using the population identity;
    it does not make the nonlinear absolute-value/sign split unbiased.
    """

    source = np.asarray(source_states, dtype=float)
    target = np.asarray(target_states, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[0] < 4 or source.shape[0] % 2:
        raise ValueError("state arrays must have equal shape [even replicate, pair], with at least four rows")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("state arrays contain non-finite values")
    if not np.isin(source, (-1, 0, 1)).all() or not np.isin(target, (-1, 0, 1)).all():
        raise ValueError("state arrays must contain only -1, 0, and +1")
    half = source.shape[0] // 2
    source_tie = (source == 0).astype(float)
    target_tie = (target == 0).astype(float)
    source_bias = source
    target_bias = target
    delta_tie_a = np.mean(source_tie[:half] - target_tie[:half], axis=0)
    delta_tie_b = np.mean(source_tie[half:] - target_tie[half:], axis=0)
    delta_bias_a = np.mean(source_bias[:half] - target_bias[:half], axis=0)
    delta_bias_b = np.mean(source_bias[half:] - target_bias[half:], axis=0)
    d_tie = 0.75 * delta_tie_a * delta_tie_b
    d_direction = 0.25 * delta_bias_a * delta_bias_b
    return {
        "d_tie_cf": d_tie,
        "d_direction_cf": d_direction,
        "d_total_cf": d_tie + d_direction,
        "delta_tie_a": delta_tie_a,
        "delta_tie_b": delta_tie_b,
        "delta_bias_a": delta_bias_a,
        "delta_bias_b": delta_bias_b,
        "pair_count": int(source.shape[1]),
        "half_replicates": int(half),
    }


def crossfit_pairwise_components(
    source_replicates: np.ndarray,
    target_replicates: np.ndarray,
    *,
    tie_tolerance: float = 0.0,
) -> dict[str, np.ndarray | int]:
    """Cross-fit the corrected two-component decomposition from risk vectors."""

    pairs_source, source_states = _state_codes(source_replicates, tie_tolerance=tie_tolerance)
    pairs_target, target_states = _state_codes(target_replicates, tie_tolerance=tie_tolerance)
    if not np.array_equal(pairs_source, pairs_target):
        raise AssertionError("source and target pair indexing differs")
    result = crossfit_pair_state_components(source_states, target_states)
    result["pairs"] = pairs_source
    return result


def _counts(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape [pair, 3] = [minus, tie, plus]")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} contains invalid counts")
    if np.any(np.sum(array, axis=1) <= 0):
        raise ValueError(f"{name} contains an empty pair")
    return array


def probabilities_from_counts(values: np.ndarray) -> np.ndarray:
    """Convert integer or real counts to ``[P(-1), P(0), P(+1)]``."""

    counts = _counts(values, "counts")
    return counts / np.sum(counts, axis=1, keepdims=True)


def decompose_pair_states(
    source_counts: np.ndarray,
    target_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the exact tie/strength/flip decomposition pair by pair.

    With ``t=P(0)`` and ``b=P(+1)-P(-1)``,

    ``1/2 ||pi_s-pi_t||² = 3/4 (t_s-t_t)²
    + 1/4 (|b_s|-|b_t|)²
    + |b_s b_t| I[b_s b_t < 0]``.
    """

    source = probabilities_from_counts(source_counts)
    target = probabilities_from_counts(target_counts)
    if source.shape != target.shape:
        raise ValueError("source and target pair laws must have equal shape")
    source_tie = source[:, 1]
    target_tie = target[:, 1]
    source_bias = source[:, 2] - source[:, 0]
    target_bias = target[:, 2] - target[:, 0]
    d_tie = 0.75 * (source_tie - target_tie) ** 2
    d_strength = 0.25 * (np.abs(source_bias) - np.abs(target_bias)) ** 2
    d_flip = np.where(source_bias * target_bias < 0.0, np.abs(source_bias * target_bias), 0.0)
    d_total = d_tie + d_strength + d_flip
    d_squared = 0.5 * np.sum((source - target) ** 2, axis=1)
    return {
        "d_tie": d_tie,
        "d_strength": d_strength,
        "d_flip": d_flip,
        "d_total": d_total,
        "d_squared": d_squared,
        "source_tie": source_tie,
        "target_tie": target_tie,
        "source_bias": source_bias,
        "target_bias": target_bias,
    }


def summarize_decomposition(
    source_counts: np.ndarray,
    target_counts: np.ndarray,
) -> dict[str, Any]:
    """Summarize components and the numerical identity error."""

    values = decompose_pair_states(source_counts, target_counts)
    identity_error = values["d_total"] - values["d_squared"]
    result: dict[str, Any] = {
        key: float(np.mean(values[key]))
        for key in ("d_tie", "d_strength", "d_flip", "d_total", "d_squared")
    }
    result.update({
        "pair_count": int(values["d_total"].size),
        "identity_abs_error_mean": float(np.mean(np.abs(identity_error))),
        "identity_abs_error_max": float(np.max(np.abs(identity_error))),
    })
    return result


def kendall_distance_no_ties(left: np.ndarray, right: np.ndarray) -> float:
    """Return ``(1-tau)/2`` for deterministic vectors with no ties.

    This is the exact normalized inversion fraction.  It is included as a
    correspondence limit, not as a replacement for the finite-measurement
    tie-aware estimand.
    """

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or left.size < 2:
        raise ValueError("left and right must be equal one-dimensional vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("deterministic vectors contain non-finite values")
    i, j = np.triu_indices(left.size, k=1)
    products = (left[i] - left[j]) * (right[i] - right[j])
    if np.any(products == 0):
        raise ValueError("kendall no-tie limit requires no ties in either vector")
    return float(np.mean(products < 0.0))


__all__ = [
    "crossfit_pair_state_components",
    "crossfit_pairwise_components",
    "decompose_pair_states",
    "kendall_distance_no_ties",
    "probabilities_from_counts",
    "summarize_decomposition",
]
