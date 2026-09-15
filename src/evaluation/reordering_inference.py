"""Perturbation-level inference for biological reliability reordering.

The estimand is defined on perturbation labels, not on the :math:`n^2`
pairwise comparisons induced by those labels.  Bootstrap draws therefore
resample labels first and reconstruct all strict pairs inside each draw.
Confidence tracking is additionally compared with a label-preserving
permutation null that shuffles confidence independently within the two
environments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr


def _as_finite_arrays(*values: np.ndarray | list[float]) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value, dtype=float) for value in values)
    if not arrays:
        raise ValueError("at least one array is required")
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("all inputs must be one-dimensional")
    if len({array.shape[0] for array in arrays}) != 1:
        raise ValueError("all inputs must have the same length")
    return arrays


def _pairwise_arrays(left_risk: np.ndarray, right_risk: np.ndarray) -> dict[str, Any]:
    left_risk, right_risk = _as_finite_arrays(left_risk, right_risk)
    if left_risk.size < 2:
        return {
            "n_shared_perturbations": int(left_risk.size),
            "n_comparable_risk_pairs": 0,
            "n_risk_inversions": 0,
            "risk_inversion_rate": float("nan"),
        }
    row_i, row_j = np.triu_indices(left_risk.size, k=1)
    left_delta = left_risk[row_i] - left_risk[row_j]
    right_delta = right_risk[row_i] - right_risk[row_j]
    finite = np.isfinite(left_delta) & np.isfinite(right_delta)
    left_sign = np.sign(left_delta)
    right_sign = np.sign(right_delta)
    comparable = finite & (left_sign != 0.0) & (right_sign != 0.0)
    inversions = comparable & (left_sign != right_sign)
    n_comparable = int(comparable.sum())
    n_inversions = int(inversions.sum())
    return {
        "n_shared_perturbations": int(left_risk.size),
        "n_comparable_risk_pairs": n_comparable,
        "n_risk_inversions": n_inversions,
        "risk_inversion_rate": float(n_inversions / n_comparable) if n_comparable else float("nan"),
    }


def _tracking_arrays(
    left_risk: np.ndarray,
    right_risk: np.ndarray,
    left_confidence: np.ndarray,
    right_confidence: np.ndarray,
) -> dict[str, Any]:
    left_risk, right_risk, left_confidence, right_confidence = _as_finite_arrays(
        left_risk, right_risk, left_confidence, right_confidence
    )
    result = _pairwise_arrays(left_risk, right_risk)
    if left_risk.size < 2:
        result.update({"confidence_tracking_rate": float("nan"), "n_confidence_comparable_inversions": 0})
        return result
    row_i, row_j = np.triu_indices(left_risk.size, k=1)
    left_delta = left_risk[row_i] - left_risk[row_j]
    right_delta = right_risk[row_i] - right_risk[row_j]
    left_sign = np.sign(left_delta)
    right_sign = np.sign(right_delta)
    comparable = (
        np.isfinite(left_delta)
        & np.isfinite(right_delta)
        & (left_sign != 0.0)
        & (right_sign != 0.0)
    )
    inversions = comparable & (left_sign != right_sign)
    confidence_left_sign = np.sign(left_confidence[row_i] - left_confidence[row_j])
    confidence_right_sign = np.sign(right_confidence[row_i] - right_confidence[row_j])
    confidence_comparable = inversions & (confidence_left_sign != 0.0) & (confidence_right_sign != 0.0)
    tracked = confidence_comparable & (confidence_left_sign == -left_sign) & (confidence_right_sign == -right_sign)
    n_inversions = int(inversions.sum())
    result.update({
        "confidence_tracking_rate": float(tracked.sum() / n_inversions) if n_inversions else float("nan"),
        "n_confidence_comparable_inversions": int(confidence_comparable.sum()),
    })
    return result


def normalized_rank_displacement(left_risk: np.ndarray, right_risk: np.ndarray) -> np.ndarray:
    """Return per-label absolute rank displacement normalized to ``[0, 1]``."""

    left_risk, right_risk = _as_finite_arrays(left_risk, right_risk)
    output = np.full(left_risk.shape, np.nan, dtype=float)
    finite = np.isfinite(left_risk) & np.isfinite(right_risk)
    count = int(finite.sum())
    if count == 0:
        return output
    if count == 1:
        output[finite] = 0.0
        return output
    output[finite] = np.abs(
        rankdata(left_risk[finite], method="average") - rankdata(right_risk[finite], method="average")
    ) / float(count - 1)
    return output


def rank_displacement_summary(left_risk: np.ndarray, right_risk: np.ndarray) -> dict[str, Any]:
    """Summarize direct rank displacement and concordance for matched labels."""

    left_risk, right_risk = _as_finite_arrays(left_risk, right_risk)
    finite = np.isfinite(left_risk) & np.isfinite(right_risk)
    if int(finite.sum()) < 2:
        return {
            "n_rank_displacement_labels": int(finite.sum()),
            "normalized_rank_displacement_mean": float("nan"),
            "normalized_rank_displacement_median": float("nan"),
            "risk_rank_spearman": float("nan"),
            "risk_rank_kendall": float("nan"),
        }
    displacement = normalized_rank_displacement(left_risk, right_risk)[finite]
    return {
        "n_rank_displacement_labels": int(finite.sum()),
        "normalized_rank_displacement_mean": float(np.nanmean(displacement)),
        "normalized_rank_displacement_median": float(np.nanmedian(displacement)),
        "risk_rank_spearman": float(spearmanr(left_risk[finite], right_risk[finite]).statistic),
        "risk_rank_kendall": float(kendalltau(left_risk[finite], right_risk[finite]).statistic),
    }


def _percentile_interval(values: list[float] | np.ndarray, *, alpha: float = 0.05) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0]))


def bootstrap_pairwise_reordering(
    left_risk: np.ndarray,
    right_risk: np.ndarray,
    left_confidence: np.ndarray | None = None,
    right_confidence: np.ndarray | None = None,
    *,
    draws: int = 2000,
    seed: int = 20260908,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap matched perturbation labels, then rebuild pairwise estimates."""

    left_risk, right_risk = _as_finite_arrays(left_risk, right_risk)
    if draws < 1:
        raise ValueError("draws must be positive")
    if left_confidence is not None or right_confidence is not None:
        if left_confidence is None or right_confidence is None:
            raise ValueError("both confidence arrays are required together")
        left_confidence, right_confidence = _as_finite_arrays(left_confidence, right_confidence)
        if left_confidence.size != left_risk.size:
            raise ValueError("confidence arrays must match risk arrays")
    point = (
        _tracking_arrays(left_risk, right_risk, left_confidence, right_confidence)
        if left_confidence is not None and right_confidence is not None
        else _pairwise_arrays(left_risk, right_risk)
    )
    if left_risk.size < 2:
        return {**point, "bootstrap_draws": int(draws), "bootstrap_seed": int(seed)}

    rng = np.random.default_rng(seed)
    inversion_rates: list[float] = []
    tracking_rates: list[float] = []
    displacement_means: list[float] = []
    for _ in range(draws):
        indices = rng.integers(0, left_risk.size, size=left_risk.size)
        sampled_left = left_risk[indices]
        sampled_right = right_risk[indices]
        displacement_means.append(float(np.nanmean(normalized_rank_displacement(sampled_left, sampled_right))))
        if left_confidence is not None and right_confidence is not None:
            stats = _tracking_arrays(
                sampled_left,
                sampled_right,
                left_confidence[indices],
                right_confidence[indices],
            )
            tracking_rates.append(float(stats["confidence_tracking_rate"]))
        else:
            stats = _pairwise_arrays(sampled_left, sampled_right)
        inversion_rates.append(float(stats["risk_inversion_rate"]))

    inversion_low, inversion_high = _percentile_interval(inversion_rates, alpha=alpha)
    displacement_low, displacement_high = _percentile_interval(displacement_means, alpha=alpha)
    result = {
        **point,
        "normalized_rank_displacement_mean": float(np.nanmean(normalized_rank_displacement(left_risk, right_risk))),
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "risk_inversion_rate_ci_low": inversion_low,
        "risk_inversion_rate_ci_high": inversion_high,
        "normalized_rank_displacement_mean_ci_low": displacement_low,
        "normalized_rank_displacement_mean_ci_high": displacement_high,
    }
    if tracking_rates:
        tracking_low, tracking_high = _percentile_interval(tracking_rates, alpha=alpha)
        result.update({
            "confidence_tracking_rate_ci_low": tracking_low,
            "confidence_tracking_rate_ci_high": tracking_high,
        })
    return result


def confidence_tracking_permutation_null(
    left_risk: np.ndarray,
    right_risk: np.ndarray,
    left_confidence: np.ndarray,
    right_confidence: np.ndarray,
    *,
    permutations: int = 2000,
    seed: int = 20260908,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Shuffle confidence labels while holding the observed risk ordering fixed."""

    left_risk, right_risk, left_confidence, right_confidence = _as_finite_arrays(
        left_risk, right_risk, left_confidence, right_confidence
    )
    if permutations < 1:
        raise ValueError("permutations must be positive")
    observed = _tracking_arrays(left_risk, right_risk, left_confidence, right_confidence)
    rng = np.random.default_rng(seed)
    null_values: list[float] = []
    for _ in range(permutations):
        null_values.append(float(_tracking_arrays(
            left_risk,
            right_risk,
            left_confidence[rng.permutation(left_confidence.size)],
            right_confidence[rng.permutation(right_confidence.size)],
        )["confidence_tracking_rate"]))
    low, high = _percentile_interval(null_values, alpha=alpha)
    null_mean = float(np.nanmean(null_values)) if np.isfinite(null_values).any() else float("nan")
    return {
        "observed_confidence_tracking_rate": float(observed["confidence_tracking_rate"]),
        "permutation_null_mean": null_mean,
        "permutation_null_median": float(np.nanmedian(null_values)) if np.isfinite(null_values).any() else float("nan"),
        "permutation_null_ci_low": low,
        "permutation_null_ci_high": high,
        "permutations": int(permutations),
        "permutation_seed": int(seed),
        "null_definition": "independent within-environment confidence permutations; observed risks and strict inversions fixed",
    }


def bootstrap_response_rank_association(
    response_shift: np.ndarray,
    left_risk: np.ndarray,
    right_risk: np.ndarray,
    *,
    draws: int = 2000,
    seed: int = 20260908,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap the direct association between response shift and rank displacement."""

    response_shift, left_risk, right_risk = _as_finite_arrays(response_shift, left_risk, right_risk)
    finite = np.isfinite(response_shift) & np.isfinite(left_risk) & np.isfinite(right_risk)
    if int(finite.sum()) < 3:
        return {
            "response_shift_rank_displacement_spearman": float("nan"),
            "response_shift_rank_displacement_spearman_ci_low": float("nan"),
            "response_shift_rank_displacement_spearman_ci_high": float("nan"),
            "response_rank_association_labels": int(finite.sum()),
            "response_rank_association_bootstrap_draws": int(draws),
            "response_rank_association_bootstrap_seed": int(seed),
        }
    response = response_shift[finite]
    left = left_risk[finite]
    right = right_risk[finite]
    displacement = normalized_rank_displacement(left, right)
    point = float(spearmanr(response, displacement).statistic)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        indices = rng.integers(0, response.size, size=response.size)
        sampled_displacement = normalized_rank_displacement(left[indices], right[indices])
        values.append(float(spearmanr(response[indices], sampled_displacement).statistic))
    low, high = _percentile_interval(values, alpha=alpha)
    return {
        "response_shift_rank_displacement_spearman": point,
        "response_shift_rank_displacement_spearman_ci_low": low,
        "response_shift_rank_displacement_spearman_ci_high": high,
        "response_rank_association_labels": int(response.size),
        "response_rank_association_bootstrap_draws": int(draws),
        "response_rank_association_bootstrap_seed": int(seed),
    }
