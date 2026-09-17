"""Block-aware inference for finite-measurement ordering transport.

The measurement-adjusted ordering divergence is an order-two U-statistic and
is degenerate at the no-transport boundary.  Consequently, an ordinary
percentile bootstrap is not a valid detector at zero.  This module keeps the
registered perturbation panel fixed, resamples complete measurement-seed
blocks, and uses paired context swaps for a finite-sample randomization test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

@dataclass(frozen=True)
class DAdjInference:
    estimate: float
    ci_low: float
    ci_high: float
    pvalue: float
    detected: bool
    bootstrap_draws: int
    variance_blocks: int
    confidence_level: float


def _validate_blocks(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("left and right must have equal [block, replicate, item] shape")
    if left.shape[0] < 4 or left.shape[1] < 1 or left.shape[2] < 2:
        raise ValueError("at least four blocks and two items are required")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("risk blocks contain non-finite values")
    return left, right


def d_adj_from_blocks(left: np.ndarray, right: np.ndarray) -> float:
    """Estimate canonical D_adj by averaging seed-level U-statistics."""

    left, right = _validate_blocks(left, right)
    return float(_d_adj_state_batch(_pair_states(left)[None, ...], _pair_states(right)[None, ...])[0])


def _pair_states(values: np.ndarray) -> np.ndarray:
    pairs = np.column_stack(np.triu_indices(values.shape[-1], k=1))
    difference = values[..., pairs[:, 0]] - values[..., pairs[:, 1]]
    return np.sign(difference).astype(np.int8)


def _d_adj_state_batch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Vectorized D_adj for ``[draw, block, replicate, pair]`` states."""

    return np.mean(_d_adj_state_blocks(left, right), axis=1)


def _d_adj_state_blocks(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return one U-corrected contribution per draw and seed block."""

    if left.shape != right.shape or left.ndim != 4:
        raise ValueError("state arrays must have equal [draw, block, replicate, pair] shape")
    _, _, replicates, _ = left.shape
    if replicates < 2:
        raise ValueError("each block needs at least two measurement replicates")
    left_counts = np.stack([(left == state).sum(axis=2) for state in (-1, 0, 1)], axis=3).astype(float)
    right_counts = np.stack([(right == state).sum(axis=2) for state in (-1, 0, 1)], axis=3).astype(float)
    left_prob = left_counts / float(replicates)
    right_prob = right_counts / float(replicates)
    cross = 1.0 - np.sum(left_prob * right_prob, axis=3)
    denominator = float(replicates * (replicates - 1))
    left_u = 1.0 - np.sum(left_counts * (left_counts - 1.0), axis=3) / denominator
    right_u = 1.0 - np.sum(right_counts * (right_counts - 1.0), axis=3) / denominator
    return np.mean(cross - 0.5 * (left_u + right_u), axis=2)


def infer_d_adj(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rng: np.random.Generator,
    bootstrap_draws: int = 399,
    permutation_draws: int = 0,
    confidence_level: float = 0.90,
) -> DAdjInference:
    """Return block-aware magnitude interval and variance-based detection test.

    The interval uses the empirical variance of the independent seed-level
    U-statistic contributions and a Student-t reference with ``blocks-1``
    degrees of freedom.  A complete-block bootstrap is also evaluated so that
    the variance calculation and downstream sensitivity share the same unit;
    it is not used as the boundary test.
    """

    left, right = _validate_blocks(left, right)
    if bootstrap_draws < 0:
        raise ValueError("bootstrap_draws must be non-negative")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0.5, 1)")

    left_states = _pair_states(left)
    right_states = _pair_states(right)
    contributions = _d_adj_state_blocks(left_states[None, ...], right_states[None, ...])[0]
    estimate = float(np.mean(contributions))
    n_blocks = left.shape[0]

    # Optional sensitivity calculation.  The Student-t interval below depends
    # only on the observed seed contributions, so calibration jobs can set
    # this to zero without changing any inferential result.
    if bootstrap_draws:
        bootstrap = np.empty(int(bootstrap_draws), dtype=float)
        for start in range(0, int(bootstrap_draws), 32):
            stop = min(start + 32, int(bootstrap_draws))
            index = rng.integers(0, n_blocks, size=(stop - start, n_blocks))
            bootstrap[start:stop] = _d_adj_state_batch(left_states[index], right_states[index])
    alpha = 1.0 - float(confidence_level)
    standard_error = float(np.std(contributions, ddof=1) / np.sqrt(n_blocks))
    critical = float(student_t.ppf(1.0 - alpha / 2.0, df=n_blocks - 1))
    lower = float(estimate - critical * standard_error)
    upper = float(estimate + critical * standard_error)
    statistic = estimate / standard_error if standard_error > 0 else np.inf if estimate > 0 else -np.inf
    pvalue = float(student_t.sf(statistic, df=n_blocks - 1))
    detected = bool(lower > 0.0)
    return DAdjInference(
        estimate=estimate,
        ci_low=lower,
        ci_high=upper,
        pvalue=pvalue,
        detected=detected,
        bootstrap_draws=int(bootstrap_draws),
        variance_blocks=int(n_blocks),
        confidence_level=float(confidence_level),
    )
