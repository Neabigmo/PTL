"""Tie-aware estimands for experimentally observed risk orderings.

The formal Claim Lock uses continuous rank displacement as a secondary
diagnostic.  The theoretical measurement-corrected decomposition is a different,
explicit estimand: it operates on the categorical pairwise ordering
``sign(R_p - R_q)`` and therefore must not be silently substituted for rank
displacement.  This module keeps the two quantities separate and provides a
small, auditable implementation of the exact identity used in the manuscript.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import beta, kendalltau


ORDERING_ESTIMAND = "pairwise_order_disagreement"
RANK_DISPLACEMENT_ESTIMAND = "normalized_rank_displacement"
DEFAULT_MINIMUM_STRICT_SUPPORT = 8


def _validate_replicates(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape [replicate, item]")
    if values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError(f"{name} needs at least two replicates and two items")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def pairwise_order_probabilities(
    risk_replicates: np.ndarray,
    *,
    tie_tolerance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the tie-aware order distribution for each unordered item pair.

    Returns ``(pairs, probabilities)`` where ``pairs`` contains the upper
    triangular ``(p, q)`` indices and probabilities has columns
    ``[P(Z=-1), P(Z=0), P(Z=+1)]``.  The risk vector is an item-level vector;
    no item-pair rows are treated as independent observations in downstream
    bootstrap code.
    """

    values = _validate_replicates(risk_replicates, "risk_replicates")
    pairs = np.column_stack(np.triu_indices(values.shape[1], k=1)).astype(np.int64)
    differences = values[:, pairs[:, 0]] - values[:, pairs[:, 1]]
    if tie_tolerance > 0:
        ties = np.abs(differences) <= float(tie_tolerance)
    else:
        ties = differences == 0.0
    probabilities = np.column_stack((
        np.mean(differences < -float(tie_tolerance), axis=0),
        np.mean(ties, axis=0),
        np.mean(differences > float(tie_tolerance), axis=0),
    )).astype(float)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("pairwise order probabilities do not sum to one")
    return pairs, probabilities


def pairwise_disagreement(probabilities_left: np.ndarray, probabilities_right: np.ndarray) -> float:
    """Expected disagreement between independent draws from two order laws."""

    left = np.asarray(probabilities_left, dtype=float)
    right = np.asarray(probabilities_right, dtype=float)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("order probability arrays must have equal shape [pair, 3]")
    return float(np.mean(1.0 - np.sum(left * right, axis=1)))


def within_disagreement(probabilities: np.ndarray) -> float:
    """Expected disagreement between two independent draws in one context."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("order probabilities must have shape [pair, 3]")
    return float(np.mean(1.0 - np.sum(values * values, axis=1)))


def within_disagreement_u(
    probabilities: np.ndarray,
    *,
    n_replicates: int | None = None,
) -> float:
    """Unbiased within disagreement from normalized probabilities and counts."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("order probabilities must have shape [pair, 3]")
    if n_replicates is None or int(n_replicates) < 2:
        raise ValueError("n_replicates>=2 is required because normalized probabilities lose counts")
    n_replicates = int(n_replicates)
    counts = np.rint(values * n_replicates).astype(np.int64)
    if np.any(counts.sum(axis=1) != n_replicates):
        raise ValueError("probabilities are not compatible with n_replicates")
    numerator = np.sum(counts * (counts - 1), axis=1)
    disagreement = 1.0 - numerator / float(n_replicates * (n_replicates - 1))
    return float(np.mean(disagreement))


def pairwise_order_counts(
    risk_replicates: np.ndarray,
    *,
    tie_tolerance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer ``[minus, tie, plus]`` counts for each item pair."""

    values = _validate_replicates(risk_replicates, "risk_replicates")
    pairs = np.column_stack(np.triu_indices(values.shape[1], k=1)).astype(np.int64)
    differences = values[:, pairs[:, 0]] - values[:, pairs[:, 1]]
    tolerance = float(tie_tolerance)
    counts = np.column_stack((
        np.sum(differences < -tolerance, axis=0),
        np.sum(np.abs(differences) <= tolerance, axis=0),
        np.sum(differences > tolerance, axis=0),
    )).astype(np.int64)
    return pairs, counts


def _tie_pair_count(values: np.ndarray) -> int:
    """Count equal-valued unordered pairs without constructing a pair matrix."""

    _, counts = np.unique(np.asarray(values), return_counts=True)
    return int(np.sum(counts * (counts - 1) // 2))


def _strict_inversion_count(left: np.ndarray, right: np.ndarray) -> int:
    """Count pairs with opposite strict directions in ``O(n log n)`` time."""

    order = np.argsort(left, kind="mergesort")
    sorted_left = left[order]
    right_values, right_codes = np.unique(right, return_inverse=True)
    sorted_codes = right_codes[order]
    tree = [0] * (len(right_values) + 1)
    seen = 0
    inversions = 0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_left[end] == sorted_left[start]:
            end += 1
        for code in sorted_codes[start:end]:
            index = int(code) + 1
            prefix = 0
            cursor = index
            while cursor:
                prefix += tree[cursor]
                cursor -= cursor & -cursor
            inversions += seen - prefix
        for code in sorted_codes[start:end]:
            cursor = int(code) + 1
            while cursor < len(tree):
                tree[cursor] += 1
                cursor += cursor & -cursor
            seen += 1
        start = end
    return int(inversions)


def _strict_inversion_count_kendall(left: np.ndarray, right: np.ndarray) -> int:
    """Count strict opposite directions through SciPy's native tau-b kernel.

    Kendall tau-b exposes ``P-Q`` while its tie counts expose the number of
    comparable pairs ``P+Q``.  Recovering ``Q`` this way is algebraically the
    same count as ``_strict_inversion_count`` but avoids a Python Fenwick loop
    in the large, repeated bootstrap path.
    """

    n_items = int(left.size)
    n_pairs = n_items * (n_items - 1) // 2
    ties_left = _tie_pair_count(left)
    ties_right = _tie_pair_count(right)
    _, joint_counts = np.unique(np.column_stack((left, right)), axis=0, return_counts=True)
    both_ties = int(np.sum(joint_counts * (joint_counts - 1) // 2))
    ties_only_left = ties_left - both_ties
    ties_only_right = ties_right - both_ties
    comparable = n_pairs - both_ties
    strict_pairs = comparable - ties_only_left - ties_only_right
    if strict_pairs <= 0:
        return 0
    tau = float(kendalltau(left, right, variant="b", method="auto").statistic)
    if not np.isfinite(tau):
        return 0
    denominator = float(np.sqrt((strict_pairs + ties_only_left) * (strict_pairs + ties_only_right)))
    concordant_minus_discordant = tau * denominator
    discordant = 0.5 * (strict_pairs - concordant_minus_discordant)
    return int(np.clip(np.rint(discordant), 0, strict_pairs))


def deterministic_pairwise_order_disagreement(
    left_risk: np.ndarray,
    right_risk: np.ndarray,
) -> float:
    """Return pairwise order disagreement for two deterministic risk vectors.

    This is the finite-item analogue of ``pairwise_disagreement``.  It counts
    strict inversions and disagreements involving ties in ``O(n log n)`` time
    rather than materializing an ``n``-by-``n`` sign matrix.  A tie in one
    vector and a strict direction in the other is a disagreement, while a tie
    in both is agreement.
    """

    left, right = (np.asarray(value, dtype=float) for value in (left_risk, right_risk))
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("deterministic risks must be equal one-dimensional arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("deterministic risks contain non-finite values")
    n_items = left.size
    n_pairs = n_items * (n_items - 1) // 2
    if n_pairs == 0:
        return float("nan")
    if n_items <= 600:
        left_difference = left[:, None] - left[None, :]
        right_difference = right[:, None] - right[None, :]
        upper = np.triu(np.ones((n_items, n_items), dtype=bool), k=1)
        mismatch = (
            ((left_difference < 0) != (right_difference < 0))
            | ((left_difference > 0) != (right_difference > 0))
        )
        return float(np.mean(mismatch[upper]))
    ties_left = _tie_pair_count(left)
    ties_right = _tie_pair_count(right)
    _, joint_counts = np.unique(np.column_stack((left, right)), axis=0, return_counts=True)
    both_ties = int(np.sum(joint_counts * (joint_counts - 1) // 2))
    discordant = _strict_inversion_count_kendall(left, right)
    disagreement = discordant + (ties_left - both_ties) + (ties_right - both_ties)
    return float(np.clip(disagreement / float(n_pairs), 0.0, 1.0))


def pairwise_disagreement_from_replicates(
    risk_left: np.ndarray,
    risk_right: np.ndarray,
) -> float:
    """Average deterministic disagreement over all independent replicate pairs."""

    left = _validate_replicates(risk_left, "risk_left")
    right = _validate_replicates(risk_right, "risk_right")
    values = [
        deterministic_pairwise_order_disagreement(left[i], right[j])
        for i in range(left.shape[0])
        for j in range(right.shape[0])
    ]
    return float(np.mean(values))


def within_disagreement_u_from_replicates(risk_replicates: np.ndarray) -> float:
    """U-statistic within disagreement directly from replicate risk vectors."""

    values = _validate_replicates(risk_replicates, "risk_replicates")
    if values.shape[0] < 2:
        raise ValueError("within U-statistic needs at least two replicates")
    disagreements = [
        deterministic_pairwise_order_disagreement(values[i], values[j])
        for i in range(values.shape[0])
        for j in range(i + 1, values.shape[0])
    ]
    return float(np.mean(disagreements))


def measurement_corrected_ordering_divergence(
    probabilities_left: np.ndarray,
    probabilities_right: np.ndarray,
) -> tuple[float, float]:
    """Return the corrected divergence and the algebraic right-hand side.

    For empirical order distributions ``pi_left`` and ``pi_right`` this is

    ``D_cross - (D_within,left + D_within,right)/2``
    ``= E ||pi_left - pi_right||_2^2 / 2``.

    The second returned value is computed independently from the squared
    distance and is used as an implementation-level identity check.
    """

    left = np.asarray(probabilities_left, dtype=float)
    right = np.asarray(probabilities_right, dtype=float)
    cross = pairwise_disagreement(left, right)
    within_left = within_disagreement(left)
    within_right = within_disagreement(right)
    corrected = cross - 0.5 * (within_left + within_right)
    rhs = float(0.5 * np.mean(np.sum((left - right) ** 2, axis=1)))
    return float(corrected), rhs


def measurement_identifiable_ordering_divergence(
    probabilities_left: np.ndarray,
    probabilities_right: np.ndarray,
    *,
    n_replicates_left: int | None = None,
    n_replicates_right: int | None = None,
) -> tuple[float, float, float]:
    """Return the U-corrected divergence and both within-context estimates.

    The population identity remains the squared-distance identity for the
    plug-in/V terms.  For inference on finite replicate data we use the
    leave-one-replicate-out U-statistic within estimates instead, so the
    returned corrected value is a finite-sample unbiased estimate of the
    population excess under independent replicate draws.
    """

    left = np.asarray(probabilities_left, dtype=float)
    right = np.asarray(probabilities_right, dtype=float)
    cross = pairwise_disagreement(left, right)
    within_left_u = within_disagreement_u(left, n_replicates=n_replicates_left)
    within_right_u = within_disagreement_u(right, n_replicates=n_replicates_right)
    return float(cross - 0.5 * (within_left_u + within_right_u)), float(within_left_u), float(within_right_u)


def stable_order_mask(
    risk_replicates: np.ndarray,
    *,
    credible_level: float = 0.95,
    tie_tolerance: float = 0.0,
    minimum_strict_support: int = DEFAULT_MINIMUM_STRICT_SUPPORT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a tie-aware stable mask, strict-direction interval, and pairs.

    Ties are not failures.  The posterior is placed on the strict direction
    conditional on a non-tie observation, ``theta=P(Z=+1 | Z!=0)``, using
    ``Beta(n_plus+1, n_minus+1)``.  A pair also needs a fixed minimum number
    of strict observations before it can be called stable.
    """

    values = _validate_replicates(risk_replicates, "risk_replicates")
    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must lie in (0, 1)")
    if int(minimum_strict_support) < 0:
        raise ValueError("minimum_strict_support must be non-negative")
    pairs, counts = pairwise_order_counts(values, tie_tolerance=tie_tolerance)
    strict_support = counts[:, 0] + counts[:, 2]
    successes = counts[:, 2]
    failures = counts[:, 0]
    tail = (1.0 - credible_level) / 2.0
    lower = beta.ppf(tail, successes + 1, failures + 1)
    upper = beta.ppf(1.0 - tail, successes + 1, failures + 1)
    stable = (strict_support >= int(minimum_strict_support)) & ((lower > 0.5) | (upper < 0.5))
    return stable, np.column_stack((lower, upper)), pairs


def summarize_fixed_predictor_ordering(
    risk_left: np.ndarray,
    risk_right: np.ndarray,
    *,
    credible_level: float = 0.95,
    tie_tolerance: float = 0.0,
    minimum_strict_support: int = DEFAULT_MINIMUM_STRICT_SUPPORT,
) -> dict[str, Any]:
    """Compute the complete measurement-corrected fixed-predictor summary."""

    left = _validate_replicates(risk_left, "risk_left")
    right = _validate_replicates(risk_right, "risk_right")
    if left.shape[1] != right.shape[1]:
        raise ValueError("fixed-predictor contexts must contain the same items")
    pairs_left, pi_left = pairwise_order_probabilities(left, tie_tolerance=tie_tolerance)
    pairs_right, pi_right = pairwise_order_probabilities(right, tie_tolerance=tie_tolerance)
    if not np.array_equal(pairs_left, pairs_right):
        raise AssertionError("pair indexing differs across contexts")
    cross = pairwise_disagreement(pi_left, pi_right)
    within_left = within_disagreement(pi_left)
    within_right = within_disagreement(pi_right)
    within_left_u = within_disagreement_u_from_replicates(left)
    within_right_u = within_disagreement_u_from_replicates(right)
    plugin_corrected, rhs = measurement_corrected_ordering_divergence(pi_left, pi_right)
    corrected = cross - 0.5 * (within_left_u + within_right_u)
    stable_left, interval_left, _ = stable_order_mask(
        left, credible_level=credible_level, tie_tolerance=tie_tolerance,
        minimum_strict_support=minimum_strict_support,
    )
    stable_right, interval_right, _ = stable_order_mask(
        right, credible_level=credible_level, tie_tolerance=tie_tolerance,
        minimum_strict_support=minimum_strict_support,
    )
    stable_both = stable_left & stable_right
    left_sign = np.where(pi_left[:, 2] > pi_left[:, 0], 1, np.where(pi_left[:, 0] > pi_left[:, 2], -1, 0))
    right_sign = np.where(pi_right[:, 2] > pi_right[:, 0], 1, np.where(pi_right[:, 0] > pi_right[:, 2], -1, 0))
    return {
        "n_items": int(left.shape[1]),
        "n_pairs": int(len(pairs_left)),
        "n_replicates_left": int(left.shape[0]),
        "n_replicates_right": int(right.shape[0]),
        "cross_pairwise_disagreement": float(cross),
        "within_pairwise_disagreement_left": float(within_left),
        "within_pairwise_disagreement_right": float(within_right),
        "within_pairwise_disagreement_u_left": float(within_left_u),
        "within_pairwise_disagreement_u_right": float(within_right_u),
        "measurement_corrected_ordering_divergence": float(corrected),
        "measurement_plugin_ordering_divergence": float(plugin_corrected),
        "measurement_identifiable_ordering_divergence": float(corrected),
        "identity_rhs_squared_distance": float(rhs),
        "identity_absolute_error": float(abs(plugin_corrected - rhs)),
        "u_minus_plugin_correction": float(corrected - plugin_corrected),
        "stable_fraction_left": float(np.mean(stable_left)),
        "stable_fraction_right": float(np.mean(stable_right)),
        "stable_fraction_both": float(np.mean(stable_both)),
        "stable_order_inversion_fraction": (
            float(np.mean(left_sign[stable_both] != right_sign[stable_both]))
            if np.any(stable_both) else float("nan")
        ),
        "stable_pairs": int(np.sum(stable_both)),
        "credible_level": float(credible_level),
        "tie_tolerance": float(tie_tolerance),
        "minimum_strict_support": int(minimum_strict_support),
        "ordering_estimand": ORDERING_ESTIMAND,
    }


def crossfit_stable_ordering_summary(
    risk_left: np.ndarray,
    risk_right: np.ndarray,
    *,
    credible_level: float = 0.95,
    tie_tolerance: float = 0.0,
    minimum_strict_support: int = DEFAULT_MINIMUM_STRICT_SUPPORT,
) -> dict[str, Any]:
    """Evaluate stable ordering on held-out measurement replicates.

    The first half of replicates selects pairs whose ordering is credibly on
    one side of 0.5 in both contexts; the second half evaluates the selected
    order.  The procedure is repeated after swapping the two halves.  This
    keeps order selection and evaluation disjoint and avoids treating the
    correlated pair rows as independent observations.
    """

    left = _validate_replicates(risk_left, "risk_left")
    right = _validate_replicates(risk_right, "risk_right")
    if left.shape != right.shape:
        raise ValueError("crossfit contexts must have equal [replicate, item] shape")
    n_replicates = left.shape[0]
    if n_replicates < 4 or n_replicates % 2:
        raise ValueError("crossfit stable ordering needs an even number of replicates >= 4")
    half = n_replicates // 2
    split_summaries: list[dict[str, float | int]] = []
    for fit_slice, eval_slice in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
        fit_left = left[fit_slice]
        fit_right = right[fit_slice]
        eval_left = left[eval_slice]
        eval_right = right[eval_slice]
        stable_left, _, pairs_left = stable_order_mask(
            fit_left, credible_level=credible_level, tie_tolerance=tie_tolerance,
            minimum_strict_support=minimum_strict_support,
        )
        stable_right, _, pairs_right = stable_order_mask(
            fit_right, credible_level=credible_level, tie_tolerance=tie_tolerance,
            minimum_strict_support=minimum_strict_support,
        )
        if not np.array_equal(pairs_left, pairs_right):
            raise AssertionError("crossfit pair indexing differs across contexts")
        _, pi_fit_left = pairwise_order_probabilities(fit_left, tie_tolerance=tie_tolerance)
        _, pi_fit_right = pairwise_order_probabilities(fit_right, tie_tolerance=tie_tolerance)
        _, pi_eval_left = pairwise_order_probabilities(eval_left, tie_tolerance=tie_tolerance)
        _, pi_eval_right = pairwise_order_probabilities(eval_right, tie_tolerance=tie_tolerance)
        stable_both = stable_left & stable_right
        fit_sign_left = np.where(pi_fit_left[:, 2] > pi_fit_left[:, 0], 1, np.where(pi_fit_left[:, 0] > pi_fit_left[:, 2], -1, 0))
        fit_sign_right = np.where(pi_fit_right[:, 2] > pi_fit_right[:, 0], 1, np.where(pi_fit_right[:, 0] > pi_fit_right[:, 2], -1, 0))
        eval_sign_left = np.where(pi_eval_left[:, 2] > pi_eval_left[:, 0], 1, np.where(pi_eval_left[:, 0] > pi_eval_left[:, 2], -1, 0))
        eval_sign_right = np.where(pi_eval_right[:, 2] > pi_eval_right[:, 0], 1, np.where(pi_eval_right[:, 0] > pi_eval_right[:, 2], -1, 0))
        fit_inversion = stable_both & (fit_sign_left != fit_sign_right)
        evaluable = stable_both & (eval_sign_left != 0) & (eval_sign_right != 0)
        eval_inversion = evaluable & (eval_sign_left != eval_sign_right)
        split_summaries.append({
            "selection_stable_both_pairs": int(np.sum(stable_both)),
            "selection_stable_both_fraction": float(np.mean(stable_both)),
            "selection_inversion_fraction": float(np.mean(fit_inversion[stable_both])) if np.any(stable_both) else float("nan"),
            "heldout_evaluable_pairs": int(np.sum(evaluable)),
            "heldout_evaluable_fraction": float(np.mean(evaluable)),
            "heldout_inversion_fraction": float(np.mean(eval_inversion[evaluable])) if np.any(evaluable) else float("nan"),
        })
    evaluable_counts = np.asarray([item["heldout_evaluable_pairs"] for item in split_summaries], dtype=float)
    inversion_counts = np.asarray([
        item["heldout_inversion_fraction"] * item["heldout_evaluable_pairs"]
        if np.isfinite(item["heldout_inversion_fraction"]) else 0.0
        for item in split_summaries
    ])
    selection_inversions = np.asarray([item["selection_inversion_fraction"] for item in split_summaries], dtype=float)
    return {
        "crossfit_fit_replicates_per_context": int(half),
        "crossfit_eval_replicates_per_context": int(half),
        "crossfit_splits": 2,
        "crossfit_selection_stable_both_fraction": float(np.mean([item["selection_stable_both_fraction"] for item in split_summaries])),
        "crossfit_selection_inversion_fraction": float(np.nanmean(selection_inversions)) if np.isfinite(selection_inversions).any() else float("nan"),
        "crossfit_heldout_evaluable_fraction": float(np.mean([item["heldout_evaluable_fraction"] for item in split_summaries])),
        "crossfit_heldout_inversion_fraction": float(inversion_counts.sum() / evaluable_counts.sum()) if evaluable_counts.sum() else float("nan"),
        "crossfit_heldout_evaluable_pairs": int(evaluable_counts.sum()),
        "crossfit_heldout_inversion_pairs": int(inversion_counts.sum()),
        "crossfit_credible_level": float(credible_level),
        "crossfit_tie_tolerance": float(tie_tolerance),
        "crossfit_minimum_strict_support": int(minimum_strict_support),
    }


def crossfit_seed_stable_ordering_summary(
    risk_left: np.ndarray,
    risk_right: np.ndarray,
    *,
    credible_level: float = 0.95,
    tie_tolerance: float = 0.0,
    minimum_strict_support: int = DEFAULT_MINIMUM_STRICT_SUPPORT,
) -> dict[str, Any]:
    """Cross-fit stable ordering by independent seed groups.

    Inputs have shape ``[seed, replicate-within-seed, item]``. All
    within-seed replicates from one seed group select the ordering and all
    replicates from the other group evaluate it, followed by the swapped
    direction. This guarantees that selection and evaluation use disjoint
    seeds while retaining both measurement halves.
    """

    left = np.asarray(risk_left, dtype=float)
    right = np.asarray(risk_right, dtype=float)
    if left.ndim != 3 or right.shape != left.shape:
        raise ValueError("seed crossfit inputs must have equal [seed, replicate, item] shape")
    if left.shape[0] < 4 or left.shape[0] % 2:
        raise ValueError("seed crossfit needs an even number of seeds >= 4")
    half = left.shape[0] // 2
    flattened_left = np.concatenate(
        (left[:half].reshape(-1, left.shape[-1]), left[half:].reshape(-1, left.shape[-1])),
        axis=0,
    )
    flattened_right = np.concatenate(
        (right[:half].reshape(-1, right.shape[-1]), right[half:].reshape(-1, right.shape[-1])),
        axis=0,
    )
    result = crossfit_stable_ordering_summary(
        flattened_left,
        flattened_right,
        credible_level=credible_level,
        tie_tolerance=tie_tolerance,
        minimum_strict_support=minimum_strict_support,
    )
    result["crossfit_fit_seeds"] = int(half)
    result["crossfit_eval_seeds"] = int(half)
    result["crossfit_within_seed_replicates"] = int(left.shape[1])
    return result
