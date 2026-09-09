import numpy as np

from src.evaluation.ordering_estimands import (
    crossfit_seed_stable_ordering_summary,
    crossfit_stable_ordering_summary,
    pairwise_order_probabilities,
    summarize_fixed_predictor_ordering,
)


def test_measurement_corrected_ordering_identity_handles_ties():
    left = np.asarray([[0.0, 1.0, 1.0], [0.0, 1.0, 2.0], [1.0, 0.0, 2.0]])
    right = np.asarray([[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [1.0, 2.0, 0.0]])
    summary = summarize_fixed_predictor_ordering(left, right)
    assert summary["n_pairs"] == 3
    assert summary["identity_absolute_error"] < 1e-12
    assert summary["measurement_corrected_ordering_divergence"] >= -1e-12


def test_pairwise_order_probabilities_are_simplex_rows():
    risks = np.asarray([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    _, probabilities = pairwise_order_probabilities(risks)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities[0], [1 / 3, 1 / 3, 1 / 3])


def test_crossfit_stable_ordering_uses_disjoint_halves():
    left = np.asarray([
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
    ])
    right = left.copy()
    summary = crossfit_stable_ordering_summary(left, right, credible_level=0.5)
    assert summary["crossfit_fit_replicates_per_context"] == 2
    assert summary["crossfit_eval_replicates_per_context"] == 2
    assert summary["crossfit_heldout_evaluable_pairs"] == 6
    assert summary["crossfit_heldout_inversion_pairs"] == 0


def test_seed_crossfit_keeps_within_seed_replicates_together():
    left = np.asarray([
        [[0.0, 1.0], [0.0, 1.0]],
        [[0.0, 1.0], [0.0, 1.0]],
        [[0.0, 1.0], [0.0, 1.0]],
        [[0.0, 1.0], [0.0, 1.0]],
    ])
    summary = crossfit_seed_stable_ordering_summary(left, left.copy(), credible_level=0.5)
    assert summary["crossfit_fit_seeds"] == 2
    assert summary["crossfit_eval_seeds"] == 2
    assert summary["crossfit_within_seed_replicates"] == 2
