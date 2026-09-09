import numpy as np

from src.evaluation.ordering_estimands import (
    crossfit_seed_stable_ordering_summary,
    crossfit_stable_ordering_summary,
    deterministic_pairwise_order_disagreement,
    _strict_inversion_count,
    _strict_inversion_count_kendall,
    pairwise_order_probabilities,
    summarize_fixed_predictor_ordering,
    within_disagreement,
    within_disagreement_u,
)


def test_measurement_corrected_ordering_identity_handles_ties():
    left = np.asarray([[0.0, 1.0, 1.0], [0.0, 1.0, 2.0], [1.0, 0.0, 2.0]])
    right = np.asarray([[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [1.0, 2.0, 0.0]])
    summary = summarize_fixed_predictor_ordering(left, right)
    assert summary["n_pairs"] == 3
    assert summary["identity_absolute_error"] < 1e-12
    assert summary["measurement_plugin_ordering_divergence"] >= -1e-12
    assert np.isfinite(summary["measurement_identifiable_ordering_divergence"])


def test_u_statistic_matches_leave_one_replicate_out_definition():
    probabilities = np.asarray([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    assert np.isclose(within_disagreement(probabilities), 0.25)
    assert np.isclose(within_disagreement_u(probabilities, n_replicates=2), 0.5)


def test_empirical_null_has_positive_plugin_bias_but_u_correction_is_centered():
    rng = np.random.default_rng(20260909)
    population = np.asarray([0.2, 0.6, 0.2])
    plugin = []
    corrected = []
    for _ in range(2500):
        left = rng.multinomial(12, population, size=64).astype(float) / 12.0
        right = rng.multinomial(12, population, size=64).astype(float) / 12.0
        cross = 1.0 - np.sum(left * right, axis=1)
        plugin.append(float(np.mean(cross) - 0.5 * (within_disagreement(left) + within_disagreement(right))))
        corrected.append(float(np.mean(cross) - 0.5 * (within_disagreement_u(left, n_replicates=12) + within_disagreement_u(right, n_replicates=12))))
    assert np.mean(plugin) > 0.015
    assert abs(np.mean(corrected)) < 0.01


def test_known_alternative_recovers_population_identity():
    left = np.asarray([0.7, 0.2, 0.1])
    right = np.asarray([0.65, 0.3, 0.05])
    expected = 0.5 * np.sum((left - right) ** 2)
    empirical = 1.0 - np.dot(left, right) - 0.5 * ((1 - np.dot(left, left)) + (1 - np.dot(right, right)))
    assert np.isclose(empirical, expected)


def test_max_floor_stress_can_be_negative_while_identifiable_excess_is_positive():
    left = np.asarray([0.7, 0.2, 0.1])
    right = np.asarray([0.65, 0.3, 0.05])
    cross = 1.0 - np.dot(left, right)
    within_left = 1.0 - np.dot(left, left)
    within_right = 1.0 - np.dot(right, right)
    assert cross - max(within_left, within_right) < 0.0
    assert cross - 0.5 * (within_left + within_right) > 0.0


def test_tie_heavy_pair_is_not_called_stable_without_strict_evidence():
    risks = np.asarray([[0.0, 0.0, 2.0]] * 28 + [[0.0, 1.0, 2.0]] * 2)
    from src.evaluation.ordering_estimands import stable_order_mask
    stable, interval, pairs = stable_order_mask(risks, minimum_strict_support=8)
    pair_index = np.flatnonzero((pairs == np.asarray([0, 1])).all(axis=1))[0]
    assert not stable[pair_index]
    assert interval[pair_index, 0] < 0.5 < interval[pair_index, 1]


def test_strict_support_allows_stable_direction_when_ties_dominate():
    risks = np.asarray([[0.0, 1.0, 2.0]] * 20 + [[0.0, 0.0, 2.0]] * 10)
    from src.evaluation.ordering_estimands import stable_order_mask
    stable, _, pairs = stable_order_mask(risks, minimum_strict_support=8)
    pair_index = np.flatnonzero((pairs == np.asarray([0, 1])).all(axis=1))[0]
    assert stable[pair_index]


def test_deterministic_tie_aware_disagreement_matches_pair_enumeration():
    left = np.asarray([0.0, 1.0, 1.0, 2.0])
    right = np.asarray([0.0, 0.0, 2.0, 2.0])
    pairs = [(i, j) for i in range(len(left)) for j in range(i + 1, len(left))]
    expected = np.mean([np.sign(left[i] - left[j]) != np.sign(right[i] - right[j]) for i, j in pairs])
    assert np.isclose(deterministic_pairwise_order_disagreement(left, right), expected)


def test_large_tie_aware_counter_matches_reference_fenwick_counter():
    rng = np.random.default_rng(20260909)
    for n_items in (601, 1024):
        left = rng.integers(0, 31, size=n_items).astype(float)
        right = rng.integers(0, 31, size=n_items).astype(float)
        assert _strict_inversion_count_kendall(left, right) == _strict_inversion_count(left, right)


def test_large_tie_aware_disagreement_is_finite_and_bounded():
    rng = np.random.default_rng(20260910)
    left = rng.integers(0, 17, size=2086).astype(float)
    right = rng.integers(0, 17, size=2086).astype(float)
    value = deterministic_pairwise_order_disagreement(left, right)
    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0


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
    summary = crossfit_stable_ordering_summary(left, right, credible_level=0.5, minimum_strict_support=0)
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
    summary = crossfit_seed_stable_ordering_summary(left, left.copy(), credible_level=0.5, minimum_strict_support=0)
    assert summary["crossfit_fit_seeds"] == 2
    assert summary["crossfit_eval_seeds"] == 2
    assert summary["crossfit_within_seed_replicates"] == 2


def test_perturbation_burden_has_exact_global_identity_and_handles_ties():
    from src.evaluation.ordering_estimands import perturbation_reordering_burden

    left = np.asarray([[0.0, 1.0, 2.0, 2.0], [0.0, 2.0, 1.0, 2.0]])
    right = np.asarray([[0.0, 2.0, 1.0, 2.0], [0.0, 1.0, 2.0, 2.0]])
    result = perturbation_reordering_burden(left, right, block_size=2)
    assert np.isclose(np.mean(result["cross_burden"]), result["cross_disagreement"])
    assert np.isclose(np.mean(result["within_burden_left"]), result["within_disagreement_left"])
    assert np.isclose(np.mean(result["within_burden_right"]), result["within_disagreement_right"])
    assert np.isclose(
        np.mean(result["identifiable_burden"]),
        result["cross_disagreement"]
        - 0.5 * (result["within_disagreement_left"] + result["within_disagreement_right"]),
    )
    assert result["cross_identity_error"] == 0.0
