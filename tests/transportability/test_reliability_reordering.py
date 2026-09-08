import numpy as np
import pandas as pd

from scripts.run_formal_v2_reliability_reordering import _pairwise_reordering, _response_distance
from src.evaluation.reordering_inference import (
    bootstrap_pairwise_reordering,
    confidence_tracking_permutation_null,
    normalized_rank_displacement,
)


def test_pairwise_reordering_counts_rank_inversion_and_confidence_tracking() -> None:
    merged = pd.DataFrame(
        {
            "perturbation_label": ["p1", "p2", "p3", "p4"],
            "risk_left": [0.1, 0.2, 0.8, 0.9],
            "risk_right": [0.9, 0.8, 0.2, 0.1],
            "confidence_left": [0.9, 0.8, 0.2, 0.1],
            "confidence_right": [0.1, 0.2, 0.8, 0.9],
        }
    )

    result = _pairwise_reordering(merged, "ptl_rf")

    assert result["method"] == "ptl_rf"
    assert result["n_comparable_risk_pairs"] == 6
    assert result["n_risk_inversions"] == 6
    assert result["risk_inversion_rate"] == 1.0
    assert result["confidence_tracking_rate"] == 1.0
    assert result["risk_rank_spearman"] == -1.0
    assert result["risk_rank_kendall"] == -1.0


def test_pairwise_reordering_excludes_tied_risk_pairs() -> None:
    merged = pd.DataFrame(
        {
            "perturbation_label": ["p1", "p2"],
            "risk_left": [0.1, 0.1],
            "risk_right": [0.2, 0.3],
            "confidence_left": [0.9, 0.8],
            "confidence_right": [0.9, 0.8],
        }
    )

    result = _pairwise_reordering(merged, "raw_normalized_uq")

    assert result["n_comparable_risk_pairs"] == 0
    assert result["n_risk_inversions"] == 0
    assert np.isnan(result["risk_inversion_rate"])


def test_response_distance_separates_direction_from_magnitude() -> None:
    distance, magnitude_change = _response_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    assert np.isclose(distance, 1.0)
    assert np.isclose(magnitude_change, 0.0)


def test_normalized_rank_displacement_is_label_level_and_bounded() -> None:
    displacement = normalized_rank_displacement(
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([0.9, 0.8, 0.2, 0.1]),
    )

    assert np.allclose(displacement, [1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0])
    assert np.all((displacement >= 0.0) & (displacement <= 1.0))


def test_perturbation_bootstrap_reconstructs_pairs_and_confidence_null() -> None:
    risk_left = np.array([0.1, 0.2, 0.8, 0.9])
    risk_right = np.array([0.9, 0.8, 0.2, 0.1])
    confidence_left = np.array([0.9, 0.8, 0.2, 0.1])
    confidence_right = np.array([0.1, 0.2, 0.8, 0.9])

    bootstrap = bootstrap_pairwise_reordering(
        risk_left, risk_right, confidence_left, confidence_right, draws=50, seed=13
    )
    null = confidence_tracking_permutation_null(
        risk_left, risk_right, confidence_left, confidence_right, permutations=50, seed=13
    )

    assert bootstrap["n_shared_perturbations"] == 4
    assert bootstrap["n_risk_inversions"] == 6
    assert bootstrap["risk_inversion_rate"] == 1.0
    assert 0.0 <= bootstrap["risk_inversion_rate_ci_low"] <= bootstrap["risk_inversion_rate_ci_high"] <= 1.0
    assert null["observed_confidence_tracking_rate"] == 1.0
    assert 0.0 <= null["permutation_null_ci_low"] <= null["permutation_null_ci_high"] <= 1.0
