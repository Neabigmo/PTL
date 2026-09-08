import numpy as np
import pandas as pd

from scripts.run_formal_v2_reliability_reordering import _pairwise_reordering, _response_distance


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
