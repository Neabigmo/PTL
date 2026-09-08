import numpy as np
import pandas as pd

from scripts.run_formal_v2_grouping_loss import assign_bins, summarize_split, _summarize_conditional_grouped


def test_confidence_bins_are_fit_from_calibration_only() -> None:
    labels, edges = assign_bins(np.asarray([0.1, 0.5, 0.9]), np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]), n_bins=4)
    assert len(labels) == 3
    assert edges[0] == -np.inf and edges[-1] == np.inf


def test_grouping_loss_matches_by_bin_without_using_outcomes_for_assignment() -> None:
    test = pd.DataFrame({
        "scenario": ["in_domain"] * 4,
        "predictor": ["p"] * 4,
        "environment_id": ["e1", "e2", "e1", "e2"],
        "biological_instance_id": ["b1", "b2", "b3", "b4"],
        "continuous_risk": [0.1, 0.8, 0.2, 0.7],
        "raw_normalized_uq": [0.9, 0.9, 0.1, 0.1],
        "u_only_rf": [0.9, 0.9, 0.1, 0.1],
        "ptl_rf": [0.9, 0.9, 0.1, 0.1],
    })
    calibration = pd.DataFrame({
        "scenario": ["in_domain"] * 4,
        "predictor": ["p"] * 4,
        "method": ["raw_normalized_uq", "u_only_rf", "ptl_rf", "ptl_rf"],
        "confidence": [0.0, 0.25, 0.75, 1.0],
    })
    rows, _ = summarize_split(test, calibration, 20260907, 2)
    assert set(rows["bin_edges_fit_on"]) == {"calibration_rows_only"}
    assert rows["outcomes_used_for_bin_construction"].eq(0).all()
    assert rows["matched_different_environment_pair_count"].sum() > 0


def test_conditional_grouping_loss_changes_when_confidence_partition_changes() -> None:
    frame = pd.DataFrame({
        "environment_id": ["e1", "e2", "e1", "e2"],
        "continuous_risk": [0.1, 0.9, 0.9, 0.1],
        "confidence_bin": [0, 0, 1, 1],
    })
    conditional = _summarize_conditional_grouped(frame)
    pooled = _summarize_conditional_grouped(frame.assign(confidence_bin=0))
    assert conditional["contextual_grouping_loss"] > pooled["contextual_grouping_loss"]
    assert conditional["matched_confidence_cross_environment_risk_gap"] == 0.8
    assert conditional["confidence_bin_weighted_max_environment_risk_gap"] == 0.8
