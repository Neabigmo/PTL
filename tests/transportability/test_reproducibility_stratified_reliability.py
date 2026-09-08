from __future__ import annotations

import pandas as pd

from scripts.run_formal_v2_reproducibility_stratified_reliability import _reliability_shift_row


def test_reproducibility_components_are_conditioned_and_evaluation_only() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["toy"] * 6,
            "predictor": ["toy_predictor"] * 6,
            "biological_instance_id": [f"bio_{index}" for index in range(6)],
            "continuous_risk": [0.10, 0.20, 0.15, 0.70, 0.80, 0.75],
            "reproducibility_stratum": ["high", "high", "high", "low", "low", "low"],
            "confidence_bin": [0, 1, 0, 0, 1, 0],
            "ptl_rf": [0.1, 0.2, 0.15, 0.7, 0.8, 0.75],
        }
    )

    row = _reliability_shift_row(frame, "toy", "toy_predictor", "ptl_rf")

    assert row["used_in_ptl_features"] == 0
    assert row["evaluation_only"] == 1
    assert row["high_n"] == 3
    assert row["low_n"] == 3
    assert row["low_high_difficulty_delta"] < 0.0
    assert row["all_strata_conditional_contextual_grouping_loss"] > 0.0
    assert row["semantic_status"] == "executed_descriptive_reproducibility_stratum_crossfit"
