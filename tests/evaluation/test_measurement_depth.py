import numpy as np
import pandas as pd

from src.evaluation.measurement_depth import (
    coverage_curve,
    depth_resolution_table,
    matched_label_universe,
    summarize_depth_rows,
    hierarchical_macro_bootstrap,
)


def _counts():
    return pd.DataFrame({
        "environment_id": ["left", "left", "left", "right", "right", "right"],
        "perturbation_label": ["control", "A", "B", "control", "A", "B"],
        "n_cells_qc": [20, 12, 6, 20, 10, 8],
    })


def test_matched_universe_requires_both_targets_and_control():
    assert matched_label_universe(_counts(), left_environment="left", right_environment="right", budget=10) == ["A"]
    curve = coverage_curve(_counts(), left_environment="left", right_environment="right", budgets=(10, 20))
    assert curve["n_labels_matched"].tolist() == [1, 0]
    assert curve.loc[0, "coverage_fraction"] == 0.5


def test_depth_summary_bootstraps_seed_level_and_resolution_uses_full_row():
    rows = []
    for seed, cross in [(1, 0.2), (2, 0.3), (3, 0.25)]:
        for label in ["A", "B"]:
            rows.append({
                "source_environment_id": "source",
                "left_target_environment_id": "left",
                "right_target_environment_id": "right",
                "metric": "delta_cosine",
                "cell_budget": 10,
                "cell_budget_order": 1,
                "split_seed": seed,
                "perturbation_label": label,
                "cross_disagreement": cross,
                "within_disagreement_left": 0.1,
                "within_disagreement_right": 0.1,
                "identifiable_divergence": cross - 0.1,
                "stable_pair_fraction": 0.8,
            })
        for label in ["A", "B"]:
            rows.append({
                "source_environment_id": "source",
                "left_target_environment_id": "left",
                "right_target_environment_id": "right",
                "metric": "delta_cosine",
                "cell_budget": 160,
                "cell_budget_order": 6,
                "split_seed": seed,
                "perturbation_label": label,
                "cross_disagreement": 0.3,
                "within_disagreement_left": 0.1,
                "within_disagreement_right": 0.1,
                "identifiable_divergence": 0.2,
                "stable_pair_fraction": 0.9,
            })
    summary = summarize_depth_rows(pd.DataFrame(rows), draws=50)
    assert len(summary) == 2
    resolved = depth_resolution_table(summary)
    assert resolved.loc[0, "resolution_90pct_full_budget"] == 160
    assert np.isclose(resolved.loc[0, "full_depth_value"], 0.2)


def test_hierarchical_macro_bootstrap_keeps_label_and_seed_layers_explicit():
    rows = []
    for seed in (1, 2, 3):
        for label, value in (("A", 0.1), ("B", 0.3)):
            rows.append({
                "source_environment_id": "left", "left_target_environment_id": "left", "right_target_environment_id": "right",
                "metric": "delta_cosine", "cell_budget": 10, "cell_budget_label": 10, "cell_budget_order": 1,
                "universe_mode": "matched_fixed", "split_seed": seed, "perturbation_label": label,
                "cross_disagreement": value, "within_disagreement_left": 0.1, "within_disagreement_right": 0.1,
                "identifiable_divergence": value - 0.1, "stable_pair_fraction": 0.8,
            })
    macro = hierarchical_macro_bootstrap(pd.DataFrame(rows), draws=20)
    assert len(macro) == 20
    assert set(macro["bootstrap_unit"]) == {"perturbation_label_then_measurement_seed"}
    assert set(macro["universe_mode"]) == {"matched_fixed"}
