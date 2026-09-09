import numpy as np

from src.evaluation.decision_theory import (
    decision_curve,
    measurement_floor_excess,
    replicate_decision_curve,
    top_k_decision_metrics,
)


def test_top_k_metrics_use_lowest_source_risk_and_report_regret():
    result = top_k_decision_metrics(
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([0.4, 0.1, 0.2, 0.3]),
        budget_fraction=0.5,
    )
    assert result["selected_indices"].tolist() == [0, 1]
    assert result["oracle_indices"].tolist() == [1, 2]
    assert result["retention"] == 0.5
    assert np.isclose(result["regret"], 0.1)
    assert result["mis_selection_count"] == 1
    assert result["boundary_bound_holds"]


def test_decision_curve_is_fixed_and_replicate_aggregation_preserves_budget_grid():
    source = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]])
    target = np.asarray([[0.4, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.4]])
    assert [row["budget_fraction"] for row in decision_curve(source[0], target[0], budgets=(0.25, 0.5))] == [0.25, 0.5]
    rows = replicate_decision_curve(source, target, budgets=(0.25, 0.5))
    assert [row["k"] for row in rows] == [1, 2]
    assert all(row["n_replicate_pairs"] == 4 for row in rows)
    assert np.isclose(measurement_floor_excess({"regret": 0.3}, {"regret": 0.1}), 0.2)
    off_diagonal = replicate_decision_curve(source, source, budgets=(0.5,), independent_only=True)
    assert off_diagonal[0]["n_replicate_pairs"] == 2
    assert off_diagonal[0]["independent_only"] is True
