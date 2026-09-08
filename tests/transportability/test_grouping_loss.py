import numpy as np
import pandas as pd

from scripts.run_formal_v2_grouping_loss import (
    _hierarchical_bootstrap_sample,
    _replicate_wise_bootstrap_means,
    _summarize_conditional_grouped,
    _summarize_arrays,
    assign_bins,
    summarize_split,
)


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


class _DeterministicRng:
    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        assert (low, high) == (0, 2)
        return np.zeros(size, dtype=int)

    def choice(self, values: list[str], size: int, replace: bool) -> np.ndarray:
        assert replace is True
        return np.asarray(values[:size], dtype=object)


def test_grouping_bootstrap_keeps_duplicate_environment_draws_as_clusters() -> None:
    group_arrays = [
        (
            np.asarray([0.1, 0.2]),
            np.asarray(["e1", "e1"], dtype=object),
            np.asarray([0, 1]),
            ["e1_a", "e1_b"],
            {"e1_a": np.asarray([0]), "e1_b": np.asarray([1])},
        ),
        (
            np.asarray([0.8, 0.9]),
            np.asarray(["e2", "e2"], dtype=object),
            np.asarray([0, 1]),
            ["e2_a", "e2_b"],
            {"e2_a": np.asarray([0]), "e2_b": np.asarray([1])},
        ),
    ]

    _, clusters, originals, _ = _hierarchical_bootstrap_sample(group_arrays, _DeterministicRng())

    assert set(clusters.tolist()) == {
        "e1__bootstrap_cluster_0",
        "e1__bootstrap_cluster_1",
    }
    assert set(originals.tolist()) == {"e1"}


def test_grouping_bootstrap_duplicate_clusters_keep_same_original_pair_identity() -> None:
    group_arrays = [
        (
            np.asarray([0.1, 0.2]),
            np.asarray(["e1", "e1"], dtype=object),
            np.asarray([0, 0]),
            ["e1_a", "e1_b"],
            {"e1_a": np.asarray([0]), "e1_b": np.asarray([1])},
        ),
        (
            np.asarray([0.8, 0.9]),
            np.asarray(["e2", "e2"], dtype=object),
            np.asarray([0, 0]),
            ["e2_a", "e2_b"],
            {"e2_a": np.asarray([0]), "e2_b": np.asarray([1])},
        ),
    ]

    risk, clusters, originals, bins = _hierarchical_bootstrap_sample(group_arrays, _DeterministicRng())
    stats = _summarize_arrays(risk, clusters, bins, originals)

    assert len(set(clusters.tolist())) == 2
    assert set(originals.tolist()) == {"e1"}
    assert np.isfinite(stats["matched_confidence_same_environment_risk_gap"])
    assert np.isnan(stats["matched_confidence_cross_environment_risk_gap"])
    assert np.isnan(stats["matched_confidence_cross_minus_same_risk_gap"])


def test_summary_ci_uses_replicate_wise_split_means_not_pooled_replicates() -> None:
    bootstrap = pd.DataFrame({
        "split_seed": [11] * 4 + [22] * 4,
        "replicate": [0, 1, 2, 3] * 2,
        "metric": [0.0, 0.0, 10.0, 10.0, 100.0, 100.0, 110.0, 110.0],
    })

    aligned_means = _replicate_wise_bootstrap_means(bootstrap, "metric", [11, 22])
    pooled = bootstrap["metric"].to_numpy(dtype=float)

    np.testing.assert_array_equal(aligned_means, np.asarray([50.0, 50.0, 60.0, 60.0]))
    assert np.quantile(aligned_means, 0.025) > np.quantile(pooled, 0.025)
    assert np.quantile(aligned_means, 0.975) < np.quantile(pooled, 0.975)
