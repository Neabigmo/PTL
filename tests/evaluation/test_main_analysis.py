from __future__ import annotations

import pandas as pd

from src.evaluation.main_analysis import (
    FORBIDDEN_NARRATIVE_TERMS,
    build_failure_mode_atlas,
    build_main_findings,
    build_selective_prediction_summary,
    build_split_difficulty_summary,
    build_transfer_decay_summary,
    generate_results_narrative,
)


def _toy_all_metrics() -> pd.DataFrame:
    rows = []
    values = {
        ("ridge_regression_baseline", "random_split"): 0.80,
        ("global_delta_baseline", "random_split"): 0.60,
        ("ridge_regression_baseline", "external_holdout"): 0.20,
        ("global_delta_baseline", "external_holdout"): 0.45,
        ("ridge_regression_baseline", "low_support_split"): 0.70,
        ("global_delta_baseline", "low_support_split"): 0.50,
    }
    for (model, family), cosine in values.items():
        rows.append(
            {
                "run_id": f"{family}__{model}",
                "model": model,
                "split_family": family,
                "mean_cosine_non_control": cosine,
                "random_to_stress_drop": None if family == "random_split" else 0.80 - cosine if model.startswith("ridge") else 0.60 - cosine,
                "stress_retention": None if family == "random_split" else cosine / (0.80 if model.startswith("ridge") else 0.60),
                "mmd_linear": 0.1,
                "energy_distance": 0.2,
                "sliced_wasserstein_128": 0.3,
            }
        )
    return pd.DataFrame(rows)


def _toy_ptl_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ablation": "naive_confidence",
                "estimator": "naive_confidence",
                "mean_false_transportability_rate": 0.50,
                "mean_selective_risk": 0.70,
                "mean_false_transportability_rate_gain_vs_naive": None,
                "mean_selective_risk_gain_vs_naive": None,
                "roc_auc": 0.45,
                "average_precision": 0.50,
                "brier": 0.40,
            },
            {
                "ablation": "full_PTL",
                "estimator": "random_forest",
                "mean_false_transportability_rate": 0.20,
                "mean_selective_risk": 0.80,
                "mean_false_transportability_rate_gain_vs_naive": 0.30,
                "mean_selective_risk_gain_vs_naive": -0.10,
                "roc_auc": 0.90,
                "average_precision": 0.92,
                "brier": 0.12,
            },
        ]
    )


def _toy_ptl_ablation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ablation": "full_PTL", "status": "completed", "reason": ""},
        ]
    )


def _toy_failure_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "ridge_regression_baseline",
                "split_family": "external_holdout",
                "failure_mode": "severe_failure",
                "n_signatures": 12,
                "transportable_rate": 0.0,
                "mean_cosine": -0.1,
                "mean_risk": 1.1,
                "mean_confidence": 0.2,
                "mean_n_cells": 15,
                "mean_perturbation_seen": 0.1,
                "mean_component_seen_fraction": 0.2,
                "mean_reference_seen": 0.4,
            },
            {
                "model": "ridge_regression_baseline",
                "split_family": "external_holdout",
                "failure_mode": "transportable",
                "n_signatures": 8,
                "transportable_rate": 1.0,
                "mean_cosine": 0.5,
                "mean_risk": 0.5,
                "mean_confidence": 0.7,
                "mean_n_cells": 30,
                "mean_perturbation_seen": 1.0,
                "mean_component_seen_fraction": 1.0,
                "mean_reference_seen": 1.0,
            },
        ]
    )


def test_transfer_decay_summary_detects_rank_changes() -> None:
    summary = build_transfer_decay_summary(_toy_all_metrics())
    ridge_external = summary[
        (summary["model"] == "ridge_regression_baseline")
        & (summary["split_family"] == "external_holdout")
    ].iloc[0]
    global_external = summary[
        (summary["model"] == "global_delta_baseline")
        & (summary["split_family"] == "external_holdout")
    ].iloc[0]
    assert int(ridge_external["random_split_rank"]) == 1
    assert int(ridge_external["family_rank"]) == 2
    assert bool(ridge_external["ranking_reversal_flag"]) is True
    assert bool(global_external["family_winner"]) is True


def test_selective_prediction_summary_calculates_naive_deltas() -> None:
    summary = build_selective_prediction_summary(_toy_ptl_metrics(), _toy_ptl_ablation())
    ptl = summary[summary["estimator"] == "random_forest"].iloc[0]
    assert ptl["primary_transportability_view"] == "improves_false_transportability_filtering"
    assert ptl["secondary_risk_view"] == "higher_or_equal_cosine_risk_than_naive"
    assert ptl["false_transportability_rate_delta_vs_naive"] == -0.30


def test_failure_mode_atlas_marks_dominant_and_context_signal() -> None:
    atlas = build_failure_mode_atlas(_toy_failure_summary())
    severe = atlas[atlas["failure_mode"] == "severe_failure"].iloc[0]
    assert bool(severe["dominant_failure"]) is True
    assert severe["failure_severity"] == 3
    assert severe["context_signal"] == "perturbation_novelty"


def test_split_difficulty_summary_normalizes_against_random_anchor() -> None:
    examples = pd.DataFrame(
        [
            {
                "model": "ridge_regression_baseline",
                "split_family": "external_holdout",
                "perturbation_train_signature_count": 0,
                "perturbation_train_cell_count": 0,
                "perturbation_seen_in_train": 0.0,
                "component_seen_fraction": 0.0,
                "unseen_component_count": 1.0,
                "delta_norm_true": 2.0,
                "train_test_centroid_l2": 3.0,
                "n_cells": 12,
            },
            {
                "model": "ridge_regression_baseline",
                "split_family": "random_split",
                "perturbation_train_signature_count": 10,
                "perturbation_train_cell_count": 100,
                "perturbation_seen_in_train": 1.0,
                "component_seen_fraction": 1.0,
                "unseen_component_count": 0.0,
                "delta_norm_true": 1.0,
                "train_test_centroid_l2": 0.1,
                "n_cells": 20,
            },
        ]
    )
    summary = build_split_difficulty_summary(_toy_all_metrics(), examples)
    ridge_external = summary[
        (summary["model"] == "ridge_regression_baseline")
        & (summary["split_family"] == "external_holdout")
    ].iloc[0]
    assert ridge_external["random_anchor_cosine"] == 0.80
    assert ridge_external["normalized_transfer_score"] == 0.25
    assert ridge_external["difficulty_index"] > 0.5


def test_narrative_is_bounded_and_avoids_forbidden_terms() -> None:
    transfer = build_transfer_decay_summary(_toy_all_metrics())
    selective = build_selective_prediction_summary(_toy_ptl_metrics(), _toy_ptl_ablation())
    atlas = build_failure_mode_atlas(_toy_failure_summary())
    pairwise = pd.DataFrame(
        [{"metric": "mean_cosine_non_control", "reject_fdr_0_05": True}]
    )
    findings = build_main_findings(transfer, selective, atlas, pairwise)
    narrative = generate_results_narrative(findings, transfer, selective)
    lower = narrative.lower()
    assert "limitations" in lower
    for term in FORBIDDEN_NARRATIVE_TERMS:
        assert term not in lower
    assert len(findings) >= 5
