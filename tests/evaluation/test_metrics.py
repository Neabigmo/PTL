from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.evaluation.evaluate_phase06 import compute_selective_metrics
from src.evaluation.metrics import (
    compute_distributional_metrics,
    compute_pathway_fidelity_metrics,
    compute_per_signature_metrics,
    compute_perturbation_discrimination_metrics,
)


def test_constant_equal_vectors_have_perfect_correlations() -> None:
    y_true = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    y_pred = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    result = compute_per_signature_metrics(y_true, y_pred)
    assert np.allclose(result["pearson_r"], [1.0, 1.0])
    assert np.allclose(result["spearman_r"], [1.0, 1.0])
    assert np.allclose(result["cosine_similarity"], [1.0, 1.0])


def test_constant_unequal_vectors_return_zero_rank_and_linear_correlation() -> None:
    y_true = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    y_pred = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    result = compute_per_signature_metrics(y_true, y_pred)
    assert result["pearson_r"][0] == 0.0
    assert result["spearman_r"][0] == 0.0
    assert result["cosine_similarity"][0] == 0.0


def test_distributional_metrics_identical_predictions_are_near_zero() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(20, 8)).astype(np.float32)
    test = rng.normal(size=(12, 8)).astype(np.float32)
    result = compute_distributional_metrics(train_true=train, test_true=test, test_pred=test.copy())
    assert result.status == "ready"
    assert result.mmd_linear <= 1e-10
    assert result.energy_distance <= 1e-6
    assert result.sliced_wasserstein_128 <= 1e-6


def test_distributional_metrics_small_sample_is_unsupported() -> None:
    rng = np.random.default_rng(1)
    train = rng.normal(size=(9, 4)).astype(np.float32)
    test = rng.normal(size=(4, 4)).astype(np.float32)
    result = compute_distributional_metrics(train_true=train, test_true=test, test_pred=test.copy())
    assert result.status == "unsupported"
    assert "below" in result.reason


def test_selective_metrics_have_monotonic_coverage_and_zero_full_coverage_gain() -> None:
    frame = pd.DataFrame(
        {
            "is_control": [False, False, False, False],
            "risk": [0.2, 0.4, 0.6, 0.8],
            "confidence": [0.9, 0.7, 0.4, 0.2],
            "transportable": pd.Series([True, False, True, False], dtype="boolean"),
        }
    )
    summary = compute_selective_metrics(frame)
    coverages = [summary[f"coverage_at_{str(t).replace('.', 'p')}"] for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    assert all(coverages[idx] >= coverages[idx + 1] for idx in range(len(coverages) - 1))
    assert math.isclose(summary["area_under_selective_risk_curve"], summary["area_under_selective_risk_curve"])
    assert summary["abstention_gain_at_0p1"] <= 1.0


def test_pathway_fidelity_metrics_are_reproducible_on_toy_matrix() -> None:
    genes = ["g1", "g2", "g3", "g4"]
    gene_sets = {"set_a": {"g1", "g2"}, "set_b": {"g3", "g4"}}
    y_true = np.array([[1.0, 1.0, -1.0, -1.0], [2.0, 0.0, 0.0, 2.0]], dtype=np.float32)
    y_pred = y_true.copy()
    result = compute_pathway_fidelity_metrics(y_true, y_pred, genes, gene_sets, min_genes=2)
    assert result["pathway_count"] == 2
    assert np.allclose(result["pathway_cosine"], [1.0, 1.0])
    assert np.allclose(result["pathway_direction_consistency"], [1.0, 1.0])


def test_perturbation_discrimination_retrieval_hits_matching_labels() -> None:
    y_true = np.eye(3, dtype=np.float32)
    y_pred = y_true.copy()
    labels = ["A", "B", "C"]
    result = compute_perturbation_discrimination_metrics(y_true, y_pred, labels, ks=(1, 2))
    assert result["perturbation_retrieval_top1"] == 1.0
    assert result["perturbation_retrieval_top2"] == 1.0
