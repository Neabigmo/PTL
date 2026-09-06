from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from src.figures.make_figures import (
    Phase09Inputs,
    figure_paths,
    run_phase09,
    validate_inputs,
    wrap_label,
    write_figure_note,
)


def _write_toy_inputs(base: Path) -> Phase09Inputs:
    tables = base / "tables"
    docs = base / "docs"
    tables.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "finding_id": "F1",
                "question": "What changes under stress?",
                "claim": "external_holdout is hardest in the toy benchmark.",
                "evidence_table": "results/tables/transfer_decay_summary.csv",
                "primary_metric": "mean_cosine_non_control",
                "primary_value": 0.20,
                "comparator": "random_split",
                "bounded_interpretation": "Stress changes ranking.",
                "limitation": "Toy data only.",
            }
        ]
    ).to_csv(tables / "main_findings.csv", index=False)

    transfer_rows = []
    for model, random_value, external_value, random_rank, external_rank, winner_external in [
        ("ridge_regression_baseline", 0.80, 0.25, 1, 2, False),
        ("global_delta_baseline", 0.60, 0.45, 2, 1, True),
    ]:
        transfer_rows.extend(
            [
                {
                    "model": model,
                    "split_family": "random_split",
                    "n_runs": 1,
                    "mean_cosine_non_control": random_value,
                    "median_cosine_non_control": random_value,
                    "mean_random_to_stress_drop": None,
                    "mean_stress_retention": None,
                    "family_rank": random_rank,
                    "family_winner": random_rank == 1,
                    "random_split_rank": random_rank,
                    "random_split_mean_cosine": random_value,
                    "rank_change_vs_random": 0,
                    "cosine_change_vs_random": 0,
                    "rank_shift_direction": "anchor",
                    "ranking_reversal_flag": False,
                    "stress_retention_flag": "anchor",
                },
                {
                    "model": model,
                    "split_family": "external_holdout",
                    "n_runs": 1,
                    "mean_cosine_non_control": external_value,
                    "median_cosine_non_control": external_value,
                    "mean_random_to_stress_drop": random_value - external_value,
                    "mean_stress_retention": external_value / random_value,
                    "family_rank": external_rank,
                    "family_winner": winner_external,
                    "random_split_rank": random_rank,
                    "random_split_mean_cosine": random_value,
                    "rank_change_vs_random": external_rank - random_rank,
                    "cosine_change_vs_random": external_value - random_value,
                    "rank_shift_direction": "worse_rank" if external_rank > random_rank else "better_rank",
                    "ranking_reversal_flag": True,
                    "stress_retention_flag": "interpretable_ratio",
                },
            ]
        )
    pd.DataFrame(transfer_rows).to_csv(tables / "transfer_decay_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "ablation": "naive_confidence",
                "estimator": "naive_confidence",
                "mean_false_transportability_rate": 0.50,
                "mean_selective_risk": 0.70,
            },
            {
                "ablation": "full_PTL",
                "estimator": "random_forest",
                "mean_false_transportability_rate": 0.20,
                "mean_selective_risk": 0.80,
            },
        ]
    ).to_csv(tables / "selective_prediction_summary.csv", index=False)
    pd.DataFrame(
        [
            {"example_id": "e1", "ablation": "naive_confidence", "estimator": "naive_confidence", "y_true": 1, "score": 0.55, "split_family": "external_holdout", "model": "ridge_regression_baseline", "signature_id": "s1", "run_id": "r1", "target_risk": 0.2, "confidence": 0.55, "failure_mode": "transportable"},
            {"example_id": "e2", "ablation": "naive_confidence", "estimator": "naive_confidence", "y_true": 0, "score": 0.70, "split_family": "external_holdout", "model": "ridge_regression_baseline", "signature_id": "s2", "run_id": "r1", "target_risk": 1.1, "confidence": 0.70, "failure_mode": "severe_failure"},
            {"example_id": "e3", "ablation": "full_PTL", "estimator": "random_forest", "y_true": 1, "score": 0.92, "split_family": "external_holdout", "model": "ridge_regression_baseline", "signature_id": "s1", "run_id": "r1", "target_risk": 0.2, "confidence": 0.55, "failure_mode": "transportable"},
            {"example_id": "e4", "ablation": "full_PTL", "estimator": "random_forest", "y_true": 0, "score": 0.12, "split_family": "external_holdout", "model": "ridge_regression_baseline", "signature_id": "s2", "run_id": "r1", "target_risk": 1.1, "confidence": 0.70, "failure_mode": "severe_failure"},
            {"example_id": "e5", "ablation": "no_context_distance", "estimator": "random_forest", "y_true": 1, "score": 0.88, "split_family": "external_holdout", "model": "ridge_regression_baseline", "signature_id": "s1", "run_id": "r1", "target_risk": 0.2, "confidence": 0.55, "failure_mode": "transportable"},
            {"example_id": "e6", "ablation": "no_context_distance", "estimator": "random_forest", "y_true": 0, "score": 0.18, "split_family": "external_holdout", "model": "ridge_regression_baseline", "signature_id": "s2", "run_id": "r1", "target_risk": 1.1, "confidence": 0.70, "failure_mode": "severe_failure"},
        ]
    ).to_csv(tables / "ptl_oof_predictions.csv", index=False)

    pd.DataFrame(
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
                "failure_mode_share": 0.6,
                "failure_severity": 3,
                "dominant_failure": True,
                "dominant_nontransportable_failure": True,
                "context_signal": "perturbation_novelty",
            },
            {
                "model": "global_delta_baseline",
                "split_family": "external_holdout",
                "failure_mode": "high_risk_failure",
                "n_signatures": 8,
                "transportable_rate": 0.0,
                "mean_cosine": 0.1,
                "mean_risk": 0.9,
                "mean_confidence": 0.3,
                "mean_n_cells": 20,
                "mean_perturbation_seen": 0.7,
                "mean_component_seen_fraction": 0.7,
                "mean_reference_seen": 0.8,
                "failure_mode_share": 0.4,
                "failure_severity": 2,
                "dominant_failure": False,
                "dominant_nontransportable_failure": False,
                "context_signal": "mixed_or_supported_context",
            },
        ]
    ).to_csv(tables / "failure_mode_atlas.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "ridge_regression_baseline",
                "split_family": "external_holdout",
                "signature_id": "s2",
                "run_id": "r1",
                "perturbation_label": "SOX2",
                "failure_mode": "severe_failure",
                "target_transportable": 0,
                "target_fidelity": -0.1,
                "target_risk": 1.1,
                "perturbation_train_signature_count": 0,
            }
        ]
    ).to_parquet(tables / "ptl_examples.parquet", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "r1",
                "output_dir": str(base / "missing_baseline_dir"),
            }
        ]
    ).to_csv(tables / "baseline_run_matrix.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": "toy_dataset",
                "n_signature_rows": 20,
                "n_unique_perturbations": 5,
            }
        ]
    ).to_csv(tables / "preprocessing_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "split_family": "external_holdout",
                "track": "signature",
                "test_units": 10,
                "train_test_centroid_l2": 1.2,
                "perturbation_overlap_train_test": 0,
                "dataset_overlap_train_test": 0,
                "reference_key_overlap_train_test": 0,
                "declared_holdout_overlap_train_test": 0,
            }
        ]
    ).to_csv(tables / "split_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "r1",
                "model": "ridge_regression_baseline",
                "split_family": "external_holdout",
                "seed": 0,
                "mean_cosine_non_control": 0.2,
                "n_test_non_control": 10,
            },
            {
                "run_id": "r2",
                "model": "global_delta_baseline",
                "split_family": "dataset_heldout_split",
                "seed": 0,
                "mean_cosine_non_control": 0.4,
                "n_test_non_control": 12,
            },
        ]
    ).to_csv(tables / "all_metrics.csv", index=False)

    (docs / "results_narrative.md").write_text(
        "# Results narrative\n\nLimitations are bounded in this toy narrative.\n",
        encoding="utf-8",
    )
    return Phase09Inputs(
        main_findings=tables / "main_findings.csv",
        transfer_decay=tables / "transfer_decay_summary.csv",
        selective_prediction=tables / "selective_prediction_summary.csv",
        failure_atlas=tables / "failure_mode_atlas.csv",
        narrative=docs / "results_narrative.md",
        ptl_predictions=tables / "ptl_oof_predictions.csv",
        ptl_examples=tables / "ptl_examples.parquet",
        all_metrics=tables / "all_metrics.csv",
        split_audit=tables / "split_audit.csv",
        preprocessing_summary=tables / "preprocessing_summary.csv",
        baseline_manifest=tables / "baseline_run_matrix.csv",
    )


def test_required_output_path_construction(tmp_path: Path) -> None:
    paths = figure_paths(tmp_path)
    assert paths["fig1_png"] == tmp_path / "fig1_study_design.png"
    assert paths["fig5_png"] == tmp_path / "fig5_failure_mode_atlas.png"
    assert paths["supp_fig1_png"] == tmp_path / "supp_fig1_robustness_audit.png"
    assert paths["fig3_pdf"].suffix == ".pdf"
    assert all("html" not in key for key in paths)


def test_non_empty_data_validation(tmp_path: Path) -> None:
    inputs = _write_toy_inputs(tmp_path)
    loaded = validate_inputs(inputs)
    assert loaded["main"].shape[0] == 1
    assert loaded["transfer"].shape[0] == 4
    assert loaded["predictions"].shape[0] == 6


def test_safe_label_wrapping_for_long_names() -> None:
    wrapped = wrap_label("very_long_model_name_without_spaces_for_wrapping", width=12)
    assert "\n" in wrapped
    assert max(len(line) for line in wrapped.splitlines()) <= 12


def test_figure_note_generation(tmp_path: Path) -> None:
    note = write_figure_note(
        tmp_path,
        "fig1",
        "Figure 1",
        "Message text.",
        "input.csv",
        "Encoding text.",
        "Limitation text.",
    )
    text = note.read_text(encoding="utf-8")
    assert "Message:" in text
    assert "Input table:" in text
    assert "Limitation:" in text


def test_phase09_smoke_run_creates_nonblank_pngs(tmp_path: Path) -> None:
    inputs = _write_toy_inputs(tmp_path)
    output_dir = tmp_path / "figures"
    notes_dir = tmp_path / "notes"
    outputs = run_phase09(inputs, output_dir, notes_dir)
    required = figure_paths(output_dir)
    for key in ["fig1_png", "fig2_png", "fig3_png", "fig4_png", "fig5_png", "supp_fig1_png"]:
        assert required[key].exists()
        with Image.open(required[key]) as image:
            assert image.size[0] > 100
            assert image.size[1] > 100
            assert image.convert("L").getextrema()[0] != image.convert("L").getextrema()[1]
    assert (notes_dir / "supp_fig1_robustness_audit.md").exists()
    assert len(outputs) == 18
