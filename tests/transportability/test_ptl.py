from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.transportability.ptl import (
    apply_ablation,
    build_feature_audit,
    build_ptl_examples,
    build_target_columns,
    infer_feature_spec,
    perturbation_novelty_features,
)
from src.transportability.train_ptl import run_phase07


def test_perturbation_novelty_features_are_deterministic() -> None:
    features = perturbation_novelty_features("A_B", {"A", "C"})
    assert features["perturbation_seen_in_train"] == 0.0
    assert features["is_combination"] == 1.0
    assert features["component_count"] == 2.0
    assert features["component_seen_fraction"] == 0.5
    assert features["unseen_component_count"] == 1.0


def test_target_columns_and_feature_spec_do_not_use_outcome_columns() -> None:
    frame = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "signature_id": ["s1", "s2"],
            "model": ["m", "m"],
            "split_family": ["low_support_split", "low_support_split"],
            "confidence": [0.2, 0.8],
            "cosine_similarity": [0.1, 0.9],
            "risk": [0.9, 0.1],
            "transport_threshold": [0.5, 0.5],
            "transportable": pd.Series([False, True], dtype="boolean"),
        }
    )
    target = build_target_columns(frame)
    assert target["target_transportable"].tolist() == [0, 1]
    assert target["failure_mode"].tolist() == ["high_risk_failure", "transportable"]
    spec = infer_feature_spec(target)
    assert "confidence" in spec.numeric
    assert "cosine_similarity" not in spec.numeric
    assert "risk" not in spec.numeric
    assert "transportable" not in spec.numeric


def test_ptl_feature_audit_blocks_target_and_evaluation_only_columns() -> None:
    frame = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "signature_id": ["s1", "s2"],
            "model": ["m1", "m2"],
            "split_family": ["external_holdout", "external_holdout"],
            "dataset_id": ["d1", "d2"],
            "confidence": [0.3, 0.7],
            "perturbation_seen_in_train": [0.0, 1.0],
            "train_test_centroid_l2": [1.0, 0.2],
            "cosine_similarity": [0.2, 0.9],
            "risk": [0.8, 0.1],
            "failure_mode": ["severe_failure", "transportable"],
            "transportable": [False, True],
            "target_transportable": [0, 1],
        }
    )
    spec = infer_feature_spec(frame, feature_mode="deployment")
    assert {"confidence", "perturbation_seen_in_train", "train_test_centroid_l2"}.issubset(spec.numeric)
    assert "cosine_similarity" not in spec.numeric
    assert "risk" not in spec.numeric
    audit = build_feature_audit(frame, feature_mode="deployment", spec=spec)
    blocked = audit.set_index("feature_name")
    assert blocked.loc["cosine_similarity", "target_or_label"]
    assert blocked.loc["risk", "target_or_label"]
    assert blocked.loc["failure_mode", "target_or_label"]
    assert blocked.loc["run_id", "identifier"]
    assert not blocked.loc["confidence", "excluded_from_ptl"]


def test_ablation_column_filtering_removes_requested_feature_family() -> None:
    frame = pd.DataFrame(
        {
            "target_transportable": [0, 1],
            "target_risk": [0.7, 0.2],
            "confidence": [0.1, 0.9],
            "train_test_centroid_l2": [1.0, 0.1],
            "perturbation_seen_in_train": [0.0, 1.0],
            "component_seen_fraction": [0.0, 1.0],
            "model": ["a", "b"],
        }
    )
    spec = infer_feature_spec(frame)
    no_context = apply_ablation(spec, "no_context_distance")
    no_perturbation = apply_ablation(spec, "no_perturbation_novelty")
    assert "train_test_centroid_l2" not in no_context.numeric
    assert "perturbation_seen_in_train" not in no_perturbation.numeric
    assert "component_seen_fraction" not in no_perturbation.numeric
    assert "confidence" in no_context.numeric


def _write_toy_phase_inputs(root: Path) -> dict[str, Path]:
    tables = root / "tables"
    data = root / "data"
    tables.mkdir(parents=True)
    data.mkdir(parents=True)

    genes = ["g1", "g2", "g3"]
    rows = []
    for ref in ("r0", "r1"):
        rows.append(
            {
                "dataset_id": "toy",
                "source_dataset": "toy_source",
                "signature_id": f"toy__{ref}__control",
                "reference_key": ref,
                "perturbation_label": "control",
                "is_control": True,
                "n_cells": 20,
                "control_label_used": "control",
                "batch": "b0",
                "timepoint": "t0",
                "g1": 0.0,
                "g2": 0.0,
                "g3": 0.0,
            }
        )
    for idx, perturbation in enumerate(["A", "B", "A", "C", "D", "E", "F", "G", "A_B", "C_D"]):
        ref = "r0" if idx < 5 else "r1"
        rows.append(
            {
                "dataset_id": "toy",
                "source_dataset": "toy_source",
                "signature_id": f"toy__{idx}__{perturbation}",
                "reference_key": ref,
                "perturbation_label": perturbation,
                "is_control": False,
                "n_cells": 5 + idx,
                "control_label_used": "control",
                "batch": "b0",
                "timepoint": "t0",
                "g1": float(idx),
                "g2": float(idx + 1),
                "g3": float(idx + 2),
            }
        )
    signatures = pd.DataFrame(rows)
    pseudobulk = signatures.copy()
    pseudobulk.insert(3, "group_key", pseudobulk["signature_id"])
    pseudobulk = pseudobulk.drop(columns=["signature_id", "control_label_used"])
    signature_path = data / "toy_delta_signatures.parquet"
    pseudobulk_path = data / "toy_pseudobulk.parquet"
    signatures.to_parquet(signature_path, index=False)
    pseudobulk.to_parquet(pseudobulk_path, index=False)

    preprocessing = pd.DataFrame(
        [
            {
                "dataset_id": "toy",
                "source_dataset": "toy_source",
                "status": "processed",
                "signature_path": str(signature_path),
                "pseudobulk_path": str(pseudobulk_path),
            }
        ]
    )
    preprocessing_path = tables / "preprocessing_summary.csv"
    preprocessing.to_csv(preprocessing_path, index=False)

    train_ids = ["toy__r0__control", "toy__r1__control", "toy__0__A", "toy__1__B"]
    test_ids = [f"toy__{idx}__{pert}" for idx, pert in enumerate(["A", "B", "A", "C", "D", "E", "F", "G", "A_B", "C_D"])]
    split_path = data / "low_support_split__toy__seed0__signature.json"
    split_path.write_text(
        json.dumps(
            {
                "split_family": "low_support_split",
                "track": "signature",
                "dataset_scope": "toy",
                "seed": 0,
                "status": "ready",
                "holdout_variable": "perturbation_label",
                "heldout_target": "",
                "unsupported_reason": "",
                "train_ids": train_ids,
                "validation_ids": ["toy__0__A"],
                "test_ids": test_ids,
            }
        ),
        encoding="utf-8",
    )

    run_id = "low_support_split__toy__seed0__signature__ridge_regression_baseline"
    manifest = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "model": "ridge_regression_baseline",
                "split_json_path": str(split_path),
                "split_family": "low_support_split",
                "dataset_scope": "toy",
                "heldout_target": "",
                "seed": 0,
                "gene_space_policy": "native",
                "expected_gene_count": 3,
                "output_dir": str(root / "baseline"),
                "log_file": str(root / "baseline.log"),
            }
        ]
    )
    manifest_path = tables / "baseline_run_matrix.csv"
    manifest.to_csv(manifest_path, index=False)

    split_audit = pd.DataFrame(
        [
            {
                "split_family": "low_support_split",
                "track": "signature",
                "dataset_scope": "toy",
                "seed": 0,
                "status": "ready",
                "heldout_target": "",
                "output_path": str(split_path),
                "train_units": len(train_ids),
                "validation_units": 1,
                "test_units": len(test_ids),
                "train_perturbations": 2,
                "validation_perturbations": 1,
                "test_perturbations": 8,
                "train_reference_keys": 2,
                "validation_reference_keys": 1,
                "test_reference_keys": 2,
                "reference_key_overlap_train_test": 2,
                "declared_holdout_overlap_train_test": 0,
                "train_test_centroid_l2": 1.5,
                "train_test_centroid_cosine": 0.7,
            }
        ]
    )
    split_audit_path = tables / "split_audit.csv"
    split_audit.to_csv(split_audit_path, index=False)

    metrics_rows = []
    labels = [True, True, False, False, True, False, True, False, True, False]
    for idx, signature_id in enumerate(test_ids):
        metrics_rows.append(
            {
                "dataset_id": "toy",
                "source_dataset": "toy_source",
                "signature_id": signature_id,
                "reference_key": "r0" if idx < 5 else "r1",
                "perturbation_label": signatures.loc[signatures["signature_id"] == signature_id, "perturbation_label"].iloc[0],
                "is_control": False,
                "n_cells": 5 + idx,
                "control_label_used": "control",
                "batch": "b0",
                "timepoint": "t0",
                "model": "ridge_regression_baseline",
                "split_family": "low_support_split",
                "seed": 0,
                "run_id": run_id,
                "gene_space_policy": "native",
                "gene_count": 3,
                "cosine_similarity": 0.85 if labels[idx] else 0.25,
                "rmse": 0.1,
                "mae": 0.1,
                "delta_norm_true": 1.0,
                "delta_norm_pred": 1.2 + idx,
                "risk": 0.15 if labels[idx] else 0.75,
                "confidence": 0.8 if labels[idx] else 0.3,
                "transportable": labels[idx],
                "anchor_signature_median_cosine": 0.8,
                "transport_threshold": 0.64,
            }
        )
    per_signature_path = tables / "per_signature_metrics.parquet"
    pd.DataFrame(metrics_rows).to_parquet(per_signature_path, index=False)
    all_metrics_path = tables / "all_metrics.csv"
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "anchor_mean_cosine": 0.8,
                "anchor_signature_median_cosine": 0.8,
                "n_test_non_control": 10,
                "n_test_control": 0,
            }
        ]
    ).to_csv(all_metrics_path, index=False)
    return {
        "per_signature": per_signature_path,
        "all_metrics": all_metrics_path,
        "manifest": manifest_path,
        "split_audit": split_audit_path,
        "preprocessing": preprocessing_path,
    }


def test_build_ptl_examples_and_cli_smoke(tmp_path: Path) -> None:
    paths = _write_toy_phase_inputs(tmp_path)
    examples = build_ptl_examples(
        per_signature_metrics_path=paths["per_signature"],
        all_metrics_path=paths["all_metrics"],
        manifest_path=paths["manifest"],
        split_audit_path=paths["split_audit"],
        preprocessing_summary_path=paths["preprocessing"],
        max_examples=8,
        random_state=1,
    )
    assert set(examples["target_transportable"]) == {0, 1}
    assert examples["perturbation_seen_in_train"].notna().all()
    assert examples["reference_seen_in_train"].notna().all()

    output_dir = tmp_path / "results"
    registry = tmp_path / "registry.csv"
    run_phase07(
        argparse.Namespace(
            per_signature_metrics=str(paths["per_signature"]),
            metrics=str(paths["all_metrics"]),
            manifest=str(paths["manifest"]),
            split_audit=str(paths["split_audit"]),
            preprocessing_summary=str(paths["preprocessing"]),
            output_dir=str(output_dir),
            registry=str(registry),
            log_file=str(tmp_path / "phase07_ptl.log"),
            max_examples=8,
            random_state=1,
            include_biology_features=False,
            feature_mode="deployment",
            write_feature_audit=True,
            run_label="toy_phase07",
        )
    )
    assert (output_dir / "tables" / "ptl_metrics.csv").exists()
    assert (output_dir / "tables" / "ptl_ablation.csv").exists()
    assert (output_dir / "tables" / "failure_mode_summary.csv").exists()
    predictions_path = output_dir / "tables" / "ptl_oof_predictions.csv"
    assert predictions_path.exists()
    predictions = pd.read_csv(predictions_path)
    assert {
        "example_id",
        "ablation",
        "estimator",
        "y_true",
        "score",
        "split_family",
        "model",
        "signature_id",
        "run_id",
    }.issubset(predictions.columns)
    assert predictions["score"].between(0, 1).all()
    audit_path = output_dir / "tables" / "ptl_feature_audit.csv"
    assert audit_path.exists()
    audit = pd.read_csv(audit_path)
    assert {"deployment_available", "evaluation_only", "target_or_label", "identifier", "excluded_from_ptl"}.issubset(audit.columns)
    assert registry.exists()
