"""Validate and summarize the remote formal GEARS predictor contract.

The adapter emits a common delta-prediction contract, native uncertainty, and
validation predictions.  Validation is required for the supported Norman and
RPE1 reliability surface; the GWPS run remains a test-only external audit
because it is not part of the canonical eight-environment surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_run(row: pd.Series, output_root: Path) -> dict[str, Any]:
    run_id = str(row["run_id"])
    run_dir = output_root / run_id
    required = ["test_predictions.npz", "test_metadata.parquet", "run_metrics.json", "genes.txt"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_id}: missing standard outputs {missing}")
    arrays = np.load(run_dir / "test_predictions.npz")
    y_true = np.asarray(arrays["y_true"])
    y_pred = np.asarray(arrays["y_pred"])
    metadata = pd.read_parquet(run_dir / "test_metadata.parquet")
    metrics_payload = json.loads((run_dir / "run_metrics.json").read_text(encoding="utf-8"))
    metrics = metrics_payload["metrics"]
    details = metrics_payload["model_details"]
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError(f"{run_id}: invalid prediction shapes {y_true.shape} and {y_pred.shape}")
    if len(metadata) != y_true.shape[0] or len(details.get("gene_panel_sha256", "")) != 64:
        raise ValueError(f"{run_id}: prediction/metadata or gene-panel contract mismatch")
    if int(details.get("epochs", 0)) != 20 or str(details.get("device")) != "cuda":
        raise ValueError(f"{run_id}: formal run budget/device contract mismatch")
    native = "native_confidence_gears_exp_neg_mean_logvar" in arrays.files
    validation_required = str(row["dataset_id"]) != "ReplogleWeissman2022_K562_gwps"
    validation_path = run_dir / "validation_predictions.npz"
    validation_metadata_path = run_dir / "validation_metadata.parquet"
    validation_present = validation_path.is_file() and validation_metadata_path.is_file()
    if validation_required and not validation_present:
        raise FileNotFoundError(f"{run_id}: supported GEARS run is missing validation predictions")
    validation_rows = 0
    validation_native = 0
    if validation_present:
        with np.load(validation_path) as validation_arrays:
            validation_true = np.asarray(validation_arrays["y_true"])
            validation_pred = np.asarray(validation_arrays["y_pred"])
            validation_native = int("native_confidence_gears_exp_neg_mean_logvar" in validation_arrays.files)
        validation_metadata = pd.read_parquet(validation_metadata_path)
        if validation_true.shape != validation_pred.shape or validation_true.ndim != 2 or len(validation_metadata) != validation_true.shape[0]:
            raise ValueError(f"{run_id}: validation prediction/metadata contract mismatch")
        validation_rows = int(validation_true.shape[0])
    return {
        "run_id": run_id,
        "dataset_id": str(row["dataset_id"]),
        "seed": int(row["seed"]),
        "split_family": str(row["split_family"]),
        "split_protocol": str(row["split_protocol"]),
        "perturbation_overlap": str(row["perturbation_overlap"]),
        "track": str(row["track"]),
        "gene_space_policy": str(row["gene_space_policy"]),
        "gene_count": int(y_true.shape[1]),
        "n_test_rows": int(y_true.shape[0]),
        "n_test_noncontrol": int(metrics["n_test_noncontrol"]),
        "mean_cosine_similarity_non_control_test": float(metrics["mean_cosine_similarity_non_control_test"]),
        "mse_non_control_test": float(metrics["mse_non_control_test"]),
        "mae_non_control_test": float(metrics["mae_non_control_test"]),
        "native_uq_present": int(native),
        "validation_predictions_present": int(validation_present),
        "validation_predictions_required": int(validation_required),
        "validation_rows": validation_rows,
        "validation_native_uq_present": validation_native,
        "reliability_surface_eligible": int(validation_required and validation_present and validation_native),
        "official_commit": str(details["official_commit"]),
        "compatibility_patch_id": str(details["compatibility_patch"]["patch_id"]),
        "gene_panel_sha256": str(details["gene_panel_sha256"]),
        "genes_sha256": sha256_file(run_dir / "genes.txt"),
        "epochs": int(details["epochs"]),
        "device": str(details["device"]),
        "output_dir": run_dir.relative_to(ROOT).as_posix(),
    }


def build_report(root: Path, output_root: Path) -> dict[str, Any]:
    matrix_path = output_root / "formal_v2_gears_run_matrix.csv"
    matrix = pd.read_csv(matrix_path)
    if len(matrix) != 9 or set(matrix["seed"].astype(int)) != {0, 1, 2}:
        raise ValueError("formal GEARS matrix must contain exactly 3 datasets x 3 seeds")
    rows = [inspect_run(row, output_root) for _, row in matrix.sort_values(["dataset_id", "seed"]).iterrows()]
    frame = pd.DataFrame(rows)
    dataset_summary = (
        frame.groupby("dataset_id", as_index=False)
        .agg(
            n_runs=("run_id", "size"),
            mean_cosine=("mean_cosine_similarity_non_control_test", "mean"),
            sd_cosine=("mean_cosine_similarity_non_control_test", "std"),
            min_cosine=("mean_cosine_similarity_non_control_test", "min"),
            max_cosine=("mean_cosine_similarity_non_control_test", "max"),
            mean_mse=("mse_non_control_test", "mean"),
            mean_mae=("mae_non_control_test", "mean"),
            total_test_rows=("n_test_rows", "sum"),
        )
    )
    row_path = root / "artifacts/manifests/formal_v2_gears_external_validity.csv"
    dataset_path = root / "artifacts/manifests/formal_v2_gears_external_validity_dataset.csv"
    summary_path = root / "artifacts/manifests/formal_v2_gears_external_validity_summary.json"
    row_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(row_path, index=False)
    dataset_summary.to_csv(dataset_path, index=False)
    payload = {
        "schema_version": 1,
        "status": "formal_v2_gears_external_contract_executed_with_validation",
        "execution": {
            "n_runs": int(len(frame)),
            "datasets": sorted(frame["dataset_id"].unique().tolist()),
            "seeds": sorted(frame["seed"].unique().tolist()),
            "epochs": sorted(frame["epochs"].unique().tolist()),
            "devices": sorted(frame["device"].unique().tolist()),
            "split_protocol": "random_condition_holdout",
            "perturbation_overlap": "zero",
            "gene_space": "dataset-native frozen train-variance panel capped at 4096 genes",
        },
        "contract": {
            "prediction_space": "delta",
            "native_uncertainty": "GEARS exp(-mean(logvar)) retained in test contract",
            "validation_predictions_present": bool(frame["validation_predictions_present"].all()),
            "supported_validation_surface": sorted(frame.loc[frame["reliability_surface_eligible"].eq(1), "dataset_id"].unique().tolist()),
            "test_only_external_audit_surface": sorted(frame.loc[frame["reliability_surface_eligible"].eq(0), "dataset_id"].unique().tolist()),
            "reliability_surface_eligible": bool(frame.loc[frame["validation_predictions_required"].eq(1), "reliability_surface_eligible"].all()),
            "reliability_surface_reason": "validation predictions are required before GEARS enters the explanatory reliability-shift analysis; GWPS remains test-only external audit",
        },
        "row_path": row_path.relative_to(root).as_posix(),
        "dataset_path": dataset_path.relative_to(root).as_posix(),
        "rows": frame.to_dict("records"),
        "dataset_summary": dataset_summary.to_dict("records"),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output_root = (args.output_root or root / "results/formal_v2/gears_remote").resolve()
    payload = build_report(root, output_root)
    print(json.dumps(payload["execution"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
