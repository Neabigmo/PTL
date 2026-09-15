"""Build the tracked audit manifest for the official GEARS scientific pilot."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "results" / "tables" / "perturbation_model_run_matrix.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "manifests" / "gears_pilot_audit.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def non_control(values: object) -> set[str]:
    return {str(value) for value in values if str(value) != "ctrl"}


def main() -> None:
    args = parse_args()
    matrix = pd.read_csv(args.run_matrix)
    rows: list[dict[str, object]] = []
    panel_hashes: dict[str, str] = {}
    for row in matrix.itertuples(index=False):
        output_dir = Path(str(row.output_dir))
        metrics_path = output_dir / "run_metrics.json"
        prediction_path = output_dir / "test_predictions.npz"
        split_path = Path(str(row.split_json_path))
        if not metrics_path.is_file() or not prediction_path.is_file() or not split_path.is_file():
            raise FileNotFoundError(f"incomplete GEARS output for {row.run_id}")
        run = json.loads(metrics_path.read_text(encoding="utf-8"))
        details = run["model_details"]
        metrics = run["metrics"]
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
        split_protocol = str(split_manifest.get("split_protocol", ""))
        perturbation_overlap = str(split_manifest.get("perturbation_overlap", ""))
        if split_protocol != "random_condition_holdout" or perturbation_overlap != "zero":
            raise AssertionError(
                f"GEARS split semantics are not explicit for {row.run_id}: "
                f"protocol={split_protocol!r}, overlap={perturbation_overlap!r}"
            )
        split_path_local = output_dir / "gears_custom_split.pkl"
        if not split_path_local.is_file():
            raise FileNotFoundError(split_path_local)
        with split_path_local.open("rb") as handle:
            split = pickle.load(handle)
        train = non_control(split.get("train", []))
        validation = non_control(split.get("val", split.get("validation", [])))
        test = non_control(split.get("test", []))
        overlaps = {
            "train_validation_condition_overlap": len(train & validation),
            "train_test_condition_overlap": len(train & test),
            "validation_test_condition_overlap": len(validation & test),
        }
        if any(overlaps.values()):
            raise AssertionError(f"condition overlap in {row.run_id}: {overlaps}")
        if details["prediction_space_output"] != "delta":
            raise AssertionError(f"non-delta output in {row.run_id}")
        panel_path = Path(str(details["gene_panel_path"]))
        panel_hash = str(details["gene_panel_sha256"])
        panel_hashes.setdefault(str(row.dataset_scope), panel_hash)
        if panel_hashes[str(row.dataset_scope)] != panel_hash:
            raise AssertionError(f"dataset gene panel changed across seeds: {row.dataset_scope}")
        with np.load(prediction_path) as arrays:
            uncertainty = np.asarray(arrays["native_uncertainty_gears_mean_logvar"])
            finite_count = int(np.isfinite(uncertainty).sum())
            total_count = int(uncertainty.size)
            required_fields = {
                "native_confidence_gears_exp_neg_mean_logvar",
                "native_uncertainty_gears_mean_logvar",
                "y_true",
                "y_pred",
            }
            if not required_fields.issubset(arrays.files):
                raise AssertionError(f"missing output fields in {row.run_id}: {arrays.files}")
        patch = details["compatibility_patch"]
        rows.append(
            {
                "run_id": str(row.run_id),
                "dataset_id": str(row.dataset_id),
                "dataset_scope": str(row.dataset_scope),
                "seed": int(row.seed),
                "split_path": str(split_path),
                "split_protocol": split_protocol,
                "perturbation_overlap": perturbation_overlap,
                "train_condition_count": len(train),
                "validation_condition_count": len(validation),
                "test_condition_count": len(test),
                **overlaps,
                "prediction_space_input": details["prediction_space_from_gears"],
                "prediction_space_output": details["prediction_space_output"],
                "official_gears_commit": details["official_commit"],
                "compatibility_patch_id": patch["patch_id"],
                "compatibility_patch_hash": patch["patch_hash"],
                "gene_panel_path": str(panel_path),
                "gene_panel_sha256": panel_hash,
                "gene_count": int(run["gene_count"]),
                "n_test_noncontrol": int(metrics["n_test_noncontrol"]),
                "native_uncertainty_finite_count": finite_count,
                "native_uncertainty_total_count": total_count,
                "mse_non_control_test": float(metrics["mse_non_control_test"]),
                "mae_non_control_test": float(metrics["mae_non_control_test"]),
                "mean_cosine_similarity_non_control_test": float(
                    metrics["mean_cosine_similarity_non_control_test"]
                ),
                "status": "success",
            }
        )
    audit = pd.DataFrame(rows).sort_values(["dataset_scope", "seed"]).reset_index(drop=True)
    if len(audit) != len(matrix):
        raise AssertionError(f"audit row count {len(audit)} != run matrix row count {len(matrix)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    print(f"Wrote {len(audit)} GEARS audit rows to {output}.")


if __name__ == "__main__":
    main()
