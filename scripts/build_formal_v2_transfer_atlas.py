"""Build the formal-v2 source-to-target reliability transfer atlas.

Each cell trains the identity-free PTL only on one source environment's
validation rows and evaluates it once on one target environment's untouched
test rows.  The source and target are therefore disjoint at the biological
instance level, while the transfer descriptors remain descriptive rather than
causal claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_reliability import (  # noqa: E402
    P_COLUMNS,
    PREDICTORS,
    UQ_COLUMNS,
    _fit_ptl,
    _selective_metrics,
    _prediction_rows,
    load_registry,
    scenario_partition,
)


BASELINES = {
    "raw_scalar_uq": "raw_normalized_uq",
    "u_only_rf": "u_only_rf",
}


def _descriptor_scale(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    scale = np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
    return np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)


def _geometry_distance(source: pd.DataFrame, target: pd.DataFrame, scale: np.ndarray) -> float:
    source_mean = source.loc[:, list(P_COLUMNS)].apply(pd.to_numeric, errors="coerce").mean().to_numpy(dtype=float)
    target_mean = target.loc[:, list(P_COLUMNS)].apply(pd.to_numeric, errors="coerce").mean().to_numpy(dtype=float)
    return float(np.linalg.norm((source_mean - target_mean) / scale))


def _uq_distribution_distance(source: pd.DataFrame, target: pd.DataFrame, scale: np.ndarray) -> float:
    distances = []
    for column, column_scale in zip(UQ_COLUMNS, scale):
        source_values = pd.to_numeric(source[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        target_values = pd.to_numeric(target[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        quantiles = np.linspace(0.05, 0.95, 19)
        distances.append(np.mean(np.abs(np.quantile(source_values, quantiles) - np.quantile(target_values, quantiles))) / column_scale)
    return float(np.mean(distances)) if distances else 0.0


def _pair_descriptors(source_registry: pd.Series, target_registry: pd.Series, source: pd.DataFrame, target: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    geometry_scale = _descriptor_scale(frame, P_COLUMNS)
    uq_scale = _descriptor_scale(frame, UQ_COLUMNS)
    same_cell = str(source_registry["cell_context"]) == str(target_registry["cell_context"])
    same_modality = str(source_registry["perturbation_modality"]) == str(target_registry["perturbation_modality"])
    same_platform = str(source_registry["platform"]) == str(target_registry["platform"])
    same_condition = str(source_registry["condition"]) == str(target_registry["condition"])
    same_dataset = str(source_registry["dataset_id"]) == str(target_registry["dataset_id"])
    return {
        "same_cell_context": int(same_cell),
        "same_perturbation_modality": int(same_modality),
        "same_platform": int(same_platform),
        "same_condition": int(same_condition),
        "same_dataset_family": int(same_dataset),
        "same_context_modality": int(same_cell and same_modality),
        "prediction_geometry_distance": _geometry_distance(source, target, geometry_scale),
        "uq_distribution_distance": _uq_distribution_distance(source, target, uq_scale),
    }


def build(root: Path) -> dict[str, Any]:
    frame = _prediction_rows(root)
    registry = load_registry(root).set_index("environment_id", drop=False)
    environment_ids = registry.index.astype(str).tolist()
    if len(environment_ids) != 8:
        raise ValueError(f"formal transfer atlas expects 8 canonical environments, found {len(environment_ids)}")

    rows: list[dict[str, Any]] = []
    for source_id in environment_ids:
        source_rows = frame.loc[frame["environment_id"].eq(source_id)]
        source_train_index, _ = scenario_partition(source_rows, "in_domain", "")
        source_train = source_rows.loc[source_train_index].copy()
        for target_id in environment_ids:
            target_rows = frame.loc[frame["environment_id"].eq(target_id)]
            _, target_test_index = scenario_partition(target_rows, "in_domain", "")
            target_test = target_rows.loc[target_test_index].copy()
            if set(source_train["biological_instance_id"].astype(str)).intersection(target_test["biological_instance_id"].astype(str)):
                raise AssertionError(f"source/target biological instances overlap: {source_id} -> {target_id}")
            scores, threshold, model_name, _, _ = _fit_ptl(source_train, target_test)
            labels = (target_test["fidelity"].to_numpy(dtype=float) >= threshold).astype(int)
            risk = target_test["continuous_risk"].to_numpy(dtype=float)
            descriptors = _pair_descriptors(registry.loc[source_id], registry.loc[target_id], source_train, target_test, frame)
            for baseline_name, baseline_method in BASELINES.items():
                baseline_metrics = _selective_metrics(risk, labels, scores[baseline_method])
                ptl_metrics = _selective_metrics(risk, labels, scores["ptl_rf"])
                rows.append({
                    "source_environment_id": source_id,
                    "source_environment_key": str(registry.loc[source_id, "environment_key"]),
                    "target_environment_id": target_id,
                    "target_environment_key": str(registry.loc[target_id, "environment_key"]),
                    "baseline": baseline_name,
                    "baseline_method": baseline_method,
                    "ptl_method": "ptl_rf",
                    "source_validation_rows": int(len(source_train)),
                    "target_test_rows": int(len(target_test)),
                    "calibration_threshold": float(threshold),
                    "ptl_model": model_name,
                    "baseline_aurc": baseline_metrics["aurc"],
                    "ptl_aurc": ptl_metrics["aurc"],
                    "baseline_excess_aurc": baseline_metrics["excess_aurc"],
                    "ptl_excess_aurc": ptl_metrics["excess_aurc"],
                    "transfer_gain": float(baseline_metrics["aurc"] - ptl_metrics["aurc"]),
                    "excess_aurc_gain": float(baseline_metrics["excess_aurc"] - ptl_metrics["excess_aurc"]),
                    "risk_at_80_gain": float(baseline_metrics["risk_at_80"] - ptl_metrics["risk_at_80"]),
                    "ftr_at_80_gain": float(baseline_metrics["ftr_at_80"] - ptl_metrics["ftr_at_80"]),
                    "same_source_target": int(source_id == target_id),
                    **descriptors,
                })

    matrix = pd.DataFrame(rows)
    if matrix.empty or matrix[["source_environment_id", "target_environment_id"]].drop_duplicates().shape[0] != 64:
        raise AssertionError("transfer atlas does not contain exactly 8x8 source-target pairs")
    manifest_dir = root / "artifacts/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = manifest_dir / "formal_v2_environment_transfer_matrix.csv"
    summary_path = manifest_dir / "formal_v2_environment_transfer_summary.csv"
    json_path = manifest_dir / "formal_v2_environment_transfer_summary.json"
    matrix.sort_values(["baseline", "source_environment_key", "target_environment_key"], kind="stable").to_csv(matrix_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    group_fields = ["baseline", "source_environment_id"]
    source_summary = matrix.groupby(group_fields, as_index=False).agg(
        scope=("source_environment_id", lambda values: "source_row"),
        n_pairs=("transfer_gain", "size"),
        mean_transfer_gain=("transfer_gain", "mean"),
        median_transfer_gain=("transfer_gain", "median"),
        mean_excess_aurc_gain=("excess_aurc_gain", "mean"),
        mean_risk_at_80_gain=("risk_at_80_gain", "mean"),
    )
    for row in source_summary.to_dict("records"):
        row["target_environment_id"] = "all"
        summary_rows.append(row)
    target_summary = matrix.groupby(["baseline", "target_environment_id"], as_index=False).agg(
        scope=("target_environment_id", lambda values: "target_column"),
        n_pairs=("transfer_gain", "size"),
        mean_transfer_gain=("transfer_gain", "mean"),
        median_transfer_gain=("transfer_gain", "median"),
        mean_excess_aurc_gain=("excess_aurc_gain", "mean"),
        mean_risk_at_80_gain=("risk_at_80_gain", "mean"),
    )
    for row in target_summary.to_dict("records"):
        row["source_environment_id"] = "all"
        summary_rows.append(row)
    for baseline, group in matrix.groupby("baseline", sort=True):
        for diagonal, subset in group.groupby("same_source_target", sort=True):
            summary_rows.append({
                "baseline": baseline,
                "source_environment_id": "diagonal" if diagonal else "off_diagonal",
                "target_environment_id": "diagonal" if diagonal else "off_diagonal",
                "scope": "diagonal_vs_off_diagonal",
                "n_pairs": int(len(subset)),
                "mean_transfer_gain": float(subset["transfer_gain"].mean()),
                "median_transfer_gain": float(subset["transfer_gain"].median()),
                "mean_excess_aurc_gain": float(subset["excess_aurc_gain"].mean()),
                "mean_risk_at_80_gain": float(subset["risk_at_80_gain"].mean()),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "matrix_shape": [8, 8],
        "matrix_rows": int(len(matrix)),
        "baselines": list(BASELINES),
        "source_split": "one source environment validation set",
        "target_split": "one target environment untouched test set",
        "transfer_gain_definition": "baseline AURC - source-trained PTL AURC; positive is useful transfer",
        "descriptors": [
            "same_cell_context", "same_perturbation_modality", "same_platform",
            "same_condition", "same_dataset_family", "prediction_geometry_distance",
            "uq_distribution_distance",
        ],
        "matrix_path": matrix_path.relative_to(root).as_posix(),
        "summary_path": summary_path.relative_to(root).as_posix(),
        "status": "formal_v2_environment_transfer_atlas_executed",
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve()), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
