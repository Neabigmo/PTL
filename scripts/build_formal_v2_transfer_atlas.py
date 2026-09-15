"""Build the formal source-to-target reliability transport atlas.

Only off-diagonal pairs are materialized. For every target, the primary gain
is measured against a pooled U-only RF trained on validation rows from all
other environments, which is the model available before target outcomes exist.
The source PTL expert is fitted on the individual source validation rows and
evaluated on the untouched target test rows. Raw normalized UQ remains a
secondary diagnostic baseline.
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
    UQ_COLUMNS,
    _fit_ptl,
    _prediction_rows,
    _selective_metrics,
    load_registry,
    scenario_partition,
)


BASELINES = {
    "u_only_rf": "u_only_rf",
    "raw_scalar_uq": "raw_normalized_uq",
}
PRIMARY_SEED = 20260907


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
    quantiles = np.linspace(0.05, 0.95, 19)
    for column, column_scale in zip(UQ_COLUMNS, scale):
        source_values = pd.to_numeric(source[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        target_values = pd.to_numeric(target[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        distances.append(np.mean(np.abs(np.quantile(source_values, quantiles) - np.quantile(target_values, quantiles))) / column_scale)
    return float(np.mean(distances)) if distances else 0.0


def _pair_descriptors(
    source_registry: pd.Series,
    target_registry: pd.Series,
    source: pd.DataFrame,
    target: pd.DataFrame,
    frame: pd.DataFrame,
) -> dict[str, Any]:
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


def build(
    root: Path,
    *,
    split_seed: int = PRIMARY_SEED,
    manifest_path: Path | None = None,
    predictor_dir: Path | None = None,
) -> dict[str, Any]:
    frame = _prediction_rows(root, manifest_path=manifest_path, predictor_dir=predictor_dir, split_seed=split_seed)
    registry = load_registry(root).set_index("environment_id", drop=False)
    environment_ids = registry.index.astype(str).tolist()
    if len(environment_ids) < 2:
        raise ValueError("formal transport atlas requires at least two canonical environments")

    rows: list[dict[str, Any]] = []
    for target_id in environment_ids:
        target_rows = frame.loc[frame["environment_id"].eq(target_id)]
        _, target_test_index = scenario_partition(target_rows, "in_domain", "")
        target_test = target_rows.loc[target_test_index].copy()
        pooled_train = frame.loc[frame["split"].eq("validation") & frame["environment_id"].ne(target_id)].copy()
        if pooled_train.empty:
            raise ValueError(f"target-excluded U-only calibration is empty for {target_id}")
        pooled_scores, pooled_threshold, pooled_model, _, _ = _fit_ptl(pooled_train, target_test)
        target_labels = (target_test["fidelity"].to_numpy(dtype=float) >= pooled_threshold).astype(int)
        target_risk = target_test["continuous_risk"].to_numpy(dtype=float)
        target_ids = set(target_test["biological_instance_id"].astype(str))

        for source_id in environment_ids:
            if source_id == target_id:
                continue
            source_rows = frame.loc[frame["environment_id"].eq(source_id)]
            source_train_index, _ = scenario_partition(source_rows, "in_domain", "")
            source_train = source_rows.loc[source_train_index].copy()
            if set(source_train["biological_instance_id"].astype(str)).intersection(target_ids):
                raise AssertionError(f"source/target biological instances overlap: {source_id} -> {target_id}")
            source_scores, source_threshold, source_model, _, _ = _fit_ptl(source_train, target_test)
            descriptors = _pair_descriptors(
                registry.loc[source_id], registry.loc[target_id], source_train, target_test, frame
            )
            for baseline_name, baseline_method in BASELINES.items():
                baseline_scores = pooled_scores[baseline_method]
                ptl_scores = source_scores["ptl_rf"]
                baseline_metrics = _selective_metrics(target_risk, target_labels, baseline_scores)
                ptl_metrics = _selective_metrics(target_risk, target_labels, ptl_scores)
                rows.append({
                    "split_seed": int(split_seed),
                    "source_environment_id": source_id,
                    "source_environment_key": str(registry.loc[source_id, "environment_key"]),
                    "target_environment_id": target_id,
                    "target_environment_key": str(registry.loc[target_id, "environment_key"]),
                    "baseline": baseline_name,
                    "baseline_method": baseline_method,
                    "ptl_method": "ptl_rf",
                    "source_validation_rows": int(len(source_train)),
                    "target_excluded_u_only_validation_rows": int(len(pooled_train)),
                    "target_test_rows": int(len(target_test)),
                    "target_excluded_u_only_threshold": float(pooled_threshold),
                    "source_ptl_threshold": float(source_threshold),
                    "target_excluded_u_only_model": pooled_model,
                    "source_ptl_model": source_model,
                    "baseline_aurc": baseline_metrics["aurc"],
                    "ptl_aurc": ptl_metrics["aurc"],
                    "baseline_excess_aurc": baseline_metrics["excess_aurc"],
                    "ptl_excess_aurc": ptl_metrics["excess_aurc"],
                    "transfer_gain": float(baseline_metrics["aurc"] - ptl_metrics["aurc"]),
                    "excess_aurc_gain": float(baseline_metrics["excess_aurc"] - ptl_metrics["excess_aurc"]),
                    "risk_at_80_gain": float(baseline_metrics["risk_at_80"] - ptl_metrics["risk_at_80"]),
                    "ftr_at_80_gain": float(baseline_metrics["ftr_at_80"] - ptl_metrics["ftr_at_80"]),
                    "same_source_target": 0,
                    **descriptors,
                })

    matrix = pd.DataFrame(rows)
    expected_pairs = len(environment_ids) * (len(environment_ids) - 1)
    unique_pairs = matrix[["source_environment_id", "target_environment_id"]].drop_duplicates()
    if matrix.empty or len(unique_pairs) != expected_pairs:
        raise AssertionError(f"off-diagonal transfer atlas has {len(unique_pairs)} pairs; expected {expected_pairs}")
    if (matrix["source_environment_id"] == matrix["target_environment_id"]).any():
        raise AssertionError("transfer atlas contains a self-source pair")

    manifest_dir = root / "artifacts/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if split_seed == PRIMARY_SEED else f"__split_{split_seed}"
    matrix_path = manifest_dir / f"formal_v2_environment_transfer_matrix{suffix}.csv"
    summary_path = manifest_dir / f"formal_v2_environment_transfer_summary{suffix}.csv"
    json_path = manifest_dir / f"formal_v2_environment_transfer_summary{suffix}.json"
    matrix.sort_values(["baseline", "source_environment_key", "target_environment_key"], kind="stable").to_csv(matrix_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    source_summary = matrix.groupby(["baseline", "source_environment_id"], as_index=False).agg(
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
        summary_rows.append({
            "baseline": baseline,
            "source_environment_id": "off_diagonal",
            "target_environment_id": "off_diagonal",
            "scope": "off_diagonal",
            "n_pairs": int(len(group)),
            "mean_transfer_gain": float(group["transfer_gain"].mean()),
            "median_transfer_gain": float(group["transfer_gain"].median()),
            "mean_excess_aurc_gain": float(group["excess_aurc_gain"].mean()),
            "mean_risk_at_80_gain": float(group["risk_at_80_gain"].mean()),
        })
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    payload = {
        "schema_version": 2,
        "benchmark_id": "ptl_context_v2",
        "split_seed": int(split_seed),
        "environment_count": len(environment_ids),
        "matrix_shape": [len(environment_ids), len(environment_ids)],
        "off_diagonal_pair_count": expected_pairs,
        "matrix_rows": int(len(matrix)),
        "baselines": list(BASELINES),
        "source_split": "source environment validation set",
        "target_split": "target environment untouched test set",
        "primary_transfer_gain_definition": "target-excluded pooled U-only AURC - source-trained PTL AURC",
        "secondary_transfer_gain_definition": "target-excluded pooled raw normalized UQ AURC - source-trained PTL AURC",
        "descriptors": [
            "same_cell_context", "same_perturbation_modality", "same_platform",
            "same_condition", "same_dataset_family", "same_context_modality",
            "prediction_geometry_distance", "uq_distribution_distance",
        ],
        "self_source_pairs_included": False,
        "matrix_path": matrix_path.relative_to(root).as_posix(),
        "summary_path": summary_path.relative_to(root).as_posix(),
        "status": "formal_v2_off_diagonal_target_excluded_transfer_atlas_executed",
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--predictor-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    print(json.dumps(build(
        root,
        split_seed=args.split_seed,
        manifest_path=args.manifest.resolve() if args.manifest else None,
        predictor_dir=args.predictor_dir.resolve() if args.predictor_dir else None,
    ), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
