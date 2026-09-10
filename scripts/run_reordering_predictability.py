"""Run prospective source-only predictability for the depth-locked burden."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.reordering_predictability import evaluate_source_only_predictability  # noqa: E402


def _prediction_features(root: Path) -> pd.DataFrame:
    payload = np.load(root / "artifacts/source_data/frangieh_source_frozen_predictions.npz", allow_pickle=False)
    labels = payload["perturbation_label"].astype(str).tolist()
    environments = payload["source_environment"].astype(str).tolist()
    prediction = payload["prediction"].astype(np.float64)
    mean = prediction.mean(axis=2)
    norm = np.sqrt(np.sum(mean * mean, axis=2))
    abs_mean = np.abs(mean)
    concentration = abs_mean.max(axis=2) / np.maximum(abs_mean.sum(axis=2), 1e-12)
    sparsity = np.mean(abs_mean <= 1e-8, axis=2)
    member_norm = np.sqrt(np.sum(prediction * prediction, axis=3))
    uq_norm = np.var(member_norm, axis=2)
    uq_gene = np.var(prediction, axis=2).mean(axis=2)
    denominator = np.maximum(np.linalg.norm(prediction, axis=3) * np.linalg.norm(mean, axis=2)[:, :, None], 1e-12)
    cosine = np.sum(prediction * mean[:, :, None, :], axis=3) / denominator
    uq_cosine = np.mean(1.0 - cosine, axis=2)
    return pd.DataFrame({
        "environment_key": np.repeat(environments, len(labels)),
        "perturbation_label": labels * len(environments),
        "prediction_norm_derived": norm.reshape(-1),
        "prediction_sparsity_derived": sparsity.reshape(-1),
        "prediction_concentration_derived": concentration.reshape(-1),
        "uq_mean_gene_variance_derived": uq_gene.reshape(-1),
        "uq_cosine_disagreement_derived": uq_cosine.reshape(-1),
        "uq_effect_norm_variance_derived": uq_norm.reshape(-1),
    })


def run(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    matched_items_path = manifests / "reliability_transport_measurement_depth_matched_fixed_items.csv"
    items_path = matched_items_path if matched_items_path.is_file() else manifests / "reliability_transport_measurement_depth_items.csv"
    features_path = root / "artifacts/source_data/formal_v2_reliability_predictions.csv"
    out_path = manifests / "reliability_transport_predictability.csv"
    report_path = manifests / "reliability_transport_predictability.json"
    if not items_path.is_file() or not features_path.is_file():
        report = {"schema_version": 1, "status": "blocked_missing_input", "items": items_path.as_posix(), "features": features_path.as_posix()}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    items = pd.read_csv(items_path, dtype={"cell_budget_label": "string"})
    features = pd.read_csv(features_path)
    selected_columns = [
        "environment_key", "perturbation_label", "prediction_norm", "prediction_sparsity", "prediction_concentration", "prediction_manifold_distance",
        "exact_training_support_fraction", "component_training_support_fraction", "training_neighborhood_density", "perturbation_seen_fraction", "component_seen_fraction", "combination_novelty", "perturbation_novelty",
        "uq_mean_gene_variance", "uq_median_gene_variance", "uq_top_effect_variance", "uq_cosine_disagreement", "uq_effect_norm_variance",
    ]
    features = features.loc[features["scenario"].eq("in_domain") & features["predictor"].eq("mean_matching"), selected_columns].drop_duplicates(["environment_key", "perturbation_label"])
    derived = _prediction_features(root)
    merged = items.merge(features, left_on=["source_environment_id", "perturbation_label"], right_on=["environment_key", "perturbation_label"], how="left")
    merged = merged.merge(derived, left_on=["source_environment_id", "perturbation_label"], right_on=["environment_key", "perturbation_label"], how="left", suffixes=("", "_derived_key"))
    for base in ("prediction_norm", "prediction_sparsity", "prediction_concentration", "uq_mean_gene_variance", "uq_cosine_disagreement", "uq_effect_norm_variance"):
        derived_name = f"{base}_derived"
        if derived_name in merged:
            merged[base] = merged[base].combine_first(merged[derived_name]) if base in merged else merged[derived_name]
    result = evaluate_source_only_predictability(merged)
    result["dataset"] = "FrangiehIzar2021_RNA"
    result["dataset_split"] = "within_dataset_leave_one_perturbation_label_out"
    result["feature_set"] = "fixed_source_only_v2"
    # A cross-dataset split is part of the prespecified validation surface.
    # Nadig has aggregate Claim-Lock outcomes but no canonical per-label
    # identifiable-divergence table, so a numerical score would be invented.
    # Preserve the requested direction as an explicit blocked audit row.
    blocked = []
    for source_dataset, target_dataset in (("FrangiehIzar2021_RNA", "NadigOConner2024"), ("NadigOConner2024", "FrangiehIzar2021_RNA")):
        for model in ("source_u_only", "linear_source_features"):
            blocked.append({
                "dataset": f"{source_dataset}__to__{target_dataset}",
                "dataset_split": "leave_dataset_out",
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "model": model,
                "status": "blocked_missing_shared_per_label_target_outcomes",
                "n_labels": 0,
                "spearman": float("nan"),
                "auroc_high_identifiable": float("nan"),
                "calibration_brier": float("nan"),
                "calibration_threshold": float("nan"),
                "calibration_status": "not_reported_no_target_labels",
                "mean_absolute_error": float("nan"),
                "source_only_features": "fixed_source_only_v2",
                "target_outcome_used_only_for_evaluation": True,
                "feature_set": "fixed_source_only_v2",
                "reason": "Nadig canonical output is aggregate-only; no shared per-label identifiable-divergence outcome exists",
            })
    result = pd.concat([result, pd.DataFrame(blocked)], ignore_index=True, sort=False)
    result.to_csv(out_path, index=False)
    report = {
        "schema_version": 2,
        "status": "executed",
        "outer_split": "leave-one-perturbation-label-out",
        "models": ["source_u_only", "linear_source_features"],
        "target_outcome_blindness": "all target risk/burden values are held out evaluation targets and never feature columns",
        "feature_set": "fixed_source_only_v2",
        "within_dataset": "Frangieh executed with source-only features; target outcome held out per label",
        "cross_dataset": {"directions": ["FrangiehIzar2021_RNA->NadigOConner2024", "NadigOConner2024->FrangiehIzar2021_RNA"], "status": "blocked_missing_shared_per_label_target_outcomes", "no_imputation": True},
        "calibration_policy": "Brier and thresholds are not reported; AUROC uses training-fold median and train-fold score scaling only",
        "output": out_path.relative_to(root).as_posix(),
        "rows": int(len(result)),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
