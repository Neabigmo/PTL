"""Run source-only perturbation heterogeneity analysis after depth lock."""

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

from src.evaluation.reordering_heterogeneity import fit_source_only_models  # noqa: E402


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
    items_path = manifests / "reliability_transport_measurement_depth_items.csv"
    feature_path = root / "artifacts/source_data/formal_v2_reliability_predictions.csv"
    if not items_path.is_file() or not feature_path.is_file():
        report = {"schema_version": 1, "status": "blocked_missing_input", "items": items_path.as_posix(), "features": feature_path.as_posix()}
        (manifests / "reliability_transport_heterogeneity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    items = pd.read_csv(items_path)
    features = pd.read_csv(feature_path)
    source_features = features.loc[
        features["scenario"].eq("in_domain") & features["predictor"].eq("mean_matching"),
        ["environment_key", "perturbation_label", "prediction_norm", "prediction_sparsity", "prediction_concentration", "prediction_manifold_distance", "exact_training_support_fraction", "component_training_support_fraction", "training_neighborhood_density", "perturbation_seen_fraction", "component_seen_fraction", "combination_novelty", "perturbation_novelty", "uq_mean_gene_variance", "uq_median_gene_variance", "uq_top_effect_variance", "uq_cosine_disagreement", "uq_effect_norm_variance"],
    ].drop_duplicates(["environment_key", "perturbation_label"])
    derived = _prediction_features(root)
    merged = items.merge(
        source_features,
        left_on=["source_environment_id", "perturbation_label"],
        right_on=["environment_key", "perturbation_label"],
        how="left",
    )
    merged = merged.merge(derived, left_on=["source_environment_id", "perturbation_label"], right_on=["environment_key", "perturbation_label"], how="left", suffixes=("", "_derived_key"))
    for base in ("prediction_norm", "prediction_sparsity", "prediction_concentration", "uq_mean_gene_variance", "uq_cosine_disagreement", "uq_effect_norm_variance"):
        derived_name = f"{base}_derived"
        if derived_name in merged:
            merged[base] = merged[base].combine_first(merged[derived_name]) if base in merged else merged[derived_name]
    result, coefficients = fit_source_only_models(merged)
    result_path = manifests / "reliability_transport_heterogeneity.csv"
    coef_path = manifests / "reliability_transport_heterogeneity_coefficients.csv"
    result.to_csv(result_path, index=False)
    coefficients.to_csv(coef_path, index=False)
    report = {
        "schema_version": 1,
        "status": "executed",
        "source_only_feature_policy": "features are frozen from the source context; target outcomes are never model inputs",
        "models": ["source_u_only", "ridge_source_features"],
        "crossfit": "leave-one-perturbation-label-out",
        "outputs": {"summary": result_path.relative_to(root).as_posix(), "coefficients": coef_path.relative_to(root).as_posix()},
        "rows": int(len(result)),
        "coefficient_rows": int(len(coefficients)),
    }
    (manifests / "reliability_transport_heterogeneity.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
