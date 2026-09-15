"""Materialize per-perturbation failure anatomy from frozen canonical outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")
ITEM_COLUMNS = [
    "source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric",
    "perturbation_label", "cell_budget", "cell_budget_label", "identifiable_divergence", "cross_disagreement", "within_disagreement_left", "within_disagreement_right",
]
FEATURE_COLUMNS = ["prediction_norm", "prediction_sparsity", "prediction_concentration", "uq_mean_gene_variance", "uq_cosine_disagreement", "uq_effect_norm_variance"]


def _summary(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"n": 0, "median": None, "q25": None, "q75": None, "mean": None, "std": None}
    return {
        "n": int(numeric.size),
        "median": float(numeric.median()),
        "q25": float(numeric.quantile(0.25)),
        "q75": float(numeric.quantile(0.75)),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=1)) if numeric.size > 1 else 0.0,
    }


def _coverage_report(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    group_columns = ["source_environment_id", "target_environment_id", "metric"]
    coverage = table.groupby(group_columns, as_index=False, sort=True).agg(
        total=("status", "size"),
        complete=("status", lambda values: int(values.eq("executed").sum())),
    )
    coverage["missing"] = coverage["total"] - coverage["complete"]
    coverage["coverage_fraction"] = coverage["complete"] / coverage["total"]
    fields = ["reordering_burden", "model_disagreement", "source_uncertainty", "prediction_norm", "prediction_sparsity", "prediction_concentration"]
    distributions: dict[str, Any] = {}
    complete_mask = table["status"].eq("executed")
    missing_mask = ~complete_mask
    for field in fields:
        complete = _summary(table.loc[complete_mask, field])
        missing = _summary(table.loc[missing_mask, field])
        pooled = np.sqrt(((max(complete["n"] - 1, 0) * (complete["std"] or 0.0) ** 2) + (max(missing["n"] - 1, 0) * (missing["std"] or 0.0) ** 2)) / max(complete["n"] + missing["n"] - 2, 1))
        difference = None
        if complete["mean"] is not None and missing["mean"] is not None and pooled > 0:
            difference = float((complete["mean"] - missing["mean"]) / pooled)
        distributions[field] = {
            "complete": complete,
            "missing": missing,
            "standardized_mean_difference_complete_minus_missing": difference,
        }
    return coverage, {
        "schema_version": 1,
        "status": "executed",
        "no_imputation": True,
        "analysis_unit": "directed source-target × metric at full matched-fixed depth",
        "coverage_by_source_target_metric": coverage.to_dict(orient="records"),
        "complete_vs_missing_distribution": distributions,
        "interpretation": "Coverage is descriptive. Complete rows are restricted to matched legacy response-displacement/effect fields; missing rows are not imputed.",
        "outputs": {
            "table": "artifacts/manifests/reliability_transport_failure_anatomy_coverage.csv",
            "report": "artifacts/manifests/reliability_transport_failure_anatomy_coverage.json",
        },
    }


def _pair_key(left: str, right: str) -> str:
    return "|".join(sorted((str(left), str(right))))


def _aggregate_items(path: Path) -> pd.DataFrame:
    sums: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    counts: dict[tuple[str, str, str, str, str], int] = {}
    for chunk in pd.read_csv(path, usecols=ITEM_COLUMNS, chunksize=200_000):
        chunk = chunk.loc[chunk["metric"].isin(METRICS)].copy()
        chunk = chunk.loc[
            chunk["source_environment_id"].eq(chunk["left_target_environment_id"])
            | chunk["source_environment_id"].eq(chunk["right_target_environment_id"])
        ].copy()
        # The matched-fixed table contains one perturbation burden per seed.
        # Keep only the full-depth endpoint and average the already materialized
        # per-seed burden; do not reconstruct a new estimand here.
        # ``cell_budget`` is used because some old CSV readers coerce "full".
        full = chunk["identifiable_divergence"].notna()
        if "cell_budget_label" in chunk.columns:
            full &= chunk["cell_budget_label"].astype(str).eq("full")
        elif "cell_budget" in chunk.columns:
            full &= chunk["cell_budget"].astype(float).ge(1e8)
        chunk = chunk.loc[full].copy()
        chunk["target_environment_id"] = np.where(
            chunk["source_environment_id"].eq(chunk["left_target_environment_id"]),
            chunk["right_target_environment_id"],
            chunk["left_target_environment_id"],
        )
        key_columns = ["source_environment_id", "target_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "perturbation_label"]
        for key, group in chunk.groupby(key_columns, sort=False, observed=True):
            values = group[["identifiable_divergence", "cross_disagreement", "within_disagreement_left", "within_disagreement_right"]].to_numpy(dtype=float)
            finite = np.isfinite(values).all(axis=1)
            if not finite.any():
                continue
            key = tuple(map(str, key))
            sums[key] = sums.get(key, np.zeros(4, dtype=float)) + np.nansum(values[finite], axis=0)
            counts[key] = counts.get(key, 0) + int(finite.sum())
    rows = []
    for key, value in sums.items():
        mean = value / float(counts[key])
        source, target, left, right, metric, label = key
        rows.append({
            "source_environment_id": source,
            "target_environment_id": target,
            "left_target_environment_id": left,
            "right_target_environment_id": right,
            "metric": metric,
            "perturbation_label": label,
            "cell_budget_label": "full",
            "reordering_burden": float(mean[0]),
            "cross_burden": float(mean[1]),
            "within_burden_left": float(mean[2]),
            "within_burden_right": float(mean[3]),
            "n_measurement_seeds": int(counts[key]),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("no full-depth endpoint perturbation burdens were materialized")
    if result.duplicated(["source_environment_id", "target_environment_id", "metric", "perturbation_label"]).any():
        raise ValueError("failure anatomy burden keys are not unique")
    return result


def _source_features(root: Path) -> pd.DataFrame:
    path = root / "artifacts/source_data/formal_v2_reliability_predictions.csv"
    frame = pd.read_csv(path, usecols=["environment_key", "perturbation_label", "predictor", "scenario", *FEATURE_COLUMNS])
    frame = frame.loc[frame["scenario"].eq("in_domain") & frame["predictor"].eq("mean_matching")].copy()
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.groupby(["environment_key", "perturbation_label"], as_index=False, sort=True)[FEATURE_COLUMNS].mean().rename(columns={"environment_key": "source_environment_id"})


def _response_features(root: Path) -> pd.DataFrame:
    oof_path = root / "artifacts/manifests/formal_v2_controlled_shift_oof_perturbations.csv"
    magnitude_path = root / "artifacts/manifests/formal_v2_reliability_reordering_perturbations.csv"
    oof = pd.read_csv(oof_path)
    oof["pair_key"] = [_pair_key(a, b) for a, b in zip(oof["left_environment_id"], oof["right_environment_id"])]
    response = oof.groupby(["pair_key", "perturbation_label"], as_index=False, sort=True).agg(
        response_displacement=("response_program_shift", "mean"),
        normalized_risk_rank_displacement=("normalized_risk_rank_displacement", "mean"),
    )
    magnitude = pd.read_csv(magnitude_path)
    magnitude = magnitude.loc[magnitude["predictor"].eq("mean_matching") & magnitude["method"].eq("ptl_rf")].copy()
    magnitude["pair_key"] = [_pair_key(a, b) for a, b in zip(magnitude["left_environment_id"], magnitude["right_environment_id"])]
    magnitude = magnitude.groupby(["pair_key", "perturbation_label"], as_index=False, sort=True).agg(effect_magnitude=("response_magnitude_change", "mean"))
    return response.merge(magnitude, on=["pair_key", "perturbation_label"], how="outer", validate="one_to_one")


def build(root: Path = ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    items_path = manifests / "reliability_transport_measurement_depth_matched_fixed_items.csv"
    burden = _aggregate_items(items_path)
    burden["pair_key"] = [_pair_key(a, b) for a, b in zip(burden["left_target_environment_id"], burden["right_target_environment_id"])]
    response = _response_features(root)
    features = _source_features(root)
    result = burden.merge(response, on=["pair_key", "perturbation_label"], how="left", validate="many_to_one")
    result = result.merge(features, on=["source_environment_id", "perturbation_label"], how="left", validate="many_to_one")
    result = result.rename(columns={"uq_cosine_disagreement": "model_disagreement", "uq_mean_gene_variance": "source_uncertainty"})
    result["target_informed_explanatory"] = 1
    result["source_only_features"] = "uq_cosine_disagreement;uq_mean_gene_variance;uq_effect_norm_variance;prediction_norm;prediction_sparsity;prediction_concentration"
    result["response_displacement_definition"] = "1 minus cosine similarity between matched perturbation response vectors"
    result["effect_magnitude_definition"] = "absolute normalized change in matched response-vector norm"
    result["model_disagreement_definition"] = "source-side uq_cosine_disagreement among frozen model members"
    result["source_uncertainty_definition"] = "source-side uq_mean_gene_variance among frozen model members"
    result["provenance"] = json.dumps(
        {
            "burden": items_path.relative_to(root).as_posix(),
            "response_displacement": "artifacts/manifests/formal_v2_controlled_shift_oof_perturbations.csv",
            "effect_magnitude": "artifacts/manifests/formal_v2_reliability_reordering_perturbations.csv",
            "source_features": "artifacts/source_data/formal_v2_reliability_predictions.csv",
            "roles": "response/effect are target-informed explanatory quantities; source features are prospective candidates and never include target outcomes",
        },
        sort_keys=True,
    )
    required = ["reordering_burden", "response_displacement", "effect_magnitude", "model_disagreement", "source_uncertainty"]
    result["status"] = np.where(result[required].notna().all(axis=1), "executed", "explicit_missing_explanatory_field")
    result = result.drop(columns=["pair_key"])
    output_columns = [
        "source_environment_id", "target_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "perturbation_label", "cell_budget_label",
        "reordering_burden", "cross_burden", "within_burden_left", "within_burden_right", "response_displacement", "effect_magnitude", "normalized_risk_rank_displacement",
        "model_disagreement", "source_uncertainty", *FEATURE_COLUMNS[:3], "uq_effect_norm_variance", "target_informed_explanatory", "source_only_features",
        "response_displacement_definition", "effect_magnitude_definition", "model_disagreement_definition", "source_uncertainty_definition", "n_measurement_seeds", "provenance", "status",
    ]
    result = result[output_columns].sort_values(["source_environment_id", "target_environment_id", "metric", "perturbation_label"], kind="stable").reset_index(drop=True)
    report = {
        "schema_version": 1,
        "status": "executed",
        "rows": int(len(result)),
        "executed_rows": int(result["status"].eq("executed").sum()),
        "missing_explanatory_rows": int(result["status"].ne("executed").sum()),
        "analysis_unit": "directed source-target × metric × perturbation label at full matched-fixed depth",
        "burden_identity": "existing per-perturbation identifiable burden from the matched-fixed canonical table; no new transport score",
        "response_displacement": "existing response_program_shift = 1 minus cosine similarity; descriptive and target-informed",
        "effect_magnitude": "existing response_magnitude_change; descriptive and target-informed",
        "prospective_feature_boundary": "uq_cosine_disagreement, uq_mean_gene_variance, uq_effect_norm_variance, prediction_norm, prediction_sparsity, prediction_concentration; target outcomes excluded",
        "outputs": {"table": "artifacts/manifests/reliability_transport_failure_anatomy.csv", "report": "artifacts/manifests/reliability_transport_failure_anatomy.json"},
    }
    return result, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifests = root / "artifacts/manifests"
    table, report = build(root)
    table_path = manifests / "reliability_transport_failure_anatomy.csv"
    report_path = manifests / "reliability_transport_failure_anatomy.json"
    table.to_csv(table_path, index=False)
    coverage_table, coverage_report = _coverage_report(table)
    coverage_table.to_csv(manifests / "reliability_transport_failure_anatomy_coverage.csv", index=False)
    (manifests / "reliability_transport_failure_anatomy_coverage.json").write_text(
        json.dumps(coverage_report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8"
    )
    report["coverage"] = {
        "total_rows": int(len(table)),
        "complete_rows": int(table["status"].eq("executed").sum()),
        "missing_rows": int(table["status"].ne("executed").sum()),
        "coverage_fraction": float(table["status"].eq("executed").mean()),
        "no_imputation": True,
        "report": "artifacts/manifests/reliability_transport_failure_anatomy_coverage.json",
        "table": "artifacts/manifests/reliability_transport_failure_anatomy_coverage.csv",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
