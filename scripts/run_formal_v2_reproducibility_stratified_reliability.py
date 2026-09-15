"""Link measured split-half reproducibility to selective reliability.

The reproducibility score is joined only for evaluation and stratification; it
is never placed in the deployment feature matrix or used to fit PTL.
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

from scripts.run_formal_v2_reliability import _aurc_only  # noqa: E402
from scripts.run_formal_v2_grouping_loss import _summarize_conditional_grouped, assign_bins  # noqa: E402
from scripts.run_formal_v2_reliability_shift import (  # noqa: E402
    _finite_correlation,
    _normalized_aurc,
    crossfit_additive_and_interaction,
)

METHODS = ("raw_normalized_uq", "u_only_rf", "ptl_rf")
RAW_METHOD = "raw_normalized_uq"


def _stratum(values: pd.Series) -> pd.Series:
    lower = float(values.quantile(1 / 3))
    upper = float(values.quantile(2 / 3))
    return pd.Series(
        np.where(values <= lower, "low", np.where(values >= upper, "high", "middle")),
        index=values.index,
    )


def _metrics(group: pd.DataFrame, method: str) -> dict[str, float]:
    risk = group["continuous_risk"].to_numpy(dtype=float)
    confidence = group[method].to_numpy(dtype=float)
    aurc = _aurc_only(risk, confidence)
    return {"n": int(len(group)), "aurc": float(aurc), "mean_risk": float(risk.mean())}


def _component_metrics(group: pd.DataFrame, method: str) -> dict[str, float]:
    """Return difficulty and within-stratum ranking metrics for one stratum."""

    risk = pd.to_numeric(group["continuous_risk"], errors="coerce").to_numpy(dtype=float)
    confidence = pd.to_numeric(group[method], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(risk) & np.isfinite(confidence)
    risk = risk[mask]
    confidence = confidence[mask]
    return {
        "n": int(len(risk)),
        "difficulty_mean_risk": float(np.mean(risk)) if len(risk) else float("nan"),
        "difficulty_risk_variance": float(np.var(risk, ddof=1)) if len(risk) > 1 else float("nan"),
        "ranking_spearman_confidence_vs_negative_risk": _finite_correlation(confidence, -risk, "spearman"),
        "ranking_kendall_confidence_vs_negative_risk": _finite_correlation(confidence, -risk, "kendall"),
        "ranking_normalized_aurc": _normalized_aurc(risk, confidence),
    }


def _reproducibility_grouping_metrics(group: pd.DataFrame) -> dict[str, float]:
    """Apply the corrected fixed-bin grouping summary with stratum as group."""

    work = group[["continuous_risk", "reproducibility_stratum", "confidence_bin"]].copy()
    work = work.rename(columns={"reproducibility_stratum": "environment_id"})
    return _summarize_conditional_grouped(work)


def _semantic_metrics(group: pd.DataFrame) -> dict[str, Any]:
    """Test whether the confidence--risk mapping differs by reproducibility stratum.

    This is an evaluation-only descriptive extension of the Reliability Shift
    decomposition.  It is intentionally marked unavailable when one stratum
    has too few biological instances to support a grouped cross-fit.
    """

    counts = group.groupby("reproducibility_stratum")["biological_instance_id"].nunique()
    if len(counts) < 2 or counts.min() < 2 or group["biological_instance_id"].nunique() < 2:
        return {
            "semantic_status": "insufficient_reproducibility_strata_for_crossfit",
            "semantic_additive_null_mse": float("nan"),
            "semantic_interaction_mse": float("nan"),
            "semantic_interaction_improvement": float("nan"),
        }
    work = group.copy()
    work["environment_id"] = "reproducibility::" + work["reproducibility_stratum"].astype(str)
    oof = crossfit_additive_and_interaction(
        work,
        n_splits=min(5, int(work["biological_instance_id"].nunique())),
        group_column="biological_instance_id",
        random_state=0,
    )
    return {
        "semantic_status": "executed_descriptive_reproducibility_stratum_crossfit",
        "semantic_additive_null_mse": float(oof["additive_squared_error"].mean()),
        "semantic_interaction_mse": float(oof["interaction_squared_error"].mean()),
        "semantic_interaction_improvement": float(oof["interaction_improvement"].mean()),
    }


def _reliability_shift_row(group: pd.DataFrame, dataset_id: str, predictor: str, method: str) -> dict[str, Any]:
    """Summarize low/high reproducibility components in one auditable row."""

    row: dict[str, Any] = {
        "dataset_id": dataset_id,
        "predictor": predictor,
        "method": method,
        "n_total": int(len(group)),
        "strata_present": "|".join(sorted(group["reproducibility_stratum"].astype(str).unique())),
        "calibration_bins_fit_on": "calibration_rows_only",
        "outcomes_used_for_bin_construction": 0,
        "used_in_ptl_features": 0,
        "evaluation_only": 1,
    }
    for stratum in ("low", "middle", "high"):
        subset = group.loc[group["reproducibility_stratum"].eq(stratum)]
        metrics = _component_metrics(subset, method) if len(subset) else {
            "n": 0,
            "difficulty_mean_risk": float("nan"),
            "difficulty_risk_variance": float("nan"),
            "ranking_spearman_confidence_vs_negative_risk": float("nan"),
            "ranking_kendall_confidence_vs_negative_risk": float("nan"),
            "ranking_normalized_aurc": float("nan"),
        }
        for key, value in metrics.items():
            row[f"{stratum}_{key}"] = value

    grouping = _reproducibility_grouping_metrics(group)
    for key, value in grouping.items():
        row[f"all_strata_conditional_{key}"] = value
    row.update(_semantic_metrics(group))
    row["low_high_difficulty_delta"] = row["high_difficulty_mean_risk"] - row["low_difficulty_mean_risk"] if np.isfinite(row["high_difficulty_mean_risk"]) and np.isfinite(row["low_difficulty_mean_risk"]) else float("nan")
    row["low_high_ranking_normalized_aurc_delta"] = row["high_ranking_normalized_aurc"] - row["low_ranking_normalized_aurc"] if np.isfinite(row["high_ranking_normalized_aurc"]) and np.isfinite(row["low_ranking_normalized_aurc"]) else float("nan")
    return row


def run(root: Path, *, input_path: Path | None = None, reproducibility_path: Path | None = None) -> dict[str, Any]:
    predictions = pd.read_csv(input_path or root / "artifacts/source_data/formal_v2_reliability_predictions.csv")
    calibration = pd.read_csv(root / "artifacts/source_data/formal_v2_reliability_calibration_scores.csv")
    reproducibility = pd.read_csv(reproducibility_path or root / "artifacts/manifests/formal_v2_raw_cell_split_half.csv")
    required_repro = {"dataset_id", "perturbation_label", "split_half_cosine"}
    if required_repro.difference(reproducibility.columns):
        raise ValueError("raw-cell reproducibility artifact is missing required join fields")
    frame = predictions.loc[predictions["scenario"].eq("in_domain")].copy()
    joined = frame.merge(
        reproducibility[["dataset_id", "perturbation_label", "split_half_cosine"]],
        on=["dataset_id", "perturbation_label"],
        how="inner",
        validate="many_to_one",
    )
    if joined.empty:
        raise ValueError("no reliability rows joined to measured raw-cell reproducibility")
    joined["reproducibility_stratum"] = _stratum(joined["split_half_cosine"])
    rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for (dataset_id, predictor, stratum), group in joined.groupby(["dataset_id", "predictor", "reproducibility_stratum"], sort=True):
        for method in METHODS:
            values = _metrics(group, method)
            rows.append({
                "dataset_id": dataset_id,
                "predictor": predictor,
                "reproducibility_stratum": stratum,
                "method": method,
                "reproducibility_min": float(group["split_half_cosine"].min()),
                "reproducibility_max": float(group["split_half_cosine"].max()),
                "reproducibility_mean": float(group["split_half_cosine"].mean()),
                **values,
                "used_in_ptl_features": 0,
                "evaluation_only": 1,
            })
    continuous_rows: list[dict[str, Any]] = []
    for (dataset_id, predictor), group in joined.groupby(["dataset_id", "predictor"], sort=True):
        risk = pd.to_numeric(group["continuous_risk"], errors="coerce")
        for method in METHODS:
            score = pd.to_numeric(group[method], errors="coerce")
            continuous_rows.append({
                "dataset_id": dataset_id,
                "predictor": predictor,
                "method": method,
                "n": int(len(group)),
                "spearman_reproducibility_risk": float(group["split_half_cosine"].corr(risk, method="spearman")),
                "spearman_reproducibility_confidence": float(group["split_half_cosine"].corr(score, method="spearman")),
            })
        for method in METHODS:
            calibration_values = calibration.loc[
                calibration["scenario"].eq("in_domain")
                & calibration["predictor"].astype(str).eq(str(predictor))
                & calibration["method"].eq(method),
                "confidence",
            ].to_numpy(dtype=float)
            if len(calibration_values) == 0:
                raise ValueError(f"missing calibration confidence for {predictor}/{method}")
            method_group = group.copy()
            method_group["confidence_bin"], _ = assign_bins(
                method_group[method].to_numpy(dtype=float), calibration_values, n_bins=10
            )
            component_rows.append(_reliability_shift_row(method_group, str(dataset_id), str(predictor), method))
    result_frame = pd.DataFrame(rows).sort_values(["dataset_id", "predictor", "reproducibility_stratum", "method"], kind="stable")
    continuous_frame = pd.DataFrame(continuous_rows).sort_values(["dataset_id", "predictor", "method"], kind="stable")
    manifest_dir = root / "artifacts/manifests"
    output_path = manifest_dir / "formal_v2_reproducibility_stratified_reliability.csv"
    summary_path = manifest_dir / "formal_v2_reproducibility_stratified_reliability.json"
    continuous_path = manifest_dir / "formal_v2_reproducibility_risk_relationship.csv"
    component_path = manifest_dir / "formal_v2_reproducibility_reliability_shift.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output_path, index=False)
    continuous_frame.to_csv(continuous_path, index=False)
    component_frame = pd.DataFrame(component_rows).sort_values(["dataset_id", "predictor", "method"], kind="stable")
    component_frame.to_csv(component_path, index=False)
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "join": "dataset_id + perturbation_label",
        "reproducibility_source": "formal_v2_raw_cell_split_half.csv",
        "strata": ["low", "middle", "high"],
        "methods": list(METHODS),
        "evaluation_only": True,
        "raw_cell_rows_joined": int(len(joined)),
        "datasets": sorted(joined["dataset_id"].unique().tolist()),
        "output_path": output_path.relative_to(root).as_posix(),
        "relationship_path": continuous_path.relative_to(root).as_posix(),
        "reliability_shift_path": component_path.relative_to(root).as_posix(),
        "reliability_shift_columns": [
            "difficulty_mean_risk", "ranking_spearman_confidence_vs_negative_risk",
            "ranking_kendall_confidence_vs_negative_risk", "ranking_normalized_aurc",
            "all_strata_conditional_contextual_grouping_loss",
            "semantic_interaction_improvement",
        ],
        "status": "formal_v2_reproducibility_stratified_reliability_executed",
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve()), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
