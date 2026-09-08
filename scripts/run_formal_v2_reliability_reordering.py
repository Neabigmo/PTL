"""Measure whether biological shifts reorder perturbation reliability.

This is a focused diagnostic layer over the materialized formal-v2 predictions.
It uses matched perturbation labels within Frangieh condition pairs and keeps
response-program distance separate from the outcome-derived risk statistics.
The analysis is descriptive: it does not claim a causal mechanism.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_grouping_loss import artifact_paths, declared_seeds  # noqa: E402
from scripts.run_formal_v2_predictors import load_panel  # noqa: E402


METHODS = ("raw_normalized_uq", "u_only_rf", "ptl_rf")
PREDICTORS = ("mean_matching", "strong_linear", "slim_string")
FRANGIEH_PAIRS = (
    ("frangieh_melanoma_control", "frangieh_melanoma_coculture"),
    ("frangieh_melanoma_control", "frangieh_melanoma_ifng"),
    ("frangieh_melanoma_coculture", "frangieh_melanoma_ifng"),
)


def _finite_correlation(left: np.ndarray, right: np.ndarray, method: str) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 4 or np.std(left[mask]) == 0.0 or np.std(right[mask]) == 0.0:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(left[mask], right[mask]).statistic)
    return float(kendalltau(left[mask], right[mask], nan_policy="omit").statistic)


def _response_distance(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return float("nan"), float("nan")
    left = left[mask]
    right = right[mask]
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        cosine_distance = float("nan")
    else:
        cosine_distance = float(1.0 - np.dot(left, right) / (left_norm * right_norm))
    magnitude_change = float(abs(right_norm - left_norm) / max(left_norm + right_norm, 1e-12))
    return cosine_distance, magnitude_change


def _response_ground_truth(root: Path) -> tuple[pd.DataFrame, list[str]]:
    path = root / "data/processed/FrangiehIzar2021_RNA_condition_ground_truth.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing Frangieh condition responses: {path}")
    genes = load_panel(root)
    metadata = ["environment_key", "perturbation_label", "condition", "total_cells"]
    frame = pd.read_parquet(path, columns=[*metadata, *genes])
    if frame.duplicated(["environment_key", "perturbation_label"]).any():
        raise ValueError("Frangieh condition responses contain duplicate environment/perturbation rows")
    return frame.set_index(["environment_key", "perturbation_label"], drop=False), genes


def _pairwise_reordering(merged: pd.DataFrame, method: str) -> dict[str, Any]:
    labels = merged["perturbation_label"].astype(str).to_numpy()
    left_risk = merged["risk_left"].to_numpy(dtype=float)
    right_risk = merged["risk_right"].to_numpy(dtype=float)
    left_conf = merged["confidence_left"].to_numpy(dtype=float)
    right_conf = merged["confidence_right"].to_numpy(dtype=float)
    if len(labels) < 2:
        return {
            "method": method,
            "n_shared_perturbations": int(len(labels)),
            "n_comparable_risk_pairs": 0,
            "n_risk_inversions": 0,
            "risk_inversion_rate": float("nan"),
            "confidence_tracking_rate": float("nan"),
            "n_confidence_comparable_inversions": 0,
            "risk_rank_spearman": float("nan"),
            "risk_rank_kendall": float("nan"),
        }
    row_i, row_j = np.triu_indices(len(labels), k=1)
    left_delta = left_risk[row_i] - left_risk[row_j]
    right_delta = right_risk[row_i] - right_risk[row_j]
    left_sign = np.sign(left_delta)
    right_sign = np.sign(right_delta)
    comparable = (left_sign != 0.0) & (right_sign != 0.0) & np.isfinite(left_delta) & np.isfinite(right_delta)
    inversions = comparable & (left_sign != right_sign)
    confidence_left_sign = np.sign(left_conf[row_i] - left_conf[row_j])
    confidence_right_sign = np.sign(right_conf[row_i] - right_conf[row_j])
    confidence_comparable = inversions & (confidence_left_sign != 0.0) & (confidence_right_sign != 0.0)
    tracked = confidence_comparable & (confidence_left_sign == -left_sign) & (confidence_right_sign == -right_sign)
    return {
        "n_shared_perturbations": int(len(labels)),
        "n_comparable_risk_pairs": int(comparable.sum()),
        "n_risk_inversions": int(inversions.sum()),
        "risk_inversion_rate": float(inversions.sum() / comparable.sum()) if comparable.any() else float("nan"),
        "confidence_tracking_rate": float(tracked.sum() / inversions.sum()) if inversions.any() else float("nan"),
        "n_confidence_comparable_inversions": int(confidence_comparable.sum()),
        "risk_rank_spearman": _finite_correlation(left_risk, right_risk, "spearman"),
        "risk_rank_kendall": _finite_correlation(left_risk, right_risk, "kendall"),
        "method": method,
    }


def _matched_pair_frame(
    test: pd.DataFrame,
    left_environment: str,
    right_environment: str,
    predictor: str,
    method: str,
) -> pd.DataFrame:
    subset = test.loc[
        test["environment_key"].isin([left_environment, right_environment])
        & test["predictor"].astype(str).eq(predictor),
        ["environment_key", "perturbation_label", "continuous_risk", method],
    ].copy()
    subset["environment_key"] = subset["environment_key"].astype(str)
    subset["perturbation_label"] = subset["perturbation_label"].astype(str)
    if subset.duplicated(["environment_key", "perturbation_label"]).any():
        raise ValueError("prediction surface contains duplicate environment/perturbation rows")
    left = subset.loc[subset["environment_key"].eq(left_environment)].rename(
        columns={"continuous_risk": "risk_left", method: "confidence_left"}
    )
    right = subset.loc[subset["environment_key"].eq(right_environment)].rename(
        columns={"continuous_risk": "risk_right", method: "confidence_right"}
    )
    return left.merge(right, on="perturbation_label", how="inner", validate="one_to_one").sort_values(
        "perturbation_label", kind="stable"
    ).reset_index(drop=True)


def run(root: Path, *, seeds: list[int] | None = None) -> dict[str, Any]:
    selected_seeds = seeds or declared_seeds(root)
    response, genes = _response_ground_truth(root)
    rows: list[dict[str, Any]] = []
    perturbation_rows: list[dict[str, Any]] = []
    for seed in selected_seeds:
        prediction_path, _ = artifact_paths(root, seed)
        test = pd.read_csv(prediction_path)
        test = test.loc[test["scenario"].eq("in_domain")].copy()
        for left_environment, right_environment in FRANGIEH_PAIRS:
            for predictor in PREDICTORS:
                for method in METHODS:
                    merged = _matched_pair_frame(test, left_environment, right_environment, predictor, method)
                    stats = _pairwise_reordering(merged, method)
                    response_shifts: list[float] = []
                    response_magnitude_changes: list[float] = []
                    risk_changes: list[float] = []
                    for label in merged["perturbation_label"]:
                        left_vector = response.loc[(left_environment, label), genes].to_numpy(dtype=float)
                        right_vector = response.loc[(right_environment, label), genes].to_numpy(dtype=float)
                        distance, magnitude_change = _response_distance(left_vector, right_vector)
                        response_shifts.append(distance)
                        response_magnitude_changes.append(magnitude_change)
                        risk_row = merged.loc[merged["perturbation_label"].eq(label)].iloc[0]
                        abs_risk_change = abs(float(risk_row["risk_left"]) - float(risk_row["risk_right"]))
                        risk_changes.append(abs_risk_change)
                        perturbation_rows.append({
                            "split_seed": int(seed),
                            "comparison": "frangieh_condition",
                            "left_environment_id": left_environment,
                            "right_environment_id": right_environment,
                            "predictor": predictor,
                            "method": method,
                            "perturbation_label": label,
                            "risk_left": float(risk_row["risk_left"]),
                            "risk_right": float(risk_row["risk_right"]),
                            "confidence_left": float(risk_row["confidence_left"]),
                            "confidence_right": float(risk_row["confidence_right"]),
                            "absolute_risk_change": abs_risk_change,
                            "response_program_shift": distance,
                            "response_magnitude_change": magnitude_change,
                            "matched_perturbation_control": "shared perturbation_label",
                            "risk_and_confidence_outcomes_used_for_evaluation_only": 1,
                            "response_program_is_descriptive": 1,
                        })
                    response_shifts_array = np.asarray(response_shifts, dtype=float)
                    risk_changes_array = np.asarray(risk_changes, dtype=float)
                    rows.append({
                        "split_seed": int(seed),
                        "comparison": "frangieh_condition",
                        "left_environment_id": left_environment,
                        "right_environment_id": right_environment,
                        "predictor": predictor,
                        **stats,
                        "response_program_shift_mean": float(np.nanmean(response_shifts_array)) if np.isfinite(response_shifts_array).any() else float("nan"),
                        "response_program_shift_median": float(np.nanmedian(response_shifts_array)) if np.isfinite(response_shifts_array).any() else float("nan"),
                        "response_magnitude_change_mean": float(np.nanmean(response_magnitude_changes)) if np.isfinite(response_magnitude_changes).any() else float("nan"),
                        "response_magnitude_change_median": float(np.nanmedian(response_magnitude_changes)) if np.isfinite(response_magnitude_changes).any() else float("nan"),
                        "response_shift_risk_change_spearman": _finite_correlation(response_shifts_array, risk_changes_array, "spearman"),
                        "n_response_shift_observations": int(np.isfinite(response_shifts_array).sum()),
                        "response_genes": int(len(genes)),
                        "matched_perturbation_control": "shared perturbation_label",
                        "risk_and_confidence_outcomes_used_for_evaluation_only": 1,
                        "response_program_is_descriptive": 1,
                    })

    detail = pd.DataFrame(rows).sort_values(
        ["split_seed", "left_environment_id", "right_environment_id", "predictor", "method"], kind="stable"
    )
    summary = detail.groupby(["comparison", "left_environment_id", "right_environment_id", "predictor", "method"], as_index=False).agg(
        n_splits=("split_seed", "nunique"),
        mean_shared_perturbations=("n_shared_perturbations", "mean"),
        mean_risk_inversion_rate=("risk_inversion_rate", "mean"),
        mean_confidence_tracking_rate=("confidence_tracking_rate", "mean"),
        mean_risk_rank_spearman=("risk_rank_spearman", "mean"),
        mean_risk_rank_kendall=("risk_rank_kendall", "mean"),
        mean_response_program_shift=("response_program_shift_mean", "mean"),
        mean_response_magnitude_change=("response_magnitude_change_mean", "mean"),
        mean_response_shift_risk_change_spearman=("response_shift_risk_change_spearman", "mean"),
    )
    ladder = [
        {
            "comparison": "frangieh_condition",
            "axis": "condition",
            "status": "executed",
            "source": "data/processed/FrangiehIzar2021_RNA_condition_ground_truth.parquet",
            "note": "matched perturbations across control, IFNγ, and TIL co-culture",
        },
        {
            "comparison": "tian_modality",
            "axis": "CRISPRa_vs_CRISPRi",
            "status": "not_identifiable",
            "reason": "the formal test split contains only one shared perturbation label",
        },
        {
            "comparison": "head_to_head_crisprko_crispri",
            "axis": "time_modality_cell_context",
            "status": "external_data_available_not_ingested",
            "source": "10.64898/2026.07.04.736492; 10.5281/zenodo.20722064",
            "reason": "aligned RDS inputs are multi-gigabyte and require a dedicated ingestion contract before model evaluation",
        },
    ]
    manifest_dir = root / "artifacts/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    detail_path = manifest_dir / "formal_v2_reliability_reordering.csv"
    perturbation_path = manifest_dir / "formal_v2_reliability_reordering_perturbations.csv"
    summary_path = manifest_dir / "formal_v2_reliability_reordering_summary.json"
    detail.to_csv(detail_path, index=False)
    pd.DataFrame(perturbation_rows).sort_values(
        ["split_seed", "left_environment_id", "right_environment_id", "predictor", "method", "perturbation_label"],
        kind="stable",
    ).to_csv(perturbation_path, index=False)
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "primary_question": "Does a biological shift reorder which matched perturbations are easy or hard, and does confidence track that reordering?",
        "estimands": {
            "risk_inversion_rate": "fraction of strict, non-tied perturbation pairs whose risk-order sign flips between matched environments",
            "confidence_tracking_rate": "fraction of risk-inverted pairs whose confidence order is correct in both environments, with higher confidence meaning lower risk",
            "response_program_shift": "1 minus cosine similarity between matched perturbation response vectors; descriptive, not causal",
        },
        "outcome_policy": "risk and confidence use held-out test outcomes for evaluation only; response vectors are descriptive evaluation artifacts",
        "ladder": ladder,
        "detail_path": detail_path.relative_to(root).as_posix(),
        "perturbation_path": perturbation_path.relative_to(root).as_posix(),
        "summary": summary.to_dict("records"),
        "status": "formal_v2_reliability_reordering_executed_frangieh_first_stage",
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seeds", type=int, nargs="*")
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), seeds=args.seeds), indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
