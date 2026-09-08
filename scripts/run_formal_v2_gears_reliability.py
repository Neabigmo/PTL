"""Bring validation-backed official GEARS runs into the reliability audit.

Only the supported Norman and RPE1 runs are eligible.  GWPS remains in the
separate external-validity contract because it is not part of the canonical
eight-environment surface.  Validation native scores define bins; test
outcomes define fidelity/risk only after the bins are fixed.
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

from scripts.run_formal_v2_grouping_loss import _summarize_conditional_grouped, assign_bins  # noqa: E402
from scripts.run_formal_v2_reliability_shift import (  # noqa: E402
    _decomposition_rows,
    _hierarchical_ci,
    _permutation_p_value,
)
from src.evaluation.metrics import safe_rowwise_cosine  # noqa: E402


DATASET_ENVIRONMENT = {
    "NormanWeissman2019_filtered": "norman_k562",
    "ReplogleWeissman2022_rpe1": "replogle_rpe1",
}
REQUIRED_DATASETS = tuple(DATASET_ENVIRONMENT)
NATIVE_CONFIDENCE = "native_confidence_gears_exp_neg_mean_logvar"


def _frame_from_run(run_dir: Path, row: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    validation_path = run_dir / "validation_predictions.npz"
    validation_metadata_path = run_dir / "validation_metadata.parquet"
    if not validation_path.is_file() or not validation_metadata_path.is_file():
        raise FileNotFoundError(f"{row['run_id']}: missing validation contract")
    with np.load(run_dir / "test_predictions.npz") as test_arrays:
        y_true = np.asarray(test_arrays["y_true"], dtype=np.float32)
        y_pred = np.asarray(test_arrays["y_pred"], dtype=np.float32)
        native = np.asarray(test_arrays[NATIVE_CONFIDENCE], dtype=float)
    with np.load(validation_path) as validation_arrays:
        validation_native = np.asarray(validation_arrays[NATIVE_CONFIDENCE], dtype=float)
    test_metadata = pd.read_parquet(run_dir / "test_metadata.parquet")
    validation_metadata = pd.read_parquet(validation_metadata_path)
    if y_true.shape != y_pred.shape or len(test_metadata) != len(y_true):
        raise ValueError(f"{row['run_id']}: GEARS test array/metadata mismatch")
    if len(validation_metadata) != len(validation_native):
        raise ValueError(f"{row['run_id']}: GEARS validation array/metadata mismatch")
    fidelity = safe_rowwise_cosine(y_true, y_pred)
    frame = pd.DataFrame({
        "biological_instance_id": [f"{row['run_id']}::{value}" for value in test_metadata["signature_id"].astype(str)],
        "environment_id": DATASET_ENVIRONMENT[str(row["dataset_id"])],
        "environment_key": DATASET_ENVIRONMENT[str(row["dataset_id"])],
        "dataset_id": str(row["dataset_id"]),
        "predictor": "official_gears",
        "method": "gears_native_uq",
        "perturbation_label": test_metadata["perturbation_label"].astype(str).to_numpy(),
        "is_control": test_metadata["is_control"].astype(bool).to_numpy(),
        "fidelity": fidelity,
        "continuous_risk": 1.0 - fidelity,
        "confidence": native,
        "gears_seed": int(row["seed"]),
        "run_id": str(row["run_id"]),
        "evaluation_only": 1,
    })
    frame = frame.loc[~frame["is_control"] & np.isfinite(frame["confidence"])].copy()
    validation_mask = ~validation_metadata["is_control"].astype(bool).to_numpy() & np.isfinite(validation_native)
    return frame, validation_native[validation_mask]


def run(root: Path, *, output_root: Path | None = None, n_bins: int = 10, bootstrap_replicates: int = 1000, permutation_repeats: int = 250) -> dict[str, Any]:
    root = root.resolve()
    output_root = (output_root or root / "results/formal_v2/gears_remote").resolve()
    matrix = pd.read_csv(output_root / "formal_v2_gears_run_matrix.csv")
    selected = matrix.loc[matrix["dataset_id"].isin(REQUIRED_DATASETS) & matrix["seed"].astype(int).isin([0, 1, 2])].copy()
    if set(selected["dataset_id"]) != set(REQUIRED_DATASETS) or len(selected) != 6:
        raise ValueError("GEARS reliability surface requires Norman/RPE1 x seeds 0/1/2")
    frames: list[pd.DataFrame] = []
    calibration_by_seed: dict[int, list[np.ndarray]] = {}
    for _, row in selected.sort_values(["seed", "dataset_id"]).iterrows():
        run_dir = output_root / str(row["run_id"])
        frame, validation_scores = _frame_from_run(run_dir, row)
        frames.append(frame)
        calibration_by_seed.setdefault(int(row["seed"]), []).append(validation_scores)

    prediction_path = root / "artifacts/source_data/formal_v2_gears_reliability_predictions.csv"
    per_environment_path = root / "artifacts/manifests/formal_v2_gears_reliability.csv"
    summary_path = root / "artifacts/manifests/formal_v2_gears_reliability_summary.json"
    all_rows: list[pd.DataFrame] = []
    environment_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    for seed in sorted(calibration_by_seed):
        seed_frame = pd.concat([frame.loc[frame["gears_seed"].eq(seed)] for frame in frames], ignore_index=True)
        calibration_scores = np.concatenate(calibration_by_seed[seed])
        seed_frame["confidence_bin"], edges = assign_bins(seed_frame["confidence"].to_numpy(dtype=float), calibration_scores, n_bins=n_bins)
        seed_frame["confidence_bin_lower"] = seed_frame["confidence_bin"].map(lambda value: float(edges[min(int(value), len(edges) - 1)]))
        seed_frame["confidence_bin_upper"] = seed_frame["confidence_bin"].map(lambda value: float(edges[min(int(value) + 1, len(edges) - 1)]))
        seed_frame["confidence_bins_fit_on"] = "validation_native_scores_only"
        seed_frame["outcomes_used_for_bin_construction"] = 0
        all_rows.append(seed_frame)
        for environment, group in seed_frame.groupby("environment_id", sort=True):
            difficulty, _, _ = _decomposition_rows(
                group,
                "official_gears",
                "gears_native_uq",
                seed,
                bootstrap_replicates=0,
                permutation_repeats=0,
            )
            first = difficulty[0]
            environment_rows.append({
                "gears_seed": seed,
                "environment_id": environment,
                **{key: value for key, value in first.items() if key not in {"split_seed", "predictor", "method", "environment_id"}},
                "validation_native_scores": int(sum(len(value) for value in calibration_by_seed[seed])),
                "evaluation_only": 1,
            })
        grouping = _summarize_conditional_grouped(seed_frame)
        _, semantic, _ = _decomposition_rows(
            seed_frame,
            "official_gears",
            "gears_native_uq",
            seed,
            bootstrap_replicates=bootstrap_replicates,
            permutation_repeats=permutation_repeats,
        )
        oof_semantic = float(semantic["semantic_shift_improvement"])
        shift_rows.append({
            "gears_seed": seed,
            "n_test_rows": int(len(seed_frame)),
            "n_environments": int(seed_frame["environment_id"].nunique()),
            "difficulty_shift_variance": float(seed_frame.groupby("environment_id")["continuous_risk"].mean().var(ddof=0)),
            "difficulty_shift_range": float(seed_frame.groupby("environment_id")["continuous_risk"].mean().max() - seed_frame.groupby("environment_id")["continuous_risk"].mean().min()),
            "ranking_shift_spearman_range": float(pd.DataFrame(environment_rows).loc[pd.DataFrame(environment_rows)["gears_seed"].eq(seed), "ranking_spearman_confidence_vs_negative_risk"].max() - pd.DataFrame(environment_rows).loc[pd.DataFrame(environment_rows)["gears_seed"].eq(seed), "ranking_spearman_confidence_vs_negative_risk"].min()),
            "contextual_grouping_loss": grouping["contextual_grouping_loss"],
            "matched_confidence_cross_environment_risk_gap": grouping["matched_confidence_cross_environment_risk_gap"],
            "matched_confidence_same_environment_risk_gap": grouping["matched_confidence_same_environment_risk_gap"],
            "semantic_shift_improvement": oof_semantic,
            "semantic_shift_ci_lower": semantic["semantic_shift_ci_lower"],
            "semantic_shift_ci_upper": semantic["semantic_shift_ci_upper"],
            "semantic_shift_permutation_p": semantic["semantic_shift_permutation_p"],
            "score_direction": "higher_native_exp_neg_mean_logvar_is_more_confident",
            "evaluation_only": 1,
        })

    prediction_frame = pd.concat(all_rows, ignore_index=True).sort_values(["gears_seed", "environment_id", "biological_instance_id"], kind="stable")
    environment_frame = pd.DataFrame(environment_rows).sort_values(["gears_seed", "environment_id"], kind="stable")
    shift_frame = pd.DataFrame(shift_rows).sort_values("gears_seed", kind="stable")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    per_environment_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(prediction_path, index=False)
    environment_frame.to_csv(per_environment_path, index=False)
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "predictor": "official_gears",
        "method": "gears_native_uq",
        "datasets": list(REQUIRED_DATASETS),
        "seeds": [0, 1, 2],
        "validation_predictions_used": True,
        "validation_scores_define_bins": True,
        "test_outcomes_evaluation_only": True,
        "prediction_path": prediction_path.relative_to(root).as_posix(),
        "per_environment_path": per_environment_path.relative_to(root).as_posix(),
        "shift_by_seed": shift_frame.to_dict("records"),
        "status": "formal_v2_gears_reliability_executed",
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--permutation-repeats", type=int, default=250)
    args = parser.parse_args()
    payload = run(
        args.root,
        output_root=args.output_root,
        n_bins=args.n_bins,
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_repeats=args.permutation_repeats,
    )
    print(json.dumps({"status": payload["status"], "shift_by_seed": payload["shift_by_seed"]}, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
