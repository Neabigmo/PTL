"""Quantify incremental decision value with context-pair-held-out validation.

The 18 directed rows are a descriptive decision surface, not 18 independent
observations. Every outer fold therefore holds out one unordered context pair
(both directions and all three metrics), so the reverse direction cannot leak
into training. Direct model contrasts are reported only when the comparison
feature set is a genuine superset of the reference feature set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.decision_analysis_common import add_context_ids, load_canonical_d_adj  # noqa: E402


METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")
METRIC_COLUMNS = [f"metric_{metric}" for metric in METRICS]
KEYS = ["source_environment_id", "target_environment_id", "metric"]
OUTER_GROUP = "unordered_context_pair_id"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260914

MODELS = {
    "M0_metric": ["metric"],
    "M1_metric_plus_mean": ["metric", "absolute_mean_risk_shift"],
    "M2_metric_plus_d_adj": ["metric", "d_meas_id"],
    "M3_metric_plus_mean_plus_d_adj": ["metric", "absolute_mean_risk_shift", "d_meas_id"],
    "M4_metric_plus_mean_plus_decomposition": ["metric", "absolute_mean_risk_shift", "d_tie_cf", "d_direction_cf"],
}

CONTRAST_SPECS = (
    ("M1-M0", "M1_metric_plus_mean", "M0_metric"),
    ("M2-M0", "M2_metric_plus_d_adj", "M0_metric"),
    ("M3-M1", "M3_metric_plus_mean_plus_d_adj", "M1_metric_plus_mean"),
    ("M4-M3", "M4_metric_plus_mean_plus_decomposition", "M3_metric_plus_mean_plus_d_adj"),
)


def _feature_matrix(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    metric = pd.get_dummies(frame["metric"], prefix="metric", dtype=float)
    metric = metric.reindex(columns=METRIC_COLUMNS, fill_value=0.0)
    pieces = [metric]
    names = METRIC_COLUMNS.copy()
    for column in columns:
        if column == "metric":
            continue
        pieces.append(frame[[column]].astype(float))
        names.append(column)
    return pd.concat(pieces, axis=1).to_numpy(dtype=float), names


def _with_context_ids(frame: pd.DataFrame) -> pd.DataFrame:
    if OUTER_GROUP not in frame.columns or "source_context_id" not in frame.columns:
        return add_context_ids(frame)
    return frame.copy()


def _cv_groups(frame: pd.DataFrame) -> list[str]:
    groups = sorted(frame[OUTER_GROUP].astype(str).unique())
    if len(groups) != 3:
        raise ValueError(f"LOPO requires three unordered context-pair blocks, found {len(groups)}")
    sizes = frame.groupby(OUTER_GROUP, sort=True).size()
    if not sizes.eq(6).all():
        raise ValueError(f"each unordered context-pair block must contain six rows, found {sizes.to_dict()}")
    return groups


def _fit_predict_oof(frame: pd.DataFrame, columns: list[str], y: np.ndarray | None = None) -> np.ndarray:
    work = _with_context_ids(frame)
    observed = work["normalized_regret"].to_numpy(dtype=float) if y is None else np.asarray(y, dtype=float)
    if len(observed) != len(work):
        raise ValueError("response vector length does not match the decision surface")
    x, _ = _feature_matrix(work, columns)
    groups = _cv_groups(work)
    predictions = np.full(len(work), np.nan, dtype=float)
    for group in groups:
        test = work[OUTER_GROUP].eq(group).to_numpy()
        train = ~test
        x_train = x[train]
        x_test = x[test]
        mean = x_train.mean(axis=0)
        scale = x_train.std(axis=0)
        scale[scale < 1e-12] = 1.0
        x_train = (x_train - mean) / scale
        x_test = (x_test - mean) / scale
        design_train = np.column_stack((np.ones(np.sum(train)), x_train))
        design_test = np.column_stack((np.ones(np.sum(test)), x_test))
        coefficients, *_ = np.linalg.lstsq(design_train, observed[train], rcond=None)
        predictions[test] = design_test @ coefficients
    if not np.isfinite(predictions).all():
        raise ValueError("LOPO OOF prediction contains non-finite values")
    return predictions


def _metrics(frame: pd.DataFrame, prediction: np.ndarray, y: np.ndarray | None = None) -> dict[str, float]:
    work = _with_context_ids(frame)
    observed = work["normalized_regret"].to_numpy(dtype=float) if y is None else np.asarray(y, dtype=float)
    residual = prediction - observed
    pair_errors = pd.DataFrame({OUTER_GROUP: work[OUTER_GROUP].to_numpy(), "abs_error": np.abs(residual)})
    held_out = pair_errors.groupby(OUTER_GROUP, sort=True)["abs_error"].mean()
    rho = spearmanr(observed, prediction).statistic if np.unique(observed).size > 1 and np.unique(prediction).size > 1 else float("nan")
    held_out_mae = float(held_out.mean())
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman_oof": float(rho),
        "held_out_unordered_pair_mae": held_out_mae,
        "held_out_transfer_mae": held_out_mae,
    }


def _is_nested(reference: str, comparison: str) -> bool:
    return set(MODELS[reference]).issubset(MODELS[comparison])


def _bootstrap_pair_delta(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    comparison: str,
    reference: str,
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[float, float, int]:
    work = _with_context_ids(frame)
    groups = np.asarray(_cv_groups(work), dtype=object)
    rng = np.random.default_rng(seed)
    y = work["normalized_regret"].to_numpy(dtype=float)
    comparison_prediction = predictions[comparison]
    reference_prediction = predictions[reference]
    values: list[float] = []
    masks = {group: np.flatnonzero(work[OUTER_GROUP].eq(group).to_numpy()) for group in groups}
    for _ in range(int(draws)):
        sample = rng.choice(groups, size=len(groups), replace=True)
        mask = np.concatenate([masks[group] for group in sample])
        reference_loss = float(np.mean(np.abs(reference_prediction[mask] - y[mask])))
        comparison_loss = float(np.mean(np.abs(comparison_prediction[mask] - y[mask])))
        values.append(reference_loss - comparison_loss)
    return float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95)), len(values)


def _contrast_row(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    contrast: str,
    comparison: str,
    reference: str,
) -> dict[str, Any] | None:
    if not _is_nested(reference, comparison):
        return None
    ci_low, ci_high, valid_draws = _bootstrap_pair_delta(frame, predictions, comparison, reference)
    return {
        "contrast": contrast,
        "reference_model": reference,
        "comparison_model": comparison,
        "nested": True,
        "valid": True,
        "n": int(len(frame)),
        "mae_reference": metrics[reference]["mae"],
        "mae_comparison": metrics[comparison]["mae"],
        "rmse_reference": metrics[reference]["rmse"],
        "rmse_comparison": metrics[comparison]["rmse"],
        "delta_mae": float(metrics[reference]["mae"] - metrics[comparison]["mae"]),
        "delta_rmse": float(metrics[reference]["rmse"] - metrics[comparison]["rmse"]),
        "bootstrap_delta_mae_q05": ci_low,
        "bootstrap_delta_mae_q95": ci_high,
        "bootstrap_draws": valid_draws,
        "cv_method": "leave-one-unordered-context-pair-out",
        "cv_blocks": 3,
        "d_adj_source": "corrected canonical summary only",
    }


def _load_frame(root: Path) -> pd.DataFrame:
    manifests = root / "artifacts/manifests"
    decision_path = manifests / "reliability_transport_decision_link.csv"
    mean_path = manifests / "metric_matched_mean_fidelity_control.csv"
    decomposition_path = manifests / "reliability_transport_decomposition.csv"
    for path in (decision_path, mean_path, decomposition_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    decision = pd.read_csv(decision_path)
    decision = decision.loc[decision["cell_budget_label"].astype(str).eq("full")].copy()
    if len(decision) != 18 or decision.duplicated(KEYS).any():
        raise ValueError("incremental value requires one full-depth decision row per directed transfer and metric")
    decision["normalized_regret"] = pd.to_numeric(decision["normalized_regret"], errors="coerce")

    # D_adj is loaded directly from the corrected canonical summary. The
    # decision-link CSV supplies only the decision response for this analysis.
    canonical = load_canonical_d_adj(root, METRICS)
    frame = canonical.merge(
        decision[KEYS + ["normalized_regret"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )

    mean = pd.read_csv(mean_path)
    mean = mean.loc[mean["cell_budget_label"].astype(str).eq("full")].copy()
    if mean.duplicated(KEYS).any():
        raise ValueError("mean-fidelity control has duplicate directed transfer x metric keys")
    frame = frame.merge(
        mean[KEYS + ["absolute_mean_risk_shift"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )

    decomposition = pd.read_csv(decomposition_path)
    required = KEYS + ["d_tie_cf", "d_direction_cf", "corrected_total_cf"]
    missing = [column for column in required if column not in decomposition.columns]
    if missing:
        raise ValueError(f"corrected decomposition is missing columns: {missing}")
    if decomposition.duplicated(KEYS).any():
        raise ValueError("corrected decomposition has duplicate directed transfer x metric keys")
    frame = frame.merge(
        decomposition[required],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    frame = frame.rename(columns={"corrected_total_cf": "d_decomposition_cf"})
    frame = add_context_ids(frame)
    if len(frame) != 18:
        raise ValueError(f"incremental value requires 18 rows, found {len(frame)}")
    if frame[OUTER_GROUP].nunique() != 3 or frame.groupby(OUTER_GROUP).size().ne(6).any():
        raise ValueError("incremental value requires three six-row unordered context-pair blocks")
    if frame["transfer_id"].nunique() != 6 or frame["metric"].nunique() != 3:
        raise ValueError("incremental value requires six directed transfers and three metrics")
    numeric = ["normalized_regret", "absolute_mean_risk_shift", "d_meas_id", "d_tie_cf", "d_direction_cf"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("incremental value inputs contain non-finite response or feature values")
    return frame.sort_values([OUTER_GROUP, "source_environment_id", "target_environment_id", "metric"], kind="stable").reset_index(drop=True)


def build(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    frame = _load_frame(root)

    predictions = {name: _fit_predict_oof(frame, columns) for name, columns in MODELS.items()}
    model_metrics = {name: _metrics(frame, prediction) for name, prediction in predictions.items()}

    model_bootstrap: dict[tuple[str, str], tuple[float, float, int]] = {}
    for model in MODELS:
        if model != "M0_metric" and _is_nested("M0_metric", model):
            model_bootstrap[(model, "M0_metric")] = _bootstrap_pair_delta(frame, predictions, model, "M0_metric")
        if model != "M1_metric_plus_mean" and _is_nested("M1_metric_plus_mean", model):
            model_bootstrap[(model, "M1_metric_plus_mean")] = _bootstrap_pair_delta(frame, predictions, model, "M1_metric_plus_mean")

    omission_metadata = json.dumps(
        {
            "status": "not_run",
            "reason": "no p-value reported with only three unordered context-pair blocks",
            "blocks": 3,
        },
        sort_keys=True,
    )
    output_rows: list[dict[str, Any]] = []
    for name, columns in MODELS.items():
        metrics = model_metrics[name]
        versus_m0 = model_bootstrap.get((name, "M0_metric"))
        versus_m1 = model_bootstrap.get((name, "M1_metric_plus_mean"))
        output_rows.append({
            "model": name,
            "feature_set": ",".join(columns),
            "n": int(len(frame)),
            "cv_method": "leave-one-unordered-context-pair-out",
            "cv_blocks": 3,
            **metrics,
            "delta_mae_vs_M0": float(model_metrics["M0_metric"]["mae"] - metrics["mae"]),
            "delta_rmse_vs_M0": float(model_metrics["M0_metric"]["rmse"] - metrics["rmse"]),
            "delta_mae_vs_M1": float(model_metrics["M1_metric_plus_mean"]["mae"] - metrics["mae"]) if _is_nested("M1_metric_plus_mean", name) else float("nan"),
            "bootstrap_delta_mae_q05": versus_m0[0] if versus_m0 else 0.0,
            "bootstrap_delta_mae_q95": versus_m0[1] if versus_m0 else 0.0,
            "bootstrap_delta_mae_vs_M1_q05": versus_m1[0] if versus_m1 else (0.0 if name == "M1_metric_plus_mean" else float("nan")),
            "bootstrap_delta_mae_vs_M1_q95": versus_m1[1] if versus_m1 else (0.0 if name == "M1_metric_plus_mean" else float("nan")),
            "permutation_test": omission_metadata,
            "d_adj_source": "corrected canonical summary only",
        })
    summary = pd.DataFrame(output_rows)

    contrast_rows: list[dict[str, Any]] = []
    omitted_contrasts: dict[str, dict[str, Any]] = {}
    for contrast, comparison, reference in CONTRAST_SPECS:
        row = _contrast_row(frame, predictions, model_metrics, contrast, comparison, reference)
        if row is None:
            omitted_contrasts[contrast] = {
                "comparison_model": comparison,
                "reference_model": reference,
                "valid": False,
                "reason": "comparison feature set is not a superset of the reference feature set",
            }
        else:
            contrast_rows.append(row)
    contrasts = pd.DataFrame(contrast_rows)

    oof = frame[["source_environment_id", "target_environment_id", "transfer_id", "source_context_id", "unordered_context_pair_id", "metric", "normalized_regret"]].copy()
    for name, prediction in predictions.items():
        oof[f"prediction_{name}"] = prediction

    summary_path = manifests / "reviewer_decision_incremental_value.csv"
    oof_path = manifests / "reviewer_decision_incremental_value_oof.csv"
    contrasts_path = manifests / "reviewer_decision_incremental_value_contrasts.csv"
    summary.to_csv(summary_path, index=False)
    oof.to_csv(oof_path, index=False)
    contrasts.to_csv(contrasts_path, index=False)

    reported_contrasts = {row["contrast"]: row for row in contrast_rows}
    report = {
        "schema_version": 2,
        "status": "executed",
        "unit": "three unordered context-pair blocks × two directions × three metrics = 18 directed rows",
        "outer_validation": "leave-one-unordered-context-pair-out; both directions and all metrics held out together; continuous scaling fit within each training fold",
        "validation_blocks": 3,
        "reverse_direction_leakage": False,
        "d_adj_source": "corrected canonical summary only",
        "models": {name: columns for name, columns in MODELS.items()},
        "nested_contrasts": {
            "reported": reported_contrasts,
            "omitted_invalid": omitted_contrasts,
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "unit": "unordered context-pair block",
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 90% interval",
        },
        "p_value_reporting": "omitted; three unordered context-pair blocks are insufficient for a headline p-value",
        "outputs": {
            "summary": summary_path.relative_to(root).as_posix(),
            "oof": oof_path.relative_to(root).as_posix(),
            "contrasts": contrasts_path.relative_to(root).as_posix(),
        },
    }
    (manifests / "reviewer_decision_incremental_value.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return summary, oof, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary, _, report = build(args.root)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
