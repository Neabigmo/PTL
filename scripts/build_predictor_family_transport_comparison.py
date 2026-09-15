"""Compare the source-only RBF family with the primary bilinear family.

This is a read-only comparison of already-frozen source-context prediction
vectors and already-computed full-size, minimum-20-cell measurement surfaces.
No target transport statistic is used to tune alpha or choose a family.
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

from scripts.build_formal_v2_controlled_shift_oof import ENVIRONMENTS, MODEL_SEEDS  # noqa: E402
from scripts.run_formal_v2_claim_lock_source_frozen import (  # noqa: E402
    _load_common_surface,
    _metric_vectors,
)
from scripts.run_frangieh_source_only_rbf_krr import (  # noqa: E402
    DIRECTED_PAIRS,
    METRICS,
    _endpoint_rows,
)


KEYS = ["source_environment_id", "target_environment_id", "metric"]
PRIMARY_FAMILY = "bilinear_ridge"
RBF_FAMILY = "rbf_kernel_ridge_source_only"
PRIMARY_PREDICTION = "artifacts/source_data/frangieh_source_frozen_predictions.npz"
PRIMARY_SOURCE_FROZEN_REPORT = "artifacts/manifests/formal_v2_claim_lock_source_frozen.json"
RBF_PREDICTION = "artifacts/source_data/frangieh_source_only_rbf_krr_predictions.npz"
PRIMARY_MEASUREMENT = "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize.json"
PRIMARY_SUMMARY = "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_summary.csv"
PRIMARY_ORDERING = "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_ordering.csv"
PRIMARY_FLOORS = "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_floors.csv"
RBF_FIT = "artifacts/manifests/frangieh_source_only_rbf_krr_fit.json"
RBF_EVALUATION = "artifacts/manifests/frangieh_source_only_rbf_krr_evaluation.json"
RBF_D_ADJ = "artifacts/manifests/frangieh_source_only_rbf_krr_d_adj_fullsize.csv"
RBF_FLOOR = "artifacts/manifests/frangieh_source_only_rbf_krr_same_context_floor_fullsize.csv"
RBF_STABLE = "artifacts/manifests/frangieh_source_only_rbf_krr_stable_inversion_fullsize.csv"
OUTPUT_CSV = "artifacts/manifests/predictor_family_transport_comparison.csv"
OUTPUT_JSON = "artifacts/manifests/predictor_family_transport_comparison.json"


def _read_json(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _validate_18_rows(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    _require_columns(frame, KEYS, label)
    output = frame.copy()
    for column in KEYS:
        output[column] = output[column].astype(str)
    output = output.sort_values(KEYS, kind="stable").reset_index(drop=True)
    expected = {
        (source, target, metric)
        for source, target in DIRECTED_PAIRS
        for metric in METRICS
    }
    observed = set(map(tuple, output[KEYS].to_records(index=False)))
    if len(output) != 18 or output.duplicated(KEYS).any() or observed != expected:
        raise ValueError(
            f"{label} must contain the exact 18 directed transfer x metric rows; "
            f"found {len(output)} rows"
        )
    return output


def _sign_label(value: float) -> str:
    value = float(value)
    if not np.isfinite(value) or np.isclose(value, 0.0, atol=1e-12):
        return "unresolved"
    return "positive" if value > 0.0 else "negative"


def _interval_sign(low: float, high: float) -> str:
    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high):
        return "unresolved"
    if low > 0.0:
        return "positive"
    if high < 0.0:
        return "negative"
    return "crosses_zero"


def _floor_lookup(floors: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, float]]:
    required = [
        "left_target_environment_id", "right_target_environment_id", "metric",
        "ordering_within_left_u", "ordering_within_right_u",
    ]
    _require_columns(floors, required, "primary floor table")
    output: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, group in floors.groupby(
        ["left_target_environment_id", "right_target_environment_id", "metric"],
        sort=False,
    ):
        left, right, metric = (str(value) for value in key)
        output[(left, right, metric)] = {
            "left": float(pd.to_numeric(group["ordering_within_left_u"], errors="raise").mean()),
            "right": float(pd.to_numeric(group["ordering_within_right_u"], errors="raise").mean()),
        }
    return output


def _load_prediction(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {"source_environment", "perturbation_label", "fold_id", "model_seed", "prediction", "evaluation_gene_symbols"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{label} prediction artifact is missing arrays: {missing}")
        output = {
            "source_environment": payload["source_environment"].astype(str).tolist(),
            "perturbation_label": payload["perturbation_label"].astype(str).tolist(),
            "fold_id": payload["fold_id"].astype(np.int16),
            "model_seed": payload["model_seed"].astype(np.int16),
            "prediction": payload["prediction"].astype(np.float32, copy=True),
            "evaluation_gene_symbols": payload["evaluation_gene_symbols"].astype(str).tolist(),
        }
    prediction = output["prediction"]
    if prediction.ndim != 4 or prediction.shape[:3] != (3, 243, 3):
        raise ValueError(f"{label} prediction tensor has unexpected shape {prediction.shape}")
    if prediction.shape[3] != len(output["evaluation_gene_symbols"]):
        raise ValueError(f"{label} prediction tensor and gene panel disagree")
    if not np.isfinite(prediction).all():
        raise ValueError(f"{label} prediction tensor contains non-finite values")
    if output["source_environment"] != list(ENVIRONMENTS):
        raise ValueError(f"{label} source contexts do not match the primary Frangieh order")
    if output["model_seed"].tolist() != list(MODEL_SEEDS):
        raise ValueError(f"{label} model-member seeds do not match the primary protocol")
    if len(output["perturbation_label"]) != len(output["fold_id"]):
        raise ValueError(f"{label} labels and folds are not aligned")
    return output


def _validate_prediction_alignment(primary: dict[str, Any], rbf: dict[str, Any]) -> None:
    for field in ("source_environment", "perturbation_label", "fold_id", "model_seed", "evaluation_gene_symbols"):
        left = primary[field]
        right = rbf[field]
        if isinstance(left, list):
            if left != right:
                raise ValueError(f"primary and RBF {field} vectors are not identical")
        elif not np.array_equal(left, right):
            raise ValueError(f"primary and RBF {field} vectors are not identical")


def _primary_transport_rows(root: Path) -> pd.DataFrame:
    summary = pd.read_csv(root / PRIMARY_SUMMARY)
    ordering = pd.read_csv(root / PRIMARY_ORDERING)
    floors = pd.read_csv(root / PRIMARY_FLOORS)
    _require_columns(
        summary,
        [
            "source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric",
            "n_perturbations_primary_min", "ordering_cross_disagreement",
            "ordering_measurement_floor", "ordering_measurement_floor_ci_low",
            "ordering_measurement_floor_ci_high", "ordering_delta_meas_id",
            "ordering_delta_meas_id_ci_low", "ordering_delta_meas_id_ci_high",
        ],
        "primary measurement summary",
    )
    _require_columns(
        ordering,
        [
            "source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric",
            "n_items", "minimum_strict_support", "stable_fraction_left", "stable_fraction_right",
            "stable_fraction_both", "stable_order_inversion_fraction", "stable_pairs",
            "crossfit_heldout_inversion_fraction", "crossfit_heldout_evaluable_fraction",
        ],
        "primary ordering table",
    )
    summary_endpoint = _endpoint_rows(summary, source_column="source_environment_id")
    ordering_endpoint = _endpoint_rows(ordering, source_column="source_environment_id")
    floor_lookup = _floor_lookup(floors)
    rows: list[dict[str, Any]] = []
    for record in summary_endpoint.to_dict(orient="records"):
        source = str(record["source_environment_id"])
        left = str(record["left_target_environment_id"])
        right = str(record["right_target_environment_id"])
        metric = str(record["metric"])
        floor = floor_lookup[(left, right, metric)]
        source_floor = floor["left"] if source == left else floor["right"]
        target_floor = floor["right"] if source == left else floor["left"]
        rows.append({
            "source_environment_id": source,
            "target_environment_id": str(record["target_environment_id"]),
            "transfer_id": str(record["transfer_id"]),
            "left_target_environment_id": left,
            "right_target_environment_id": right,
            "metric": metric,
            "cell_budget_label": "full_min20",
            "n_perturbations": int(float(record["n_perturbations_primary_min"])),
            "d_adj": float(record["ordering_delta_meas_id"]),
            "d_adj_ci_low": float(record["ordering_delta_meas_id_ci_low"]),
            "d_adj_ci_high": float(record["ordering_delta_meas_id_ci_high"]),
            "cross_disagreement": float(record["ordering_cross_disagreement"]),
            "same_context_floor": float(record["ordering_measurement_floor"]),
            "same_context_floor_ci_low": float(record["ordering_measurement_floor_ci_low"]),
            "same_context_floor_ci_high": float(record["ordering_measurement_floor_ci_high"]),
            "source_same_context_floor": source_floor,
            "target_same_context_floor": target_floor,
            "same_context_floor_definition": "mean of the two endpoint U-statistic within-context order floors at the matched full-size budget",
            "measurement_resampling": "full_size_nonparametric",
            "prediction_refit_per_metric": 0,
            "provenance": "frangieh_source_frozen_predictions.npz; formal_v2_claim_lock_measurement_fullsize_summary.csv; formal_v2_claim_lock_measurement_fullsize_ordering.csv; formal_v2_claim_lock_measurement_fullsize_floors.csv",
        })
    base = _validate_18_rows(pd.DataFrame(rows), "primary transport summary")
    stable = _validate_18_rows(
        _endpoint_rows(ordering, source_column="source_environment_id"),
        "primary endpoint ordering",
    )
    stable = stable[
        KEYS + [
            "n_items", "minimum_strict_support", "stable_fraction_left", "stable_fraction_right",
            "stable_fraction_both", "stable_order_inversion_fraction", "stable_pairs",
            "crossfit_heldout_inversion_fraction", "crossfit_heldout_evaluable_fraction",
        ]
    ].copy()
    merged = base.merge(stable, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_ordering"))
    if "left_target_environment_id" not in merged.columns or "right_target_environment_id" not in merged.columns:
        raise ValueError("primary endpoint summary did not retain its context-pair identifiers")
    source_is_left = merged["source_environment_id"].eq(merged["left_target_environment_id"])
    merged["stable_fraction_source"] = np.where(
        source_is_left, merged["stable_fraction_left"], merged["stable_fraction_right"]
    )
    merged["stable_fraction_target"] = np.where(
        source_is_left, merged["stable_fraction_right"], merged["stable_fraction_left"]
    )
    # The ordering endpoint table retains endpoint identifiers, but they are
    # not needed in the normalized comparison table.
    return merged.drop(
        columns=[
            column for column in [
                "left_target_environment_id", "right_target_environment_id",
                "stable_fraction_left", "stable_fraction_right",
            ] if column in merged.columns
        ]
    )


def _rbf_transport_rows(root: Path) -> pd.DataFrame:
    d_adj = pd.read_csv(root / RBF_D_ADJ)
    floor = pd.read_csv(root / RBF_FLOOR)
    stable = pd.read_csv(root / RBF_STABLE)
    _require_columns(
        d_adj,
        KEYS + [
            "cell_budget_label", "n_perturbations", "d_adj", "d_adj_ci_low", "d_adj_ci_high",
            "cross_disagreement", "same_context_floor", "same_context_floor_ci_low",
            "same_context_floor_ci_high", "source_same_context_floor", "target_same_context_floor",
        ],
        "RBF D_adj table",
    )
    _require_columns(
        stable,
        KEYS + [
            "n_items", "minimum_strict_support", "stable_fraction_source", "stable_fraction_target",
            "stable_fraction_both", "stable_order_inversion_fraction", "stable_pairs",
            "crossfit_heldout_inversion_fraction", "crossfit_heldout_evaluable_fraction",
        ],
        "RBF stable inversion table",
    )
    base = _validate_18_rows(d_adj, "RBF D_adj")
    stable = _validate_18_rows(stable, "RBF stable inversion")
    _validate_18_rows(floor, "RBF same-context floor")
    merged = base.merge(
        stable[
            KEYS + [
                "n_items", "minimum_strict_support", "stable_fraction_source", "stable_fraction_target",
                "stable_fraction_both", "stable_order_inversion_fraction", "stable_pairs",
                "crossfit_heldout_inversion_fraction", "crossfit_heldout_evaluable_fraction",
            ]
        ],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    merged["cell_budget_label"] = "full_min20"
    if merged[["n_items", "stable_order_inversion_fraction"]].isna().any().any():
        raise ValueError("RBF D_adj and stable inversion rows do not match")
    return merged


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 2 or np.isclose(np.std(left), 0.0) or np.isclose(np.std(right), 0.0):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 2:
        return None
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    return _pearson(left_rank, right_rank)


def _correlation_stats(left: pd.Series, right: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    n = int(np.sum(np.isfinite(x) & np.isfinite(y)))
    return {
        "n": n,
        "spearman": _spearman(x, y),
        "pearson": _pearson(x, y),
    }


def _risk_surfaces(
    root: Path,
    labels: list[str],
    panel: list[str],
    predictions: np.ndarray,
    truths: dict[str, np.ndarray],
) -> tuple[dict[tuple[str, str, str], np.ndarray], dict[str, dict[str, float]]]:
    risks: dict[tuple[str, str, str], np.ndarray] = {}
    fidelity: dict[str, dict[str, float]] = {}
    for source_index, source in enumerate(ENVIRONMENTS):
        source_fidelity: dict[str, float] = {}
        for target in ENVIRONMENTS:
            vectors = _metric_vectors(truths[target], predictions[source_index], labels, panel)
            for metric, values in vectors.items():
                risks[(source, target, metric)] = np.asarray(values, dtype=float)
                if target == source:
                    source_fidelity[metric] = float(1.0 - np.mean(values))
        fidelity[source] = source_fidelity
    return risks, fidelity


def _attach_prediction_evidence(
    table: pd.DataFrame,
    risks: dict[tuple[str, str, str], np.ndarray],
    fidelity: dict[str, dict[str, float]],
) -> pd.DataFrame:
    output = table.copy()
    risk_delta: list[float] = []
    risk_direction: list[str] = []
    for record in output.to_dict(orient="records"):
        source = str(record["source_environment_id"])
        target = str(record["target_environment_id"])
        metric = str(record["metric"])
        source_mean = float(np.mean(risks[(source, source, metric)]))
        target_mean = float(np.mean(risks[(source, target, metric)]))
        delta = target_mean - source_mean
        risk_delta.append(delta)
        risk_direction.append(_sign_label(delta))
    output["source_target_mean_risk_change"] = risk_delta
    output["source_target_risk_direction"] = risk_direction
    for metric in METRICS:
        output[f"source_oof_fidelity_{metric}"] = [
            float(fidelity[str(source)][metric])
            for source in output["source_environment_id"]
        ]
    output["d_adj_point_sign"] = [_sign_label(value) for value in output["d_adj"]]
    output["d_adj_ci_sign"] = [
        _interval_sign(low, high)
        for low, high in zip(output["d_adj_ci_low"], output["d_adj_ci_high"], strict=True)
    ]
    output["same_context_floor_ci_sign"] = [
        _interval_sign(low, high)
        for low, high in zip(output["same_context_floor_ci_low"], output["same_context_floor_ci_high"], strict=True)
    ]
    return output


def _rowwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    numerator = np.sum(left * right, axis=-1)
    left_norm = np.linalg.norm(left, axis=-1)
    right_norm = np.linalg.norm(right, axis=-1)
    denominator = left_norm * right_norm
    result = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0)
    result[(left_norm == 0.0) & (right_norm == 0.0)] = 1.0
    return result


def _prediction_similarity(primary: np.ndarray, rbf: np.ndarray) -> dict[str, Any]:
    primary_mean = np.mean(primary, axis=2)
    rbf_mean = np.mean(rbf, axis=2)
    per_source: dict[str, dict[str, Any]] = {}
    for source_index, source in enumerate(ENVIRONMENTS):
        left = primary_mean[source_index]
        right = rbf_mean[source_index]
        cosine = _rowwise_cosine(left, right)
        flat_left, flat_right = left.reshape(-1), right.reshape(-1)
        scale = float(np.sqrt(np.mean(flat_left * flat_left)))
        per_source[source] = {
            "n_prediction_vectors": int(left.shape[0]),
            "mean_row_cosine": float(np.mean(cosine)),
            "median_row_cosine": float(np.median(cosine)),
            "q05_row_cosine": float(np.quantile(cosine, 0.05)),
            "q95_row_cosine": float(np.quantile(cosine, 0.95)),
            "min_row_cosine": float(np.min(cosine)),
            "flattened_pearson": _pearson(flat_left, flat_right),
            "normalized_rmse_by_primary_rms": float(
                np.sqrt(np.mean((flat_left - flat_right) ** 2)) / scale
            ) if scale > 0.0 else None,
        }
    all_left = primary_mean.reshape(-1)
    all_right = rbf_mean.reshape(-1)
    all_cosine = _rowwise_cosine(primary_mean.reshape(-1, primary_mean.shape[-1]), rbf_mean.reshape(-1, rbf_mean.shape[-1]))
    scale = float(np.sqrt(np.mean(all_left * all_left)))
    return {
        "comparison_unit": "matched source x perturbation mean prediction vector over the three frozen ensemble members",
        "per_source": per_source,
        "overall": {
            "n_prediction_vectors": int(all_cosine.size),
            "mean_row_cosine": float(np.mean(all_cosine)),
            "median_row_cosine": float(np.median(all_cosine)),
            "q05_row_cosine": float(np.quantile(all_cosine, 0.05)),
            "q95_row_cosine": float(np.quantile(all_cosine, 0.95)),
            "min_row_cosine": float(np.min(all_cosine)),
            "flattened_pearson": _pearson(all_left, all_right),
            "normalized_rmse_by_primary_rms": float(
                np.sqrt(np.mean((all_left - all_right) ** 2)) / scale
            ) if scale > 0.0 else None,
        },
    }


def _cross_family_correlations(paired: pd.DataFrame) -> dict[str, Any]:
    fields = [
        "d_adj", "same_context_floor", "stable_fraction_both",
        "stable_order_inversion_fraction", "crossfit_heldout_inversion_fraction",
        "source_target_mean_risk_change",
    ]
    overall: dict[str, Any] = {}
    by_metric: dict[str, dict[str, Any]] = {metric: {} for metric in METRICS}
    for field in fields:
        overall[field] = _correlation_stats(paired[f"{field}_bilinear"], paired[f"{field}_rbf"])
        for metric in METRICS:
            subset = paired.loc[paired["metric"].eq(metric)]
            by_metric[metric][field] = _correlation_stats(
                subset[f"{field}_bilinear"], subset[f"{field}_rbf"]
            )
    return {
        "overall_profile": overall,
        "by_metric": by_metric,
        "d_adj_profile_spearman": overall["d_adj"]["spearman"],
        "d_adj_metric_spearman": {
            metric: by_metric[metric]["d_adj"]["spearman"] for metric in METRICS
        },
    }


def _direction_consistency(
    paired: pd.DataFrame,
    *,
    include_by_metric: bool = True,
) -> dict[str, Any]:
    comparisons = {
        "d_adj_point_sign": (
            paired["d_adj_point_sign_bilinear"], paired["d_adj_point_sign_rbf"]
        ),
        "d_adj_ci_sign": (
            paired["d_adj_ci_sign_bilinear"], paired["d_adj_ci_sign_rbf"]
        ),
        "same_context_floor_ci_sign": (
            paired["same_context_floor_ci_sign_bilinear"], paired["same_context_floor_ci_sign_rbf"]
        ),
        "source_target_risk_direction": (
            paired["source_target_risk_direction_bilinear"], paired["source_target_risk_direction_rbf"]
        ),
    }
    output: dict[str, Any] = {"n_rows": int(len(paired))}
    for name, (left, right) in comparisons.items():
        output[name] = {
            "agree_count": int(np.sum(left.to_numpy() == right.to_numpy())),
            "agree_fraction": float(np.mean(left.to_numpy() == right.to_numpy())),
        }
    if include_by_metric:
        output["by_metric"] = {}
        for metric in METRICS:
            subset = paired.loc[paired["metric"].eq(metric)]
            output["by_metric"][metric] = _direction_consistency(
                subset, include_by_metric=False
            )
    return output


def _ci_sign_counts(table: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        field: {str(key): int(value) for key, value in table[field].value_counts().to_dict().items()}
        for field in ("d_adj_ci_sign", "same_context_floor_ci_sign")
    }


def _attach_cross_family_row_evidence(
    bilinear: pd.DataFrame,
    rbf: pd.DataFrame,
    similarity: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = bilinear.merge(
        rbf,
        on=KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_bilinear", "_rbf"),
    )
    if len(paired) != 18:
        raise ValueError(f"cross-family comparison must have 18 paired rows; found {len(paired)}")
    paired["d_adj_point_sign_agree"] = (
        paired["d_adj_point_sign_bilinear"] == paired["d_adj_point_sign_rbf"]
    )
    paired["d_adj_ci_sign_agree"] = (
        paired["d_adj_ci_sign_bilinear"] == paired["d_adj_ci_sign_rbf"]
    )
    paired["source_target_risk_direction_agree"] = (
        paired["source_target_risk_direction_bilinear"]
        == paired["source_target_risk_direction_rbf"]
    )
    per_source_similarity = similarity["per_source"]
    for table, family in ((bilinear, PRIMARY_FAMILY), (rbf, RBF_FAMILY)):
        table["predictor_family"] = family
        table["comparison_rows_per_family"] = 18
        table["prediction_vector_mean_row_cosine_by_source"] = [
            float(per_source_similarity[str(source)]["mean_row_cosine"])
            for source in table["source_environment_id"]
        ]
        table["cross_family_d_adj_point_sign_agree"] = [
            bool(paired.set_index(KEYS).loc[tuple(key), "d_adj_point_sign_agree"])
            for key in map(tuple, table[KEYS].to_records(index=False))
        ]
        table["cross_family_d_adj_ci_sign_agree"] = [
            bool(paired.set_index(KEYS).loc[tuple(key), "d_adj_ci_sign_agree"])
            for key in map(tuple, table[KEYS].to_records(index=False))
        ]
        table["cross_family_source_target_direction_agree"] = [
            bool(paired.set_index(KEYS).loc[tuple(key), "source_target_risk_direction_agree"])
            for key in map(tuple, table[KEYS].to_records(index=False))
        ]
    return bilinear, rbf, paired


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    root = root.resolve()
    primary_fit = _load_prediction(root / PRIMARY_PREDICTION, PRIMARY_FAMILY)
    rbf_fit = _load_prediction(root / RBF_PREDICTION, RBF_FAMILY)
    _validate_prediction_alignment(primary_fit, rbf_fit)

    primary_source_report = _read_json(root, PRIMARY_SOURCE_FROZEN_REPORT)
    if primary_source_report.get("status") != "source_frozen_primary_executed":
        raise ValueError("primary prediction report is not the executed source-frozen primary")
    if primary_source_report.get("prediction_npz") != PRIMARY_PREDICTION:
        raise ValueError("primary prediction report does not point to the canonical bilinear artifact")
    primary_measurement = _read_json(root, PRIMARY_MEASUREMENT)
    primary_min_cells = int(primary_measurement.get("min_cells_primary", -1))
    if primary_min_cells != 20 or primary_measurement.get("resampling_mode") != "full_size_nonparametric":
        raise ValueError("primary bilinear measurement is not the declared full-size min20 surface")
    rbf_evaluation = _read_json(root, RBF_EVALUATION)
    rbf_measurement = rbf_evaluation.get("measurement_report", {})
    if int(rbf_measurement.get("min_cells_primary", -1)) != 20 or rbf_measurement.get("resampling_mode") != "full_size_nonparametric":
        raise ValueError("RBF measurement is not the declared full-size min20 surface")
    rbf_fit_report = _read_json(root, RBF_FIT)
    alpha_extension = rbf_fit_report.get("alpha_extension", {})
    if not alpha_extension:
        raise ValueError("RBF fit report does not contain the source-only alpha audit")

    panel = primary_fit["evaluation_gene_symbols"]
    labels = primary_fit["perturbation_label"]
    common, _, truths = _load_common_surface(root, panel)
    if common != labels:
        raise ValueError("truth surface labels do not match the frozen prediction labels")

    primary_risks, primary_fidelity = _risk_surfaces(
        root, labels, panel, primary_fit["prediction"], truths
    )
    rbf_risks, rbf_fidelity = _risk_surfaces(
        root, labels, panel, rbf_fit["prediction"], truths
    )
    bilinear = _attach_prediction_evidence(
        _primary_transport_rows(root), primary_risks, primary_fidelity
    )
    rbf = _attach_prediction_evidence(
        _rbf_transport_rows(root), rbf_risks, rbf_fidelity
    )
    similarity = _prediction_similarity(primary_fit["prediction"], rbf_fit["prediction"])
    bilinear, rbf, paired = _attach_cross_family_row_evidence(bilinear, rbf, similarity)

    bilinear["measurement_surface"] = "primary_bilinear_full_size_min20"
    rbf["measurement_surface"] = "source_only_rbf_full_size_min20"
    comparison = pd.concat([bilinear, rbf], ignore_index=True, sort=False)
    comparison = comparison.sort_values(
        ["predictor_family", "source_environment_id", "target_environment_id", "metric"],
        kind="stable",
    ).reset_index(drop=True)
    comparison_path = root / OUTPUT_CSV
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)

    correlations = _cross_family_correlations(paired)
    directions = _direction_consistency(paired)
    source_oof_fidelity = {
        PRIMARY_FAMILY: {
            "by_source": primary_fidelity,
            "macro_by_metric": {
                metric: float(np.mean([primary_fidelity[source][metric] for source in ENVIRONMENTS]))
                for metric in METRICS
            },
        },
        RBF_FAMILY: {
            "by_source": rbf_fidelity,
            "macro_by_metric": {
                metric: float(np.mean([rbf_fidelity[source][metric] for source in ENVIRONMENTS]))
                for metric in METRICS
            },
        },
    }
    report = {
        "schema_version": 1,
        "status": "executed",
        "comparison_surface": {
            "rows_per_family": 18,
            "total_rows": int(len(comparison)),
            "directed_transfer_count": len(DIRECTED_PAIRS),
            "metric_count": len(METRICS),
            "metrics": list(METRICS),
            "common_perturbations": len(labels),
            "evaluation_genes": len(panel),
            "primary_min_cells": 20,
            "measurement_resampling": "full_size_nonparametric",
            "families": [PRIMARY_FAMILY, RBF_FAMILY],
        },
        "inputs": {
            "primary_prediction": PRIMARY_PREDICTION,
            "primary_source_frozen_report": PRIMARY_SOURCE_FROZEN_REPORT,
            "primary_measurement": PRIMARY_MEASUREMENT,
            "primary_summary": PRIMARY_SUMMARY,
            "primary_ordering": PRIMARY_ORDERING,
            "primary_floors": PRIMARY_FLOORS,
            "rbf_prediction": RBF_PREDICTION,
            "rbf_fit_report": RBF_FIT,
            "rbf_evaluation": RBF_EVALUATION,
            "rbf_d_adj": RBF_D_ADJ,
            "rbf_floor": RBF_FLOOR,
            "rbf_stable": RBF_STABLE,
        },
        "alpha_extension_audit": alpha_extension,
        "selection_independence": {
            "target_outcomes_used_for_alpha_tuning": False,
            "target_transport_used_for_family_selection": False,
            "prediction_vectors_frozen_before_target_evaluation": True,
            "source_oof_fidelity_and_transport_comparison_are_evaluation_only": True,
        },
        "row_comparison": {
            "csv": OUTPUT_CSV,
            "ci_sign_counts": {
                PRIMARY_FAMILY: _ci_sign_counts(bilinear),
                RBF_FAMILY: _ci_sign_counts(rbf),
            },
            "stable_inversion_fields": [
                "stable_fraction_source", "stable_fraction_target", "stable_fraction_both",
                "stable_order_inversion_fraction", "crossfit_heldout_inversion_fraction",
                "crossfit_heldout_evaluable_fraction",
            ],
        },
        "cross_family_profile_spearman_and_metric_correlations": correlations,
        "direction_consistency": directions,
        "source_oof_fidelity": source_oof_fidelity,
        "prediction_vector_similarity": similarity,
        "support_conclusion": {
            "supports": [
                "The source-only RBF family has a prespecified source-inner-CV alpha extension audit and a matched 18-row full-size min20 comparison surface.",
                "The two families can be compared descriptively on the same Frangieh labels, evaluation genes, metrics, transfer directions, and measurement-floor estimand.",
            ],
            "does_not_support": [
                "The audit does not support selecting RBF versus bilinear from target transport results; no such selection was performed.",
                "The audit does not establish predictor-family equivalence, superiority, causality, or generalization beyond this matched Frangieh surface.",
                "Spearman correlations and direction agreement are descriptive profile concordance, not a replacement for independent replication or a target-free model-selection criterion.",
            ],
            "stable_inversion_note": "Stable inversion fractions are reported as point diagnostics from the existing strict-support ordering artifact; no unsupported confidence interval is invented for them.",
        },
        "reproduction_commands": [
            "python scripts/run_frangieh_source_only_rbf_krr.py --measurement --draws 2000",
            "python scripts/build_predictor_family_transport_comparison.py",
        ],
    }
    report_path = root / OUTPUT_JSON
    report_path.write_text(
        json.dumps(_json_clean(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return comparison_path, report_path, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    comparison_path, report_path, report = build(args.root)
    print(comparison_path)
    print(report_path)
    print(json.dumps(_json_clean(report), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
