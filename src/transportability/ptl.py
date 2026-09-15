from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.run_baseline import DataRepository, parse_combination_components, prepare_split_data


STRESS_FAMILIES = {
    "unseen_perturbation_split",
    "unseen_combination_split",
    "low_support_split",
    "dataset_heldout_split",
    "external_holdout",
}

TARGET_COLUMNS = {
    "cosine_similarity",
    "pearson_r",
    "spearman_r",
    "rmse",
    "mae",
    "delta_norm_true",
    "log1p_delta_norm_true",
    "risk",
    "transportable",
    "target_transportable",
    "target_fidelity",
    "target_risk",
    "failure_mode",
    "transport_threshold",
    "top_deg_overlap_at_20",
    "top_deg_overlap_at_50",
    "top_deg_overlap_at_100",
    "deg_direction_consistency_at_20",
    "deg_direction_consistency_at_50",
    "deg_direction_consistency_at_100",
    "recall_at_20",
    "recall_at_50",
    "recall_at_100",
    "ndcg_at_20",
    "ndcg_at_50",
    "ndcg_at_100",
    "map_at_20",
    "map_at_50",
    "map_at_100",
    "pathway_cosine",
    "pathway_direction_consistency",
    "mean_pathway_cosine",
    "mean_pathway_direction_consistency",
    "perturbation_retrieval_top1",
    "perturbation_retrieval_top5",
    "perturbation_retrieval_top10",
}

IDENTIFIER_COLUMNS = {
    "run_id",
    "signature_id",
    "split_json_path",
    "output_dir",
    "log_file",
    "control_label_used",
}

CATEGORICAL_FEATURE_COLUMNS = {
    "model",
    "split_family",
    "dataset_id",
    "source_dataset",
    "dataset_scope",
    "heldout_target",
    "gene_space_policy",
    "batch",
    "timepoint",
}

DEPLOYMENT_NUMERIC_COLUMNS = {
    "confidence",
    "delta_norm_pred",
    "log1p_delta_norm_pred",
    "n_cells",
    "stress_family_flag",
    "reference_seen_in_train",
    "dataset_seen_in_train",
    "batch_seen_in_train",
    "timepoint_seen_in_train",
    "perturbation_seen_in_train",
    "is_combination",
    "component_count",
    "component_seen_fraction",
    "unseen_component_count",
    "perturbation_train_signature_count",
    "perturbation_train_cell_count",
    "reference_train_signature_count",
    "reference_train_cell_count",
    "dataset_train_signature_count",
    "dataset_train_cell_count",
    "log1p_perturbation_train_signature_count",
    "log1p_perturbation_train_cell_count",
    "log1p_reference_train_signature_count",
    "log1p_reference_train_cell_count",
    "log1p_dataset_train_signature_count",
    "log1p_dataset_train_cell_count",
    "train_units",
    "validation_units",
    "test_units",
    "train_perturbations",
    "validation_perturbations",
    "test_perturbations",
    "train_reference_keys",
    "validation_reference_keys",
    "test_reference_keys",
    "reference_key_overlap_train_test",
    "declared_holdout_overlap_train_test",
    "train_test_centroid_l2",
    "train_test_centroid_cosine",
    "n_test_non_control",
    "n_test_control",
    "bio_feature_available",
    "bio_pathway_support_count",
}

DEPLOYMENT_PREFIXES = (
    "log1p_",
    "bio_",
)

TARGET_OR_OUTCOME_TERMS = (
    "cosine",
    "pearson",
    "spearman",
    "rmse",
    "mae",
    "risk",
    "transport",
    "target_",
    "failure",
    "deg_",
    "recall_at_",
    "ndcg_at_",
    "map_at_",
    "pathway_cosine",
    "retrieval",
)


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]

    @property
    def all_columns(self) -> list[str]:
        return self.numeric + self.categorical


@dataclass(frozen=True)
class EvaluationSplit:
    train_index: np.ndarray
    test_index: np.ndarray


def _as_bool_series(series: pd.Series) -> pd.Series:
    return series.astype("boolean")


def _clean_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _safe_log1p(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(np.log1p(max(float(value), 0.0)))
    except (TypeError, ValueError):
        return 0.0


def perturbation_novelty_features(label: str, train_perturbations: set[str]) -> dict[str, float]:
    components = parse_combination_components(label)
    is_combination = bool(components)
    if not components and label and label.lower() != "control":
        components = [label]
    seen_components = sum(1 for component in components if component in train_perturbations)
    component_count = len(components)
    component_seen_fraction = seen_components / component_count if component_count else 0.0
    exact_seen = label in train_perturbations
    return {
        "perturbation_seen_in_train": float(exact_seen),
        "is_combination": float(is_combination),
        "component_count": float(component_count),
        "component_seen_fraction": float(component_seen_fraction),
        "unseen_component_count": float(component_count - seen_components),
    }


def build_target_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "transportable" not in output.columns:
        raise ValueError("PTL examples require a transportable label from Phase 06.")
    known = output["transportable"].notna()
    output = output.loc[known].copy()
    output["target_transportable"] = _as_bool_series(output["transportable"]).astype(int)
    output["target_fidelity"] = output["cosine_similarity"].astype(float)
    output["target_risk"] = output["risk"].astype(float)
    output["failure_mode"] = np.select(
        [
            output["target_transportable"].eq(1),
            output["target_risk"].ge(1.0),
            output["target_risk"].ge(0.75),
            output["target_fidelity"].lt(output["transport_threshold"].astype(float)),
        ],
        ["transportable", "severe_failure", "high_risk_failure", "below_transport_threshold"],
        default="non_transportable",
    )
    return output


def sample_examples(frame: pd.DataFrame, max_examples: int | None, random_state: int) -> pd.DataFrame:
    if max_examples is None or max_examples <= 0 or len(frame) <= max_examples:
        return frame.reset_index(drop=True)
    parts = []
    for _, part in frame.groupby("target_transportable", sort=False):
        take_n = max(1, min(len(part), int(round(max_examples * len(part) / len(frame)))))
        parts.append(part.sample(n=take_n, random_state=random_state))
    sampled = pd.concat(parts)
    if len(sampled) > max_examples:
        sampled = sampled.sample(n=max_examples, random_state=random_state)
    elif len(sampled) < max_examples:
        remainder = frame.drop(index=sampled.index, errors="ignore")
        if not remainder.empty:
            sampled = pd.concat([sampled, remainder.sample(n=min(max_examples - len(sampled), len(remainder)), random_state=random_state)])
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_manifest_context(manifest_path: Path, split_audit_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    audit = pd.read_csv(split_audit_path)
    audit = audit[audit["track"] == "signature"].copy()
    audit["split_json_path"] = audit["output_path"].map(lambda value: str(Path(str(value))))
    manifest["split_json_path"] = manifest["split_json_path"].map(lambda value: str(Path(str(value))))
    audit_columns = [
        "split_json_path",
        "train_units",
        "validation_units",
        "test_units",
        "train_perturbations",
        "validation_perturbations",
        "test_perturbations",
        "train_reference_keys",
        "validation_reference_keys",
        "test_reference_keys",
        "reference_key_overlap_train_test",
        "declared_holdout_overlap_train_test",
        "train_test_centroid_l2",
        "train_test_centroid_cosine",
    ]
    available = [column for column in audit_columns if column in audit.columns]
    merged = manifest.merge(audit[available], on="split_json_path", how="left", suffixes=("", "_split_audit"))
    return merged


def _build_support_features_for_split(repository: DataRepository, split_json_path: Path) -> pd.DataFrame:
    prepared = prepare_split_data(repository, split_json_path)
    train = prepared.train_frame.reset_index(drop=True)
    test = prepared.test_frame.reset_index(drop=True)
    non_control_train = train.loc[~train["is_control"].astype(bool)].copy()
    train_perturbations = set(non_control_train["perturbation_label"].astype(str))

    perturbation_counts = non_control_train.groupby("perturbation_label", sort=False).agg(
        perturbation_train_signature_count=("signature_id", "count"),
        perturbation_train_cell_count=("n_cells", "sum"),
    )
    reference_counts = train.groupby("reference_key", sort=False).agg(
        reference_train_signature_count=("signature_id", "count"),
        reference_train_cell_count=("n_cells", "sum"),
    )
    dataset_counts = train.groupby("dataset_id", sort=False).agg(
        dataset_train_signature_count=("signature_id", "count"),
        dataset_train_cell_count=("n_cells", "sum"),
    )
    batch_values = set(train["batch"].astype(str)) if "batch" in train.columns else set()
    timepoint_values = set(train["timepoint"].astype(str)) if "timepoint" in train.columns else set()

    rows: list[dict[str, Any]] = []
    for row in test.itertuples(index=False):
        perturbation = str(row.perturbation_label)
        reference_key = str(row.reference_key)
        dataset_id = str(row.dataset_id)
        features = {
            "signature_id": str(row.signature_id),
            "split_json_path": str(split_json_path),
            "reference_seen_in_train": float(reference_key in reference_counts.index),
            "dataset_seen_in_train": float(dataset_id in dataset_counts.index),
            "batch_seen_in_train": float(str(getattr(row, "batch", "")) in batch_values),
            "timepoint_seen_in_train": float(str(getattr(row, "timepoint", "")) in timepoint_values),
        }
        features.update(perturbation_novelty_features(perturbation, train_perturbations))
        if perturbation in perturbation_counts.index:
            p_count = perturbation_counts.loc[perturbation]
            features["perturbation_train_signature_count"] = float(p_count["perturbation_train_signature_count"])
            features["perturbation_train_cell_count"] = float(p_count["perturbation_train_cell_count"])
        else:
            features["perturbation_train_signature_count"] = 0.0
            features["perturbation_train_cell_count"] = 0.0
        if reference_key in reference_counts.index:
            r_count = reference_counts.loc[reference_key]
            features["reference_train_signature_count"] = float(r_count["reference_train_signature_count"])
            features["reference_train_cell_count"] = float(r_count["reference_train_cell_count"])
        else:
            features["reference_train_signature_count"] = 0.0
            features["reference_train_cell_count"] = 0.0
        if dataset_id in dataset_counts.index:
            d_count = dataset_counts.loc[dataset_id]
            features["dataset_train_signature_count"] = float(d_count["dataset_train_signature_count"])
            features["dataset_train_cell_count"] = float(d_count["dataset_train_cell_count"])
        else:
            features["dataset_train_signature_count"] = 0.0
            features["dataset_train_cell_count"] = 0.0
        rows.append(features)

    feature_frame = pd.DataFrame(rows)
    for column in [
        "perturbation_train_signature_count",
        "perturbation_train_cell_count",
        "reference_train_signature_count",
        "reference_train_cell_count",
        "dataset_train_signature_count",
        "dataset_train_cell_count",
    ]:
        feature_frame[f"log1p_{column}"] = feature_frame[column].map(_safe_log1p)
    return feature_frame


def build_ptl_examples(
    per_signature_metrics_path: Path,
    all_metrics_path: Path,
    manifest_path: Path,
    split_audit_path: Path,
    preprocessing_summary_path: Path,
    max_examples: int | None = None,
    random_state: int = 7,
    include_biology_features: bool = False,
) -> pd.DataFrame:
    per_signature = pd.read_parquet(per_signature_metrics_path)
    per_signature = per_signature[
        (~per_signature["is_control"].astype(bool))
        & per_signature["split_family"].isin(STRESS_FAMILIES)
        & per_signature["transportable"].notna()
    ].copy()
    per_signature = build_target_columns(per_signature)

    all_metrics = pd.read_csv(all_metrics_path)
    safe_run_columns = [
        "run_id",
        "anchor_mean_cosine",
        "anchor_signature_median_cosine",
        "n_test_non_control",
        "n_test_control",
    ]
    safe_run_columns = [column for column in safe_run_columns if column in all_metrics.columns]
    per_signature = per_signature.merge(all_metrics[safe_run_columns], on="run_id", how="left", suffixes=("", "_run"))

    manifest_context = load_manifest_context(manifest_path, split_audit_path)
    manifest_columns = [
        "run_id",
        "split_json_path",
        "dataset_scope",
        "heldout_target",
        "gene_space_policy",
        "expected_gene_count",
        "train_units",
        "validation_units",
        "test_units",
        "train_perturbations",
        "validation_perturbations",
        "test_perturbations",
        "train_reference_keys",
        "validation_reference_keys",
        "test_reference_keys",
        "reference_key_overlap_train_test",
        "declared_holdout_overlap_train_test",
        "train_test_centroid_l2",
        "train_test_centroid_cosine",
    ]
    manifest_columns = [column for column in manifest_columns if column in manifest_context.columns]
    per_signature = per_signature.merge(manifest_context[manifest_columns], on="run_id", how="left", suffixes=("", "_manifest"))
    per_signature["split_json_path"] = per_signature["split_json_path"].map(lambda value: str(Path(str(value))))

    repository = DataRepository(preprocessing_summary_path)
    support_frames = []
    for split_path in sorted(per_signature["split_json_path"].dropna().unique()):
        support_frames.append(_build_support_features_for_split(repository, Path(split_path)))
    support = pd.concat(support_frames, ignore_index=True) if support_frames else pd.DataFrame()
    if not support.empty:
        per_signature = per_signature.merge(support, on=["split_json_path", "signature_id"], how="left")

    per_signature["log1p_n_cells"] = per_signature["n_cells"].map(_safe_log1p)
    per_signature["log1p_gene_count"] = per_signature["gene_count"].map(_safe_log1p)
    per_signature["log1p_delta_norm_pred"] = per_signature["delta_norm_pred"].map(_safe_log1p)
    per_signature["stress_family_flag"] = per_signature["split_family"].isin(STRESS_FAMILIES).astype(float)

    if include_biology_features:
        per_signature["bio_feature_available"] = 0.0
        per_signature["bio_pathway_support_count"] = 0.0

    return sample_examples(per_signature, max_examples=max_examples, random_state=random_state)


def _feature_group(column: str) -> str:
    if column in CATEGORICAL_FEATURE_COLUMNS:
        return "model_split_dataset_indicator"
    if "centroid" in column or "reference" in column:
        return "context_distance"
    if "perturbation" in column or "component" in column or "combination" in column:
        return "perturbation_novelty"
    if "confidence" in column or "delta_norm_pred" in column:
        return "prediction_confidence_uncertainty"
    if "cell_count" in column or "signature_count" in column or column in {"n_cells", "train_units", "validation_units", "test_units", "n_test_non_control", "n_test_control"}:
        return "support_availability"
    if column.startswith("bio_") or "pathway" in column:
        return "optional_biology"
    return "other"


def classify_feature_column(column: str, series: pd.Series | None = None) -> dict[str, Any]:
    lower = column.lower()
    is_identifier = column in IDENTIFIER_COLUMNS or lower.endswith("_path") or lower.endswith("_file")
    is_target = column in TARGET_COLUMNS or any(term in lower for term in TARGET_OR_OUTCOME_TERMS)
    is_categorical = column in CATEGORICAL_FEATURE_COLUMNS
    is_numeric = bool(series is not None and (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)))
    deployment_available = (
        (is_categorical or column in DEPLOYMENT_NUMERIC_COLUMNS or any(column.startswith(prefix) for prefix in DEPLOYMENT_PREFIXES))
        and not is_identifier
        and not is_target
    )
    evaluation_only = (not deployment_available) and (not is_identifier) and (not is_target)
    if is_identifier:
        reason = "identifier excluded to avoid memorizing runs or signatures"
    elif is_target:
        reason = "target, label, or evaluation outcome excluded from PTL features"
    elif deployment_available:
        reason = "available at deployment from model output, split context, support, novelty, or optional annotation"
    else:
        reason = "not in deployment feature allowlist"
    if is_identifier:
        source_group = "identifier"
    elif is_target:
        source_group = "target_or_evaluation_outcome"
    else:
        source_group = _feature_group(column)
    return {
        "feature_name": column,
        "feature_type": "categorical" if is_categorical else ("numeric" if is_numeric else "non_feature"),
        "source_group": source_group,
        "deployment_available": bool(deployment_available),
        "evaluation_only": bool(evaluation_only),
        "target_or_label": bool(is_target),
        "identifier": bool(is_identifier),
        "ablation_group": _feature_group(column),
        "reason": reason,
    }


def build_feature_audit(frame: pd.DataFrame, feature_mode: str = "deployment", spec: FeatureSpec | None = None) -> pd.DataFrame:
    if feature_mode not in {"deployment", "all"}:
        raise ValueError("feature_mode must be 'deployment' or 'all'.")
    if spec is None:
        spec = infer_feature_spec(frame, feature_mode=feature_mode)
    selected = set(spec.all_columns)
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        row = classify_feature_column(column, frame[column])
        row["feature_mode"] = feature_mode
        row["excluded_from_ptl"] = column not in selected
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["excluded_from_ptl", "source_group", "feature_name"]).reset_index(drop=True)


def infer_feature_spec(frame: pd.DataFrame, feature_mode: str = "deployment") -> FeatureSpec:
    if feature_mode not in {"deployment", "all"}:
        raise ValueError("feature_mode must be 'deployment' or 'all'.")
    excluded = TARGET_COLUMNS | IDENTIFIER_COLUMNS
    numeric: list[str] = []
    categorical: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        audit = classify_feature_column(column, frame[column])
        if feature_mode == "deployment" and not audit["deployment_available"]:
            continue
        if column in CATEGORICAL_FEATURE_COLUMNS:
            categorical.append(column)
        elif pd.api.types.is_bool_dtype(frame[column]):
            numeric.append(column)
        elif pd.api.types.is_numeric_dtype(frame[column]):
            numeric.append(column)
    numeric = sorted(set(numeric))
    categorical = sorted(set(categorical))
    return FeatureSpec(numeric=numeric, categorical=categorical)


def apply_ablation(spec: FeatureSpec, ablation: str) -> FeatureSpec:
    drop_terms: tuple[str, ...]
    if ablation == "full_PTL":
        drop_terms = ()
    elif ablation == "no_context_distance":
        drop_terms = ("centroid", "reference_", "reference_key", "reference_seen", "train_reference")
    elif ablation == "no_perturbation_novelty":
        drop_terms = ("perturbation_", "component", "combination", "unseen_component")
    elif ablation == "no_uncertainty_features":
        drop_terms = ("confidence", "delta_norm_pred")
    elif ablation == "no_pathway_features":
        drop_terms = ("bio_", "pathway", "string", "reactome")
    else:
        raise ValueError(f"Unknown PTL ablation '{ablation}'.")

    def keep(column: str) -> bool:
        return not any(term in column for term in drop_terms)

    return FeatureSpec(
        numeric=[column for column in spec.numeric if keep(column)],
        categorical=[column for column in spec.categorical if keep(column)],
    )


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    transformers = []
    if spec.numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                spec.numeric,
            )
        )
    if spec.categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing", keep_empty_features=True)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
                    ]
                ),
                spec.categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_estimator(name: str, spec: FeatureSpec, random_state: int) -> Pipeline:
    if name == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    elif name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported estimator '{name}'.")
    return Pipeline([("preprocessor", build_preprocessor(spec)), ("classifier", classifier)])


def make_evaluation_split(frame: pd.DataFrame, random_state: int = 7) -> EvaluationSplit:
    y = frame["target_transportable"].to_numpy(dtype=int)
    groups = frame["run_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 4:
        for seed_offset in range(20):
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state + seed_offset)
            train_idx, test_idx = next(splitter.split(frame, y, groups))
            if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[test_idx])) == 2:
                return EvaluationSplit(train_index=train_idx, test_index=test_idx)
    stratify = y if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2 else None
    train_idx, test_idx = train_test_split(
        np.arange(len(frame)),
        test_size=0.25,
        random_state=random_state,
        stratify=stratify,
    )
    return EvaluationSplit(train_index=np.asarray(train_idx), test_index=np.asarray(test_idx))


def selective_profile(y_true: np.ndarray, risk: np.ndarray, scores: np.ndarray, coverages: Iterable[float] = (0.2, 0.4, 0.6, 0.8)) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    risk = np.asarray(risk, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    base_risk = float(np.mean(risk)) if len(risk) else float("nan")
    result: dict[str, float] = {"base_risk": base_risk}
    coverage_values: list[float] = []
    risk_values: list[float] = []
    for coverage in coverages:
        keep_n = max(1, int(math.ceil(coverage * len(order))))
        selected = order[:keep_n]
        observed_coverage = float(keep_n / len(order)) if len(order) else float("nan")
        selective_risk = float(np.mean(risk[selected])) if len(selected) else float("nan")
        false_transportability = float(np.mean(y_true[selected] == 0)) if len(selected) else float("nan")
        label = str(coverage).replace(".", "p")
        result[f"coverage_at_{label}"] = observed_coverage
        result[f"selective_risk_at_{label}"] = selective_risk
        result[f"false_transportability_rate_at_{label}"] = false_transportability
        result[f"abstention_gain_at_{label}"] = base_risk - selective_risk
        coverage_values.append(observed_coverage)
        risk_values.append(selective_risk)
    result["mean_selective_risk"] = float(np.nanmean(risk_values)) if risk_values else float("nan")
    result["mean_abstention_gain"] = base_risk - result["mean_selective_risk"]
    false_rate_columns = [
        value
        for key, value in result.items()
        if key.startswith("false_transportability_rate_at_") and not math.isnan(float(value))
    ]
    result["mean_false_transportability_rate"] = float(np.mean(false_rate_columns)) if false_rate_columns else float("nan")
    order_cov = np.argsort(coverage_values)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    result["area_under_selective_risk_curve"] = float(
        trapezoid(np.asarray(risk_values, dtype=float)[order_cov], np.asarray(coverage_values, dtype=float)[order_cov])
    )
    return result


def score_binary_classifier(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= 0.5).astype(int)
    output = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier": float(brier_score_loss(y_true, np.clip(scores, 0.0, 1.0))),
    }
    if len(np.unique(y_true)) == 2:
        output["roc_auc"] = float(roc_auc_score(y_true, scores))
        output["average_precision"] = float(average_precision_score(y_true, scores))
    else:
        output["roc_auc"] = float("nan")
        output["average_precision"] = float("nan")
    return output


def evaluate_scores(y_true: np.ndarray, risk: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    output = score_binary_classifier(y_true, scores)
    output.update(selective_profile(y_true=y_true, risk=risk, scores=scores))
    return output


def build_prediction_rows(
    examples: pd.DataFrame,
    test_index: np.ndarray,
    ablation: str,
    estimator: str,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scored = examples.iloc[test_index].reset_index(drop=False).rename(columns={"index": "source_index"})
    for row, score in zip(scored.itertuples(index=False), scores):
        rows.append(
            {
                "example_id": f"{row.run_id}::{row.signature_id}::{int(row.source_index)}",
                "ablation": ablation,
                "estimator": estimator,
                "y_true": int(row.target_transportable),
                "score": float(score),
                "split_family": str(row.split_family),
                "model": str(row.model),
                "signature_id": str(row.signature_id),
                "run_id": str(row.run_id),
                "target_risk": float(row.target_risk),
                "confidence": float(row.confidence) if not pd.isna(row.confidence) else 0.0,
                "failure_mode": str(row.failure_mode),
            }
        )
    return rows


def run_ptl_ablation_suite(
    examples: pd.DataFrame,
    random_state: int = 7,
    ablations: Iterable[str] = ("full_PTL", "no_context_distance", "no_perturbation_novelty", "no_uncertainty_features", "no_pathway_features"),
    feature_mode: str = "deployment",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if examples.empty:
        raise ValueError("No PTL examples available after filtering.")
    if examples["target_transportable"].nunique() < 2:
        raise ValueError("PTL requires both transportable and non-transportable examples.")

    split = make_evaluation_split(examples, random_state=random_state)
    y_test = examples.iloc[split.test_index]["target_transportable"].to_numpy(dtype=int)
    risk_test = examples.iloc[split.test_index]["target_risk"].to_numpy(dtype=float)

    base_spec = infer_feature_spec(examples, feature_mode=feature_mode)
    metrics_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    naive_scores = examples.iloc[split.test_index]["confidence"].fillna(0.0).to_numpy(dtype=float)
    naive_metrics = evaluate_scores(y_test, risk_test, naive_scores)
    metrics_rows.append({"ablation": "naive_confidence", "estimator": "naive_confidence", "n_features": 1, **naive_metrics})
    prediction_rows.extend(
        build_prediction_rows(
            examples=examples,
            test_index=split.test_index,
            ablation="naive_confidence",
            estimator="naive_confidence",
            scores=naive_scores,
        )
    )
    naive_mean_risk = naive_metrics["mean_selective_risk"]
    naive_false_rate = naive_metrics["mean_false_transportability_rate"]

    for ablation in ablations:
        spec = apply_ablation(base_spec, ablation)
        if ablation == "no_pathway_features" and not any(column.startswith("bio_") or "pathway" in column for column in base_spec.all_columns):
            ablation_rows.append(
                {
                    "ablation": ablation,
                    "status": "skipped",
                    "reason": "optional biological support features were unavailable",
                    "best_estimator": "",
                    "best_mean_selective_risk": float("nan"),
                    "mean_selective_risk_gain_vs_naive": float("nan"),
                    "n_features": 0,
                }
            )
            continue
        for estimator_name in ("logistic_regression", "random_forest"):
            estimator = make_estimator(estimator_name, spec, random_state=random_state)
            estimator.fit(examples.iloc[split.train_index][spec.all_columns], examples.iloc[split.train_index]["target_transportable"].astype(int))
            scores = estimator.predict_proba(examples.iloc[split.test_index][spec.all_columns])[:, 1]
            row = {
                "ablation": ablation,
                "estimator": estimator_name,
                "n_features": len(spec.all_columns),
                **evaluate_scores(y_test, risk_test, scores),
            }
            row["mean_selective_risk_gain_vs_naive"] = naive_mean_risk - row["mean_selective_risk"]
            row["mean_false_transportability_rate_gain_vs_naive"] = naive_false_rate - row["mean_false_transportability_rate"]
            metrics_rows.append(row)
            prediction_rows.extend(
                build_prediction_rows(
                    examples=examples,
                    test_index=split.test_index,
                    ablation=ablation,
                    estimator=estimator_name,
                    scores=scores,
                )
            )

    metrics = pd.DataFrame(metrics_rows)
    for ablation, group in metrics[metrics["ablation"] != "naive_confidence"].groupby("ablation", sort=False):
        best = group.sort_values(
            ["mean_false_transportability_rate", "mean_selective_risk", "brier"],
            ascending=[True, True, True],
        ).iloc[0]
        ablation_rows.append(
            {
                "ablation": ablation,
                "status": "completed",
                "reason": "",
                "best_estimator": best["estimator"],
                "best_mean_selective_risk": float(best["mean_selective_risk"]),
                "mean_selective_risk_gain_vs_naive": float(best["mean_selective_risk_gain_vs_naive"]),
                "best_mean_false_transportability_rate": float(best["mean_false_transportability_rate"]),
                "mean_false_transportability_rate_gain_vs_naive": float(best["mean_false_transportability_rate_gain_vs_naive"]),
                "n_features": int(best["n_features"]),
                "roc_auc": float(best["roc_auc"]),
                "average_precision": float(best["average_precision"]),
                "brier": float(best["brier"]),
            }
        )
    ablation_summary = pd.DataFrame(ablation_rows).sort_values(["status", "ablation"]).reset_index(drop=True)
    failure_summary = build_failure_mode_summary(examples)
    predictions = pd.DataFrame(prediction_rows)
    return metrics.reset_index(drop=True), ablation_summary, failure_summary, predictions.reset_index(drop=True)


def build_failure_mode_summary(examples: pd.DataFrame) -> pd.DataFrame:
    grouped = examples.groupby(["model", "split_family", "failure_mode"], dropna=False)
    return grouped.agg(
        n_signatures=("signature_id", "count"),
        transportable_rate=("target_transportable", "mean"),
        mean_cosine=("target_fidelity", "mean"),
        mean_risk=("target_risk", "mean"),
        mean_confidence=("confidence", "mean"),
        mean_n_cells=("n_cells", "mean"),
        mean_perturbation_seen=("perturbation_seen_in_train", "mean"),
        mean_component_seen_fraction=("component_seen_fraction", "mean"),
        mean_reference_seen=("reference_seen_in_train", "mean"),
    ).reset_index().sort_values(["split_family", "model", "failure_mode"]).reset_index(drop=True)


def write_json_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
