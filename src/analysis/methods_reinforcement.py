from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transportability.ptl import (  # noqa: E402
    FeatureSpec,
    evaluate_scores,
    make_estimator,
    make_evaluation_split,
    run_ptl_ablation_suite,
)


TABLES_DIR = ROOT / "results" / "tables"
LOG_DIR = ROOT / "results" / "logs" / "analysis"
MODELS_DIR = ROOT / "results" / "models"

THRESHOLDS = (0.6, 0.7, 0.8, 0.9)
RANDOM_STATE = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Methods reinforcement analyses.")
    parser.add_argument("--ptl-examples", default=str(TABLES_DIR / "ptl_examples.parquet"))
    parser.add_argument("--ptl-predictions", default=str(TABLES_DIR / "ptl_oof_predictions.csv"))
    parser.add_argument("--pathway-summary", default=str(TABLES_DIR / "pathway_metric_summary.csv"))
    parser.add_argument("--retrieval-summary", default=str(TABLES_DIR / "perturbation_discrimination_summary.csv"))
    parser.add_argument("--gears-matrix", default=str(TABLES_DIR / "perturbation_model_run_matrix.csv"))
    parser.add_argument("--threshold-output", default=str(TABLES_DIR / "threshold_sensitivity_summary.csv"))
    parser.add_argument("--baseline-output", default=str(TABLES_DIR / "reliability_baseline_summary.csv"))
    parser.add_argument("--biology-output", default=str(TABLES_DIR / "ptl_retention_biology_summary.csv"))
    parser.add_argument("--gears-output", default=str(TABLES_DIR / "gears_validation_summary.csv"))
    parser.add_argument("--log-file", default=str(LOG_DIR / "methods_reinforcement.log"))
    parser.add_argument("--max-examples", type=int, default=0)
    return parser.parse_args()


def log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def load_examples(path: Path, max_examples: int = 0) -> pd.DataFrame:
    examples = pd.read_parquet(path)
    if max_examples and len(examples) > max_examples:
        examples = (
            examples.groupby("target_transportable", group_keys=False)
            .apply(lambda frame: frame.sample(min(len(frame), max(1, int(max_examples * len(frame) / len(examples)))), random_state=RANDOM_STATE))
            .sample(frac=1.0, random_state=RANDOM_STATE)
            .head(max_examples)
            .reset_index(drop=True)
        )
    return examples.reset_index(drop=True)


def apply_relative_threshold(examples: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    frame = examples.copy()
    anchor = frame["anchor_signature_median_cosine"].fillna(frame["anchor_signature_median_cosine_run"])
    threshold = multiplier * anchor.astype(float)
    frame["transport_threshold"] = threshold
    frame["target_transportable"] = (frame["target_fidelity"].astype(float) >= threshold).astype(int)
    frame["transportable"] = frame["target_transportable"].astype(bool)
    frame["target_risk"] = 1.0 - frame["target_fidelity"].astype(float)
    frame["failure_mode"] = np.select(
        [
            frame["target_transportable"].eq(1),
            frame["target_risk"].ge(1.0),
            frame["target_risk"].ge(0.75),
            frame["target_fidelity"].astype(float).lt(frame["transport_threshold"].astype(float)),
        ],
        ["transportable", "severe_failure", "high_risk_failure", "below_transport_threshold"],
        default="non_transportable",
    )
    return frame


def build_threshold_sensitivity(examples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for multiplier in THRESHOLDS:
        thresholded = apply_relative_threshold(examples, multiplier)
        metrics, _, _, _ = run_ptl_ablation_suite(
            thresholded,
            random_state=RANDOM_STATE,
            ablations=("full_PTL", "no_context_distance"),
            feature_mode="deployment",
        )
        for row in metrics.itertuples(index=False):
            rows.append(
                {
                    "threshold_multiplier": multiplier,
                    "ablation": row.ablation,
                    "estimator": row.estimator,
                    "transportable_rate": float(thresholded["target_transportable"].mean()),
                    "accuracy": float(row.accuracy),
                    "roc_auc": float(row.roc_auc),
                    "average_precision": float(row.average_precision),
                    "mean_false_transportability_rate": float(row.mean_false_transportability_rate),
                    "mean_selective_risk": float(row.mean_selective_risk),
                    "coverage_at_0p8": float(row.coverage_at_0p8),
                    "false_transportability_rate_at_0p8": float(row.false_transportability_rate_at_0p8),
                }
            )
    return pd.DataFrame(rows)


def _numeric_existing(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column])]


def _categorical_existing(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _evaluate_model_baseline(examples: pd.DataFrame, name: str, spec: FeatureSpec, estimator_name: str) -> dict[str, Any]:
    split = make_evaluation_split(examples, random_state=RANDOM_STATE)
    y_test = examples.iloc[split.test_index]["target_transportable"].to_numpy(dtype=int)
    risk_test = examples.iloc[split.test_index]["target_risk"].to_numpy(dtype=float)
    model = make_estimator(estimator_name, spec, random_state=RANDOM_STATE)
    model.fit(examples.iloc[split.train_index][spec.all_columns], examples.iloc[split.train_index]["target_transportable"].astype(int))
    scores = model.predict_proba(examples.iloc[split.test_index][spec.all_columns])[:, 1]
    metrics = evaluate_scores(y_test, risk_test, scores)
    return {"baseline": name, "estimator": estimator_name, "n_features": len(spec.all_columns), **metrics}


def _evaluate_score_baseline(examples: pd.DataFrame, name: str, score_column: str = "confidence") -> dict[str, Any]:
    split = make_evaluation_split(examples, random_state=RANDOM_STATE)
    y_test = examples.iloc[split.test_index]["target_transportable"].to_numpy(dtype=int)
    risk_test = examples.iloc[split.test_index]["target_risk"].to_numpy(dtype=float)
    scores = examples.iloc[split.test_index][score_column].fillna(0.0).to_numpy(dtype=float)
    metrics = evaluate_scores(y_test, risk_test, scores)
    return {"baseline": name, "estimator": score_column, "n_features": 1, **metrics}


def _evaluate_conformal_proxy(examples: pd.DataFrame) -> dict[str, Any]:
    split = make_evaluation_split(examples, random_state=RANDOM_STATE)
    train = examples.iloc[split.train_index]
    test = examples.iloc[split.test_index]
    nonconformity = 1.0 - train["confidence"].fillna(0.0).to_numpy(dtype=float)
    cutoff = float(np.quantile(nonconformity, 0.2))
    scores = 1.0 - np.maximum(0.0, (1.0 - test["confidence"].fillna(0.0).to_numpy(dtype=float)) - cutoff)
    scores = np.clip(scores, 0.0, 1.0)
    metrics = evaluate_scores(
        test["target_transportable"].to_numpy(dtype=int),
        test["target_risk"].to_numpy(dtype=float),
        scores,
    )
    return {"baseline": "conformal_style_confidence_proxy", "estimator": "confidence_quantile", "n_features": 1, **metrics}


def build_reliability_baselines(examples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        _evaluate_score_baseline(examples, "naive_confidence"),
        _evaluate_conformal_proxy(examples),
    ]
    specs = {
        "split_family_only": FeatureSpec(numeric=[], categorical=_categorical_existing(examples, ["split_family"])),
        "support_only": FeatureSpec(
            numeric=_numeric_existing(
                examples,
                [
                    "n_cells",
                    "perturbation_train_signature_count",
                    "perturbation_train_cell_count",
                    "reference_train_signature_count",
                    "reference_train_cell_count",
                    "dataset_train_signature_count",
                    "dataset_train_cell_count",
                ],
            ),
            categorical=[],
        ),
        "novelty_only": FeatureSpec(
            numeric=_numeric_existing(
                examples,
                [
                    "perturbation_seen_in_train",
                    "is_combination",
                    "component_count",
                    "component_seen_fraction",
                    "unseen_component_count",
                    "reference_seen_in_train",
                    "dataset_seen_in_train",
                ],
            ),
            categorical=[],
        ),
        "support_plus_novelty": FeatureSpec(
            numeric=_numeric_existing(
                examples,
                [
                    "n_cells",
                    "perturbation_train_signature_count",
                    "perturbation_train_cell_count",
                    "reference_train_signature_count",
                    "reference_train_cell_count",
                    "dataset_train_signature_count",
                    "dataset_train_cell_count",
                    "perturbation_seen_in_train",
                    "is_combination",
                    "component_count",
                    "component_seen_fraction",
                    "unseen_component_count",
                    "reference_seen_in_train",
                    "dataset_seen_in_train",
                ],
            ),
            categorical=[],
        ),
        "calibrated_logistic_deployment": FeatureSpec(
            numeric=_numeric_existing(
                examples,
                [
                    "confidence",
                    "delta_norm_pred",
                    "n_cells",
                    "perturbation_train_signature_count",
                    "reference_train_signature_count",
                    "dataset_train_signature_count",
                    "perturbation_seen_in_train",
                    "component_seen_fraction",
                    "unseen_component_count",
                    "train_test_centroid_l2",
                ],
            ),
            categorical=_categorical_existing(examples, ["model", "split_family", "dataset_id", "dataset_scope"]),
        ),
        "full_PTL_random_forest": FeatureSpec(
            numeric=_numeric_existing(
                examples,
                [
                    "confidence",
                    "delta_norm_pred",
                    "n_cells",
                    "stress_family_flag",
                    "reference_seen_in_train",
                    "dataset_seen_in_train",
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
                    "train_units",
                    "test_units",
                    "train_perturbations",
                    "test_perturbations",
                    "reference_key_overlap_train_test",
                    "declared_holdout_overlap_train_test",
                    "train_test_centroid_l2",
                    "train_test_centroid_cosine",
                ],
            ),
            categorical=_categorical_existing(examples, ["model", "split_family", "dataset_id", "dataset_scope", "heldout_target", "gene_space_policy"]),
        ),
    }
    for name, spec in specs.items():
        if not spec.all_columns:
            continue
        estimator = "random_forest" if name == "full_PTL_random_forest" else "logistic_regression"
        rows.append(_evaluate_model_baseline(examples, name, spec, estimator))
    output = pd.DataFrame(rows)
    naive = output.loc[output["baseline"].eq("naive_confidence"), "mean_false_transportability_rate"].iloc[0]
    output["false_transportability_gain_vs_naive"] = naive - output["mean_false_transportability_rate"]
    return output.sort_values(["mean_false_transportability_rate", "baseline"]).reset_index(drop=True)


def build_retention_biology(examples: pd.DataFrame, predictions_path: Path, pathway_path: Path, retrieval_path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    keep_scores = predictions[
        predictions["ablation"].eq("full_PTL") & predictions["estimator"].eq("random_forest")
    ][["run_id", "signature_id", "score"]].copy()
    if keep_scores.empty:
        keep_scores = predictions[
            predictions["ablation"].eq("no_context_distance") & predictions["estimator"].eq("random_forest")
        ][["run_id", "signature_id", "score"]].copy()
    merged = examples.merge(keep_scores, on=["run_id", "signature_id"], how="inner")
    merged["ptl_decision"] = np.where(merged["score"].ge(0.5), "retained", "rejected_high_risk")
    pathway = pd.read_csv(pathway_path)
    retrieval = pd.read_csv(retrieval_path)
    merged = merged.merge(pathway[["run_id", "mean_pathway_cosine", "mean_pathway_direction_consistency"]], on="run_id", how="left")
    merged = merged.merge(retrieval[["run_id", "perturbation_retrieval_top1", "perturbation_retrieval_top5"]], on="run_id", how="left")
    metrics = [
        "target_fidelity",
        "deg_direction_consistency_at_50",
        "mean_pathway_cosine",
        "mean_pathway_direction_consistency",
        "perturbation_retrieval_top1",
        "perturbation_retrieval_top5",
    ]
    rows: list[dict[str, Any]] = []
    for decision, group in merged.groupby("ptl_decision"):
        row: dict[str, Any] = {
            "ptl_decision": decision,
            "n_signatures": int(len(group)),
            "mean_ptl_score": float(group["score"].mean()),
            "transportable_rate": float(group["target_transportable"].mean()),
        }
        for metric in metrics:
            if metric in group.columns:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_gears(matrix_path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(matrix_path)
    rows: list[dict[str, Any]] = []
    for row in matrix.itertuples(index=False):
        output_dir = Path(row.output_dir)
        metrics_path = output_dir / "run_metrics.json"
        failure_path = output_dir / "adapter_failure.json"
        base = {
            "run_id": row.run_id,
            "dataset_scope": row.dataset_scope,
            "split_family": row.split_family,
            "seed": int(row.seed),
            "output_dir": str(output_dir),
        }
        if metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            details = payload.get("model_details", {})
            rows.append(
                {
                    **base,
                    "status": "success",
                    "mean_cosine_non_control": metrics.get("mean_cosine_similarity_non_control_test", math.nan),
                    "n_test_noncontrol": metrics.get("n_test_noncontrol", math.nan),
                    "gene_count": metrics.get("gene_count", payload.get("gene_count", math.nan)),
                    "epochs": details.get("epochs", math.nan),
                    "graph_adapter": details.get("graph_adapter", ""),
                    "missing_graph_conditions": details.get("n_missing_perturbation_graph_conditions", 0),
                }
            )
        elif failure_path.exists():
            payload = json.loads(failure_path.read_text(encoding="utf-8"))
            rows.append({**base, "status": "failed", "error": payload.get("error", "")})
        else:
            rows.append({**base, "status": "missing"})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    log(log_path, "Starting Methods reinforcement analyses.")
    examples = load_examples(Path(args.ptl_examples), max_examples=args.max_examples)
    log(log_path, f"Loaded PTL examples: rows={len(examples)}.")

    threshold = build_threshold_sensitivity(examples)
    Path(args.threshold_output).parent.mkdir(parents=True, exist_ok=True)
    threshold.to_csv(args.threshold_output, index=False)
    log(log_path, f"Wrote threshold sensitivity rows={len(threshold)} to {args.threshold_output}.")

    baselines = build_reliability_baselines(examples)
    baselines.to_csv(args.baseline_output, index=False)
    log(log_path, f"Wrote reliability baseline rows={len(baselines)} to {args.baseline_output}.")

    biology = build_retention_biology(examples, Path(args.ptl_predictions), Path(args.pathway_summary), Path(args.retrieval_summary))
    biology.to_csv(args.biology_output, index=False)
    log(log_path, f"Wrote retention biology rows={len(biology)} to {args.biology_output}.")

    gears = summarize_gears(Path(args.gears_matrix))
    gears.to_csv(args.gears_output, index=False)
    log(log_path, f"Wrote GEARS validation summary rows={len(gears)} to {args.gears_output}.")
    log(log_path, "Methods reinforcement analyses completed.")


if __name__ == "__main__":
    main()
