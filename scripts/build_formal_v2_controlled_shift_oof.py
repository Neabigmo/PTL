"""Build a full-surface, perturbation-label OOF Frangieh shift audit.

This is a separate analysis estimand from the canonical 60/20/20 formal split.
Each Frangieh condition is scored by five-fold perturbation-label cross-fitting;
the strong-linear predictor and its three bootstrap members are fit inside each
fold.  The resulting surface is used only for the matched-condition diagnostic,
never to overwrite the canonical formal-v2 predictions or PTL artifacts.
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

from scripts.run_formal_v2_predictors import (  # noqa: E402
    fit_ahlmann_eltze_bilinear_ridge,
    load_panel,
    load_registry,
    METADATA_COLUMNS,
    read_ground_truth,
    read_training_post_expression,
)
from scripts.run_formal_v2_reliability_reordering import (  # noqa: E402
    FRANGIEH_PAIRS,
    _pairwise_reordering,
    _response_distance,
    _response_ground_truth,
)
from src.evaluation.metrics import safe_rowwise_cosine  # noqa: E402
from src.ptl.uncertainty.uq import summarize_ensemble  # noqa: E402


ENVIRONMENTS = (
    "frangieh_melanoma_control",
    "frangieh_melanoma_coculture",
    "frangieh_melanoma_ifng",
)
SPLIT_SEED = 20260908
MODEL_SEEDS = (0, 1, 2)


def _fold_indices(labels: list[str], *, folds: int, seed: int) -> list[np.ndarray]:
    if folds < 2 or len(labels) < folds:
        raise ValueError("cross-fit folds must be at least two and no larger than the label count")
    order = np.random.default_rng(seed).permutation(len(labels))
    return [np.asarray(part, dtype=int) for part in np.array_split(order, folds)]


def _environment_rows(root: Path, environment_key: str) -> tuple[pd.DataFrame, pd.Series]:
    registry = load_registry(root)
    registry_row = registry.loc[registry["environment_key"].eq(environment_key)]
    if len(registry_row) != 1:
        raise ValueError(f"expected one registry row for {environment_key}")
    manifest = pd.read_csv(root / "artifacts/manifests/biological_instance_registry.csv", dtype=str)
    manifest_row = manifest.loc[manifest["environment_key"].eq(environment_key)]
    if manifest_row.empty:
        raise ValueError(f"formal manifest has no rows for {environment_key}")
    source = str(manifest_row["ground_truth_path"].iloc[0])
    ground_truth = read_ground_truth(root, source, genes=None)
    environment_id = str(registry_row["environment_id"].iloc[0])
    frame = ground_truth.loc[ground_truth["environment_id"].astype(str).eq(environment_id)].copy()
    if frame.empty or frame["perturbation_label"].astype(str).duplicated().any():
        raise ValueError(f"invalid Frangieh ground-truth surface for {environment_key}")
    return frame, registry_row.iloc[0]


def _score_environment(root: Path, environment_key: str, panel: list[str], *, folds: int) -> pd.DataFrame:
    frame, registry_row = _environment_rows(root, environment_key)
    metadata = {
        "environment_key": environment_key,
        "environment_id": str(registry_row["environment_id"]),
        "dataset_id": str(registry_row["dataset_id"]),
        "condition": str(registry_row["condition"]),
        "condition_field": str(registry_row["condition_field"]),
    }
    native_genes = [column for column in frame.columns if column not in METADATA_COLUMNS]
    missing = set(panel).difference(native_genes)
    if missing:
        raise ValueError(f"{environment_key} ground truth misses evaluation genes: {sorted(missing)[:5]}")
    eval_indices = [native_genes.index(gene) for gene in panel]
    labels = frame["perturbation_label"].astype(str).to_numpy()
    native_values = frame[native_genes].to_numpy(dtype=np.float32, copy=True)
    weights = pd.to_numeric(frame["total_cells"], errors="coerce").fillna(1).to_numpy(dtype=np.float64)
    folds_indices = _fold_indices(labels.tolist(), folds=folds, seed=SPLIT_SEED + sum(ord(c) for c in environment_key))
    rows: list[dict[str, Any]] = []
    for fold_id, test_indices in enumerate(folds_indices):
        train_mask = np.ones(len(frame), dtype=bool)
        train_mask[test_indices] = False
        train_indices = np.flatnonzero(train_mask)
        train_labels = labels[train_indices]
        train_label_set = set(train_labels.tolist())
        raw_condition_field = registry_row.get("condition_field", "")
        raw_condition = registry_row.get("condition", "")
        condition_field = "" if pd.isna(raw_condition_field) else str(raw_condition_field).strip()
        condition = "" if pd.isna(raw_condition) else str(raw_condition).strip()
        post_by_label, _ = read_training_post_expression(
            root,
            str(registry_row["dataset_id"]),
            train_label_set,
            native_genes,
            condition_field=condition_field,
            condition=condition,
        )
        members: list[np.ndarray] = []
        query_labels = labels[test_indices]
        for model_seed in MODEL_SEEDS:
            rng = np.random.default_rng(model_seed + 1009 * (sum(ord(c) for c in environment_key) + 1) + 100000 * fold_id)
            bootstrap = rng.choice(train_indices, size=len(train_indices), replace=True)
            bootstrap_labels = labels[bootstrap]
            bootstrap_native_values = native_values[bootstrap]
            post_train_values = np.stack([post_by_label[str(label)] for label in bootstrap_labels])
            prediction = fit_ahlmann_eltze_bilinear_ridge(
                bootstrap_labels,
                bootstrap_native_values,
                query_labels,
                gene_names=native_genes,
                post_train_values=post_train_values,
                alpha=0.1,
                n_components=10,
            )[:, eval_indices]
            members.append(prediction)
        stack = np.stack(members, axis=0)
        mean_prediction = stack.mean(axis=0)
        uq = summarize_ensemble(stack)
        risk = 1.0 - safe_rowwise_cosine(native_values[test_indices][:, eval_indices], mean_prediction)
        confidence = 1.0 - uq.cosine_disagreement
        for row_index, biological_index in enumerate(test_indices):
            rows.append({
                **metadata,
                "perturbation_label": str(labels[biological_index]),
                "oof_fold": int(fold_id),
                "model_seeds": ",".join(str(seed) for seed in MODEL_SEEDS),
                "continuous_risk": float(risk[row_index]),
                "confidence": float(confidence[row_index]),
                "uq_cosine_disagreement": float(uq.cosine_disagreement[row_index]),
                "uq_mean_gene_variance": float(uq.mean_gene_variance[row_index]),
                "total_cells": float(weights[biological_index]),
                "predictor": "strong_linear",
                "prediction_protocol": "five_fold_perturbation_label_OOF",
                "outcomes_used_for_training": 0,
                "outcomes_used_for_confidence": 0,
            })
    result = pd.DataFrame(rows)
    if result["perturbation_label"].duplicated().any():
        raise ValueError(f"OOF surface contains duplicate labels for {environment_key}")
    return result


def _build_pair_summary(root: Path, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    response, genes = _response_ground_truth(root)
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for left_environment, right_environment in FRANGIEH_PAIRS:
        left = oof.loc[oof["environment_key"].eq(left_environment)].rename(columns={"confidence": "confidence_left", "continuous_risk": "risk_left"})
        right = oof.loc[oof["environment_key"].eq(right_environment)].rename(columns={"confidence": "confidence_right", "continuous_risk": "risk_right"})
        merged = left.merge(right, on="perturbation_label", how="inner", validate="one_to_one", suffixes=("_left", "_right"))
        stats = _pairwise_reordering(merged.rename(columns={"risk_left": "risk_left", "risk_right": "risk_right"}), "confidence")
        shifts: list[float] = []
        changes: list[float] = []
        for _, row in merged.iterrows():
            label = str(row["perturbation_label"])
            left_vector = response.loc[(left_environment, label), genes].to_numpy(dtype=float)
            right_vector = response.loc[(right_environment, label), genes].to_numpy(dtype=float)
            shift, _ = _response_distance(left_vector, right_vector)
            change = abs(float(row["risk_left"]) - float(row["risk_right"]))
            shifts.append(shift)
            changes.append(change)
            detail_rows.append({
                "comparison": "frangieh_condition_oof",
                "left_environment_id": left_environment,
                "right_environment_id": right_environment,
                "perturbation_label": label,
                "risk_left": float(row["risk_left"]),
                "risk_right": float(row["risk_right"]),
                "confidence_left": float(row["confidence_left"]),
                "confidence_right": float(row["confidence_right"]),
                "absolute_risk_change": change,
                "response_program_shift": shift,
                "response_program_is_descriptive": 1,
                "risk_and_confidence_outcomes_used_for_evaluation_only": 1,
                "matching": "exact_perturbation_label_intersection",
            })
        stats.update({
            "comparison": "frangieh_condition_oof",
            "left_environment_id": left_environment,
            "right_environment_id": right_environment,
            "response_program_shift_mean": float(np.nanmean(shifts)),
            "response_shift_risk_change_spearman": float(pd.Series(shifts).corr(pd.Series(changes), method="spearman")),
            "response_genes": int(len(genes)),
            "prediction_protocol": "five_fold_perturbation_label_OOF",
            "response_program_is_descriptive": 1,
        })
        summary_rows.append(stats)
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def run(root: Path, *, folds: int = 5) -> dict[str, Any]:
    panel = load_panel(root)
    surfaces = [_score_environment(root, env, panel, folds=folds) for env in ENVIRONMENTS]
    oof = pd.concat(surfaces, ignore_index=True).sort_values(["environment_key", "perturbation_label"], kind="stable")
    summary, detail = _build_pair_summary(root, oof)
    out_dir = root / "artifacts/manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "formal_v2_controlled_shift_oof_predictions.csv"
    summary_path = out_dir / "formal_v2_controlled_shift_oof_summary.csv"
    detail_path = out_dir / "formal_v2_controlled_shift_oof_perturbations.csv"
    report_path = out_dir / "formal_v2_controlled_shift_oof.json"
    oof.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2_frangieh_controlled_shift_oof_v1",
        "estimand": "full-surface matched-condition reliability reordering under perturbation-label OOF cross-fitting",
        "predictor": "strong_linear",
        "prediction_protocol": "five_fold_perturbation_label_OOF",
        "folds": int(folds),
        "model_seeds": list(MODEL_SEEDS),
        "n_prediction_rows": int(len(oof)),
        "n_unique_labels_by_environment": {env: int(oof.loc[oof["environment_key"].eq(env), "perturbation_label"].nunique()) for env in ENVIRONMENTS},
        "summary_path": summary_path.relative_to(root).as_posix(),
        "detail_path": detail_path.relative_to(root).as_posix(),
        "predictions_path": predictions_path.relative_to(root).as_posix(),
        "outcome_policy": "outcomes are used only after each fold's prediction and confidence are materialized; no target outcome enters a deployment feature",
        "canonical_artifacts_untouched": True,
        "response_program_policy": "descriptive response distance, not causal mechanism evidence",
        "status": "executed",
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), folds=args.folds), indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
