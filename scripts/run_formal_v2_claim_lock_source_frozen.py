"""Build the Claim Lock source-frozen Frangieh diagnostic.

The existing controlled-shift OOF surface fits one model per condition.  This
runner adds the primary Claim Lock estimand: for each source condition and
global perturbation-label fold, fit once and score the identical prediction
vector against all three target-condition outcomes.  Metric changes therefore
cannot be explained by per-target refitting.

The raw prediction vectors are kept in a compressed NPZ because the vectors
are the auditable boundary between fitting and evaluation.  Small CSV/JSON
manifests contain only summaries and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_formal_v2_controlled_shift_oof import (  # noqa: E402
    ENVIRONMENTS,
    MODEL_SEEDS,
    _environment_rows,
)
from scripts.run_formal_v2_predictors import (  # noqa: E402
    fit_ahlmann_eltze_bilinear_ridge,
    load_panel,
    read_training_post_expression,
)
from src.evaluation.metrics import safe_rowwise_cosine  # noqa: E402
from src.evaluation.reordering_inference import (  # noqa: E402
    _pairwise_arrays,
    normalized_rank_displacement,
)
from src.ptl.evaluation.fidelity import absolute_effect_rank_agreement  # noqa: E402
from src.ptl.uncertainty.uq import summarize_ensemble  # noqa: E402

SYSTEMA_ROOT = ROOT / "third_party/systema"
if str(SYSTEMA_ROOT / "evaluation") not in sys.path:
    sys.path.insert(0, str(SYSTEMA_ROOT / "evaluation"))
from centroid_accuracy import calculate_centroid_accuracies  # noqa: E402


SPLIT_SEED = 20260908
FOLDS = 5
BOOTSTRAP_DRAWS = 2000
INFERENCE_SEED = 20260908
PAIR_ORDER = (
    ("frangieh_melanoma_control", "frangieh_melanoma_coculture"),
    ("frangieh_melanoma_control", "frangieh_melanoma_ifng"),
    ("frangieh_melanoma_coculture", "frangieh_melanoma_ifng"),
)


def _fold_ids(labels: list[str], *, folds: int = FOLDS, seed: int = SPLIT_SEED) -> np.ndarray:
    if len(labels) < folds:
        raise ValueError("the common perturbation surface is smaller than the fold count")
    fold = np.empty(len(labels), dtype=np.int16)
    order = np.random.default_rng(seed).permutation(len(labels))
    for fold_id, indices in enumerate(np.array_split(order, folds)):
        fold[indices] = fold_id
    return fold


def _gene_checksum(panel: list[str]) -> str:
    payload = "\n".join(panel).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_common_surface(root: Path, panel: list[str]) -> tuple[list[str], dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    frames: dict[str, pd.DataFrame] = {}
    values: dict[str, np.ndarray] = {}
    label_sets: list[set[str]] = []
    for environment in ENVIRONMENTS:
        frame, _ = _environment_rows(root, environment)
        frame = frame.copy()
        frame["perturbation_label"] = frame["perturbation_label"].astype(str)
        genes = [column for column in frame.columns if column not in {
            "biological_instance_id", "environment_id", "environment_key", "dataset_id",
            "condition", "condition_field", "perturbation_label", "dose", "timepoint",
            "n_reference_groups", "total_cells", "source_reference_keys", "aggregation_rule",
            "cell_line", "celltype", "cell_context", "target", "guide_id", "perturbation_type",
        }]
        missing = set(panel).difference(genes)
        if missing:
            raise ValueError(f"{environment} is missing evaluation genes: {sorted(missing)[:5]}")
        frames[environment] = frame.set_index("perturbation_label", drop=False)
        values[environment] = frame.set_index("perturbation_label").loc[:, panel].to_numpy(dtype=np.float32)
        label_sets.append(set(frame["perturbation_label"]))
    common = sorted(set.intersection(*label_sets))
    if len(common) < FOLDS:
        raise ValueError("no usable common perturbation-label surface")
    for environment in ENVIRONMENTS:
        frames[environment] = frames[environment].loc[common].copy()
        values[environment] = frames[environment].loc[:, panel].to_numpy(dtype=np.float32)
    return common, frames, values


def _fit_source_surface(
    root: Path,
    source_environment: str,
    labels: list[str],
    source_frame: pd.DataFrame,
    panel: list[str],
    fold_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit source-only fold members and return [label, member, gene] arrays."""

    metadata = source_frame.iloc[0]
    native_genes = [column for column in source_frame.columns if column not in {
        "biological_instance_id", "environment_id", "environment_key", "dataset_id",
        "condition", "condition_field", "perturbation_label", "dose", "timepoint",
        "n_reference_groups", "total_cells", "source_reference_keys", "aggregation_rule",
        "cell_line", "celltype", "cell_context", "target", "guide_id", "perturbation_type",
    }]
    eval_indices = [native_genes.index(gene) for gene in panel]
    labels_array = np.asarray(labels, dtype=str)
    native_values = source_frame.loc[labels, native_genes].to_numpy(dtype=np.float32)
    predictions = np.zeros((len(labels), len(MODEL_SEEDS), len(panel)), dtype=np.float32)
    for fold_id in range(FOLDS):
        test_indices = np.flatnonzero(fold_ids == fold_id)
        train_indices = np.flatnonzero(fold_ids != fold_id)
        train_labels = labels_array[train_indices]
        post_by_label, _ = read_training_post_expression(
            root,
            str(metadata["dataset_id"]),
            set(train_labels.tolist()),
            native_genes,
            condition_field=str(metadata.get("condition_field", "") or ""),
            condition=str(metadata.get("condition", "") or ""),
        )
        query_labels = labels_array[test_indices]
        for member_index, model_seed in enumerate(MODEL_SEEDS):
            rng = np.random.default_rng(
                int(model_seed) + 1009 * (sum(ord(c) for c in source_environment) + 1) + 100000 * fold_id
            )
            bootstrap = rng.choice(train_indices, size=len(train_indices), replace=True)
            bootstrap_labels = labels_array[bootstrap]
            post_train_values = np.stack([post_by_label[str(label)] for label in bootstrap_labels])
            predictions[test_indices, member_index] = fit_ahlmann_eltze_bilinear_ridge(
                bootstrap_labels,
                native_values[bootstrap],
                query_labels,
                gene_names=native_genes,
                post_train_values=post_train_values,
                alpha=0.1,
                n_components=10,
            )[:, eval_indices]
    if not np.isfinite(predictions).all():
        raise ValueError(f"non-finite source-frozen predictions for {source_environment}")
    # The uncertainty helper uses [member, row, gene] ordering, whereas the
    # persisted audit uses [row, member, gene] so each frozen vector is easy
    # to address by perturbation label.
    confidence = 1.0 - summarize_ensemble(np.transpose(predictions, (1, 0, 2))).cosine_disagreement
    return predictions, confidence.astype(np.float32)


def _bootstrap_mean(values: np.ndarray, *, seed: int, draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, values.size, size=(draws, values.size))
    means = np.mean(values[samples], axis=1)
    return float(np.mean(values)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _metric_vectors(truth: np.ndarray, prediction_members: np.ndarray, labels: list[str], panel: list[str]) -> dict[str, np.ndarray]:
    mean_prediction = prediction_members.mean(axis=1)
    metric: dict[str, np.ndarray] = {
        "delta_cosine": 1.0 - safe_rowwise_cosine(truth, mean_prediction),
        "absolute_effect_rank_agreement": 1.0 - absolute_effect_rank_agreement(truth, mean_prediction),
    }
    prediction_frame = pd.DataFrame(mean_prediction, index=pd.MultiIndex.from_tuples(
        [(label, "source_frozen") for label in labels], names=["condition", "method"]), columns=panel)
    truth_frame = pd.DataFrame(truth, index=pd.Index(labels, name="condition"), columns=panel)
    centroid = calculate_centroid_accuracies(prediction_frame, truth_frame)["source_frozen"]
    metric["systema_centroid_accuracy"] = 1.0 - centroid.to_numpy(dtype=float)
    return metric


def _reordering_rows(
    predictions: dict[str, np.ndarray],
    truths: dict[str, np.ndarray],
    labels: list[str],
    panel: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_seeds = {"delta_cosine": 11, "systema_centroid_accuracy": 17, "absolute_effect_rank_agreement": 23}
    for source_index, source in enumerate(ENVIRONMENTS):
        source_prediction = predictions[source]
        metric_risks = {target: _metric_vectors(truths[target], source_prediction, labels, panel) for target in ENVIRONMENTS}
        for pair_index, (left, right) in enumerate(PAIR_ORDER):
            for metric in ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement"):
                left_risk = metric_risks[left][metric]
                right_risk = metric_risks[right][metric]
                displacement = normalized_rank_displacement(left_risk, right_risk)
                strict_rate = float(_pairwise_arrays(left_risk, right_risk)["risk_inversion_rate"])
                mean, low, high = _bootstrap_mean(
                    displacement,
                    seed=INFERENCE_SEED + source_index * 1000 + pair_index * 100 + metric_seeds[metric],
                )
                rows.append({
                    "estimand": "source_frozen_primary",
                    "source_environment_id": source,
                    "left_target_environment_id": left,
                    "right_target_environment_id": right,
                    "metric": metric,
                    "n_common_perturbations": len(labels),
                    "strict_inversion_rate": strict_rate,
                    "normalized_rank_displacement_mean": mean,
                    "normalized_rank_displacement_ci_low": low,
                    "normalized_rank_displacement_ci_high": high,
                    "bootstrap_unit": "matched perturbation label",
                    "fold_contract": "one global 5-fold assignment shared by all source contexts",
                    "prediction_contract": "one source-context fitted predictor scored unchanged against both target outcomes",
                    "refit_per_metric": 0,
                })
    return pd.DataFrame(rows)


def run(root: Path, *, bootstrap_draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    global BOOTSTRAP_DRAWS
    BOOTSTRAP_DRAWS = int(bootstrap_draws)
    panel = load_panel(root)
    labels, frames, truths = _load_common_surface(root, panel)
    fold_ids = _fold_ids(labels)
    predictions: dict[str, np.ndarray] = {}
    confidences: dict[str, np.ndarray] = {}
    for source in ENVIRONMENTS:
        predictions[source], confidences[source] = _fit_source_surface(
            root, source, labels, frames[source], panel, fold_ids
        )
    output_dir = root / "artifacts/manifests"
    source_dir = root / "artifacts/source_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    stacked = np.stack([predictions[environment] for environment in ENVIRONMENTS], axis=0)
    confidence = np.stack([confidences[environment] for environment in ENVIRONMENTS], axis=0)
    npz_path = source_dir / "frangieh_source_frozen_predictions.npz"
    np.savez_compressed(
        npz_path,
        source_environment=np.asarray(ENVIRONMENTS),
        perturbation_label=np.asarray(labels),
        fold_id=fold_ids,
        model_seed=np.asarray(MODEL_SEEDS, dtype=np.int16),
        prediction=stacked,
        confidence=confidence,
        evaluation_gene_symbols=np.asarray(panel),
    )
    summary = _reordering_rows(predictions, truths, labels, panel)
    summary_path = output_dir / "formal_v2_claim_lock_source_frozen_reordering.csv"
    summary.to_csv(summary_path, index=False)
    metric_rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(ENVIRONMENTS):
        for target in ENVIRONMENTS:
            vectors = _metric_vectors(truths[target], predictions[source], labels, panel)
            for metric, risk in vectors.items():
                metric_rows.append({
                    "estimand": "source_frozen",
                    "source_environment_id": source,
                    "target_environment_id": target,
                    "metric": metric,
                    "risk_mean": float(np.mean(risk)),
                    "risk_median": float(np.median(risk)),
                    "n_common_perturbations": len(labels),
                    "refit_per_metric": 0,
                    "prediction_artifact": npz_path.relative_to(root).as_posix(),
                })
    metric_path = output_dir / "formal_v2_claim_lock_metric_robustness.csv"
    pd.DataFrame(metric_rows).to_csv(metric_path, index=False)
    report_path = output_dir / "formal_v2_claim_lock_source_frozen.json"
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2_frangieh_claim_lock_v1",
        "status": "source_frozen_primary_executed",
        "estimands": {
            "primary": "source-context predictor fit once per global perturbation-label fold and scored unchanged against all target contexts",
            "secondary": "context-adapted OOF reordering remains in formal_v2_controlled_shift_oof_* and is not pooled with primary",
        },
        "environments": list(ENVIRONMENTS),
        "common_perturbation_count": len(labels),
        "folds": FOLDS,
        "split_seed": SPLIT_SEED,
        "model_seeds": list(MODEL_SEEDS),
        "evaluation_gene_count": len(panel),
        "evaluation_gene_checksum_sha256": _gene_checksum(panel),
        "prediction_npz": npz_path.relative_to(root).as_posix(),
        "reordering_summary": summary_path.relative_to(root).as_posix(),
        "metric_robustness_summary": metric_path.relative_to(root).as_posix(),
        "metrics": ["delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement"],
        "metric_policy": "same frozen prediction vectors, target labels, target truth, and gene panel; no per-metric refit",
        "response_and_rank_policy": "rank displacement is the causal-diagnostic estimand; response shifts remain descriptive",
        "source_frozen_contract": "target outcomes are read only after source prediction vectors are materialized",
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), bootstrap_draws=args.bootstrap_draws), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
