"""Run the preregistered source-only RBF kernel-ridge Frangieh family.

The prediction NPZ is materialized before any target outcome is loaded.  The
optional measurement stage reuses the existing raw-cell resampling engine with
an isolated output stem; canonical linear-family artifacts are never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_formal_v2_controlled_shift_oof import ENVIRONMENTS, MODEL_SEEDS, _environment_rows  # noqa: E402
from scripts.run_formal_v2_claim_lock_source_frozen import (  # noqa: E402
    FOLDS,
    SPLIT_SEED,
    _fold_ids,
)
from scripts.run_formal_v2_predictors import (  # noqa: E402
    components,
    load_panel,
    read_training_post_expression,
)

INITIAL_ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
ALPHA_EXTENSION_VALUES = (30.0, 100.0, 300.0, 1000.0)
EXTENDED_ALPHA_GRID = INITIAL_ALPHA_GRID + ALPHA_EXTENSION_VALUES
# Preserve the original public name for focused callers/tests.  The
# materialization workflow decides whether to use this grid or the
# deterministic extension after the initial source-only audit.
ALPHA_GRID = INITIAL_ALPHA_GRID
ALPHA_EXTENSION_TRIGGER_FRACTION = 0.50
PREDICTION_NAME = "frangieh_source_only_rbf_krr_predictions.npz"
OUTPUT_STEM = "frangieh_source_only_rbf_krr_measurement_fullsize"
METRICS = (
    "delta_cosine",
    "systema_centroid_accuracy",
    "absolute_effect_rank_agreement",
)
DIRECTED_PAIRS = tuple(
    (source, target)
    for source in ENVIRONMENTS
    for target in ENVIRONMENTS
    if source != target
)
D_ADJ_NAME = "frangieh_source_only_rbf_krr_d_adj_fullsize.csv"
SAME_CONTEXT_FLOOR_NAME = "frangieh_source_only_rbf_krr_same_context_floor_fullsize.csv"
STABLE_INVERSION_NAME = "frangieh_source_only_rbf_krr_stable_inversion_fullsize.csv"
REGRET_NAME = "frangieh_source_only_rbf_krr_top10_regret.csv"
REGRET_REPORT_NAME = "frangieh_source_only_rbf_krr_top10_regret.json"
EVALUATION_REPORT_NAME = "frangieh_source_only_rbf_krr_evaluation.json"


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return np.maximum(
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T,
        0.0,
    )


def median_distance_gamma(embedding: np.ndarray) -> float:
    """Return ``1 / (2 median(||xi-xj||)^2)`` on distinct training pairs."""

    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2 or embedding.shape[0] == 0:
        raise ValueError("embedding must be a non-empty two-dimensional array")
    if not np.isfinite(embedding).all():
        raise ValueError("embedding contains non-finite values")
    distance2 = _squared_distances(embedding, embedding)
    values = distance2[np.triu_indices(len(embedding), k=1)]
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return 1.0
    return float(1.0 / (2.0 * np.median(positive)))


def fit_rbf_krr(
    train_embedding: np.ndarray,
    train_response: np.ndarray,
    query_embedding: np.ndarray,
    *,
    alpha: float,
    gamma: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Fit multi-output RBF KRR with a symmetric eigensolve and eigenvalue floor."""

    x = np.asarray(train_embedding, dtype=np.float64)
    y = np.asarray(train_response, dtype=np.float64)
    q = np.asarray(query_embedding, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or q.ndim != 2:
        raise ValueError("train_embedding, train_response, and query_embedding must be two-dimensional")
    if x.shape[0] == 0 or x.shape[0] != y.shape[0] or x.shape[1] != q.shape[1]:
        raise ValueError("RBF KRR training/query dimensions are inconsistent")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError("RBF KRR inputs contain non-finite values")
    if not np.isfinite(alpha) or float(alpha) <= 0.0:
        raise ValueError("alpha must be a finite positive value")
    selected_gamma = median_distance_gamma(x) if gamma is None else float(gamma)
    if not np.isfinite(selected_gamma) or selected_gamma <= 0.0:
        raise ValueError("gamma must be a finite positive value")
    kernel = np.exp(-selected_gamma * _squared_distances(x, x))
    regularized = (kernel + kernel.T) * 0.5 + float(alpha) * np.eye(len(x))
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    floor = max(np.finfo(np.float64).eps * max(1.0, float(eigenvalues[-1])) * len(x), 1e-12)
    coefficients = eigenvectors @ ((eigenvectors.T @ y) / np.maximum(eigenvalues, floor)[:, None])
    prediction = np.exp(-selected_gamma * _squared_distances(q, x)) @ coefficients
    if not np.isfinite(prediction).all():
        raise FloatingPointError("non-finite RBF KRR prediction")
    condition = float(np.max(np.maximum(eigenvalues, floor)) / np.min(np.maximum(eigenvalues, floor)))
    return prediction.astype(np.float32), selected_gamma, condition


def _validated_alpha_grid(alpha_grid: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    grid = tuple(float(value) for value in alpha_grid)
    if not grid or any(not np.isfinite(value) or value <= 0.0 for value in grid):
        raise ValueError("alpha_grid must contain finite positive values")
    if len(set(grid)) != len(grid) or tuple(sorted(grid)) != grid:
        raise ValueError("alpha_grid must be strictly increasing without duplicates")
    return grid


def select_alpha_source_inner_cv(
    embedding: np.ndarray,
    response: np.ndarray,
    *,
    fold_ids: np.ndarray,
    alpha_grid: tuple[float, ...] | list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Tune alpha using source responses only; gamma is recomputed per inner train fold."""

    embedding = np.asarray(embedding, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    fold_ids = np.asarray(fold_ids)
    if embedding.ndim != 2 or response.ndim != 2 or embedding.shape[0] != response.shape[0]:
        raise ValueError("source inner-CV arrays have inconsistent dimensions")
    if fold_ids.ndim != 1 or len(fold_ids) != len(embedding):
        raise ValueError("fold_ids must align with source training rows")
    unique_folds = np.unique(fold_ids)
    if len(unique_folds) < 2 or any(np.sum(fold_ids == fold) == 0 for fold in unique_folds):
        raise ValueError("source inner-CV requires at least two non-empty folds")
    grid = _validated_alpha_grid(ALPHA_GRID if alpha_grid is None else alpha_grid)
    scores: dict[str, float] = {}
    for alpha in grid:
        losses = []
        for fold in sorted(np.unique(fold_ids).tolist()):
            validation = fold_ids == fold
            training = ~validation
            prediction, _, _ = fit_rbf_krr(
                embedding[training], response[training], embedding[validation], alpha=alpha
            )
            losses.append(float(np.mean((prediction.astype(np.float64) - response[validation]) ** 2)))
        scores[f"{alpha:g}"] = float(np.mean(losses))
    selected = min(grid, key=lambda value: (scores[f"{value:g}"], value))
    return float(selected), scores


def _source_embeddings(
    labels: list[str],
    genes: list[str],
    post_values: np.ndarray,
    *,
    n_components: int = 10,
) -> np.ndarray:
    """Build the legal source-only perturbation embeddings used by the reference.

    This is the existing Ahlmann/Elze source-side basis construction: native
    post-perturbation expression is used only for the supplied source-training
    labels, gene rows are SVD embedded, and a perturbation label is represented
    by the mean of its known component-gene vectors. Query labels never
    contribute expression values to this basis.
    """

    post_values = np.asarray(post_values, dtype=np.float64)
    genes = [str(gene) for gene in genes]
    if post_values.ndim != 2 or post_values.shape[0] == 0 or post_values.shape[1] != len(genes):
        raise ValueError("post_values must be non-empty and aligned with genes")
    if not np.isfinite(post_values).all() or len(set(genes)) != len(genes):
        raise ValueError("source post-expression values or gene names are invalid")
    if int(n_components) <= 0:
        raise ValueError("n_components must be positive")
    centered = post_values.T - post_values.T.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    rank = min(int(n_components), u.shape[0], u.shape[1])
    if rank == 0:
        raise ValueError("source post-expression basis has zero rank")
    gene_embedding = u[:, :rank] * singular_values[:rank]
    index = {gene: position for position, gene in enumerate(genes)}
    rows = []
    for label in labels:
        known = [gene_embedding[index[part]] for part in components(label) if part in index]
        rows.append(np.mean(known, axis=0) if known else np.zeros(rank, dtype=np.float64))
    return np.stack(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_protocol(root: Path) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Read and validate the shared contexts/folds/panel protocol."""

    protocol_path = root / "artifacts/source_data/frangieh_source_frozen_predictions.npz"
    with np.load(protocol_path, allow_pickle=False) as protocol:
        required = {"source_environment", "perturbation_label", "fold_id", "model_seed", "evaluation_gene_symbols"}
        missing = sorted(required.difference(protocol.files))
        if missing:
            raise ValueError(f"shared Frangieh protocol is missing arrays: {missing}")
        sources = protocol["source_environment"].astype(str).tolist()
        labels = protocol["perturbation_label"].astype(str).tolist()
        folds = protocol["fold_id"].astype(np.int16)
        seeds = protocol["model_seed"].astype(np.int16)
        panel = protocol["evaluation_gene_symbols"].astype(str).tolist()
    if sources != list(ENVIRONMENTS) or seeds.tolist() != list(MODEL_SEEDS):
        raise ValueError("shared Frangieh context/member protocol changed")
    if len(labels) != len(folds) or len(set(labels)) != len(labels):
        raise ValueError("shared Frangieh labels and folds are not a unique aligned surface")
    if set(np.unique(folds).tolist()) != set(range(FOLDS)):
        raise ValueError("shared Frangieh outer folds are not the declared five-fold surface")
    if panel != load_panel(root):
        raise ValueError("RBF evaluation panel differs from the frozen context-v2 evaluation panel")
    return sources, labels, folds, np.asarray(panel, dtype=str)


def materialize_predictions(root: Path) -> tuple[Path, dict[str, Any]]:
    """Fit all outer folds without loading any target-context outcome arrays."""

    root = root.resolve()
    _, labels, outer_folds, panel_array = _prediction_protocol(root)
    panel = panel_array.tolist()
    labels_array = np.asarray(labels, dtype=str)
    predictions = np.zeros((len(ENVIRONMENTS), len(labels), len(MODEL_SEEDS), len(panel)), dtype=np.float32)
    # First pass: audit the original grid on unique source x outer-fold
    # choices.  This pass only reads source responses and source post-expression.
    selection_jobs: list[dict[str, Any]] = []
    context_data: dict[str, dict[str, Any]] = {}
    tuning_rows: list[dict[str, Any]] = []
    excluded = {
        "biological_instance_id", "environment_id", "environment_key", "dataset_id", "condition",
        "condition_field", "perturbation_label", "dose", "timepoint", "n_reference_groups", "total_cells",
        "source_reference_keys", "aggregation_rule", "cell_line", "celltype", "cell_context", "target",
        "guide_id", "perturbation_type",
    }
    for source_index, source in enumerate(ENVIRONMENTS):
        frame, _ = _environment_rows(root, source)
        frame = frame.copy()
        frame["perturbation_label"] = frame["perturbation_label"].astype(str)
        frame = frame.set_index("perturbation_label", drop=False).loc[labels]
        genes = [column for column in frame.columns if column not in excluded]
        eval_indices = [genes.index(gene) for gene in panel]
        source_response = frame.loc[labels, genes].to_numpy(dtype=np.float32)
        metadata = frame.iloc[0]
        for outer_fold in range(FOLDS):
            test_indices = np.flatnonzero(outer_folds == outer_fold)
            train_indices = np.flatnonzero(outer_folds != outer_fold)
            train_labels = labels_array[train_indices]
            post_by_label, _ = read_training_post_expression(
                root, str(metadata["dataset_id"]), set(train_labels.tolist()), genes,
                condition_field=str(metadata.get("condition_field", "") or ""),
                condition=str(metadata.get("condition", "") or ""),
            )
            post = np.stack([post_by_label[str(label)] for label in train_labels])
            train_embedding = _source_embeddings(train_labels.tolist(), genes, post)
            inner_folds = _fold_ids(train_labels.tolist(), folds=FOLDS, seed=SPLIT_SEED + 7919 * (outer_fold + 1))
            initial_alpha, initial_scores = select_alpha_source_inner_cv(
                train_embedding,
                source_response[train_indices],
                fold_ids=inner_folds,
                alpha_grid=INITIAL_ALPHA_GRID,
            )
            selection_jobs.append({
                "source_index": source_index,
                "source": source,
                "outer_fold": outer_fold,
                "test_indices": test_indices,
                "train_indices": train_indices,
                "train_labels": train_labels,
                "train_embedding": train_embedding,
                "inner_folds": inner_folds,
                "initial_selected_alpha": initial_alpha,
                "initial_inner_cv_mse": initial_scores,
            })
        context_data[source] = {
            "frame": frame,
            "genes": genes,
            "eval_indices": eval_indices,
            "source_response": source_response,
            "metadata": metadata,
        }

    initial_max_alpha = float(max(INITIAL_ALPHA_GRID))
    initial_choice_count = len(selection_jobs)
    initial_max_alpha_hits = sum(
        float(job["initial_selected_alpha"]) == initial_max_alpha for job in selection_jobs
    )
    initial_max_alpha_fraction = (
        float(initial_max_alpha_hits) / float(initial_choice_count)
        if initial_choice_count else float("nan")
    )
    extension_triggered = bool(
        initial_choice_count > 0
        and initial_max_alpha_fraction > ALPHA_EXTENSION_TRIGGER_FRACTION
    )
    selected_alpha_grid = EXTENDED_ALPHA_GRID if extension_triggered else INITIAL_ALPHA_GRID
    final_selection_by_key: dict[tuple[str, int], dict[str, Any]] = {}

    # Second pass: refit only after the source-only selection audit is complete.
    # Post-expression is reloaded per outer fold so held-out source labels never
    # enter that fold's predictor basis.
    for job in selection_jobs:
        source_index = int(job["source_index"])
        source = str(job["source"])
        outer_fold = int(job["outer_fold"])
        source_data = context_data[source]
        genes = source_data["genes"]
        eval_indices = source_data["eval_indices"]
        source_response = source_data["source_response"]
        metadata = source_data["metadata"]
        test_indices = np.asarray(job["test_indices"], dtype=int)
        train_indices = np.asarray(job["train_indices"], dtype=int)
        train_labels = np.asarray(job["train_labels"], dtype=str)
        train_embedding = np.asarray(job["train_embedding"], dtype=np.float64)
        inner_folds = np.asarray(job["inner_folds"])
        initial_alpha = float(job["initial_selected_alpha"])
        initial_scores = dict(job["initial_inner_cv_mse"])
        if extension_triggered:
            selected_alpha, selected_scores = select_alpha_source_inner_cv(
                train_embedding,
                source_response[train_indices],
                fold_ids=inner_folds,
                alpha_grid=EXTENDED_ALPHA_GRID,
            )
        else:
            selected_alpha, selected_scores = initial_alpha, initial_scores
        final_selection_by_key[(source, outer_fold)] = {
            "initial_selected_alpha": initial_alpha,
            "final_selected_alpha": float(selected_alpha),
            "initial_inner_cv_mse": initial_scores,
            "final_inner_cv_mse": selected_scores,
        }
        post_by_label, _ = read_training_post_expression(
            root, str(metadata["dataset_id"]), set(train_labels.tolist()), genes,
            condition_field=str(metadata.get("condition_field", "") or ""),
            condition=str(metadata.get("condition", "") or ""),
        )
        post = np.stack([post_by_label[str(label)] for label in train_labels])
        for member_index, model_seed in enumerate(MODEL_SEEDS):
            rng = np.random.default_rng(int(model_seed) + 1009 * (source_index + 1) + 100000 * outer_fold)
            bootstrap = rng.choice(len(train_indices), size=len(train_indices), replace=True)
            bootstrap_labels = train_labels[bootstrap]
            bootstrap_post = np.stack([post_by_label[str(label)] for label in bootstrap_labels])
            member_embedding = _source_embeddings(bootstrap_labels.tolist(), genes, bootstrap_post)
            query_embedding = _source_embeddings(labels, genes, bootstrap_post)
            member_prediction, gamma, condition = fit_rbf_krr(
                member_embedding, source_response[train_indices][bootstrap],
                query_embedding[test_indices], alpha=selected_alpha,
            )
            predictions[source_index, test_indices, member_index] = member_prediction[:, eval_indices]
            tuning_rows.append({
                "source_environment": source,
                "outer_fold": outer_fold,
                "model_seed": int(model_seed),
                "selected_alpha": float(selected_alpha),
                "initial_selected_alpha": initial_alpha,
                "selection_stage": "extended_source_inner_cv" if extension_triggered else "initial_source_inner_cv",
                "alpha_grid_used": list(selected_alpha_grid),
                "gamma": gamma,
                "regularized_condition_number": condition,
                "inner_cv_mse": selected_scores,
                "initial_inner_cv_mse": initial_scores,
            })
    if not np.isfinite(predictions).all():
        raise ValueError("RBF prediction tensor contains non-finite values")
    selection_audit_rows = []
    for job in selection_jobs:
        key = (str(job["source"]), int(job["outer_fold"]))
        final_selection = final_selection_by_key[key]
        selection_audit_rows.append({
            "source_environment": key[0],
            "outer_fold": key[1],
            "initial_selected_alpha": float(final_selection["initial_selected_alpha"]),
            "initial_hit_largest_alpha": bool(
                float(final_selection["initial_selected_alpha"]) == initial_max_alpha
            ),
            "final_selected_alpha": float(final_selection["final_selected_alpha"]),
            "selection_stage": "extended_source_inner_cv" if extension_triggered else "initial_source_inner_cv",
            "initial_inner_cv_mse": final_selection["initial_inner_cv_mse"],
            "final_inner_cv_mse": final_selection["final_inner_cv_mse"],
        })
    out_dir = root / "artifacts/source_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / PREDICTION_NAME
    np.savez_compressed(
        path, source_environment=np.asarray(ENVIRONMENTS), perturbation_label=labels_array,
        fold_id=outer_folds, model_seed=np.asarray(MODEL_SEEDS, dtype=np.int16), prediction=predictions,
        evaluation_gene_symbols=np.asarray(panel),
    )
    report = {
        "schema_version": 2, "status": "source_only_predictions_materialized", "model": "rbf_kernel_ridge",
        "target_outcomes_visible_during_fit_or_tuning": False,
        "target_outcomes_used_for_fit_or_tuning": False,
        "alpha_grid": list(selected_alpha_grid),
        "initial_alpha_grid": list(INITIAL_ALPHA_GRID),
        "extended_alpha_grid": list(EXTENDED_ALPHA_GRID),
        "selected_alpha_grid": list(selected_alpha_grid),
        "alpha_extension_triggered": extension_triggered,
        "alpha_extension": {
            "triggered": extension_triggered,
            "rule": "extend when the fraction of unique source x outer-fold initial choices at the largest initial alpha is strictly greater than 0.50",
            "initial_choice_count": initial_choice_count,
            "initial_max_alpha": initial_max_alpha,
            "initial_max_alpha_hit_count": int(initial_max_alpha_hits),
            "initial_max_alpha_hit_fraction": initial_max_alpha_fraction,
            "trigger_fraction": ALPHA_EXTENSION_TRIGGER_FRACTION,
            "extension_values": list(ALPHA_EXTENSION_VALUES),
            "unique_outer_fold_choices": selection_audit_rows,
        },
        "gamma_policy": "1 / (2 * median positive squared distance) on the applicable source-training embeddings",
        "outer_folds": FOLDS, "outer_split_seed": SPLIT_SEED, "contexts": list(ENVIRONMENTS),
        "perturbations": len(labels), "evaluation_genes": len(panel),
        "prediction_artifact": path.relative_to(root).as_posix(), "tuning": tuning_rows,
        "shared_protocol_artifact": "artifacts/source_data/frangieh_source_frozen_predictions.npz",
        "shared_protocol_sha256": _sha256_file(root / "artifacts/source_data/frangieh_source_frozen_predictions.npz"),
        "source_embedding_policy": "existing Ahlmann/Elze source-only native post-expression SVD gene basis; perturbation component means",
        "source_post_expression_rows": "outer-fold training labels only; each bootstrap member rebuilds its source-only basis",
        "prediction_materialization": "source-only outer-fold predictions are frozen before target outcome loading",
    }
    report["prediction_artifact_sha256"] = _sha256_file(path)
    report_path = root / "artifacts/manifests/frangieh_source_only_rbf_krr_fit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path, report


def _load_truths_after_fit(
    root: Path,
    labels: list[str],
    panel: list[str],
) -> dict[str, np.ndarray]:
    """Load the full target outcome surface only after prediction materialization."""

    from scripts.run_formal_v2_claim_lock_source_frozen import _load_common_surface

    common, _, truths = _load_common_surface(root, panel)
    if common != labels:
        raise ValueError("evaluation outcome surface differs from the frozen prediction labels")
    return truths


def _endpoint_rows(
    table: pd.DataFrame,
    *,
    source_column: str,
    left_column: str = "left_target_environment_id",
    right_column: str = "right_target_environment_id",
) -> pd.DataFrame:
    """Keep the six directed endpoint transfers from a three-context table."""

    required = {source_column, left_column, right_column, "metric"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"measurement table is missing endpoint columns: {missing}")
    rows: list[dict[str, Any]] = []
    for record in table.to_dict(orient="records"):
        source = str(record[source_column])
        left = str(record[left_column])
        right = str(record[right_column])
        if source == left:
            target = right
        elif source == right:
            target = left
        else:
            continue
        output = dict(record)
        output.update({
            "source_environment_id": source,
            "target_environment_id": target,
            "transfer_id": f"{source}->{target}",
        })
        rows.append(output)
    output = pd.DataFrame(rows)
    keys = ["source_environment_id", "target_environment_id", "metric"]
    expected = len(DIRECTED_PAIRS) * len(METRICS)
    if len(output) != expected or output.duplicated(keys).any():
        raise ValueError(f"endpoint transfer surface must have {expected} unique rows; found {len(output)}")
    return output.sort_values(keys, kind="stable").reset_index(drop=True)


def _floor_by_context_pair(floors: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, float]]:
    required = {
        "left_target_environment_id", "right_target_environment_id", "metric",
        "ordering_within_left_u", "ordering_within_right_u",
    }
    missing = sorted(required.difference(floors.columns))
    if missing:
        raise ValueError(f"measurement floor table is missing columns: {missing}")
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


def _measurement_summary(
    root: Path,
    measurement_report: dict[str, Any],
    summary_path: Path,
) -> tuple[pd.DataFrame, bool]:
    """Read the summary, or rebuild it from the persisted streaming inputs."""

    try:
        summary = pd.read_csv(summary_path)
    except pd.errors.EmptyDataError:
        summary = pd.DataFrame()
    if not summary.empty:
        return summary, False
    bootstrap_relative = measurement_report.get("bootstrap_input_path")
    if not bootstrap_relative:
        raise ValueError("measurement summary is empty and no persisted bootstrap inputs are available")
    bootstrap_path = root / str(bootstrap_relative)
    if not bootstrap_path.is_file():
        raise FileNotFoundError(bootstrap_path)
    from scripts.run_formal_v2_claim_lock_measurement import _bootstrap_floor

    input_fields = (
        "cross_left_a", "cross_right_a", "cross_left_b", "cross_right_b",
        "meas_left_a", "meas_left_b", "meas_right_a", "meas_right_b",
        "joint_left_a", "joint_left_b", "joint_right_a", "joint_right_b",
    )
    rows: list[dict[str, Any]] = []
    bootstrap_keys = list(measurement_report.get("bootstrap_input_keys", []))
    with np.load(bootstrap_path, allow_pickle=False) as payload:
        for encoded in bootstrap_keys:
            parts = str(encoded).split("__")
            if len(parts) != 4:
                raise ValueError(f"invalid persisted measurement input key: {encoded}")
            source, left, right, metric = parts
            first = payload[f"{encoded}__cross_left_a"]
            inputs = [
                {field: payload[f"{encoded}__{field}"][index] for field in input_fields}
                for index in range(first.shape[0])
            ]
            stats = _bootstrap_floor(
                inputs,
                seed=SPLIT_SEED + sum(ord(char) for char in f"{source}|{left}|{right}|{metric}"),
                draws=int(measurement_report.get("bootstrap_draws", 2000)),
                return_draws=True,
            )
            stats.pop("_draws", None)
            stats.update({
                "estimand": "matched_budget_source_frozen",
                "source_environment_id": source,
                "left_target_environment_id": left,
                "right_target_environment_id": right,
                "metric": metric,
                "n_split_seeds": int(first.shape[0]),
                "n_perturbations_primary_min": int(first.shape[1]),
                "min_cells_primary": 20,
                "eligibility_min_cells": 20,
                "sensitivity_min_cells": 40,
                "bootstrap_unit": "matched perturbation label; each draw also selects one of 30 measurement seeds and one unordered model-member pair",
                "measurement_definition": "fixed source-frozen mean prediction versus two independent full-size with-replacement raw-cell truth pseudoreplicates",
                "joint_definition": "source-frozen model member pairs crossed with two independent full-size with-replacement truth pseudoreplicates",
                "ordering_estimator": "pairwise-order disagreement with U-statistic within-context floors",
                "rank_displacement_estimator": "normalized rank displacement; secondary magnitude diagnostic only",
                "metric_entrypoint": "delta_cosine; chunked centroid distance; absolute_effect_rank_agreement",
                "refit_per_metric": 0,
            })
            rows.append(stats)
    rebuilt = pd.DataFrame(rows)
    if rebuilt.empty:
        raise ValueError(f"no measurement summary rows rebuilt from {bootstrap_path}")
    rebuilt.to_csv(summary_path, index=False)
    return rebuilt, True


def _build_full_depth_ordering_artifacts(root: Path, measurement_report: dict[str, Any]) -> dict[str, Any]:
    """Materialize directed 18-row D_adj, floor, and stable-inversion outputs."""

    manifests = root / "artifacts/manifests"
    summary_path = root / measurement_report["summary_rows"]
    ordering_path = root / measurement_report["ordering_rows"]
    floor_path = root / measurement_report["floor_rows"]
    summary_frame, summary_rebuilt = _measurement_summary(
        root, measurement_report, summary_path
    )
    summary = _endpoint_rows(summary_frame, source_column="source_environment_id")
    ordering = _endpoint_rows(
        pd.read_csv(ordering_path), source_column="source_environment_id"
    )
    floors = pd.read_csv(floor_path)
    floor_lookup = _floor_by_context_pair(floors)
    provenance = (
        f"{PREDICTION_NAME}; {summary_path.relative_to(root).as_posix()}; "
        f"{ordering_path.relative_to(root).as_posix()}; {floor_path.relative_to(root).as_posix()}"
    )
    d_rows: list[dict[str, Any]] = []
    floor_rows: list[dict[str, Any]] = []
    for record in summary.to_dict(orient="records"):
        left = str(record["left_target_environment_id"])
        right = str(record["right_target_environment_id"])
        source = str(record["source_environment_id"])
        target = str(record["target_environment_id"])
        pair_floor = floor_lookup[(left, right, str(record["metric"]))]
        source_floor = pair_floor["left"] if source == left else pair_floor["right"]
        target_floor = pair_floor["right"] if source == left else pair_floor["left"]
        base = {
            "source_environment_id": source,
            "target_environment_id": target,
            "transfer_id": str(record["transfer_id"]),
            "metric": str(record["metric"]),
            "cell_budget_label": "full",
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
            "provenance": provenance,
        }
        d_rows.append(base)
        floor_rows.append({
            key: base[key]
            for key in (
                "source_environment_id", "target_environment_id", "transfer_id", "metric",
                "cell_budget_label", "n_perturbations", "source_same_context_floor",
                "target_same_context_floor", "same_context_floor",
                "same_context_floor_ci_low", "same_context_floor_ci_high",
                "same_context_floor_definition", "measurement_resampling", "provenance",
            )
        })
    d_adj = pd.DataFrame(d_rows).sort_values(
        ["source_environment_id", "target_environment_id", "metric"], kind="stable"
    ).reset_index(drop=True)
    same_floor = pd.DataFrame(floor_rows).sort_values(
        ["source_environment_id", "target_environment_id", "metric"], kind="stable"
    ).reset_index(drop=True)
    ordering_by_key = ordering.set_index(
        ["source_environment_id", "target_environment_id", "metric"]
    )
    stable_rows: list[dict[str, Any]] = []
    for record in d_rows:
        key = (record["source_environment_id"], record["target_environment_id"], record["metric"])
        order = ordering_by_key.loc[key]
        left = str(order["left_target_environment_id"])
        source = str(record["source_environment_id"])
        source_stable = float(order["stable_fraction_left"] if source == left else order["stable_fraction_right"])
        target_stable = float(order["stable_fraction_right"] if source == left else order["stable_fraction_left"])
        stable_rows.append({
            "source_environment_id": source,
            "target_environment_id": record["target_environment_id"],
            "transfer_id": record["transfer_id"],
            "metric": record["metric"],
            "cell_budget_label": "full",
            "n_items": int(float(order["n_items"])),
            "minimum_strict_support": int(float(order["minimum_strict_support"])),
            "stable_fraction_source": source_stable,
            "stable_fraction_target": target_stable,
            "stable_fraction_both": float(order["stable_fraction_both"]),
            "stable_order_inversion_fraction": float(order["stable_order_inversion_fraction"]),
            "stable_pairs": int(float(order["stable_pairs"])),
            "crossfit_heldout_inversion_fraction": float(order["crossfit_heldout_inversion_fraction"]),
            "crossfit_heldout_evaluable_fraction": float(order["crossfit_heldout_evaluable_fraction"]),
            "stable_inversion_definition": "opposite strict directions among pairs stable in both endpoint contexts; minimum strict support >= 8",
            "measurement_resampling": "full_size_nonparametric",
            "provenance": provenance,
        })
    stable = pd.DataFrame(stable_rows).sort_values(
        ["source_environment_id", "target_environment_id", "metric"], kind="stable"
    ).reset_index(drop=True)
    outputs = {
        "d_adj": manifests / D_ADJ_NAME,
        "same_context_floor": manifests / SAME_CONTEXT_FLOOR_NAME,
        "stable_inversion": manifests / STABLE_INVERSION_NAME,
    }
    d_adj.to_csv(outputs["d_adj"], index=False)
    same_floor.to_csv(outputs["same_context_floor"], index=False)
    stable.to_csv(outputs["stable_inversion"], index=False)
    report = {
        "schema_version": 1,
        "status": "executed",
        "predictor_family": "frangieh_source_only_rbf_krr",
        "analysis": "full_depth_directed_transfer_ordering",
        "rows": {key: int(len(value)) for key, value in (("d_adj", d_adj), ("same_context_floor", same_floor), ("stable_inversion", stable))},
        "transfer_count": len(DIRECTED_PAIRS),
        "metric_count": len(METRICS),
        "outputs": {key: path.relative_to(root).as_posix() for key, path in outputs.items()},
        "measurement_report": measurement_report,
        "summary_rebuilt_from_persisted_bootstrap_inputs": summary_rebuilt,
        "target_outcomes_loaded_after_prediction_materialization": True,
        "target_outcomes_used_for_fit_or_tuning": False,
        "provenance": provenance,
    }
    report["output_sha256"] = {key: _sha256_file(path) for key, path in outputs.items()}
    report_path = manifests / EVALUATION_REPORT_NAME
    report["report"] = report_path.relative_to(root).as_posix()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _build_top10_regret(root: Path) -> dict[str, Any]:
    """Write a deterministic optional top-10% regret diagnostic after fitting."""

    from scripts.run_formal_v2_claim_lock_source_frozen import _metric_vectors
    from src.evaluation.decision_theory import top_k_decision_metrics

    prediction_path = root / "artifacts/source_data" / PREDICTION_NAME
    with np.load(prediction_path, allow_pickle=False) as payload:
        labels = payload["perturbation_label"].astype(str).tolist()
        panel = payload["evaluation_gene_symbols"].astype(str).tolist()
        predictions = payload["prediction"].astype(np.float32, copy=False)
    truths = _load_truths_after_fit(root, labels, panel)
    risks: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for source_index, source in enumerate(ENVIRONMENTS):
        risks[source] = {
            target: _metric_vectors(truths[target], predictions[source_index], labels, panel)
            for target in ENVIRONMENTS
        }
    rows: list[dict[str, Any]] = []
    for source, target in DIRECTED_PAIRS:
        for metric in METRICS:
            decision = top_k_decision_metrics(
                risks[source][source][metric],
                risks[source][target][metric],
                budget_fraction=0.10,
            )
            selected_labels = [labels[int(index)] for index in decision["selected_indices"]]
            oracle_labels = [labels[int(index)] for index in decision["oracle_indices"]]
            rows.append({
                "source_environment_id": source,
                "target_environment_id": target,
                "transfer_id": f"{source}->{target}",
                "metric": metric,
                "cell_budget_label": "full",
                "decision_budget_fraction": 0.10,
                "k": int(decision["k"]),
                "n_items": int(decision["n_items"]),
                "retention": float(decision["retention"]),
                "selected_target_risk": float(decision["selected_target_risk"]),
                "oracle_target_risk": float(decision["oracle_target_risk"]),
                "random_target_risk": float(decision["random_target_risk"]),
                "regret": float(decision["regret"]),
                "normalized_regret": float(decision["normalized_regret"]),
                "boundary_inversion": float(decision["boundary_inversion"]),
                "mis_selection_count": int(decision["mis_selection_count"]),
                "selected_labels": "|".join(selected_labels),
                "oracle_labels": "|".join(oracle_labels),
                "decision_definition": "deterministic lowest source-risk top-10% selection evaluated on target risk; point diagnostic without measurement bootstrap",
                "target_outcomes_used_for_fit_or_tuning": 0,
                "provenance": f"{PREDICTION_NAME}; target truth surface loaded after prediction materialization; shared 243-label/8229-gene evaluation universe",
            })
    output = pd.DataFrame(rows).sort_values(
        ["source_environment_id", "target_environment_id", "metric"], kind="stable"
    ).reset_index(drop=True)
    if len(output) != len(DIRECTED_PAIRS) * len(METRICS):
        raise ValueError(f"top-10 regret surface must have 18 rows; found {len(output)}")
    path = root / "artifacts/manifests" / REGRET_NAME
    output.to_csv(path, index=False)
    report = {
        "schema_version": 1,
        "status": "executed",
        "predictor_family": "frangieh_source_only_rbf_krr",
        "analysis": "full_depth_top10_percent_decision_diagnostic",
        "rows": int(len(output)),
        "decision_budget_fraction": 0.10,
        "outputs": {"top10_regret": path.relative_to(root).as_posix()},
        "prediction_artifact": prediction_path.relative_to(root).as_posix(),
        "prediction_artifact_sha256": _sha256_file(prediction_path),
        "target_outcomes_loaded_after_prediction_materialization": True,
        "target_outcomes_used_for_fit_or_tuning": False,
        "provenance": "source-only OOF predictions are frozen before target truth is loaded; no target statistics enter fitting, tuning, or selection",
    }
    report_path = root / "artifacts/manifests" / REGRET_REPORT_NAME
    report["output_sha256"] = {"top10_regret": _sha256_file(path)}
    report["report"] = report_path.relative_to(root).as_posix()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def run_measurement(root: Path, *, draws: int = 2000) -> dict[str, Any]:
    """Run isolated full-size measurement and directed artifact evaluation."""

    import scripts.run_formal_v2_claim_lock_measurement as measurement

    original = measurement.PREDICTION_PATH
    try:
        root = root.resolve()
        # The canonical measurement module resolves this path relative to its
        # module-level ROOT; the run root is then prepended inside measurement.run.
        measurement.PREDICTION_PATH = root / "artifacts/source_data" / PREDICTION_NAME
        measurement_report = measurement.run(
            root, draws=draws, output_stem=OUTPUT_STEM,
            resampling_mode="full_size_nonparametric", stream_bootstrap_inputs=False,
        )
        return {
            "measurement": measurement_report,
            "directed_outputs": _build_full_depth_ordering_artifacts(root, measurement_report),
        }
    finally:
        measurement.PREDICTION_PATH = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--measurement", action="store_true")
    parser.add_argument("--regret", action="store_true", help="also write the optional deterministic top-10%% regret artifact")
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    root = args.root.resolve()
    _, report = materialize_predictions(root)
    if args.measurement:
        report["measurement"] = run_measurement(root, draws=args.draws)
    if args.regret:
        report["top10_regret"] = _build_top10_regret(root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
