"""Run the reproducible formal-v2 simple and bilinear-linear predictors.

The runner consumes only the condition-level ground truth and the frozen
biological-instance split. It writes local prediction arrays under ``results``
and small, tracked contract/metric manifests under ``artifacts/manifests``.
No outcome is used to construct a deployment feature or an uncertainty score.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import safe_rowwise_cosine, rowwise_spearman
from src.ptl.data.ids import prediction_id
from src.ptl.uncertainty.uq import summarize_ensemble


SPLIT_ID = "ptl_biological_instance_split_v1"
PREDICTOR_VERSION = "formal_v2_exact_linear_20260907"
PREDICTOR_VERSIONS = {
    "mean_matching": "formal_v2_mean_matching_bootstrap_20260907",
    "strong_linear": PREDICTOR_VERSION,
    "slim_string": "formal_v2_slim_string_v0.3_20260907",
}
COMPONENT_PATTERN = re.compile(r"[_+|;]+")
METADATA_COLUMNS = {
    "biological_instance_id", "environment_id", "environment_key", "dataset_id",
    "condition", "condition_field", "perturbation_label", "dose", "timepoint",
    "n_reference_groups", "total_cells", "source_reference_keys", "aggregation_rule",
    "cell_line", "celltype", "cell_context", "target", "guide_id", "perturbation_type",
}
POST_METADATA_COLUMNS = {
    "dataset_id", "source_dataset", "group_key", "reference_key", "perturbation_label",
    "is_control", "cell_line", "cell_type", "celltype", "cell_context", "target", "guide_id",
    "perturbation_type", "perturbation_2", "batch", "replicate",
    "time", "timepoint", "dose", "n_cells", "control_label_used",
}


def current_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "working-tree"


def load_slim_string_embeddings(root: Path) -> tuple[dict[str, np.ndarray], str]:
    """Load the pinned public SLIM STRING embedding artifact."""

    path = root / "data" / "external" / "slim" / "gene_string_embeddings.v0.3.h5"
    if not path.is_file():
        raise FileNotFoundError(f"missing SLIM STRING embedding artifact: {path}")
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("SLIM requires h5py to load the public STRING embedding artifact") from exc
    with h5py.File(path, "r") as handle:
        embeddings = {
            str(gene): np.asarray(handle[gene][()], dtype=np.float64)
            for gene in handle.keys()
        }
    dimensions = {value.shape for value in embeddings.values()}
    if not embeddings or len(dimensions) != 1 or next(iter(dimensions))[0] <= 0:
        raise ValueError("SLIM STRING embedding artifact is empty or dimensionally inconsistent")
    return embeddings, str(path.relative_to(root).as_posix())


def load_panel(root: Path) -> list[str]:
    panel = pd.read_csv(root / "artifacts/manifests/evaluation_gene_space.csv")
    required = {"panel_id", "gene_index", "gene_symbol"}
    if required.difference(panel.columns):
        raise ValueError("evaluation gene panel is missing required columns")
    panel = panel.sort_values("gene_index", kind="stable")
    if panel["gene_symbol"].duplicated().any() or not panel["panel_id"].eq("ptl_context_v2_intersection").all():
        raise ValueError("evaluation gene panel is not the frozen context-v2 panel")
    return panel["gene_symbol"].astype(str).tolist()


def load_registry(root: Path) -> pd.DataFrame:
    registry = pd.read_csv(root / "artifacts/manifests/environment_registry.csv")
    required = {"environment_id", "environment_key", "dataset_id"}
    if required.difference(registry.columns):
        raise ValueError("environment registry is missing canonical identity columns")
    if registry["environment_id"].duplicated().any() or registry["environment_key"].duplicated().any():
        raise ValueError("environment registry IDs and keys must be unique")
    return registry


def load_manifest(root: Path, path: Path | None = None, *, split_seed: int = 20260907) -> pd.DataFrame:
    manifest = pd.read_csv(path or root / "artifacts/manifests/biological_instance_registry.csv", dtype=str)
    required = {
        "biological_instance_id", "environment_id", "perturbation_label", "dose", "timepoint",
        "split", "split_seed", "model_seeds", "ground_truth_path",
    }
    if required.difference(manifest.columns):
        raise ValueError("biological-instance manifest is missing formal split columns")
    if manifest["biological_instance_id"].duplicated().any():
        raise ValueError("biological-instance manifest contains duplicate IDs")
    if not manifest["split_seed"].eq(str(split_seed)).all() or not manifest["model_seeds"].eq("0,1,2").all():
        raise ValueError(f"formal split seed/model seed contract changed for split_seed={split_seed}")
    return manifest


def read_ground_truth(root: Path, path: str, genes: list[str] | None = None) -> pd.DataFrame:
    source = root / path
    schema_columns = pq.read_schema(source).names
    columns = [column for column in METADATA_COLUMNS if column in schema_columns]
    if genes is None:
        genes = [column for column in schema_columns if column not in METADATA_COLUMNS]
    columns.extend(gene for gene in genes if gene not in columns)
    frame = pd.read_parquet(source, columns=columns)
    missing = set(genes).difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing frozen evaluation genes: {sorted(missing)[:5]}")
    for column in ("dose", "timepoint"):
        if column not in frame.columns:
            frame[column] = ""
    return frame


def read_training_post_expression(
    root: Path,
    dataset_id: str,
    labels: set[str],
    genes: list[str],
    *,
    condition_field: str = "",
    condition: str = "",
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load train-condition post-expression for the official Ahlmann baseline.

    The publication constructs ``G`` from post-perturbation pseudobulk ``X``
    and fits the two-sided ridge to ``change``. The formal ground-truth table
    stores only the latter, so this reads native pseudobulk X for train labels
    only, using Parquet predicate pushdown to keep held-out conditions out.
    """

    if not labels:
        raise ValueError(f"cannot build a post-expression embedding without training labels for {dataset_id}")
    source = root / "data" / "processed" / f"{dataset_id}_pseudobulk.parquet"
    if not source.is_file():
        raise FileNotFoundError(f"missing native post-expression pseudobulk: {source}")
    schema_columns = pq.read_schema(source).names
    required = {"perturbation_label", "is_control", "n_cells"}
    missing_metadata = required.difference(schema_columns)
    if missing_metadata:
        raise ValueError(f"{source} is missing post-expression metadata: {sorted(missing_metadata)}")
    missing_genes = set(genes).difference(schema_columns)
    if missing_genes:
        raise ValueError(f"{source} is missing native genes required by the linear model: {sorted(missing_genes)[:5]}")
    condition_field = str(condition_field or "").strip()
    read_metadata = ["perturbation_label", "is_control", "n_cells"]
    if condition_field:
        if condition_field not in schema_columns:
            raise ValueError(f"{source} is missing formal condition field {condition_field!r}")
        read_metadata.append(condition_field)
    filters: list[tuple[str, str, object]] = [("perturbation_label", "in", sorted(labels))]
    if condition_field and condition:
        filters.append((condition_field, "==", str(condition)))
    frame = pq.read_table(
        source,
        columns=read_metadata + list(genes),
        filters=filters,
        use_threads=True,
    ).to_pandas()
    frame = frame.loc[~frame["is_control"].astype(bool)].copy()
    frame["n_cells"] = pd.to_numeric(frame["n_cells"], errors="coerce")
    frame = frame.loc[frame["n_cells"].gt(0)].copy()
    observed = set(frame["perturbation_label"].astype(str))
    missing_labels = set(labels).difference(observed)
    if missing_labels:
        raise ValueError(f"native post-expression surface has no train rows for {dataset_id}: {sorted(missing_labels)[:10]}")
    result: dict[str, np.ndarray] = {}
    for label, group in frame.groupby("perturbation_label", sort=True):
        weights = group["n_cells"].to_numpy(dtype=np.float64)
        values = group[genes].to_numpy(dtype=np.float64, copy=False)
        result[str(label)] = weighted_mean(values, weights).astype(np.float32)
    control_filters: list[tuple[str, str, object]] = [("is_control", "==", True)]
    if condition_field and condition:
        control_filters.append((condition_field, "==", str(condition)))
    control_frame = pq.read_table(
        source,
        columns=read_metadata + list(genes),
        filters=control_filters,
        use_threads=True,
    ).to_pandas()
    control_frame["n_cells"] = pd.to_numeric(control_frame["n_cells"], errors="coerce")
    control_frame = control_frame.loc[control_frame["n_cells"].gt(0)].copy()
    if control_frame.empty:
        raise ValueError(f"native post-expression surface has no matched controls for {dataset_id}")
    control_values = control_frame[genes].to_numpy(dtype=np.float64, copy=False)
    control_weights = control_frame["n_cells"].to_numpy(dtype=np.float64)
    control_mean = weighted_mean(control_values, control_weights).astype(np.float32)
    return result, control_mean


def components(label: object) -> list[str]:
    value = str(label).strip()
    if not value or value.casefold() in {"control", "ctrl", "non-targeting", "ntc"}:
        return []
    return [part for part in COMPONENT_PATTERN.split(value) if part]


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    return (np.asarray(values, dtype=np.float64) * weights[:, None]).sum(axis=0) / weights.sum()


def fit_matching_mean(
    train_labels: np.ndarray,
    train_values: np.ndarray,
    train_weights: np.ndarray,
    query_labels: np.ndarray,
) -> np.ndarray:
    global_mean = weighted_mean(train_values, train_weights)
    exact: dict[str, list[int]] = {}
    by_component: dict[str, list[int]] = {}
    for index, label in enumerate(train_labels.astype(str)):
        exact.setdefault(label, []).append(index)
        for component in components(label):
            by_component.setdefault(component, []).append(index)
    exact_means = {label: weighted_mean(train_values[indexes], train_weights[indexes]) for label, indexes in exact.items()}
    component_means = {
        label: weighted_mean(train_values[indexes], train_weights[indexes])
        for label, indexes in by_component.items()
    }
    predictions = np.empty((len(query_labels), train_values.shape[1]), dtype=np.float32)
    for row, label in enumerate(query_labels.astype(str)):
        if label in exact_means:
            predictions[row] = exact_means[label]
            continue
        means = [component_means[part] for part in components(label) if part in component_means]
        predictions[row] = np.mean(means, axis=0) if means else global_mean
    return predictions


def fit_ahlmann_eltze_bilinear_ridge(
    train_labels: np.ndarray,
    train_values: np.ndarray,
    query_labels: np.ndarray,
    gene_names: list[str] | None = None,
    post_train_values: np.ndarray | None = None,
    alpha: float = 0.1,
    n_components: int = 10,
) -> np.ndarray:
    """Fit the publication-style two-sided ridge predictor in native gene space.

    The official publication code fits PCA to post-perturbation pseudobulk
    expression ``X`` (training conditions only), then fits the two-sided ridge
    to the corresponding ``change`` response. ``post_train_values`` supplies X
    in the same row order as ``train_values``. ``G`` contains the first
    ``n_components`` PCA score coordinates for every native gene, while ``P``
    is formed by averaging the rows of ``G`` for the genes in each perturbation
    label. The fitted map is

    ``W = (G'G + alpha I)^-1 G'(Y-b)P(P'P + alpha I)^-1``

    and queries are reconstructed as ``G W P_query' + b``. Unknown query
    components contribute no embedding, which yields the training centroid for
    an entirely unsupported query rather than leaking evaluation information.
    """

    train_values = np.asarray(train_values, dtype=np.float64)
    if train_values.ndim != 2 or train_values.shape[0] == 0:
        raise ValueError("train_values must be a non-empty 2D array")
    if len(train_labels) != train_values.shape[0]:
        raise ValueError("train_labels and train_values row counts differ")
    if post_train_values is None:
        raise ValueError("post_train_values is required: the official baseline fits G from post-expression X")
    post_train_values = np.asarray(post_train_values, dtype=np.float64)
    if post_train_values.shape != train_values.shape:
        raise ValueError("post_train_values must have the same shape as train_values")
    if alpha <= 0 or n_components <= 0:
        raise ValueError("alpha and n_components must be positive")
    if gene_names is None:
        gene_names = [str(index) for index in range(train_values.shape[1])]
    gene_names = [str(gene) for gene in gene_names]
    if len(gene_names) != train_values.shape[1] or len(set(gene_names)) != len(gene_names):
        raise ValueError("gene_names must uniquely identify every native response column")

    intercept = train_values.mean(axis=0)
    centered = train_values - intercept
    # R's prcomp() is applied to genes x training-conditions. Its default
    # centering subtracts the mean of each training condition across genes.
    post_by_gene = post_train_values.T
    post_centered = post_by_gene - post_by_gene.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(post_centered, full_matrices=False)
    rank = min(int(n_components), u.shape[0], u.shape[1])
    if rank == 0:
        return np.tile(intercept, (len(query_labels), 1)).astype(np.float32)
    gene_embedding = u[:, :rank] * singular_values[:rank]
    gene_index = {gene: index for index, gene in enumerate(gene_names)}

    def perturbation_embedding(label: object) -> np.ndarray:
        known = [gene_embedding[gene_index[part]] for part in components(label) if part in gene_index]
        return np.mean(known, axis=0) if known else np.zeros(rank, dtype=np.float64)

    train_p = np.stack([perturbation_embedding(label) for label in train_labels])
    query_p = np.stack([perturbation_embedding(label) for label in query_labels])
    identity = np.eye(rank, dtype=np.float64)
    left_gram = gene_embedding.T @ gene_embedding + float(alpha) * identity
    right_gram = train_p.T @ train_p + float(alpha) * identity
    left = np.linalg.solve(left_gram, gene_embedding.T @ centered.T @ train_p)
    try:
        weights = np.linalg.solve(right_gram, left.T).T
    except np.linalg.LinAlgError:
        weights = left @ np.linalg.pinv(right_gram)
    return ((gene_embedding @ weights @ query_p.T).T + intercept).astype(np.float32)


def fit_slim_string_bilinear_ridge(
    train_labels: np.ndarray,
    train_post_values: np.ndarray,
    control_mean: np.ndarray,
    query_labels: np.ndarray,
    gene_names: list[str],
    string_embeddings: dict[str, np.ndarray],
    *,
    alpha: float = 0.1,
    n_components: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Fit the published SLIM basis/STRING bilinear model and return deltas.

    SLIM fits its PCA basis to post-perturbation pseudobulk expression, uses
    an external per-gene embedding for ``P``, solves the closed-form bilinear
    ridge, and predicts post-expression. We subtract the matched control mean
    here because the formal benchmark scores change-from-control responses.
    Combination labels use the explicit mean of available component vectors;
    absent components remain zero, matching SLIM's zero-fill inference policy.
    """

    post_values = np.asarray(train_post_values, dtype=np.float64)
    control_mean = np.asarray(control_mean, dtype=np.float64)
    if post_values.ndim != 2 or post_values.shape[0] == 0:
        raise ValueError("train_post_values must be a non-empty 2D array")
    if post_values.shape[1] != len(gene_names) or control_mean.shape != (len(gene_names),):
        raise ValueError("SLIM post/control gene dimensions do not match gene_names")
    if len(train_labels) != post_values.shape[0]:
        raise ValueError("SLIM train labels and post-expression rows differ")
    rank = min(int(n_components), post_values.shape[0] - 1, post_values.shape[1])
    if rank <= 0:
        return np.tile(post_values.mean(axis=0) - control_mean, (len(query_labels), 1)).astype(np.float32)
    # This is the default SLIM pca_basis path: gene rows are samples and the
    # train-mean-centered post-expression columns are the features.
    post_centered = post_values.T - post_values.mean(axis=0)[:, None]
    gene_embedding = PCA(n_components=rank, random_state=int(seed), svd_solver="auto").fit_transform(post_centered)
    gene_index = {gene: index for index, gene in enumerate(gene_names)}
    embedding_dim = len(next(iter(string_embeddings.values()))) if string_embeddings else 0

    def perturbation_embedding(label: object) -> np.ndarray:
        rows = [
            np.asarray(string_embeddings[part], dtype=np.float64)
            for part in components(label)
            if part in string_embeddings
        ]
        return np.mean(rows, axis=0) if rows else np.zeros(embedding_dim, dtype=np.float64)

    if embedding_dim == 0:
        raise ValueError("SLIM requires a non-empty STRING embedding table")
    train_p = np.stack([perturbation_embedding(label) for label in train_labels])
    query_p = np.stack([perturbation_embedding(label) for label in query_labels])
    response_bias = post_values.mean(axis=0)
    centered_response = post_values.T - response_bias[:, None]
    left = np.linalg.solve(
        gene_embedding.T @ gene_embedding + float(alpha) * np.eye(rank),
        gene_embedding.T @ centered_response,
    )
    weights = np.linalg.solve(
        train_p.T @ train_p + float(alpha) * np.eye(embedding_dim),
        (left @ train_p).T,
    ).T
    predicted_post = (gene_embedding @ weights @ query_p.T).T + response_bias
    return (predicted_post - control_mean).astype(np.float32)


def split_frame(frame: pd.DataFrame, manifest: pd.DataFrame, environment_id: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    local_manifest = manifest[manifest["environment_id"].eq(environment_id)].copy()
    key = ["environment_id", "perturbation_label", "dose", "timepoint"]
    local_manifest["dose"] = local_manifest["dose"].fillna("")
    local_manifest["timepoint"] = local_manifest["timepoint"].fillna("")
    frame = frame.copy()
    frame = frame.drop(columns=["biological_instance_id"], errors="ignore")
    for column in ("dose", "timepoint"):
        frame[column] = frame[column].fillna("").astype(str)
    joined = frame.merge(local_manifest[key + ["biological_instance_id", "split"]], on=key, how="inner", validate="one_to_one")
    if len(joined) != len(local_manifest):
        raise ValueError(f"ground truth/manifest mismatch for {environment_id}")
    if joined["biological_instance_id"].duplicated().any():
        raise ValueError(f"duplicate biological instances after join for {environment_id}")
    indices = {split: joined.index[joined["split"].eq(split)].to_numpy() for split in ("train", "validation", "test")}
    if any(len(indexes) == 0 for indexes in indices.values()):
        raise ValueError(f"empty formal split for {environment_id}")
    return joined, indices


def metric_row(true_values: np.ndarray, predicted: np.ndarray, labels: np.ndarray, uq: np.ndarray | None) -> dict[str, float]:
    cosine = safe_rowwise_cosine(true_values, predicted)
    spearman = rowwise_spearman(true_values, predicted)
    true_norm = np.linalg.norm(true_values, axis=1)
    pred_norm = np.linalg.norm(predicted, axis=1)
    top_overlaps = []
    for true_row, pred_row in zip(true_values, predicted):
        k = min(50, true_row.shape[0])
        true_top = set(np.argpartition(np.abs(true_row), -k)[-k:].tolist())
        pred_top = set(np.argpartition(np.abs(pred_row), -k)[-k:].tolist())
        top_overlaps.append(len(true_top & pred_top) / max(1, len(true_top | pred_top)))
    row = {
        "n": float(len(true_values)),
        "mean_fidelity": float(np.mean(cosine)),
        "median_fidelity": float(np.median(cosine)),
        "mean_spearman": float(np.mean(spearman)),
        "mean_rmse": float(np.sqrt(np.mean(np.square(predicted - true_values, dtype=np.float64)))),
        "mean_true_norm": float(np.mean(true_norm)),
        "mean_pred_norm": float(np.mean(pred_norm)),
        "mean_top_deg_overlap_at_50": float(np.mean(top_overlaps)),
    }
    if uq is not None:
        row.update({
            "uq_mean_gene_variance": float(np.mean(uq.mean_gene_variance)),
            "uq_median_gene_variance": float(np.mean(uq.median_gene_variance)),
            "uq_top_effect_variance": float(np.mean(uq.top_effect_variance)),
            "uq_cosine_disagreement": float(np.mean(uq.cosine_disagreement)),
            "uq_effect_norm_variance": float(np.mean(uq.effect_norm_variance)),
        })
    return row


def run(
    root: Path,
    output_dir: Path,
    metrics_path: Path,
    contract_path: Path,
    *,
    split_seed: int = 20260907,
    manifest_path: Path | None = None,
    isolated_output_dir: bool = False,
) -> dict[str, Any]:
    evaluation_genes = load_panel(root)
    registry = load_registry(root)
    manifest = load_manifest(root, manifest_path, split_seed=split_seed)
    commit = current_commit(root)
    if split_seed != 20260907 and not isolated_output_dir:
        output_dir = output_dir / f"split_seed_{split_seed}"
    if split_seed == 20260907 and isolated_output_dir and output_dir.name != "predictors":
        raise ValueError("primary split must use the canonical predictors directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    training_gene_counts_by_dataset: dict[str, int] = {}
    predictor_names = ("mean_matching", "strong_linear", "slim_string")
    slim_embeddings, slim_embedding_path = load_slim_string_embeddings(root)

    for dataset_id, registry_group in registry.groupby("dataset_id", sort=True):
        source = str(manifest.loc[manifest["environment_id"].isin(registry_group["environment_id"]), "ground_truth_path"].iloc[0])
        ground_truth = read_ground_truth(root, source, genes=None)
        native_genes = [column for column in ground_truth.columns if column not in METADATA_COLUMNS]
        missing_evaluation_genes = set(evaluation_genes).difference(native_genes)
        if missing_evaluation_genes:
            raise ValueError(
                f"{source} is missing frozen evaluation genes: {sorted(missing_evaluation_genes)[:5]}"
            )
        training_gene_counts_by_dataset[str(dataset_id)] = len(native_genes)
        evaluation_indices = [native_genes.index(gene) for gene in evaluation_genes]
        post_expression_cache: dict[
            tuple[str, str, tuple[str, ...]], tuple[dict[str, np.ndarray], np.ndarray]
        ] = {}
        for _, registry_row in registry_group.iterrows():
            environment_id = str(registry_row["environment_id"])
            environment_key = str(registry_row["environment_key"])
            joined, indices = split_frame(ground_truth[ground_truth["environment_id"].eq(environment_id)], manifest, environment_id)
            native_values = joined[native_genes].to_numpy(dtype=np.float32, copy=True)
            y = native_values[:, evaluation_indices]
            labels = joined["perturbation_label"].astype(str).to_numpy()
            weights = pd.to_numeric(joined["total_cells"], errors="coerce").fillna(1).to_numpy(dtype=np.float64)
            raw_condition_field = registry_row.get("condition_field", "")
            raw_condition = registry_row.get("condition", "")
            condition_field = "" if pd.isna(raw_condition_field) else str(raw_condition_field).strip()
            condition = "" if pd.isna(raw_condition) else str(raw_condition).strip()
            train_label_set = set(labels[indices["train"]].astype(str))
            post_cache_key = (condition_field, condition, tuple(sorted(train_label_set)))
            if post_cache_key not in post_expression_cache:
                post_expression_cache[post_cache_key] = read_training_post_expression(
                    root,
                    str(dataset_id),
                    train_label_set,
                    native_genes,
                    condition_field=condition_field,
                    condition=condition,
                )
            post_by_label, control_mean = post_expression_cache[post_cache_key]
            seed_values = (0, 1, 2)
            saved: dict[str, dict[str, Any]] = {}
            for predictor in predictor_names:
                validation_members: list[np.ndarray] = []
                test_members: list[np.ndarray] = []
                for seed in seed_values:
                    rng = np.random.default_rng(seed + 1009 * (list(registry["environment_id"]).index(environment_id) + 1))
                    train_indices = indices["train"]
                    bootstrap_indices = rng.choice(train_indices, size=len(train_indices), replace=True)
                    train_labels = labels[bootstrap_indices]
                    train_values = y[bootstrap_indices]
                    native_train_values = native_values[bootstrap_indices]
                    post_train_values = np.stack(
                        [post_by_label[str(label)] for label in train_labels]
                    )
                    train_weights = weights[bootstrap_indices]
                    query_validation = labels[indices["validation"]]
                    query_test = labels[indices["test"]]
                    if predictor == "mean_matching":
                        validation_prediction = fit_matching_mean(train_labels, train_values, train_weights, query_validation)
                        test_prediction = fit_matching_mean(train_labels, train_values, train_weights, query_test)
                        uq_source = "training_bootstrap_disagreement"
                    elif predictor == "strong_linear":
                        validation_prediction = fit_ahlmann_eltze_bilinear_ridge(
                            train_labels,
                            native_train_values,
                            query_validation,
                            gene_names=native_genes,
                            post_train_values=post_train_values,
                            alpha=0.1,
                            n_components=10,
                        )[:, evaluation_indices]
                        test_prediction = fit_ahlmann_eltze_bilinear_ridge(
                            train_labels,
                            native_train_values,
                            query_test,
                            gene_names=native_genes,
                            post_train_values=post_train_values,
                            alpha=0.1,
                            n_components=10,
                        )[:, evaluation_indices]
                        uq_source = "training_bootstrap_disagreement"
                    else:
                        validation_prediction = fit_slim_string_bilinear_ridge(
                            train_labels,
                            post_train_values,
                            control_mean,
                            query_validation,
                            native_genes,
                            slim_embeddings,
                            alpha=0.1,
                            n_components=10,
                            seed=seed,
                        )[:, evaluation_indices]
                        test_prediction = fit_slim_string_bilinear_ridge(
                            train_labels,
                            post_train_values,
                            control_mean,
                            query_test,
                            native_genes,
                            slim_embeddings,
                            alpha=0.1,
                            n_components=10,
                            seed=seed,
                        )[:, evaluation_indices]
                        uq_source = "training_bootstrap_disagreement"
                    validation_members.append(validation_prediction)
                    test_members.append(test_prediction)
                    metric = metric_row(y[indices["test"]], test_prediction, query_test, None)
                    metric.update({
                        "aggregation_level": "environment_predictor_seed",
                        "environment_id": environment_id,
                        "environment_key": environment_key,
                        "dataset_id": dataset_id,
                        "predictor": predictor,
                        "seed": str(seed),
                        "split_id": SPLIT_ID,
                        "split_seed": split_seed,
                        "uncertainty_source": uq_source,
                        "linear_embedding_source": "native_post_expression_X_train_only" if predictor == "strong_linear" else "not_applicable",
                        "linear_reference_release": "const-ae/linear_perturbation_prediction-Paper_publication_runner" if predictor == "strong_linear" else "not_applicable",
                        "slim_embedding_path": slim_embedding_path if predictor == "slim_string" else "not_applicable",
                        "training_gene_space_policy": "native_predictor_gene_space",
                        "training_gene_count": len(native_genes),
                        "evaluation_gene_space": "ptl_context_v2_intersection",
                        "evaluation_gene_count": len(evaluation_genes),
                    })
                    metric_rows.append(metric)
                validation_stack = np.stack(validation_members, axis=0)
                test_stack = np.stack(test_members, axis=0)
                validation_uq = summarize_ensemble(validation_stack)
                test_uq = summarize_ensemble(test_stack)
                array_path = output_dir / f"{environment_key}__{predictor}.npz"
                np.savez_compressed(
                    array_path,
                    validation_predictions=validation_stack,
                    test_predictions=test_stack,
                    validation_biological_instance_ids=joined.loc[indices["validation"], "biological_instance_id"].astype(str).to_numpy(),
                    test_biological_instance_ids=joined.loc[indices["test"], "biological_instance_id"].astype(str).to_numpy(),
                    model_seeds=np.asarray(seed_values, dtype=np.int64),
                    genes=np.asarray(evaluation_genes, dtype=str),
                )
                saved[predictor] = {"array_path": array_path, "validation_uq": validation_uq, "test_uq": test_uq}
                aggregate = metric_row(y[indices["test"]], test_stack.mean(axis=0), query_test, test_uq)
                aggregate.update({
                    "aggregation_level": "environment_predictor_ensemble_mean",
                    "environment_id": environment_id,
                    "environment_key": environment_key,
                    "dataset_id": dataset_id,
                    "predictor": predictor,
                    "seed": "ensemble_mean",
                    "split_id": SPLIT_ID,
                    "split_seed": split_seed,
                    "uncertainty_source": uq_source,
                    "linear_embedding_source": "native_post_expression_X_train_only" if predictor == "strong_linear" else "not_applicable",
                    "linear_reference_release": "const-ae/linear_perturbation_prediction-Paper_publication_runner" if predictor == "strong_linear" else "not_applicable",
                    "slim_embedding_path": slim_embedding_path if predictor == "slim_string" else "not_applicable",
                    "training_gene_space_policy": "native_predictor_gene_space",
                    "training_gene_count": len(native_genes),
                    "evaluation_gene_space": "ptl_context_v2_intersection",
                    "evaluation_gene_count": len(evaluation_genes),
                })
                metric_rows.append(aggregate)
                for split_name, split_indices, stack in (
                    ("validation", indices["validation"], validation_stack),
                    ("test", indices["test"], test_stack),
                ):
                    for member_index, seed in enumerate(seed_values):
                        prediction_array = str(array_path.relative_to(root).as_posix())
                        for biological_id in joined.loc[split_indices, "biological_instance_id"].astype(str):
                            contract_rows.append({
                                "prediction_id": prediction_id(biological_id, predictor, SPLIT_ID, seed),
                                "biological_instance_id": biological_id,
                                "environment_id": environment_id,
                                "environment_key": environment_key,
                                "predictor": predictor,
                                "predictor_version": PREDICTOR_VERSIONS[predictor],
                                "predictor_commit": commit,
                                "predictor_split_id": SPLIT_ID,
                                "split_seed": split_seed,
                                "seed": seed,
                                "split": split_name,
                                "prediction_array_path": prediction_array,
                                "array_key": f"{split_name}_predictions[{member_index}]",
                                "gene_list_path": "artifacts/manifests/evaluation_gene_space.csv",
                                "training_gene_space_policy": "native_predictor_gene_space",
                                "training_gene_count": len(native_genes),
                                "evaluation_gene_space": "ptl_context_v2_intersection",
                                "evaluation_gene_count": len(evaluation_genes),
                                "uncertainty_source": uq_source,
                                "ensemble_member_seeds": "0,1,2",
                            })

    metric_frame = pd.DataFrame(metric_rows)
    metric_frame = metric_frame.sort_values(["aggregation_level", "environment_id", "predictor", "seed"], kind="stable")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metric_frame.to_csv(metrics_path, index=False)
    contract_frame = pd.DataFrame(contract_rows)
    contract_frame = contract_frame.sort_values(["environment_id", "predictor", "split", "seed", "biological_instance_id"], kind="stable")
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_frame.to_csv(contract_path, index=False)
    summary = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "split_id": SPLIT_ID,
        "split_seed": split_seed,
        "predictors": list(predictor_names),
        "environments": int(registry["environment_id"].nunique()),
        "model_seeds": list(seed_values),
        "metric_rows": int(len(metric_frame)),
        "contract_rows": int(len(contract_frame)),
        "predictor_version": PREDICTOR_VERSION,
        "predictor_versions": PREDICTOR_VERSIONS,
        "slim_embedding_path": slim_embedding_path,
        "metrics_path": metrics_path.relative_to(root).as_posix(),
        "contract_path": contract_path.relative_to(root).as_posix(),
        "evaluation_gene_space": "ptl_context_v2_intersection",
        "evaluation_gene_count": len(evaluation_genes),
        "training_gene_space_policy": "native_predictor_gene_space",
        "training_gene_counts_by_dataset": training_gene_counts_by_dataset,
        "status": "formal_v2_simple_predictors_executed",
    }
    summary_name = "formal_v2_predictor_summary.json" if split_seed == 20260907 else f"formal_v2_predictor_summary__split_{split_seed}.json"
    (metrics_path.parent / summary_name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/formal_v2/predictors")
    parser.add_argument("--metrics", type=Path, default=ROOT / "artifacts/manifests/formal_v2_predictor_metrics.csv")
    parser.add_argument("--contract", type=Path, default=ROOT / "artifacts/manifests/formal_v2_prediction_index.csv")
    parser.add_argument("--split-seed", type=int, default=20260907)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--isolated-output-dir", action="store_true")
    args = parser.parse_args()
    result = run(
        args.root.resolve(), args.output_dir.resolve(), args.metrics.resolve(), args.contract.resolve(),
        split_seed=args.split_seed,
        manifest_path=args.manifest.resolve() if args.manifest else None,
        isolated_output_dir=args.isolated_output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
