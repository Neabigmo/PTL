from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
TRAINING_LOG_DIR = RESULTS_DIR / "logs" / "training"
BASELINE_OUTPUT_DIR = RESULTS_DIR / "baselines"
SPLIT_AUDIT_PATH = TABLES_DIR / "split_audit.csv"
PREPROCESSING_SUMMARY_PATH = TABLES_DIR / "preprocessing_summary.csv"
BASELINE_RUN_MATRIX_PATH = TABLES_DIR / "baseline_run_matrix.csv"
EXPERIMENT_REGISTRY_PATH = TABLES_DIR / "experiment_registry.csv"
BASELINE_METRICS_PATH = TABLES_DIR / "baseline_metrics_by_split.csv"
RANKING_INSTABILITY_PATH = TABLES_DIR / "model_ranking_instability.csv"
GLOBAL_GENE_SPACE_PATH = TABLES_DIR / "phase05_global_gene_intersection_6257.txt"

BASELINE_MODELS = [
    "control_mean_baseline",
    "global_delta_baseline",
    "perturbation_mean_delta_baseline",
    "cell_context_knn_delta_baseline",
    "ridge_regression_baseline",
]
K_GRID = [3, 5, 10, 20]
RIDGE_ALPHA_GRID = [0.1, 1.0, 10.0, 100.0]
PRIMARY_METRIC = "mean_cosine_similarity_non_control_test"
PHASE_NAME = "Phase 05"
EPS = 1e-8


def resolve_artifact_path(value: object) -> Path:
    """Resolve current-workspace artifacts while migrating old summaries."""

    raw = Path(str(value))
    if raw.is_file():
        return raw
    normalized = str(value).replace("\\", "/")
    for marker in ("data/processed/", "results/"):
        if marker in normalized:
            candidate = ROOT / marker.rstrip("/") / normalized.split(marker, 1)[1]
            if candidate.is_file():
                return candidate
    return raw

SIGNATURE_METADATA_COLUMNS = {
    "dataset_id",
    "source_dataset",
    "signature_id",
    "reference_key",
    "perturbation_label",
    "is_control",
    "n_cells",
    "control_label_used",
    "batch",
    "timepoint",
}
PSEUDOBULK_METADATA_COLUMNS = {
    "dataset_id",
    "source_dataset",
    "group_key",
    "reference_key",
    "perturbation_label",
    "is_control",
    "batch",
    "timepoint",
    "n_cells",
}


@dataclass
class DatasetArtifacts:
    dataset_id: str
    source_dataset: str
    signature_path: Path
    pseudobulk_path: Path
    gene_columns: list[str]
    signature_df: pd.DataFrame
    pseudobulk_df: pd.DataFrame
    signature_index: dict[str, int]
    control_lookup: dict[tuple[str, str], np.ndarray]
    gene_index: dict[str, int]


@dataclass
class PreparedSplitData:
    split_payload: dict[str, Any]
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    gene_columns: list[str]
    ref_train: np.ndarray
    ref_val: np.ndarray
    ref_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


@dataclass
class RunSpec:
    run_id: str
    model: str
    split_json_path: Path
    split_family: str
    dataset_scope: str
    heldout_target: str
    seed: int
    gene_space_policy: str
    expected_gene_count: int
    output_dir: Path
    log_file: Path


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


class DataRepository:
    def __init__(self, preprocessing_summary_path: Path) -> None:
        self.summary_df = pd.read_csv(preprocessing_summary_path)
        self.summary_df = self.summary_df[self.summary_df["status"] == "processed"].copy()
        self._cache: dict[tuple[str, tuple[str, ...] | None], DatasetArtifacts] = {}
        self._global_gene_space: list[str] | None = None
        self._available_gene_cache: dict[str, list[str]] = {}

    def dataset_ids(self) -> list[str]:
        return self.summary_df["dataset_id"].tolist()

    def summary_record(self, dataset_id: str) -> pd.Series:
        row = self.summary_df.loc[self.summary_df["dataset_id"] == dataset_id]
        if row.empty:
            raise KeyError(f"Dataset '{dataset_id}' not present in preprocessing summary.")
        return row.iloc[0]

    def feature_metadata_path_for(self, record: pd.Series) -> Path:
        feature_metadata_value = record["feature_metadata_path"] if "feature_metadata_path" in record.index else ""
        return resolve_artifact_path(feature_metadata_value) if str(feature_metadata_value) and str(feature_metadata_value) != "nan" else Path("")

    def available_genes(self, dataset_id: str) -> list[str]:
        if dataset_id in self._available_gene_cache:
            return self._available_gene_cache[dataset_id]
        record = self.summary_record(dataset_id)
        feature_metadata_path = self.feature_metadata_path_for(record)
        if feature_metadata_path.is_file():
            feature_frame = pd.read_parquet(feature_metadata_path)
            genes = feature_frame.index.astype(str).tolist()
        else:
            signature_df = pd.read_parquet(resolve_artifact_path(record["signature_path"]))
            genes = [
                column
                for column in signature_df.columns
                if column not in SIGNATURE_METADATA_COLUMNS and pd.api.types.is_numeric_dtype(signature_df[column])
            ]
        self._available_gene_cache[dataset_id] = genes
        return genes

    def load_dataset(self, dataset_id: str, gene_subset: list[str] | None = None) -> DatasetArtifacts:
        cache_key = (dataset_id, tuple(gene_subset) if gene_subset is not None else None)
        if cache_key in self._cache:
            return self._cache[cache_key]

        record = self.summary_record(dataset_id)
        signature_path = resolve_artifact_path(record["signature_path"])
        pseudobulk_path = resolve_artifact_path(record["pseudobulk_path"])

        available_genes = self.available_genes(dataset_id)
        if gene_subset is not None:
            requested = set(gene_subset)
            gene_columns = [gene for gene in available_genes if gene in requested]
        else:
            gene_columns = available_genes
        if not gene_columns:
            raise ValueError(f"No numeric gene columns could be resolved for {dataset_id}.")

        signature_schema_columns = set(pq.read_schema(signature_path).names)
        pseudobulk_schema_columns = set(pq.read_schema(pseudobulk_path).names)
        signature_columns = [column for column in list(SIGNATURE_METADATA_COLUMNS) + gene_columns if column in signature_schema_columns]
        pseudobulk_columns = [column for column in list(PSEUDOBULK_METADATA_COLUMNS) + ["control_label_used"] + gene_columns if column in pseudobulk_schema_columns]
        signature_df = pd.read_parquet(signature_path, columns=signature_columns)
        pseudobulk_df = pd.read_parquet(pseudobulk_path, columns=pseudobulk_columns)
        gene_columns = [gene for gene in gene_columns if gene in signature_df.columns and gene in pseudobulk_df.columns]
        if not gene_columns:
            raise ValueError(f"No requested gene columns were present in both signature and pseudobulk for {dataset_id}.")
        signature_df[gene_columns] = signature_df[gene_columns].astype(np.float32)
        pseudobulk_gene_columns = [gene for gene in gene_columns if gene in pseudobulk_df.columns]
        if gene_columns != pseudobulk_gene_columns:
            raise ValueError(f"Gene columns mismatch between signature and pseudobulk for {dataset_id}.")
        pseudobulk_df[gene_columns] = pseudobulk_df[gene_columns].astype(np.float32)

        signature_index = {signature_id: idx for idx, signature_id in enumerate(signature_df["signature_id"].tolist())}
        controls = pseudobulk_df[pseudobulk_df["is_control"]].copy()
        control_lookup: dict[tuple[str, str], np.ndarray] = {}
        for _, control_row in controls.iterrows():
            key = (str(control_row["reference_key"]), str(control_row["perturbation_label"]))
            control_lookup[key] = control_row[gene_columns].to_numpy(dtype=np.float32, copy=True)

        artifacts = DatasetArtifacts(
            dataset_id=dataset_id,
            source_dataset=str(record["source_dataset"]),
            signature_path=signature_path,
            pseudobulk_path=pseudobulk_path,
            gene_columns=gene_columns,
            signature_df=signature_df,
            pseudobulk_df=pseudobulk_df,
            signature_index=signature_index,
            control_lookup=control_lookup,
            gene_index={gene: idx for idx, gene in enumerate(gene_columns)},
        )
        self._cache[cache_key] = artifacts
        return artifacts

    def global_gene_space(self) -> list[str]:
        if self._global_gene_space is None:
            gene_sets = [set(self.available_genes(dataset_id)) for dataset_id in self.dataset_ids()]
            intersection = set.intersection(*gene_sets)
            self._global_gene_space = sorted(intersection)
        return self._global_gene_space


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_gene_list(path: Path, genes: list[str]) -> None:
    path.write_text("\n".join(json.dumps(str(gene)) for gene in genes) + "\n", encoding="utf-8")


def read_gene_list(path: Path) -> list[str]:
    genes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('"'):
            genes.append(str(json.loads(stripped)))
        else:
            genes.append(stripped)
    return genes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 05 baseline benchmark runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_manifest = subparsers.add_parser("build-run-matrix", help="Build the 525-run campaign manifest.")
    build_manifest.add_argument("--overwrite", action="store_true", help="Overwrite existing run matrix.")

    run_all = subparsers.add_parser("run-all", help="Build the manifest if needed and execute the full campaign.")
    run_all.add_argument("--manifest", default=str(BASELINE_RUN_MATRIX_PATH), help="Path to the run manifest CSV.")
    run_all.add_argument("--force", action="store_true", help="Re-run runs even if outputs already exist.")

    run_one = subparsers.add_parser("run-one", help="Run a single baseline on one split JSON.")
    run_one.add_argument("--model", choices=BASELINE_MODELS, required=True)
    run_one.add_argument("--split-json", required=True)
    run_one.add_argument("--output-dir", required=True)
    run_one.add_argument("--seed", type=int, required=True)
    run_one.add_argument("--run-id", default="")

    summarize = subparsers.add_parser("summarize", help="Build aggregate Phase 05 summary tables from registry/metrics.")
    summarize.add_argument("--manifest", default=str(BASELINE_RUN_MATRIX_PATH))

    return parser.parse_args()


def load_split_audit_ready_signature_rows() -> pd.DataFrame:
    split_audit = pd.read_csv(SPLIT_AUDIT_PATH)
    ready = split_audit[(split_audit["track"] == "signature") & (split_audit["status"] == "ready")].copy()
    ready = ready.sort_values(["split_family", "dataset_scope", "heldout_target", "seed", "output_path"]).reset_index(drop=True)
    return ready


def build_run_matrix(repository: DataRepository, overwrite: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if BASELINE_RUN_MATRIX_PATH.exists() and not overwrite:
        return pd.read_csv(BASELINE_RUN_MATRIX_PATH)

    ready = load_split_audit_ready_signature_rows()
    global_gene_space = repository.global_gene_space()

    planned_rows: list[dict[str, Any]] = []
    for _, row in ready.iterrows():
        split_json_path = Path(row["output_path"])
        split_payload = read_json(split_json_path)
        gene_space_policy = (
            f"global_intersection_{len(global_gene_space)}"
            if row["split_family"] in {"dataset_heldout_split", "external_holdout"}
            else "native"
        )
        expected_gene_count = len(global_gene_space) if gene_space_policy.startswith("global") else infer_native_gene_count(repository, split_payload)
        split_stem = split_json_path.stem
        for model in BASELINE_MODELS:
            run_id = f"{split_stem}__{model}"
            output_dir = BASELINE_OUTPUT_DIR / model / split_stem
            log_file = TRAINING_LOG_DIR / f"{run_id}.log"
            planned_rows.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "split_json_path": str(split_json_path),
                    "split_family": row["split_family"],
                    "dataset_scope": row["dataset_scope"],
                    "heldout_target": row["heldout_target"],
                    "seed": int(row["seed"]),
                    "gene_space_policy": gene_space_policy,
                    "expected_gene_count": int(expected_gene_count),
                    "output_dir": str(output_dir),
                    "log_file": str(log_file),
                }
            )

    manifest = pd.DataFrame(planned_rows)
    manifest.to_csv(BASELINE_RUN_MATRIX_PATH, index=False)
    GLOBAL_GENE_SPACE_PATH.write_text("\n".join(global_gene_space) + "\n", encoding="utf-8")
    return manifest


def infer_native_gene_count(repository: DataRepository, split_payload: dict[str, Any]) -> int:
    if split_payload["dataset_scope"] == "all_datasets":
        heldout_target = split_payload["heldout_target"]
        if split_payload["split_family"] == "external_holdout":
            dataset_id = heldout_target
        else:
            dataset_ids = [dataset_id for dataset_id in split_payload.get("dataset_members", []) if dataset_id != heldout_target]
            dataset_id = dataset_ids[0]
    else:
        dataset_id = split_payload["dataset_scope"]
    return len(repository.available_genes(dataset_id))


def dataset_id_from_signature_id(signature_id: str) -> str:
    if "__" not in signature_id:
        raise ValueError(f"Malformed signature_id '{signature_id}'.")
    return signature_id.split("__", 1)[0]


def select_rows_for_ids(repository: DataRepository, ids: list[str], gene_columns: list[str] | None = None) -> pd.DataFrame:
    grouped: dict[str, list[str]] = {}
    for signature_id in ids:
        dataset_id = dataset_id_from_signature_id(signature_id)
        grouped.setdefault(dataset_id, []).append(signature_id)

    frames: list[pd.DataFrame] = []
    for dataset_id, signature_ids in grouped.items():
        dataset = repository.load_dataset(dataset_id, gene_subset=gene_columns)
        indices = [dataset.signature_index[signature_id] for signature_id in signature_ids]
        subset = dataset.signature_df.iloc[indices].copy()
        frames.append(subset)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_reference_feature_matrix(frame: pd.DataFrame, repository: DataRepository, gene_columns: list[str]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for _, row in frame.iterrows():
        dataset = repository.load_dataset(str(row["dataset_id"]), gene_subset=gene_columns)
        key = (str(row["reference_key"]), str(row["control_label_used"]))
        if key not in dataset.control_lookup:
            available = sorted(dataset.control_lookup.keys())
            raise KeyError(f"Missing control profile for {dataset.dataset_id} key={key}. Available examples: {available[:5]}")
        control_vector_full = dataset.control_lookup[key]
        if gene_columns == dataset.gene_columns:
            vector = control_vector_full
        else:
            positions = [dataset.gene_index[gene] for gene in gene_columns]
            vector = control_vector_full[positions]
        rows.append(vector.astype(np.float32, copy=False))
    return np.vstack(rows).astype(np.float32, copy=False)


def build_target_matrix(frame: pd.DataFrame, gene_columns: list[str]) -> np.ndarray:
    return frame[gene_columns].to_numpy(dtype=np.float32, copy=True)


def metadata_frame(frame: pd.DataFrame, include_perturbation: bool) -> pd.DataFrame:
    columns = ["dataset_id", "batch"]
    if "timepoint" in frame.columns:
        columns.append("timepoint")
    meta = frame.reindex(columns=columns, fill_value="")
    if "timepoint" not in meta.columns:
        meta["timepoint"] = ""
    if include_perturbation:
        meta["perturbation_label"] = frame["perturbation_label"].astype(str)
    return meta.fillna("").astype(str)


def build_ridge_features(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    train_ref: np.ndarray,
    val_ref: np.ndarray,
    test_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], OneHotEncoder]:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    train_meta = metadata_frame(train_frame, include_perturbation=True)
    val_meta = metadata_frame(val_frame, include_perturbation=True)
    test_meta = metadata_frame(test_frame, include_perturbation=True)
    train_encoded = encoder.fit_transform(train_meta)
    val_encoded = encoder.transform(val_meta)
    test_encoded = encoder.transform(test_meta)
    train_features = np.hstack([train_ref, train_encoded]).astype(np.float32, copy=False)
    val_features = np.hstack([val_ref, val_encoded]).astype(np.float32, copy=False)
    test_features = np.hstack([test_ref, test_encoded]).astype(np.float32, copy=False)
    feature_names = [f"gene::{name}" for name in range(train_ref.shape[1])] + encoder.get_feature_names_out(train_meta.columns.tolist()).tolist()
    return train_features, val_features, test_features, feature_names, encoder


def fit_predict_dual_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)
    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    x_centered = x_train - x_mean
    y_centered = y_train - y_mean
    gram = x_centered @ x_centered.T
    base_alpha = float(alpha)
    gram.flat[:: gram.shape[0] + 1] += base_alpha
    solve_jitter = 0.0
    jitter_grid = [0.0, max(base_alpha * 1e-6, 1e-8), max(base_alpha * 1e-4, 1e-6), max(base_alpha * 1e-2, 1e-4)]
    dual_coef = None
    for solve_jitter in jitter_grid:
        try:
            if solve_jitter > 0:
                gram.flat[:: gram.shape[0] + 1] += solve_jitter
            dual_coef = linalg.solve(gram, y_centered, assume_a="sym", check_finite=False)
            break
        except np.linalg.LinAlgError:
            if solve_jitter > 0:
                gram.flat[:: gram.shape[0] + 1] -= solve_jitter
            dual_coef = None
            continue
    if dual_coef is None:
        dual_coef = linalg.lstsq(gram, y_centered, check_finite=False)[0]
    weights = x_centered.T @ dual_coef
    predictions = (x_eval - x_mean) @ weights + y_mean
    return predictions.astype(np.float32, copy=False)


def prepare_dual_ridge_state(x_train: np.ndarray, y_train: np.ndarray) -> dict[str, np.ndarray]:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    x_centered = x_train - x_mean
    y_centered = y_train - y_mean
    gram = x_centered @ x_centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    projected_targets = eigenvectors.T @ y_centered
    return {
        "x_mean": x_mean,
        "y_mean": y_mean,
        "x_centered": x_centered,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "projected_targets": projected_targets,
    }


def predict_dual_ridge_from_state(state: dict[str, np.ndarray], x_eval: np.ndarray, alpha: float) -> np.ndarray:
    x_eval = np.asarray(x_eval, dtype=np.float64)
    x_eval_centered = x_eval - state["x_mean"]
    cross_kernel = x_eval_centered @ state["x_centered"].T
    denominator = state["eigenvalues"] + float(alpha)
    denominator = np.where(denominator <= EPS, denominator + max(float(alpha) * 1e-4, 1e-6), denominator)
    scaled = state["projected_targets"] / denominator[:, None]
    dual_eval = cross_kernel @ state["eigenvectors"]
    predictions = dual_eval @ scaled + state["y_mean"]
    return predictions.astype(np.float32, copy=False)


def weighted_mean(matrix: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if matrix.size == 0:
        raise ValueError("Cannot compute a weighted mean on an empty matrix.")
    if weights is None:
        return matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        weights = np.ones_like(weights, dtype=np.float64)
        total = float(weights.sum())
    mean = (matrix.astype(np.float64) * weights[:, None]).sum(axis=0) / total
    return mean.astype(np.float32)


def safe_rowwise_cosine(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_norm = np.linalg.norm(y_true, axis=1)
    pred_norm = np.linalg.norm(y_pred, axis=1)
    numerator = np.sum(y_true * y_pred, axis=1)
    denominator = true_norm * pred_norm
    cosine = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator > EPS
    cosine[valid] = numerator[valid] / denominator[valid]
    both_zero = (true_norm <= EPS) & (pred_norm <= EPS)
    cosine[both_zero] = 1.0
    cosine = np.clip(cosine, -1.0, 1.0)
    return cosine


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_control: np.ndarray) -> dict[str, Any]:
    non_control_mask = ~is_control
    control_mask = is_control

    metrics: dict[str, Any] = {
        "n_test_noncontrol": int(non_control_mask.sum()),
        "n_test_control": int(control_mask.sum()),
    }

    if non_control_mask.any():
        nc_true = y_true[non_control_mask]
        nc_pred = y_pred[non_control_mask]
        nc_cos = safe_rowwise_cosine(nc_true, nc_pred)
        diff = nc_pred - nc_true
        metrics["mean_cosine_similarity_non_control_test"] = float(np.mean(nc_cos))
        metrics["mse_non_control_test"] = float(np.mean(np.square(diff, dtype=np.float64)))
        metrics["mae_non_control_test"] = float(np.mean(np.abs(diff, dtype=np.float64)))
    else:
        metrics["mean_cosine_similarity_non_control_test"] = float("nan")
        metrics["mse_non_control_test"] = float("nan")
        metrics["mae_non_control_test"] = float("nan")

    if control_mask.any():
        c_true = y_true[control_mask]
        c_pred = y_pred[control_mask]
        c_cos = safe_rowwise_cosine(c_true, c_pred)
        metrics["mean_cosine_similarity_control_test"] = float(np.mean(c_cos))
    else:
        metrics["mean_cosine_similarity_control_test"] = float("nan")
    return metrics


def primary_metric_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, is_control: np.ndarray) -> float:
    metrics = summarize_metrics(y_true, y_pred, is_control)
    value = metrics[PRIMARY_METRIC]
    if isinstance(value, float) and math.isnan(value):
        return float("-inf")
    return float(value)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return matrix / norms


def predict_knn(
    train_ref: np.ndarray,
    train_target: np.ndarray,
    train_perturbations: np.ndarray,
    train_weights: np.ndarray,
    query_ref: np.ndarray,
    query_perturbations: np.ndarray,
    k: int,
    global_mean: np.ndarray,
) -> np.ndarray:
    normalized_train = normalize_rows(train_ref.astype(np.float32, copy=False))
    normalized_query = normalize_rows(query_ref.astype(np.float32, copy=False))
    predictions = np.zeros((query_ref.shape[0], train_target.shape[1]), dtype=np.float32)
    for idx, perturbation in enumerate(query_perturbations):
        same_mask = train_perturbations == perturbation
        if same_mask.any():
            candidate_ref = normalized_train[same_mask]
            candidate_target = train_target[same_mask]
            candidate_weights = train_weights[same_mask]
        else:
            predictions[idx] = global_mean
            continue

        similarities = candidate_ref @ normalized_query[idx]
        distances = 1.0 - np.clip(similarities, -1.0, 1.0)
        neighbor_count = min(k, candidate_target.shape[0])
        nearest = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
        nearest_distances = distances[nearest]
        nearest_targets = candidate_target[nearest]
        nearest_support = candidate_weights[nearest]

        if np.any(nearest_distances <= EPS):
            zero_dist_mask = nearest_distances <= EPS
            weights = nearest_support[zero_dist_mask].astype(np.float64)
            if weights.sum() <= 0:
                weights = np.ones_like(weights, dtype=np.float64)
            prediction = np.average(nearest_targets[zero_dist_mask], axis=0, weights=weights)
        else:
            distance_weights = (1.0 / np.maximum(nearest_distances, EPS)) * nearest_support
            if distance_weights.sum() <= 0:
                distance_weights = np.ones_like(distance_weights, dtype=np.float64)
            prediction = np.average(nearest_targets, axis=0, weights=distance_weights)
        predictions[idx] = prediction.astype(np.float32)
    return predictions


def parse_combination_components(label: str) -> list[str]:
    if not label or label in {"control", "NT", "WT", "MULTI_TARGET"}:
        return []
    if "_" not in label:
        return []
    return [part for part in label.split("_") if part]


def build_perturbation_mean_lookup(train_frame: pd.DataFrame, y_train: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    train_non_control = train_frame[~train_frame["is_control"]].reset_index(drop=True)
    y_non_control = y_train[~train_frame["is_control"].to_numpy(dtype=bool)]
    weights = train_non_control["n_cells"].to_numpy(dtype=np.float64)
    global_mean = weighted_mean(y_non_control, weights)
    perturbation_means: dict[str, np.ndarray] = {}
    singleton_means: dict[str, np.ndarray] = {}
    grouped = train_non_control.groupby("perturbation_label", sort=False)
    for perturbation, subset in grouped:
        subset_indices = subset.index.to_numpy(dtype=int)
        subset_weights = weights[subset_indices]
        mean_vector = weighted_mean(y_non_control[subset_indices], subset_weights)
        perturbation_means[str(perturbation)] = mean_vector
        if not parse_combination_components(str(perturbation)):
            singleton_means[str(perturbation)] = mean_vector
    return perturbation_means, singleton_means, global_mean


def predict_perturbation_mean(
    frame: pd.DataFrame,
    perturbation_means: dict[str, np.ndarray],
    singleton_means: dict[str, np.ndarray],
    global_mean: np.ndarray,
    output_dim: int,
) -> np.ndarray:
    predictions = np.zeros((len(frame), output_dim), dtype=np.float32)
    for idx, row in enumerate(frame.itertuples(index=False)):
        if bool(row.is_control):
            continue
        perturbation = str(row.perturbation_label)
        if perturbation in perturbation_means:
            predictions[idx] = perturbation_means[perturbation]
            continue
        components = parse_combination_components(perturbation)
        if components and all(component in singleton_means for component in components):
            predictions[idx] = np.mean([singleton_means[component] for component in components], axis=0, dtype=np.float64).astype(np.float32)
            continue
        predictions[idx] = global_mean
    return predictions


def run_model(
    model_name: str,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    ref_train: np.ndarray,
    ref_val: np.ndarray,
    ref_test: np.ndarray,
    logger: RunLogger,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_is_control = train_frame["is_control"].to_numpy(dtype=bool)
    val_is_control = val_frame["is_control"].to_numpy(dtype=bool)
    test_is_control = test_frame["is_control"].to_numpy(dtype=bool)
    output_dim = y_test.shape[1]

    train_non_control_frame = train_frame.loc[~train_is_control].reset_index(drop=True)
    y_train_non_control = y_train[~train_is_control]
    ref_train_non_control = ref_train[~train_is_control]
    train_weights = train_non_control_frame["n_cells"].to_numpy(dtype=np.float64)

    model_details: dict[str, Any] = {"model_name": model_name}

    if model_name == "control_mean_baseline":
        predictions = np.zeros((len(test_frame), output_dim), dtype=np.float32)
        model_details["selection_metric"] = None
        return predictions, model_details

    if y_train_non_control.shape[0] == 0:
        raise ValueError(f"{model_name} requires at least one non-control training row.")

    global_mean = weighted_mean(y_train_non_control, train_weights)
    model_details["global_non_control_mean_norm"] = float(np.linalg.norm(global_mean))

    if model_name == "global_delta_baseline":
        predictions = np.tile(global_mean[None, :], (len(test_frame), 1)).astype(np.float32)
        predictions[test_is_control] = 0.0
        return predictions, model_details

    if model_name == "perturbation_mean_delta_baseline":
        perturbation_means, singleton_means, computed_global_mean = build_perturbation_mean_lookup(train_frame, y_train)
        predictions = predict_perturbation_mean(test_frame, perturbation_means, singleton_means, computed_global_mean, output_dim)
        model_details["n_exact_train_perturbations"] = int(len(perturbation_means))
        model_details["n_singleton_train_perturbations"] = int(len(singleton_means))
        return predictions, model_details

    if model_name == "cell_context_knn_delta_baseline":
        candidate_metrics: list[tuple[float, int]] = []
        train_perturbations = train_non_control_frame["perturbation_label"].astype(str).to_numpy()
        val_perturbations = val_frame["perturbation_label"].astype(str).to_numpy()
        for candidate_k in K_GRID:
            val_pred = predict_knn(
                train_ref=ref_train_non_control,
                train_target=y_train_non_control,
                train_perturbations=train_perturbations,
                train_weights=train_weights,
                query_ref=ref_val,
                query_perturbations=val_perturbations,
                k=candidate_k,
                global_mean=global_mean,
            )
            val_pred[val_is_control] = 0.0
            metric_value = primary_metric_from_predictions(y_val, val_pred, val_is_control)
            candidate_metrics.append((metric_value, candidate_k))
            logger.log(f"kNN validation metric at k={candidate_k}: {metric_value:.6f}")
        best_metric, best_k = max(candidate_metrics, key=lambda item: (item[0], -item[1]))
        test_predictions = predict_knn(
            train_ref=ref_train_non_control,
            train_target=y_train_non_control,
            train_perturbations=train_perturbations,
            train_weights=train_weights,
            query_ref=ref_test,
            query_perturbations=test_frame["perturbation_label"].astype(str).to_numpy(),
            k=best_k,
            global_mean=global_mean,
        )
        test_predictions[test_is_control] = 0.0
        model_details["selected_k"] = int(best_k)
        model_details["validation_metric"] = float(best_metric)
        model_details["grid"] = K_GRID
        return test_predictions, model_details

    if model_name == "ridge_regression_baseline":
        train_x, val_x, test_x, feature_names, encoder = build_ridge_features(
            train_non_control_frame,
            val_frame.reset_index(drop=True),
            test_frame.reset_index(drop=True),
            ref_train_non_control,
            ref_val,
            ref_test,
        )
        val_targets = y_val
        logger.log("Preparing dual ridge eigensystem.")
        ridge_state = prepare_dual_ridge_state(train_x, y_train_non_control)
        candidate_metrics = []
        for alpha in RIDGE_ALPHA_GRID:
            val_predictions = predict_dual_ridge_from_state(ridge_state, val_x, alpha=alpha)
            val_predictions[val_is_control] = 0.0
            metric_value = primary_metric_from_predictions(val_targets, val_predictions, val_is_control)
            candidate_metrics.append((metric_value, alpha))
            logger.log(f"Ridge validation metric at alpha={alpha}: {metric_value:.6f}")
        best_metric, best_alpha = max(candidate_metrics, key=lambda item: (item[0], -item[1]))
        test_predictions = predict_dual_ridge_from_state(ridge_state, test_x, alpha=best_alpha)
        test_predictions[test_is_control] = 0.0
        model_details["selected_alpha"] = float(best_alpha)
        model_details["validation_metric"] = float(best_metric)
        model_details["grid"] = RIDGE_ALPHA_GRID
        model_details["n_feature_columns"] = int(train_x.shape[1])
        model_details["n_encoded_metadata_columns"] = int(train_x.shape[1] - ref_train_non_control.shape[1])
        model_details["metadata_feature_names"] = encoder.get_feature_names_out().tolist()
        model_details["feature_names_summary"] = {
            "n_gene_features": int(ref_train_non_control.shape[1]),
            "n_total_features": int(train_x.shape[1]),
        }
        model_details["ridge_solver"] = "custom_dual_closed_form"
        _ = feature_names  # keeps the variable explicit for future debugging without storing the entire list here
        return test_predictions, model_details

    raise ValueError(f"Unknown model '{model_name}'.")


def materialize_run_data(
    repository: DataRepository,
    split_payload: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    gene_columns = (
        repository.global_gene_space()
        if split_payload["split_family"] in {"dataset_heldout_split", "external_holdout"}
        else repository.load_dataset(split_payload["dataset_scope"]).gene_columns
    )
    train_frame = select_rows_for_ids(repository, split_payload["train_ids"], gene_columns=gene_columns)
    val_frame = select_rows_for_ids(repository, split_payload["validation_ids"], gene_columns=gene_columns)
    test_frame = select_rows_for_ids(repository, split_payload["test_ids"], gene_columns=gene_columns)
    return train_frame, val_frame, test_frame, gene_columns


def prepare_split_data(repository: DataRepository, split_json_path: Path) -> PreparedSplitData:
    split_payload = read_json(split_json_path)
    train_frame, val_frame, test_frame, gene_columns = materialize_run_data(repository, split_payload)
    ref_train = build_reference_feature_matrix(train_frame, repository, gene_columns)
    ref_val = build_reference_feature_matrix(val_frame, repository, gene_columns)
    ref_test = build_reference_feature_matrix(test_frame, repository, gene_columns)
    y_train = build_target_matrix(train_frame, gene_columns)
    y_val = build_target_matrix(val_frame, gene_columns)
    y_test = build_target_matrix(test_frame, gene_columns)
    return PreparedSplitData(
        split_payload=split_payload,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        gene_columns=gene_columns,
        ref_train=ref_train,
        ref_val=ref_val,
        ref_test=ref_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def manifest_row_to_run_spec(row: pd.Series) -> RunSpec:
    return RunSpec(
        run_id=str(row["run_id"]),
        model=str(row["model"]),
        split_json_path=Path(row["split_json_path"]),
        split_family=str(row["split_family"]),
        dataset_scope=str(row["dataset_scope"]),
        heldout_target=str(row["heldout_target"]) if not pd.isna(row["heldout_target"]) else "",
        seed=int(row["seed"]),
        gene_space_policy=str(row["gene_space_policy"]),
        expected_gene_count=int(row["expected_gene_count"]),
        output_dir=Path(row["output_dir"]),
        log_file=Path(row["log_file"]),
    )


def append_registry_row(row: dict[str, Any]) -> None:
    registry_row = pd.DataFrame([row])
    if EXPERIMENT_REGISTRY_PATH.exists():
        registry_row.to_csv(EXPERIMENT_REGISTRY_PATH, mode="a", header=False, index=False)
    else:
        registry_row.to_csv(EXPERIMENT_REGISTRY_PATH, index=False)


def save_predictions_bundle(
    run_spec: RunSpec,
    gene_columns: list[str],
    test_frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, Any],
    model_details: dict[str, Any],
    command: str,
) -> None:
    run_spec.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_spec.output_dir / "test_predictions.npz", y_true=y_true.astype(np.float32), y_pred=y_pred.astype(np.float32))
    metadata_columns = [column for column in test_frame.columns if column in SIGNATURE_METADATA_COLUMNS]
    metadata = test_frame[metadata_columns].copy()
    metadata["model"] = run_spec.model
    metadata["split_family"] = run_spec.split_family
    metadata["seed"] = run_spec.seed
    metadata["run_id"] = run_spec.run_id
    metadata["gene_space_policy"] = run_spec.gene_space_policy
    metadata["gene_count"] = len(gene_columns)
    metadata.to_parquet(run_spec.output_dir / "test_metadata.parquet", index=False)
    write_gene_list(run_spec.output_dir / "genes.txt", gene_columns)
    metrics_payload = {
        "run_id": run_spec.run_id,
        "model": run_spec.model,
        "split_family": run_spec.split_family,
        "dataset_scope": run_spec.dataset_scope,
        "heldout_target": run_spec.heldout_target,
        "seed": run_spec.seed,
        "command": command,
        "metrics": metrics,
        "model_details": model_details,
        "gene_space_policy": run_spec.gene_space_policy,
        "gene_count": len(gene_columns),
        "timestamp": now_iso(),
    }
    write_json(run_spec.output_dir / "run_metrics.json", metrics_payload)


def collect_environment_snapshot() -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": sys.platform,
        "cwd": str(Path.cwd()),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "script_sha256": sha256_file(script_path),
    }


def build_command_string(run_spec: RunSpec) -> str:
    return (
        f"{sys.executable} {Path(__file__).resolve()} run-one "
        f"--model {run_spec.model} --split-json \"{run_spec.split_json_path}\" "
        f"--output-dir \"{run_spec.output_dir}\" --seed {run_spec.seed} --run-id {run_spec.run_id}"
    )


def execute_run(
    repository: DataRepository,
    run_spec: RunSpec,
    force: bool = False,
    prepared_data: PreparedSplitData | None = None,
) -> dict[str, Any]:
    logger = RunLogger(run_spec.log_file)
    command = build_command_string(run_spec)
    logger.log(f"Starting run {run_spec.run_id}")
    logger.log(f"Command: {command}")
    env_snapshot = collect_environment_snapshot()
    logger.log(f"Environment: {json.dumps(env_snapshot, ensure_ascii=False)}")
    start = time.time()
    split_payload = prepared_data.split_payload if prepared_data is not None else read_json(run_spec.split_json_path)
    gene_columns: list[str] = []
    metrics: dict[str, Any] = {}
    model_details: dict[str, Any] = {}
    registry_status = "success"
    notes = ""

    if run_spec.output_dir.exists() and (run_spec.output_dir / "run_metrics.json").exists() and not force:
        logger.log("Run output already exists and --force was not supplied; loading cached metrics.")
        cached = read_json(run_spec.output_dir / "run_metrics.json")
        metrics = cached["metrics"]
        model_details = cached.get("model_details", {})
        runtime_seconds = time.time() - start
        registry_row = build_registry_row(
            run_spec=run_spec,
            command=command,
            status="cached",
            metrics=metrics,
            runtime_seconds=runtime_seconds,
            notes="Loaded cached run_metrics.json",
        )
        append_registry_row(registry_row)
        return registry_row

    try:
        if prepared_data is None:
            prepared_data = prepare_split_data(repository, run_spec.split_json_path)
        train_frame = prepared_data.train_frame
        val_frame = prepared_data.val_frame
        test_frame = prepared_data.test_frame
        gene_columns = prepared_data.gene_columns
        if len(gene_columns) != run_spec.expected_gene_count:
            raise ValueError(
                f"Gene count mismatch for {run_spec.run_id}: manifest expected {run_spec.expected_gene_count}, got {len(gene_columns)}."
            )

        ref_train = prepared_data.ref_train
        ref_val = prepared_data.ref_val
        ref_test = prepared_data.ref_test
        y_train = prepared_data.y_train
        y_val = prepared_data.y_val
        y_test = prepared_data.y_test

        logger.log(
            f"Loaded run data: train={train_frame.shape[0]}, val={val_frame.shape[0]}, test={test_frame.shape[0]}, genes={len(gene_columns)}."
        )
        predictions, model_details = run_model(
            model_name=run_spec.model,
            train_frame=train_frame,
            val_frame=val_frame,
            test_frame=test_frame,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            ref_train=ref_train,
            ref_val=ref_val,
            ref_test=ref_test,
            logger=logger,
        )
        metrics = summarize_metrics(y_test, predictions, test_frame["is_control"].to_numpy(dtype=bool))
        metrics["gene_count"] = int(len(gene_columns))
        metrics["train_non_control_rows"] = int((~train_frame["is_control"]).sum())
        metrics["validation_non_control_rows"] = int((~val_frame["is_control"]).sum())
        metrics["test_non_control_rows"] = int((~test_frame["is_control"]).sum())
        metrics["train_rows_total"] = int(len(train_frame))
        metrics["validation_rows_total"] = int(len(val_frame))
        metrics["test_rows_total"] = int(len(test_frame))
        logger.log(f"Test metrics: {json.dumps(metrics, ensure_ascii=False)}")

        save_predictions_bundle(
            run_spec=run_spec,
            gene_columns=gene_columns,
            test_frame=test_frame,
            y_true=y_test,
            y_pred=predictions,
            metrics=metrics,
            model_details=model_details,
            command=command,
        )
    except Exception as exc:  # noqa: BLE001
        registry_status = "failed"
        notes = str(exc)
        trace = traceback.format_exc()
        logger.log(f"Run failed: {exc}")
        logger.log(trace)
        metrics = {
            PRIMARY_METRIC: float("nan"),
            "mse_non_control_test": float("nan"),
            "mae_non_control_test": float("nan"),
            "mean_cosine_similarity_control_test": float("nan"),
            "n_test_noncontrol": 0,
            "n_test_control": 0,
        }
        model_details = {"error": str(exc), "traceback": trace}

    runtime_seconds = time.time() - start
    registry_row = build_registry_row(
        run_spec=run_spec,
        command=command,
        status=registry_status,
        metrics=metrics,
        runtime_seconds=runtime_seconds,
        notes=notes,
    )
    append_registry_row(registry_row)
    return registry_row


def build_registry_row(
    run_spec: RunSpec,
    command: str,
    status: str,
    metrics: dict[str, Any],
    runtime_seconds: float,
    notes: str,
) -> dict[str, Any]:
    main_metric = metrics.get(PRIMARY_METRIC, float("nan"))
    if isinstance(main_metric, (np.floating,)):
        main_metric = float(main_metric)
    return {
        "run_id": run_spec.run_id,
        "timestamp": now_iso(),
        "phase": PHASE_NAME,
        "dataset": run_spec.dataset_scope,
        "split": run_spec.split_json_path.stem,
        "model": run_spec.model,
        "seed": run_spec.seed,
        "command": command,
        "status": status,
        "main_metric": main_metric,
        "output_dir": str(run_spec.output_dir),
        "log_file": str(run_spec.log_file),
        "notes": notes,
        "split_family": run_spec.split_family,
        "heldout_target": run_spec.heldout_target,
        "gene_space_policy": run_spec.gene_space_policy,
        "expected_gene_count": run_spec.expected_gene_count,
        "runtime_seconds": float(runtime_seconds),
    }


def aggregate_baseline_metrics(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        run_spec = manifest_row_to_run_spec(row)
        metrics_path = run_spec.output_dir / "run_metrics.json"
        if not metrics_path.exists():
            continue
        payload = read_json(metrics_path)
        metrics = payload["metrics"]
        model_details = payload.get("model_details", {})
        rows.append(
            {
                "run_id": run_spec.run_id,
                "model": run_spec.model,
                "split_family": run_spec.split_family,
                "dataset_scope": run_spec.dataset_scope,
                "heldout_target": run_spec.heldout_target,
                "seed": run_spec.seed,
                "split_json_path": str(run_spec.split_json_path),
                "gene_space_policy": run_spec.gene_space_policy,
                "gene_count": payload.get("gene_count", metrics.get("gene_count", run_spec.expected_gene_count)),
                PRIMARY_METRIC: metrics.get(PRIMARY_METRIC, float("nan")),
                "mse_non_control_test": metrics.get("mse_non_control_test", float("nan")),
                "mae_non_control_test": metrics.get("mae_non_control_test", float("nan")),
                "mean_cosine_similarity_control_test": metrics.get("mean_cosine_similarity_control_test", float("nan")),
                "n_test_noncontrol": metrics.get("n_test_noncontrol", 0),
                "n_test_control": metrics.get("n_test_control", 0),
                "selected_k": model_details.get("selected_k", np.nan),
                "selected_alpha": model_details.get("selected_alpha", np.nan),
                "output_dir": str(run_spec.output_dir),
            }
        )
    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(["split_family", "dataset_scope", "heldout_target", "seed", PRIMARY_METRIC], ascending=[True, True, True, True, False])
        metrics_df.to_csv(BASELINE_METRICS_PATH, index=False)
    else:
        pd.DataFrame(columns=["run_id"]).to_csv(BASELINE_METRICS_PATH, index=False)
    return metrics_df


def build_model_ranking_instability(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        empty = pd.DataFrame(
            columns=["model", "mean_rank", "rank_std", "best_rank", "worst_rank", "rank_reversal_count", "contexts_observed"]
        )
        empty.to_csv(RANKING_INSTABILITY_PATH, index=False)
        return empty

    ranking_context_columns = ["split_family", "dataset_scope", "heldout_target", "seed"]
    valid = metrics_df.dropna(subset=[PRIMARY_METRIC]).copy()
    valid["rank"] = valid.groupby(ranking_context_columns)[PRIMARY_METRIC].rank(method="min", ascending=False)

    overall_mean_rank = valid.groupby("model")["rank"].mean().sort_values()
    overall_order = overall_mean_rank.index.tolist()
    overall_position = {model: idx for idx, model in enumerate(overall_order)}
    reversal_counts = {model: 0 for model in overall_order}

    for _, subset in valid.groupby(ranking_context_columns, dropna=False):
        context_order = subset.sort_values("rank")["model"].tolist()
        context_position = {model: idx for idx, model in enumerate(context_order)}
        for left_index in range(len(context_order)):
            for right_index in range(left_index + 1, len(context_order)):
                left_model = context_order[left_index]
                right_model = context_order[right_index]
                overall_delta = overall_position[left_model] - overall_position[right_model]
                context_delta = context_position[left_model] - context_position[right_model]
                if overall_delta == 0:
                    continue
                if np.sign(overall_delta) != np.sign(context_delta):
                    reversal_counts[left_model] += 1
                    reversal_counts[right_model] += 1

    summary = (
        valid.groupby("model")
        .agg(
            mean_rank=("rank", "mean"),
            rank_std=("rank", "std"),
            best_rank=("rank", "min"),
            worst_rank=("rank", "max"),
            contexts_observed=("rank", "count"),
        )
        .reset_index()
    )
    summary["rank_reversal_count"] = summary["model"].map(reversal_counts).fillna(0).astype(int)
    summary = summary.sort_values("mean_rank").reset_index(drop=True)
    summary.to_csv(RANKING_INSTABILITY_PATH, index=False)
    return summary


def run_campaign(manifest_path: Path, force: bool = False) -> None:
    ensure_dirs()
    repository = DataRepository(PREPROCESSING_SUMMARY_PATH)
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = build_run_matrix(repository=repository, overwrite=not BASELINE_RUN_MATRIX_PATH.exists())
        if manifest_path != BASELINE_RUN_MATRIX_PATH:
            manifest.to_csv(manifest_path, index=False)
    campaign_logger = RunLogger(TRAINING_LOG_DIR / "phase05_campaign.log")
    campaign_logger.log(f"Prepared manifest with {len(manifest)} runs.")
    current_split_path: Path | None = None
    current_prepared_data: PreparedSplitData | None = None
    for _, row in manifest.iterrows():
        run_spec = manifest_row_to_run_spec(row)
        if current_split_path != run_spec.split_json_path:
            campaign_logger.log(f"Preparing split payload once for {run_spec.split_json_path.name}")
            current_split_path = run_spec.split_json_path
            current_prepared_data = prepare_split_data(repository, run_spec.split_json_path)
        campaign_logger.log(f"Dispatching {run_spec.run_id}")
        execute_run(repository=repository, run_spec=run_spec, force=force, prepared_data=current_prepared_data)
    metrics_df = aggregate_baseline_metrics(manifest)
    build_model_ranking_instability(metrics_df)
    campaign_logger.log(
        f"Campaign complete. metrics_rows={len(metrics_df)}, registry_path={EXPERIMENT_REGISTRY_PATH}, metrics_path={BASELINE_METRICS_PATH}"
    )


def run_single(
    model: str,
    split_json: Path,
    output_dir: Path,
    seed: int,
    run_id: str,
) -> None:
    ensure_dirs()
    repository = DataRepository(PREPROCESSING_SUMMARY_PATH)
    split_payload = read_json(split_json)
    gene_space_policy = (
        f"global_intersection_{len(repository.global_gene_space())}"
        if split_payload["split_family"] in {"dataset_heldout_split", "external_holdout"}
        else "native"
    )
    expected_gene_count = len(repository.global_gene_space()) if gene_space_policy.startswith("global") else infer_native_gene_count(repository, split_payload)
    if not run_id:
        run_id = f"{split_json.stem}__{model}"
    run_spec = RunSpec(
        run_id=run_id,
        model=model,
        split_json_path=split_json,
        split_family=str(split_payload["split_family"]),
        dataset_scope=str(split_payload["dataset_scope"]),
        heldout_target=str(split_payload.get("heldout_target", "")),
        seed=seed,
        gene_space_policy=gene_space_policy,
        expected_gene_count=expected_gene_count,
        output_dir=output_dir,
        log_file=TRAINING_LOG_DIR / f"{run_id}.log",
    )
    execute_run(repository=repository, run_spec=run_spec, force=True)


def summarize_only(manifest_path: Path) -> None:
    manifest = pd.read_csv(manifest_path)
    metrics_df = aggregate_baseline_metrics(manifest)
    build_model_ranking_instability(metrics_df)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    repository = DataRepository(PREPROCESSING_SUMMARY_PATH)
    if args.command == "build-run-matrix":
        manifest = build_run_matrix(repository, overwrite=args.overwrite)
        print(f"Built run matrix with {len(manifest)} rows at {BASELINE_RUN_MATRIX_PATH}.")
        return
    if args.command == "run-all":
        run_campaign(Path(args.manifest), force=args.force)
        return
    if args.command == "run-one":
        run_single(
            model=args.model,
            split_json=Path(args.split_json),
            output_dir=Path(args.output_dir),
            seed=args.seed,
            run_id=args.run_id,
        )
        return
    if args.command == "summarize":
        summarize_only(Path(args.manifest))
        return
    raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    main()
