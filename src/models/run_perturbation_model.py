from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import pickle
import shutil
import sys
import tarfile
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
TRAINING_LOG_DIR = RESULTS_DIR / "logs" / "training"
MANUAL_INTERVENTION = ROOT / "docs" / "manual_intervention_needed.md"
PHASE_NAME = "Phase 11 reinforcement"
OFFICIAL_GEARS_COMMIT = "f374e43e197b295016d80395d7a54ddb81cc6769"
GEARS_COMPATIBILITY_PATCH_ID = "gears-0.7.2-torch-leaf-inplace-algebraic-rewrite-v1"

GEARS_DATASETS = {
    "NormanWeissman2019_filtered",
    "ReplogleWeissman2022_K562_essential",
    "ReplogleWeissman2022_K562_gwps",
    "ReplogleWeissman2022_rpe1",
}

GEARS_SPLIT_FAMILIES = {
    "random_split",
    "unseen_perturbation_split",
    "unseen_combination_split",
    "dataset_heldout_split",
}

REGISTRY_COLUMNS = [
    "run_id",
    "timestamp",
    "phase",
    "dataset",
    "split",
    "model",
    "seed",
    "command",
    "status",
    "main_metric",
    "output_dir",
    "log_file",
    "notes",
]

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


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def check_gears_dependencies() -> dict[str, Any]:
    required = ["torch", "torch_geometric", "gears"]
    status = {}
    for module in required:
        status[module] = importlib.util.find_spec(module) is not None
    return {
        "available": all(status.values()),
        "modules": status,
        "missing": [name for name, ok in status.items() if not ok],
    }


def append_manual_intervention(title: str, payload: dict[str, Any]) -> None:
    MANUAL_INTERVENTION.parent.mkdir(parents=True, exist_ok=True)
    with MANUAL_INTERVENTION.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n## {title}\n\n")
        handle.write(f"- Timestamp: {now_iso()}\n")
        for key, value in payload.items():
            handle.write(f"- {key}: {value}\n")


def append_registry(registry_path: Path, row: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{column: row.get(column, "") for column in REGISTRY_COLUMNS}])
    if registry_path.exists():
        frame.to_csv(registry_path, mode="a", header=False, index=False)
    else:
        frame.to_csv(registry_path, index=False)


def import_gears_stack() -> tuple[Any, Any, Any]:
    import anndata as ad
    from gears import GEARS, PertData

    return ad, PertData, GEARS


def normalize_gears_condition(label: Any, is_control: Any = False) -> str:
    text = str(label)
    control_flag = str(is_control).lower() in {"true", "1", "yes"}
    if control_flag or text.lower() in {"control", "ctrl", "non-targeting", "non_targeting", "ntc"}:
        return "ctrl"
    return text


def perturbation_to_gears_list(label: str) -> list[str]:
    condition = normalize_gears_condition(label)
    if condition == "ctrl":
        return []
    return [part for part in condition.split("_") if part]


def _select_prepared_gene_columns(prepared_data: Any, selected: list[str]) -> Any:
    gene_columns = [str(gene) for gene in prepared_data.gene_columns]
    positions = [gene_columns.index(gene) for gene in selected]
    indices = np.asarray(positions, dtype=int)
    return replace(
        prepared_data,
        gene_columns=list(selected),
        ref_train=np.asarray(prepared_data.ref_train)[:, indices],
        ref_val=np.asarray(prepared_data.ref_val)[:, indices],
        ref_test=np.asarray(prepared_data.ref_test)[:, indices],
        y_train=np.asarray(prepared_data.y_train)[:, indices],
        y_val=np.asarray(prepared_data.y_val)[:, indices],
        y_test=np.asarray(prepared_data.y_test)[:, indices],
    )


def read_frozen_gene_panel(path: Path) -> list[str]:
    genes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    genes = [gene for gene in genes if gene and not gene.startswith("#")]
    if not genes or len(genes) != len(set(genes)):
        raise ValueError(f"frozen GEARS gene panel is empty or duplicated: {path}")
    return genes


def select_pilot_gene_space(
    prepared_data: Any,
    max_genes: int,
    gene_panel: list[str] | None = None,
) -> Any:
    """Select a train-ranked, perturbation-preserving pilot gene space.

    Official GEARS builds a dense co-expression similarity matrix. Limiting
    this explicitly scoped pilot to 4,096 genes keeps that official graph
    construction tractable while preserving every gene named by a requested
    perturbation. Ranking the remaining genes by training delta variance
    avoids using held-out expression values for feature selection.
    """

    gene_columns = [str(gene) for gene in prepared_data.gene_columns]
    if gene_panel is not None:
        if max_genes > 0 and len(gene_panel) != min(max_genes, len(gene_columns)):
            raise ValueError(
                f"frozen GEARS gene panel has {len(gene_panel)} genes; "
                f"expected {min(max_genes, len(gene_columns))} for this dataset"
            )
        missing = sorted(set(gene_panel) - set(gene_columns))
        if missing:
            raise ValueError(f"frozen GEARS gene panel contains unavailable genes: {missing[:10]}")
        return _select_prepared_gene_columns(prepared_data, gene_panel)
    if max_genes <= 0 or len(gene_columns) <= max_genes:
        return prepared_data
    required_genes: set[str] = set()
    for frame in (prepared_data.train_frame, prepared_data.val_frame, prepared_data.test_frame):
        for label, is_control in zip(frame["perturbation_label"], frame["is_control"]):
            required_genes.update(perturbation_to_gears_list(normalize_gears_condition(label, is_control)))
    gene_index = {gene: index for index, gene in enumerate(gene_columns)}
    required_indices = {gene_index[gene] for gene in required_genes if gene in gene_index}
    variance = np.var(np.asarray(prepared_data.y_train, dtype=np.float32), axis=0)
    ranked_indices = np.argsort(-variance, kind="mergesort").tolist()
    selected = list(sorted(required_indices))
    selected_set = set(selected)
    for index in ranked_indices:
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
        if len(selected) >= max_genes:
            break
    selected = np.asarray(sorted(selected), dtype=int)
    return _select_prepared_gene_columns(
        prepared_data,
        [gene_columns[index] for index in selected],
    )


def build_signature_metadata(test_frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    columns = [column for column in test_frame.columns if column in SIGNATURE_METADATA_COLUMNS]
    metadata = test_frame[columns].copy()
    metadata["model"] = "gears"
    metadata["split_family"] = row["split_family"]
    metadata["seed"] = int(row["seed"])
    metadata["run_id"] = row["run_id"]
    metadata["gene_space_policy"] = row.get("gene_space_policy", "native")
    try:
        gears_version = version("gears")
    except PackageNotFoundError:
        gears_version = "unknown"
    metadata["predictor_version"] = f"official-gears-{gears_version}"
    metadata["predictor_commit"] = row.get(
        "predictor_commit", "f374e43e197b295016d80395d7a54ddb81cc6769"
    )
    metadata["predictor_split_id"] = row["run_id"]
    return metadata


def build_gears_adata_from_prepared(prepared_data: Any, dataset_id: str) -> Any:
    ad, _, _ = import_gears_stack()
    frames = [
        prepared_data.train_frame.copy(),
        prepared_data.val_frame.copy(),
        prepared_data.test_frame.copy(),
    ]
    arrays = [prepared_data.y_train, prepared_data.y_val, prepared_data.y_test]
    roles = ["train", "val", "test"]
    obs_parts: list[pd.DataFrame] = []
    for frame, role in zip(frames, roles):
        obs = frame[["perturbation_label", "is_control"]].copy()
        obs["condition"] = [
            normalize_gears_condition(label, is_control)
            for label, is_control in zip(obs["perturbation_label"], obs["is_control"])
        ]
        obs["cell_type"] = str(dataset_id)
        obs["split_role"] = role
        obs_parts.append(obs[["condition", "cell_type", "split_role", "perturbation_label", "is_control"]])
    obs_all = pd.concat(obs_parts, ignore_index=True)
    expression_arrays = [
        np.asarray(reference, dtype=np.float32) + np.asarray(delta, dtype=np.float32)
        for reference, delta in zip(
            [prepared_data.ref_train, prepared_data.ref_val, prepared_data.ref_test], arrays
        )
    ]
    x_all = sparse.csr_matrix(np.vstack(expression_arrays))
    var = pd.DataFrame({"gene_name": [str(gene) for gene in prepared_data.gene_columns]}, index=[str(gene) for gene in prepared_data.gene_columns])
    return ad.AnnData(X=x_all, obs=obs_all, var=var)


def custom_condition_split_from_prepared(prepared_data: Any) -> dict[str, list[str]]:
    split: dict[str, list[str]] = {}
    for role, frame in [
        ("train", prepared_data.train_frame),
        ("val", prepared_data.val_frame),
        ("test", prepared_data.test_frame),
    ]:
        conditions = [
            normalize_gears_condition(label, is_control)
            for label, is_control in zip(frame["perturbation_label"], frame["is_control"])
        ]
        split[role] = sorted(set(conditions))
    for role in ("train", "val"):
        if "ctrl" not in split[role]:
            split[role].append("ctrl")
    condition_sets = {
        role: set(values) - {"ctrl"}
        for role, values in split.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(condition_sets[left] & condition_sets[right])
        if overlap:
            raise AssertionError(
                f"GEARS condition split overlap between {left} and {right}: {overlap[:10]}"
            )
    return split


def _gears_gene_panel_path(dataset_id: str) -> Path:
    return ROOT / "artifacts" / "manifests" / f"gears_pilot_gene_space__{dataset_id}.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_gears_gene_panel(
    repository: Any,
    dataset_id: str,
    split_paths: list[Path],
    max_genes: int,
) -> Path:
    """Freeze one train-only gene panel shared by all pilot seeds."""

    from src.baselines.run_baseline import prepare_split_data

    prepared_by_seed = [prepare_split_data(repository, path) for path in sorted(split_paths)]
    if not prepared_by_seed:
        raise ValueError(f"no split paths available to freeze GEARS gene panel for {dataset_id}")
    base_genes = [str(gene) for gene in prepared_by_seed[0].gene_columns]
    if any([str(gene) for gene in prepared.gene_columns] != base_genes for prepared in prepared_by_seed[1:]):
        raise ValueError(f"GEARS pilot split seeds do not share a gene universe for {dataset_id}")
    train_id_sets = [set(prepared.train_frame["signature_id"].astype(str)) for prepared in prepared_by_seed]
    shared_train_ids = set.intersection(*train_id_sets)
    if not shared_train_ids:
        raise ValueError(f"GEARS pilot has no shared train signatures for {dataset_id}")
    shared_training_values = []
    required_genes: set[str] = set()
    for prepared in prepared_by_seed:
        shared_mask = prepared.train_frame["signature_id"].astype(str).isin(shared_train_ids).to_numpy()
        shared_training_values.append(np.asarray(prepared.y_train[shared_mask], dtype=np.float32))
        for frame in (prepared.train_frame, prepared.val_frame, prepared.test_frame):
            for label, is_control in zip(frame["perturbation_label"], frame["is_control"]):
                required_genes.update(perturbation_to_gears_list(normalize_gears_condition(label, is_control)))
    training_values = np.vstack(shared_training_values)
    variance = np.var(training_values, axis=0, dtype=np.float64)
    gene_index = {gene: index for index, gene in enumerate(base_genes)}
    required_indices = {gene_index[gene] for gene in required_genes if gene in gene_index}
    if max_genes > 0 and len(required_indices) > max_genes:
        raise ValueError(
            f"{dataset_id} requires {len(required_indices)} perturbation genes, exceeding max_genes={max_genes}"
        )
    ranked_indices = np.argsort(-variance, kind="mergesort").tolist()
    selected = sorted(required_indices)
    selected_set = set(selected)
    target_count = len(base_genes) if max_genes <= 0 else min(max_genes, len(base_genes))
    for index in ranked_indices:
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
        if len(selected) >= target_count:
            break
    selected_genes = [base_genes[index] for index in sorted(selected)]
    path = _gears_gene_panel_path(dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        frozen = read_frozen_gene_panel(path)
        if frozen != selected_genes:
            raise ValueError(f"frozen GEARS gene panel changed for {dataset_id}: {path}")
    else:
        path.write_text("\n".join(selected_genes) + "\n", encoding="utf-8")
    return path


def ensure_frozen_gears_gene_panel(
    repository: Any,
    row: pd.Series,
    args: argparse.Namespace,
) -> tuple[list[str], Path]:
    path = _gears_gene_panel_path(str(row["dataset_id"]))
    if not path.exists():
        matrix = pd.read_csv(args.run_matrix)
        rows = matrix[matrix["dataset_scope"].astype(str).eq(str(row["dataset_id"]))]
        split_paths = [Path(value) for value in rows["split_json_path"].astype(str).tolist()]
        if len(split_paths) < 3:
            split_paths = sorted(
                (ROOT / "data" / "processed" / "splits").glob(
                    f"gears_condition_split__{row['dataset_id']}__seed*.json"
                )
            )
        if len(split_paths) < 3:
            raise ValueError(f"need all three GEARS seed splits to freeze {row['dataset_id']} panel")
        freeze_gears_gene_panel(repository, str(row["dataset_id"]), split_paths, int(args.max_genes))
    panel = read_frozen_gene_panel(path)
    return panel, path


def ensure_gears_filter_metadata(adata: Any) -> None:
    required_obs = {"condition_name", "control"}
    required_uns = {"non_zeros_gene_idx", "rank_genes_groups_cov_all"}
    missing_obs = sorted(required_obs - set(adata.obs.columns))
    missing_uns = sorted(required_uns - set(adata.uns))
    if missing_obs or missing_uns:
        raise RuntimeError(
            "official GEARS preprocessing did not produce required analysis metadata: "
            f"missing_obs={missing_obs}, missing_uns={missing_uns}"
        )


def ensure_official_gears_pickle(data_path: Path, filename: str, url: str) -> None:
    """Materialize a valid official GEARS resource before PertData reads it.

    The GEARS 0.7.2 helper uses ``requests`` without a User-Agent. Harvard
    Dataverse currently returns a small 403 HTML document to that request,
    which otherwise surfaces later as an opaque pickle error. We keep the
    source and filename prescribed by GEARS, add a browser-like User-Agent,
    validate the pickle, and atomically replace only the invalid local cache.
    """

    path = data_path / filename
    if path.is_file():
        try:
            with path.open("rb") as handle:
                pickle.load(handle)
            return
        except Exception:  # noqa: BLE001
            pass
    data_path.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 PTL-GEARS-adapter"},
        stream=True,
        timeout=120,
    )
    response.raise_for_status()
    with temporary.open("wb") as handle:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                handle.write(chunk)
    try:
        with temporary.open("rb") as handle:
            pickle.load(handle)
    except Exception as exc:  # noqa: BLE001
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"official GEARS resource {filename} is not a valid pickle") from exc
    temporary.replace(path)


def ensure_official_gears_tar(data_path: Path, directory_name: str, url: str) -> None:
    """Download and extract the official GEARS GO graph resource."""

    directory = data_path / directory_name
    expected = directory / f"{directory_name}.csv"
    archive = data_path / f"{directory_name}.tar.gz"
    if expected.is_file():
        return
    if archive.is_file() and not tarfile.is_tarfile(archive):
        archive.unlink()
    if not archive.is_file():
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 PTL-GEARS-adapter"},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        temporary = archive.with_suffix(archive.suffix + ".download")
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
        if not tarfile.is_tarfile(temporary):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"official GEARS resource {directory_name}.tar.gz is not a valid tar archive")
        temporary.replace(archive)
    with tarfile.open(archive) as tar:
        tar.extractall(path=data_path)
    if not expected.is_file():
        raise RuntimeError(f"official GEARS GO resource is missing {expected}")


def materialize_official_gears_go_subgraph(
    data_path: Path,
    relevant_genes: set[str],
) -> None:
    """Keep the official GO edge values while avoiding a full CSV load.

    The official GEARS archive contains a large all-essential-gene edge list.
    GEARS later loads that CSV into pandas even when a pilot uses a smaller
    expression gene space.  Stream-filtering both endpoints to the genes that
    can occur in this run preserves the official graph semantics and prevents
    an avoidable multi-hundred-megabyte temporary dataframe.
    """

    graph_dir = data_path / "go_essential_all"
    source = graph_dir / "go_essential_all.csv"
    marker = graph_dir / "go_essential_all_pilot_subset.marker"
    backup = graph_dir / "go_essential_all_full.csv"
    if marker.is_file():
        return
    if not source.is_file():
        raise RuntimeError(f"official GEARS GO edge list is missing {source}")
    if not relevant_genes:
        raise ValueError("cannot materialize an official GEARS GO subgraph without genes")

    graph_dir.mkdir(parents=True, exist_ok=True)
    if not backup.is_file():
        shutil.copyfile(source, backup)
    temporary = source.with_suffix(source.suffix + ".pilot.tmp")
    kept = 0
    with source.open("r", encoding="utf-8", newline="") as input_handle, temporary.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.reader(input_handle)
        writer = csv.writer(output_handle)
        header = next(reader, None)
        if header != ["source", "target", "importance"]:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"unexpected official GEARS GO header in {source}: {header}")
        writer.writerow(header)
        for row in reader:
            if len(row) >= 2 and row[0] in relevant_genes and row[1] in relevant_genes:
                writer.writerow(row)
                kept += 1
    if kept == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("official GEARS GO subgraph is empty for the pilot gene universe")
    temporary.replace(source)
    marker.write_text(
        f"official_source={backup.name}\nrelevant_genes={len(relevant_genes)}\nkept_edges={kept}\n",
        encoding="utf-8",
    )


def invalidate_incompatible_gears_coexpression_cache(
    pert_data: Any,
    gene_names: set[str],
) -> None:
    """Force a train-only coexpression graph when an old cache mismatches."""

    def mark_stale(cache_path: Path, suffix: str) -> None:
        stale_path = cache_path.with_name(cache_path.name + suffix)
        # A rerun may already have left the same stale marker behind.  Replace
        # it atomically so the adapter remains idempotent on Windows as well.
        cache_path.replace(stale_path)

    dataset_path = Path(str(pert_data.dataset_path))
    cache_name = (
        f"{pert_data.split}_{pert_data.seed}_{pert_data.train_gene_set_size}"
        "_0.4_20_co_expression_network.csv"
    )
    cache = dataset_path / cache_name
    if not cache.is_file():
        return
    cached = pd.read_csv(cache, usecols=["source", "target"])
    if cached.empty:
        mark_stale(cache, ".empty")
        return
    endpoints = set(cached["source"].astype(str)) | set(cached["target"].astype(str))
    if endpoints.issubset(gene_names):
        return
    mark_stale(cache, ".incompatible")


def patch_gears_uncertainty_loss_for_torch() -> dict[str, str]:
    """Preserve the official uncertainty loss without a PyTorch leaf mutation.

    GEARS 0.7.2 initializes its accumulator with ``requires_grad=True`` and
    then updates it in-place. Newer PyTorch rejects that operation. The
    expression below is algebraically identical, but builds a differentiable
    sum from ordinary tensor terms before training starts.
    """

    import importlib
    import torch

    def uncertainty_loss_fct(pred, logvar, y, perts, reg, ctrl, dict_filter, direction_lambda):
        gamma = 2
        perts_array = np.asarray(perts)
        terms = []
        for perturbation in set(perts_array):
            if perturbation != "ctrl":
                retain_idx = dict_filter[perturbation]
                indices = np.where(perts_array == perturbation)[0]
                pred_p = pred[indices][:, retain_idx]
                y_p = y[indices][:, retain_idx]
                logvar_p = logvar[indices][:, retain_idx]
                direction_target = ctrl[retain_idx]
            else:
                indices = np.where(perts_array == perturbation)[0]
                pred_p = pred[indices]
                y_p = y[indices]
                logvar_p = logvar[indices]
                direction_target = ctrl
            terms.append(
                torch.sum(
                    (pred_p - y_p) ** (2 + gamma)
                    + reg * torch.exp(-logvar_p) * (pred_p - y_p) ** (2 + gamma)
                )
                / pred_p.shape[0]
                / pred_p.shape[1]
            )
            terms.append(
                torch.sum(
                    direction_lambda
                    * (
                        torch.sign(y_p - direction_target)
                        - torch.sign(pred_p - direction_target)
                    )
                    ** 2
                )
                / pred_p.shape[0]
                / pred_p.shape[1]
            )
        return torch.stack(terms).sum() / len(set(perts_array))

    utils_module = importlib.import_module("gears.utils")
    gears_module = importlib.import_module("gears.gears")
    utils_module.uncertainty_loss_fct = uncertainty_loss_fct
    gears_module.uncertainty_loss_fct = uncertainty_loss_fct
    patch_hash = hashlib.sha256(inspect.getsource(uncertainty_loss_fct).encode("utf-8")).hexdigest()
    return {
        "patch_id": GEARS_COMPATIBILITY_PATCH_ID,
        "patch_hash": patch_hash,
        "reason": "algebraically equivalent uncertainty loss without PyTorch leaf in-place mutation",
    }


def materialize_delta_predictions(
    prepared_data: Any,
    requested_conditions: list[str],
    predictions: dict[str, np.ndarray],
    *,
    prediction_space: str,
) -> np.ndarray:
    """Expand condition predictions to rows and enforce the delta contract."""

    y_pred = np.zeros_like(np.asarray(prepared_data.y_test), dtype=np.float32)
    for idx, condition in enumerate(requested_conditions):
        if condition == "ctrl":
            continue
        if condition not in predictions:
            raise KeyError(f"GEARS did not return prediction for condition {condition!r}")
        value = np.asarray(predictions[condition], dtype=np.float32)
        if prediction_space == "absolute_expression":
            value = value - np.asarray(prepared_data.ref_test[idx], dtype=np.float32)
        elif prediction_space != "delta":
            raise ValueError(f"unsupported prediction space: {prediction_space}")
        y_pred[idx] = value
    return y_pred


def write_standard_outputs(
    row: pd.Series,
    prepared_data: Any,
    predictions: dict[str, np.ndarray],
    native_uq: dict[str, float],
    command: str,
    metrics: dict[str, Any],
    model_details: dict[str, Any],
    prediction_space: str,
) -> None:
    from src.baselines.run_baseline import write_gene_list, write_json

    output_dir = Path(row["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(prepared_data.y_test, dtype=np.float32)
    requested_conditions = [
        normalize_gears_condition(label, is_control)
        for label, is_control in zip(
            prepared_data.test_frame["perturbation_label"],
            prepared_data.test_frame["is_control"],
        )
    ]
    y_pred = materialize_delta_predictions(
        prepared_data,
        requested_conditions,
        predictions,
        prediction_space=prediction_space,
    )
    y_native_confidence = np.full(len(prepared_data.test_frame), np.nan, dtype=np.float32)
    y_native_uncertainty = np.full(len(prepared_data.test_frame), np.nan, dtype=np.float32)
    for idx, record in prepared_data.test_frame.reset_index(drop=True).iterrows():
        condition = normalize_gears_condition(record["perturbation_label"], record["is_control"])
        if condition in native_uq:
            confidence = float(native_uq[condition])
            if not np.isfinite(confidence) or confidence <= 0:
                raise ValueError(f"GEARS returned invalid native confidence for {condition!r}: {confidence}")
            y_native_confidence[idx] = confidence
            y_native_uncertainty[idx] = -np.log(confidence)

    np.savez_compressed(
        output_dir / "test_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        native_confidence_gears_exp_neg_mean_logvar=y_native_confidence,
        native_uncertainty_gears_mean_logvar=y_native_uncertainty,
    )
    metadata = build_signature_metadata(prepared_data.test_frame, row)
    metadata["gene_count"] = len(prepared_data.gene_columns)
    metadata["prediction_space"] = "delta"
    metadata["native_confidence_gears_exp_neg_mean_logvar"] = y_native_confidence
    metadata["native_uncertainty_gears_mean_logvar"] = y_native_uncertainty
    metadata.to_parquet(output_dir / "test_metadata.parquet", index=False)
    write_gene_list(output_dir / "genes.txt", [str(gene) for gene in prepared_data.gene_columns])
    write_json(
        output_dir / "run_metrics.json",
        {
            "run_id": row["run_id"],
            "model": "gears",
            "split_family": row["split_family"],
            "dataset_scope": row["dataset_scope"],
            "dataset_id": row["dataset_id"],
            "seed": int(row["seed"]),
            "command": command,
            "metrics": metrics,
            "model_details": model_details,
            "gene_space_policy": row.get("gene_space_policy", "native"),
            "gene_count": len(prepared_data.gene_columns),
            "timestamp": now_iso(),
        },
    )


def run_gears_adapter(row: pd.Series, args: argparse.Namespace, command: str, registry_path: Path) -> None:
    from src.baselines.run_baseline import DataRepository, prepare_split_data, summarize_metrics

    _, PertData, GEARS = import_gears_stack()
    output_dir = Path(row["output_dir"])
    log_file = Path(row["log_file"])
    logger = RunLogger(log_file)
    logger.log("Starting GEARS adapter run.")
    logger.log(f"Command: {command}")
    logger.log(f"Run id: {row['run_id']}")
    logger.log(f"Dataset: {row['dataset_id']}; split: {row['split_family']}; seed: {row['seed']}")

    try:
        repository = DataRepository(TABLES_DIR / "preprocessing_summary.csv")
        gene_panel, gene_panel_path = ensure_frozen_gears_gene_panel(repository, row, args)
        prepared = prepare_split_data(repository, Path(row["split_json_path"]))
        original_gene_count = len(prepared.gene_columns)
        prepared = select_pilot_gene_space(prepared, int(args.max_genes), gene_panel=gene_panel)
        if len(prepared.gene_columns) != original_gene_count:
            logger.log(
                f"Applied explicit pilot gene space: {original_gene_count} -> {len(prepared.gene_columns)} genes."
            )
        logger.log(f"Using frozen dataset gene panel: {gene_panel_path} ({sha256_file(gene_panel_path)})")
        adata = build_gears_adata_from_prepared(prepared, str(row["dataset_id"]))
        logger.log(f"Built AnnData for GEARS: cells={adata.n_obs}, genes={adata.n_vars}, conditions={adata.obs['condition'].nunique()}.")
        data_path = output_dir / "gears_data"
        data_path.mkdir(parents=True, exist_ok=True)
        ensure_official_gears_pickle(
            data_path,
            "gene2go_all.pkl",
            "https://dataverse.harvard.edu/api/access/datafile/6153417",
        )
        ensure_official_gears_pickle(
            data_path,
            "essential_all_data_pert_genes.pkl",
            "https://dataverse.harvard.edu/api/access/datafile/6934320",
        )
        ensure_official_gears_tar(
            data_path,
            "go_essential_all",
            "https://dataverse.harvard.edu/api/access/datafile/6934319",
        )
        pert_data = PertData(str(data_path), default_pert_graph=True)
        pert_data.new_data_process(dataset_name=str(row["run_id"]).lower(), adata=adata, skip_calc_de=False)
        ensure_gears_filter_metadata(pert_data.adata)
        active_perturbation_genes = {
            gene
            for condition in pert_data.adata.obs["condition"].astype(str)
            if condition != "ctrl"
            for gene in condition.split("+")
        }
        relevant_genes = {
            str(gene) for gene in pert_data.adata.var["gene_name"].astype(str)
        } | active_perturbation_genes
        materialize_official_gears_go_subgraph(data_path, relevant_genes)
        pert_data.create_dataset_file()

        split_dict = custom_condition_split_from_prepared(prepared)
        split_path = output_dir / "gears_custom_split.pkl"
        with split_path.open("wb") as handle:
            pickle.dump(split_dict, handle)
        pert_data.prepare_split(split="custom", seed=int(row["seed"]), split_dict_path=str(split_path))
        train_conditions = len(split_dict.get("train", []))
        batch_size = max(1, min(int(args.batch_size), train_conditions))
        pert_data.get_dataloader(batch_size=batch_size, test_batch_size=max(1, min(int(args.test_batch_size), max(1, len(split_dict.get("test", []))))))
        invalidate_incompatible_gears_coexpression_cache(
            pert_data,
            {str(gene) for gene in pert_data.adata.var["gene_name"].astype(str)},
        )

        gears_model = GEARS(pert_data, device=str(args.device), weight_bias_track=False, proj_name="PTL", exp_name=str(row["run_id"]))
        gears_model.model_initialize(
            hidden_size=int(args.hidden_size),
            uncertainty=True,
        )
        compatibility_patch = patch_gears_uncertainty_loss_for_torch()
        logger.log(f"Training GEARS: epochs={args.epochs}, batch_size={batch_size}, device={args.device}.")
        training_warning = ""
        try:
            stderr_path = output_dir / "gears_train_stderr.log"
            with stderr_path.open("a", encoding="utf-8", errors="replace") as stderr_handle:
                with contextlib.redirect_stderr(stderr_handle):
                    gears_model.train(epochs=int(args.epochs), lr=float(args.lr), weight_decay=float(args.weight_decay))
        except (ZeroDivisionError, KeyError) as exc:
            if not hasattr(gears_model, "best_model"):
                raise
            if isinstance(exc, KeyError) and "ctrl_1" not in str(exc):
                raise
            training_warning = f"GEARS post-test deeper_analysis skipped: {exc}"
            logger.log(training_warning)

        requested_conditions = [
            normalize_gears_condition(label, is_control)
            for label, is_control in zip(prepared.test_frame["perturbation_label"], prepared.test_frame["is_control"])
        ]
        known_perts = set(str(pert) for pert in gears_model.pert_list)
        prediction_conditions = []
        missing_graph_conditions = []
        for condition in sorted(set(requested_conditions)):
            if condition == "ctrl":
                continue
            pert_list = perturbation_to_gears_list(condition)
            if all(pert in known_perts for pert in pert_list):
                prediction_conditions.append(condition)
            else:
                missing_graph_conditions.append(condition)
        if missing_graph_conditions:
            raise RuntimeError(
                "official GEARS graph cannot represent requested test conditions: "
                + ", ".join(missing_graph_conditions[:20])
            )
        pert_lists = [perturbation_to_gears_list(condition) for condition in prediction_conditions]
        raw_predictions, native_uq = gears_model.predict(pert_lists) if pert_lists else ({}, {})
        predictions = {
            condition: np.asarray(value, dtype=np.float32)
            for condition, value in raw_predictions.items()
        }
        y_pred_for_metrics = materialize_delta_predictions(
            prepared,
            requested_conditions,
            predictions,
            prediction_space="absolute_expression",
        )
        metrics = summarize_metrics(
            prepared.y_test,
            y_pred_for_metrics,
            prepared.test_frame["is_control"].to_numpy(dtype=bool),
        )
        metrics["gene_count"] = int(len(prepared.gene_columns))
        model_details = {
            "adapter": "gears_signature_contract",
            "epochs": int(args.epochs),
            "batch_size": int(batch_size),
            "device": str(args.device),
            "n_conditions": int(adata.obs["condition"].nunique()),
            "n_test_conditions": int(len(pert_lists)),
            "n_missing_perturbation_graph_conditions": int(len(missing_graph_conditions)),
            "missing_perturbation_graph_conditions": missing_graph_conditions[:50],
            "custom_split_path": str(split_path),
            "graph_adapter": "official_gears_default_gene2go_and_official_go_subgraph_and_coexpression",
            "prediction_space_from_gears": "absolute_expression",
            "prediction_space_output": "delta",
            "native_uq": {
                "returned_by_gears": "exp(-mean(logvar))",
                "confidence_field": "native_confidence_gears_exp_neg_mean_logvar",
                "uncertainty_field": "native_uncertainty_gears_mean_logvar",
                "direction": "higher_is_more_uncertain",
            },
            "official_commit": OFFICIAL_GEARS_COMMIT,
            "compatibility_patch": compatibility_patch,
            "gene_panel_path": str(gene_panel_path),
            "gene_panel_sha256": sha256_file(gene_panel_path),
            "gene_space_policy": row.get("gene_space_policy", "native"),
            "gene_space_max_genes": int(args.max_genes),
            "gene_space_original_gene_count": int(original_gene_count),
            "gene_space_selected_gene_count": int(len(prepared.gene_columns)),
            "training_warning": training_warning,
        }
        write_standard_outputs(
            row,
            prepared,
            predictions,
            native_uq,
            command,
            metrics,
            model_details,
            prediction_space="absolute_expression",
        )
        logger.log(f"GEARS metrics: {json.dumps(metrics, ensure_ascii=False)}")
        append_registry(
            registry_path,
            {
                "run_id": row["run_id"],
                "timestamp": now_iso(),
                "phase": PHASE_NAME,
                "dataset": row["dataset_id"],
                "split": row["split_family"],
                "model": "gears",
                "seed": row["seed"],
                "command": command,
                "status": "success",
                "main_metric": metrics.get("mean_cosine_similarity_non_control_test", ""),
                "output_dir": str(output_dir),
                "log_file": str(log_file),
                "notes": "GEARS adapter completed with project signature-level output contract",
            },
        )
        logger.log("GEARS adapter completed.")
    except Exception as exc:  # noqa: BLE001
        output_dir.mkdir(parents=True, exist_ok=True)
        trace = traceback.format_exc()
        logger.log(f"GEARS adapter failed: {exc}")
        logger.log(trace)
        (output_dir / "adapter_failure.json").write_text(
            json.dumps({"timestamp": now_iso(), "run_id": row["run_id"], "error": str(exc), "traceback": trace}, indent=2),
            encoding="utf-8",
        )
        append_registry(
            registry_path,
            {
                "run_id": row["run_id"],
                "timestamp": now_iso(),
                "phase": PHASE_NAME,
                "dataset": row["dataset_id"],
                "split": row["split_family"],
                "model": "gears",
                "seed": row["seed"],
                "command": command,
                "status": "failed",
                "main_metric": "",
                "output_dir": str(output_dir),
                "log_file": str(log_file),
                "notes": str(exc),
            },
        )
        raise


def load_split_candidates(split_audit_path: Path) -> pd.DataFrame:
    audit = pd.read_csv(split_audit_path)
    required = {
        "split_family",
        "split_protocol",
        "perturbation_overlap",
        "track",
        "dataset_scope",
        "seed",
        "status",
        "output_path",
    }
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(f"Split audit missing required columns for GEARS run matrix: {sorted(missing)}")
    candidates = audit[
        audit["track"].eq("signature")
        & audit["status"].eq("ready")
        & audit["dataset_scope"].astype(str).isin(GEARS_DATASETS)
        & audit["split_family"].astype(str).isin(GEARS_SPLIT_FAMILIES)
        & audit["split_protocol"].astype(str).eq("random_condition_holdout")
        & audit["perturbation_overlap"].astype(str).eq("zero")
    ].copy()
    return candidates.sort_values(["dataset_scope", "split_family", "seed"]).reset_index(drop=True)


def build_gears_run_matrix(
    split_audit_path: Path,
    output_path: Path,
    gene_space_policy: str = "native",
) -> pd.DataFrame:
    candidates = load_split_candidates(split_audit_path)
    rows: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        dataset = str(row.dataset_scope)
        split_family = str(row.split_family)
        seed = int(row.seed)
        run_id = f"{split_family}__{dataset}__seed{seed}__signature__gears"
        output_dir = RESULTS_DIR / "models" / run_id
        rows.append(
            {
                "run_id": run_id,
                "model": "gears",
                "dataset_id": dataset,
                "dataset_scope": dataset,
                "split_family": split_family,
                "split_protocol": str(row.split_protocol),
                "perturbation_overlap": str(row.perturbation_overlap),
                "seed": seed,
                "split_json_path": str(row.output_path),
                "track": "signature",
                "gene_space_policy": gene_space_policy,
                "output_dir": str(output_dir),
                "prediction_path": str(output_dir / "test_predictions.npz"),
                "metadata_path": str(output_dir / "test_metadata.parquet"),
                "genes_path": str(output_dir / "genes.txt"),
                "log_file": str(TRAINING_LOG_DIR / f"{run_id}.log"),
                "status": "planned",
            }
        )
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def _write_blocked_outputs(row: pd.Series, dependency_status: dict[str, Any], command: str, registry_path: Path) -> None:
    output_dir = Path(row["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(row["log_file"])
    logger = RunLogger(log_file)
    logger.log("Starting GEARS-compatible adapter run.")
    logger.log(f"Command: {command}")
    logger.log(f"Run id: {row['run_id']}")
    logger.log(f"Dataset: {row['dataset_id']}; split: {row['split_family']}; seed: {row['seed']}")
    logger.log(f"Dependency status: {json.dumps(dependency_status, sort_keys=True)}")
    logger.log("Blocked before training because mandatory GEARS dependencies are unavailable.")
    (output_dir / "blocked_missing_dependency.json").write_text(
        json.dumps(
            {
                "timestamp": now_iso(),
                "run_id": row["run_id"],
                "missing_dependencies": dependency_status["missing"],
                "required_outputs": ["test_predictions.npz", "test_metadata.parquet", "genes.txt"],
                "status": "blocked_missing_dependency",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    append_registry(
        registry_path,
        {
            "run_id": row["run_id"],
            "timestamp": now_iso(),
            "phase": PHASE_NAME,
            "dataset": row["dataset_id"],
            "split": row["split_family"],
            "model": "gears",
            "seed": row["seed"],
            "command": command,
            "status": "blocked_missing_dependency",
            "main_metric": "",
            "output_dir": str(output_dir),
            "log_file": str(log_file),
            "notes": f"Mandatory GEARS run blocked; missing {', '.join(dependency_status['missing'])}",
        },
    )
    append_manual_intervention(
        "GEARS mandatory model blocked",
        {
            "run_id": row["run_id"],
            "dataset": row["dataset_id"],
            "split": row["split_family"],
            "missing_dependencies": ", ".join(dependency_status["missing"]),
            "recommended_action": "Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.",
            "log_file": str(log_file),
        },
    )


def run_one(args: argparse.Namespace) -> None:
    if args.model != "gears":
        raise ValueError("Only --model gears is currently supported by this adapter.")
    matrix = pd.read_csv(args.run_matrix)
    if args.run_id:
        matrix = matrix[matrix["run_id"].astype(str).eq(args.run_id)]
    if matrix.empty:
        raise ValueError("No GEARS run rows selected.")
    row = matrix.iloc[0]
    deps = check_gears_dependencies()
    command = " ".join(sys.argv)
    if not deps["available"]:
        _write_blocked_outputs(row, deps, command, Path(args.registry))
        raise RuntimeError(f"Mandatory GEARS dependencies are unavailable: {deps['missing']}")
    run_gears_adapter(row, args, command, Path(args.registry))


def run_all(args: argparse.Namespace) -> None:
    matrix_path = Path(args.run_matrix)
    if not matrix_path.exists():
        build_gears_run_matrix(Path(args.split_audit), matrix_path)
    matrix = pd.read_csv(matrix_path)
    if matrix.empty:
        raise ValueError("GEARS run matrix is empty; expected Norman/Replogle K562/RPE1 ready split rows.")
    deps = check_gears_dependencies()
    command = " ".join(sys.argv)
    blocked = 0
    if not deps["available"]:
        for _, row in matrix.iterrows():
            _write_blocked_outputs(row, deps, command, Path(args.registry))
            blocked += 1
        raise RuntimeError(f"GEARS mandatory model blocked for {blocked} runs; missing {deps['missing']}.")
    selected = matrix.copy()
    if getattr(args, "dataset_scopes", None):
        selected = selected[selected["dataset_scope"].astype(str).isin(args.dataset_scopes)]
    if getattr(args, "split_families", None):
        selected = selected[selected["split_family"].astype(str).isin(args.split_families)]
    if getattr(args, "seeds", None):
        selected = selected[selected["seed"].astype(int).isin(args.seeds)]
    if getattr(args, "run_ids", None):
        selected = selected[selected["run_id"].astype(str).isin(args.run_ids)]
    if not getattr(args, "force", False):
        selected = selected[~selected["output_dir"].map(lambda value: (Path(str(value)) / "run_metrics.json").exists())]
    if getattr(args, "max_runs", 0):
        selected = selected.head(int(args.max_runs))
    if selected.empty:
        print("No GEARS rows selected for execution.")
        return
    for row in selected.itertuples(index=False):
        run_gears_adapter(pd.Series(row._asdict()), args, command, Path(args.registry))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mandatory perturbation-model adapters for the transportability benchmark.")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-deps")
    check.add_argument("--model", default="gears", choices=["gears"])

    build = sub.add_parser("build-run-matrix")
    build.add_argument("--split-audit", default=str(TABLES_DIR / "split_audit.csv"))
    build.add_argument("--output", default=str(TABLES_DIR / "perturbation_model_run_matrix.csv"))
    build.add_argument("--gene-space-policy", default="native")

    common: dict[str, Any] = {}
    run_one_parser = sub.add_parser("run-one")
    run_one_parser.add_argument("--model", default="gears", choices=["gears"])
    run_one_parser.add_argument("--run-id", default="")
    run_one_parser.add_argument("--run-matrix", default=str(TABLES_DIR / "perturbation_model_run_matrix.csv"))
    run_one_parser.add_argument("--registry", default=str(TABLES_DIR / "experiment_registry.csv"))
    run_one_parser.add_argument("--split-audit", default=str(TABLES_DIR / "split_audit.csv"))
    run_one_parser.add_argument("--epochs", type=int, default=1)
    run_one_parser.add_argument("--batch-size", type=int, default=16)
    run_one_parser.add_argument("--test-batch-size", type=int, default=64)
    run_one_parser.add_argument("--hidden-size", type=int, default=16)
    run_one_parser.add_argument("--lr", type=float, default=0.001)
    run_one_parser.add_argument("--weight-decay", type=float, default=0.0005)
    run_one_parser.add_argument("--device", default="cpu")
    run_one_parser.add_argument("--max-genes", type=int, default=4096)

    run_all_parser = sub.add_parser("run-all")
    run_all_parser.add_argument("--model", default="gears", choices=["gears"])
    run_all_parser.add_argument("--run-matrix", default=str(TABLES_DIR / "perturbation_model_run_matrix.csv"))
    run_all_parser.add_argument("--registry", default=str(TABLES_DIR / "experiment_registry.csv"))
    run_all_parser.add_argument("--split-audit", default=str(TABLES_DIR / "split_audit.csv"))
    run_all_parser.add_argument("--dataset-scopes", nargs="*")
    run_all_parser.add_argument("--split-families", nargs="*")
    run_all_parser.add_argument("--seeds", nargs="*", type=int)
    run_all_parser.add_argument("--run-ids", nargs="*")
    run_all_parser.add_argument("--max-runs", type=int, default=0)
    run_all_parser.add_argument("--force", action="store_true")
    run_all_parser.add_argument("--epochs", type=int, default=1)
    run_all_parser.add_argument("--batch-size", type=int, default=16)
    run_all_parser.add_argument("--test-batch-size", type=int, default=64)
    run_all_parser.add_argument("--hidden-size", type=int, default=16)
    run_all_parser.add_argument("--lr", type=float, default=0.001)
    run_all_parser.add_argument("--weight-decay", type=float, default=0.0005)
    run_all_parser.add_argument("--device", default="cpu")
    run_all_parser.add_argument("--max-genes", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "check-deps":
        print(json.dumps(check_gears_dependencies(), indent=2, sort_keys=True))
    elif args.command == "build-run-matrix":
        frame = build_gears_run_matrix(
            Path(args.split_audit),
            Path(args.output),
            gene_space_policy=args.gene_space_policy,
        )
        print(f"Wrote {args.output} with {len(frame)} planned GEARS runs.")
    elif args.command == "run-one":
        run_one(args)
    elif args.command == "run-all":
        run_all(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
