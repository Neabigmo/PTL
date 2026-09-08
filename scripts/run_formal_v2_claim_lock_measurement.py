"""Run the matched-budget Frangieh measurement and joint noise Claim Lock.

The source-frozen prediction vectors are never refit here.  QC-passed raw
cells are split independently within environment/perturbation strata for 30
prespecified seeds.  For each ordered Frangieh pair, both halves use the same
cell budget ``floor(min(n_left, n_right) / 2)``; controls are matched in the
same way.  The sparse CSC source is scanned in gene blocks so the full raw
matrix is never materialized in memory.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_formal_v2_controlled_shift_oof import ENVIRONMENTS  # noqa: E402
from scripts.run_formal_v2_claim_lock_source_frozen import PAIR_ORDER  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    _rowwise_centered_pearson,
    _rowwise_rankdata_average,
    safe_rowwise_cosine,
)
from src.evaluation.reordering_inference import normalized_rank_displacement  # noqa: E402
from src.ptl.evaluation.fidelity import absolute_effect_rank_agreement  # noqa: E402


RAW_PATH = ROOT / "data/raw/scperturb_v1.4/FrangiehIzar2021_RNA.h5ad"
CELL_METADATA_PATH = ROOT / "data/processed/FrangiehIzar2021_RNA_cell_metadata.parquet"
FEATURE_METADATA_PATH = ROOT / "data/processed/FrangiehIzar2021_RNA_feature_metadata.parquet"
PREDICTION_PATH = ROOT / "artifacts/source_data/frangieh_source_frozen_predictions.npz"
SPLIT_SEED = 20260908
SPLIT_SEEDS = tuple(SPLIT_SEED + index for index in range(30))
MIN_CELLS_PRIMARY = 20
MIN_CELLS_SENSITIVITY = 40
MODEL_PAIRS = ((0, 1), (0, 2), (1, 2))
METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")
CONDITION_BY_ENVIRONMENT = {
    "frangieh_melanoma_control": "Control",
    "frangieh_melanoma_coculture": "Co-culture",
    "frangieh_melanoma_ifng": "IFNγ",
}


def _stable_seed(*values: object) -> int:
    return int(SPLIT_SEED + sum((index + 1) * sum(ord(char) for char in str(value)) for index, value in enumerate(values)))


def _load_prediction_payload(root: Path) -> tuple[list[str], np.ndarray, list[str]]:
    path = root / PREDICTION_PATH.relative_to(ROOT)
    if not path.is_file():
        raise FileNotFoundError(f"source-frozen prediction artifact is missing: {path}")
    payload = np.load(path, allow_pickle=False)
    labels = payload["perturbation_label"].astype(str).tolist()
    environments = payload["source_environment"].astype(str).tolist()
    predictions = payload["prediction"].astype(np.float32, copy=False)
    if environments != list(ENVIRONMENTS) or predictions.shape[:3] != (len(ENVIRONMENTS), len(labels), 3):
        raise ValueError("source-frozen NPZ has an unexpected environment/label/member shape")
    return labels, predictions, environments


def _load_metadata(root: Path, labels: list[str]) -> pd.DataFrame:
    columns = [
        "row_index", "perturbation_label", "is_control", "ncounts", "ngenes",
        "percent_mito", "passes_qc", "guide_id", "perturbation_2",
    ]
    frame = pd.read_parquet(root / CELL_METADATA_PATH.relative_to(ROOT), columns=columns)
    frame["perturbation_label"] = frame["perturbation_label"].astype(str)
    frame["condition"] = frame["perturbation_2"].astype(str)
    frame["environment_key"] = frame["condition"].map({value: key for key, value in CONDITION_BY_ENVIRONMENT.items()})
    frame = frame.loc[
        frame["passes_qc"].astype(bool)
        & frame["environment_key"].notna()
        & (frame["perturbation_label"].isin(labels) | frame["is_control"].astype(bool))
    ].copy()
    frame["row_index"] = pd.to_numeric(frame["row_index"], errors="raise").astype(np.int64)
    frame["ncounts"] = pd.to_numeric(frame["ncounts"], errors="coerce").fillna(0.0).astype(np.float32)
    frame["ncounts"] = frame["ncounts"].clip(lower=1.0)
    frame["perturbation_label"] = frame["perturbation_label"].where(~frame["is_control"].astype(bool), "control")
    if frame["row_index"].duplicated().any():
        raise ValueError("Frangieh cell metadata contains duplicate raw row indices")
    return frame.sort_values("row_index", kind="stable").reset_index(drop=True)


def _load_panel_positions(root: Path, panel: list[str]) -> np.ndarray:
    feature = pd.read_parquet(root / FEATURE_METADATA_PATH.relative_to(ROOT), columns=["feature_name", "feature_order"])
    feature["feature_name"] = feature["feature_name"].astype(str)
    feature = feature.drop_duplicates("feature_name", keep="first").set_index("feature_name")
    missing = sorted(set(panel).difference(feature.index))
    if missing:
        raise ValueError(f"raw Frangieh matrix misses panel genes: {missing[:5]}")
    return feature.loc[panel, "feature_order"].to_numpy(dtype=np.int64)


def _groups(metadata: pd.DataFrame, labels: list[str]) -> dict[tuple[str, str], np.ndarray]:
    result: dict[tuple[str, str], np.ndarray] = {}
    for (environment, label), group in metadata.groupby(["environment_key", "perturbation_label"], sort=True):
        if environment in ENVIRONMENTS and (label in labels or label == "control"):
            result[(str(environment), str(label))] = group["row_index"].to_numpy(dtype=np.int64)
    missing = [
        (environment, label)
        for environment in ENVIRONMENTS
        for label in [*labels, "control"]
        if (environment, label) not in result
    ]
    if missing:
        raise ValueError(f"raw Frangieh matrix has no QC-passed cells for {missing[:5]}")
    return result


def _cell_budgets(groups: dict[tuple[str, str], np.ndarray], labels: list[str], minimum: int = MIN_CELLS_PRIMARY) -> dict[tuple[str, str, str], int]:
    budgets: dict[tuple[str, str, str], int] = {}
    for left, right in PAIR_ORDER:
        for label in [*labels, "control"]:
            count = min(len(groups[(left, label)]), len(groups[(right, label)]))
            budget = count // 2
            if budget >= minimum:
                budgets[(left, right, label)] = int(budget)
    return budgets


def _seed_plan(
    groups: dict[tuple[str, str], np.ndarray],
    labels: list[str],
    budgets: dict[tuple[str, str, str], int],
    seed: int,
) -> dict[str, Any]:
    """Create independent half assignments plus virtual budget groups."""

    virtual_keys: list[tuple[str, str, int, int]] = []
    selected_rows: list[np.ndarray] = []
    counts: list[int] = []
    for environment_index, environment in enumerate(ENVIRONMENTS):
        for label in [*labels, "control"]:
            requested = sorted({
                budget
                for left, right, budget_label in budgets
                if budget_label == label and environment in (left, right)
                for budget in [budgets[(left, right, budget_label)]]
            })
            if not requested:
                continue
            source_rows = groups[(environment, label)]
            permutation = np.random.default_rng(_stable_seed(seed, environment_index, environment, label)).permutation(source_rows)
            halves = np.array_split(permutation, 2)
            for half in (0, 1):
                for budget in requested:
                    chosen = halves[half][:budget]
                    if len(chosen) != budget:
                        raise ValueError("a matched raw-cell budget exceeds one assigned split half")
                    virtual_keys.append((environment, label, half, int(budget)))
                    selected_rows.append(chosen.astype(np.int64, copy=False))
                    counts.append(int(budget))
    if not selected_rows:
        raise ValueError("no primary matched cell budgets were available")
    return {
        "seed": int(seed),
        "virtual_keys": virtual_keys,
        "selected_rows": selected_rows,
        "counts": np.asarray(counts, dtype=np.int32),
    }


def _aggregate_seed_plans(
    handle: h5py.File,
    plans: list[dict[str, Any]],
    metadata_by_row: pd.DataFrame,
    panel_positions: np.ndarray,
    *,
    panel_size: int,
    gene_chunk_size: int = 256,
) -> list[np.ndarray]:
    """Aggregate selected cells for a small seed chunk in one CSC pass."""

    x_group = handle["X"]
    shape = tuple(int(value) for value in x_group.attrs["shape"])
    if str(x_group.attrs.get("encoding-type", "")).lower() != "csc_matrix":
        raise ValueError("the raw Frangieh matrix must remain CSC for the low-memory audit")
    data_ds = x_group["data"]
    indices_ds = x_group["indices"]
    indptr_ds = x_group["indptr"]
    sorted_panel_order = np.argsort(panel_positions)
    sorted_panel = panel_positions[sorted_panel_order]
    outputs = [
        np.zeros((len(plan["virtual_keys"]), panel_size), dtype=np.float32)
        for plan in plans
    ]
    counts_by_plan = [plan["counts"] for plan in plans]
    # All seed plans use almost the same QC-passed cells.  Prepare one union
    # row list and one joint design matrix so each CSC gene block is sliced and
    # normalized exactly once, instead of once per seed.
    plan_rows: list[np.ndarray] = []
    plan_groups: list[np.ndarray] = []
    for plan in plans:
        rows = np.concatenate(plan["selected_rows"])
        group_ids = np.concatenate([
            np.full(len(selected), group_index, dtype=np.int32)
            for group_index, selected in enumerate(plan["selected_rows"])
        ])
        order = np.argsort(rows, kind="stable")
        plan_rows.append(rows[order])
        plan_groups.append(group_ids[order])
    union_rows = np.unique(np.concatenate(plan_rows))
    design_row_parts: list[np.ndarray] = []
    design_column_parts: list[np.ndarray] = []
    n_total_groups = sum(len(plan["virtual_keys"]) for plan in plans)
    group_offset = 0
    for plan_rows_one, plan_groups_one, plan in zip(plan_rows, plan_groups, plans):
        design_row_parts.append(np.searchsorted(union_rows, plan_rows_one))
        design_column_parts.append(plan_groups_one + group_offset)
        group_offset += len(plan["virtual_keys"])
    design_rows = np.concatenate(design_row_parts)
    design_columns = np.concatenate(design_column_parts)
    joint_design = sparse.csr_matrix(
        (np.ones(len(design_rows), dtype=np.float32), (design_rows, design_columns)),
        shape=(len(union_rows), n_total_groups),
    )
    for gene_start in range(0, shape[1], int(gene_chunk_size)):
        gene_end = min(gene_start + int(gene_chunk_size), shape[1])
        panel_mask = (sorted_panel >= gene_start) & (sorted_panel < gene_end)
        if not panel_mask.any():
            continue
        local_panel = (sorted_panel[panel_mask] - gene_start).astype(np.int64)
        output_columns = sorted_panel_order[panel_mask]
        data_start = int(indptr_ds[gene_start])
        data_end = int(indptr_ds[gene_end])
        indptr = np.asarray(indptr_ds[gene_start:gene_end + 1], dtype=np.int64) - data_start
        data = np.asarray(data_ds[data_start:data_end], dtype=np.float32)
        indices = np.asarray(indices_ds[data_start:data_end], dtype=np.int32)
        block = sparse.csc_matrix((data, indices, indptr), shape=(shape[0], gene_end - gene_start))
        selected = block[union_rows][:, local_panel].tocsr().astype(np.float32)
        scales = 10000.0 / metadata_by_row.loc[union_rows, "ncounts"].to_numpy(dtype=np.float32)
        selected = selected.multiply(scales[:, None]).tocsr()
        selected.data = np.log1p(selected.data)
        grouped = joint_design.T @ selected
        grouped_array = grouped.toarray().astype(np.float32, copy=False)
        group_offset = 0
        for plan_index, plan in enumerate(plans):
            n_groups = len(plan["virtual_keys"])
            outputs[plan_index][:, output_columns] = grouped_array[group_offset:group_offset + n_groups]
            group_offset += n_groups
    for plan_index, output in enumerate(outputs):
        output /= counts_by_plan[plan_index][:, None]
    return outputs


def _profile_lookup(plan: dict[str, Any], output: np.ndarray) -> dict[tuple[str, str, int, int], np.ndarray]:
    return {key: output[index] for index, key in enumerate(plan["virtual_keys"])}


def _systema_risk_chunked(prediction: np.ndarray, truth: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Reproduce pinned centroid_accuracy.py without allocating an n-by-n matrix."""

    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("Systema centroid inputs must have equal two-dimensional shapes")
    n_rows = prediction.shape[0]
    if n_rows < 2:
        return np.zeros(n_rows, dtype=np.float64)
    pred_norm = np.einsum("ij,ij->i", prediction, prediction)
    truth_norm = np.einsum("ij,ij->i", truth, truth)
    risk = np.zeros(n_rows, dtype=np.float64)
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        distances = pred_norm[start:end, None] + truth_norm[None, :] - 2.0 * prediction[start:end] @ truth.T
        distances = np.maximum(distances, 0.0)
        local_rows = np.arange(end - start)
        self_distances = distances[local_rows, np.arange(start, end)].copy()
        distances[local_rows, np.arange(start, end)] = -np.inf
        accuracy = np.sum(distances > self_distances[:, None], axis=1) / float(n_rows - 1)
        risk[start:end] = 1.0 - accuracy
    return risk


def _systema_risk_batch(prediction_versions: np.ndarray, truths: np.ndarray) -> np.ndarray:
    """Score all model versions and split seeds in one BLAS-backed distance pass.

    ``prediction_versions`` is [version, label, gene] and ``truths`` is
    [split_seed, label, gene].  The result is [split_seed, version, label].
    This is algebraically the pinned Systema centroid-accuracy definition but
    avoids thousands of small Python-level calls during the 30-seed audit.
    """

    prediction_versions = np.asarray(prediction_versions, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.float64)
    if prediction_versions.ndim != 3 or truths.ndim != 3 or prediction_versions.shape[1:] != truths.shape[1:]:
        raise ValueError("batched Systema inputs have incompatible shapes")
    n_versions, n_labels, n_genes = prediction_versions.shape
    n_seeds = truths.shape[0]
    if n_labels < 2:
        return np.zeros((n_seeds, n_versions, n_labels), dtype=np.float64)
    # The distance comparison is exactly the pinned Systema definition.  CuPy
    # is used when the configured CUDA device is available; all persisted
    # outputs are copied back to NumPy before any rank/bootstrap operation.
    try:
        import cupy as cp
        xp = cp
        pred_flat = cp.asarray(prediction_versions, dtype=cp.float32).reshape(n_versions * n_labels, n_genes)
        truth_flat = cp.asarray(truths, dtype=cp.float32).reshape(n_seeds * n_labels, n_genes)
        pred_norm = xp.einsum("ij,ij->i", pred_flat, pred_flat)
        truth_norm = xp.einsum("ij,ij->i", truth_flat, truth_flat)
        distances = pred_norm[:, None] + truth_norm[None, :] - 2.0 * (pred_flat @ truth_flat.T)
        distances = xp.maximum(distances, 0.0).reshape(n_versions, n_labels, n_seeds, n_labels).transpose(2, 0, 1, 3)
        self_distances = xp.diagonal(distances, axis1=2, axis2=3).copy()
        comparison = distances > self_distances[..., None]
        label_index = xp.arange(n_labels)
        comparison[:, :, label_index, label_index] = False
        accuracy = comparison.sum(axis=-1) / float(n_labels - 1)
        return cp.asnumpy(1.0 - accuracy)
    except (ImportError, RuntimeError):
        pred_flat = prediction_versions.reshape(n_versions * n_labels, n_genes)
        truth_flat = truths.reshape(n_seeds * n_labels, n_genes)
        pred_norm = np.einsum("ij,ij->i", pred_flat, pred_flat)
        truth_norm = np.einsum("ij,ij->i", truth_flat, truth_flat)
        distances = pred_norm[:, None] + truth_norm[None, :] - 2.0 * (pred_flat @ truth_flat.T)
        distances = np.maximum(distances, 0.0).reshape(n_versions, n_labels, n_seeds, n_labels).transpose(2, 0, 1, 3)
        self_distances = np.diagonal(distances, axis1=2, axis2=3).copy()
        comparison = distances > self_distances[..., None]
        label_index = np.arange(n_labels)
        comparison[:, :, label_index, label_index] = False
        accuracy = comparison.sum(axis=-1) / float(n_labels - 1)
        return 1.0 - accuracy


def _metric_risks(prediction_members: np.ndarray, truth: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    mean_prediction = prediction_members.mean(axis=1)
    mean_risk = {
        "delta_cosine": 1.0 - safe_rowwise_cosine(truth, mean_prediction),
        "systema_centroid_accuracy": _systema_risk_chunked(mean_prediction, truth),
        "absolute_effect_rank_agreement": 1.0 - absolute_effect_rank_agreement(truth, mean_prediction),
    }
    member_risks = {metric: np.zeros((prediction_members.shape[1], truth.shape[0]), dtype=np.float64) for metric in METRICS}
    for member_index in range(prediction_members.shape[1]):
        member_prediction = prediction_members[:, member_index, :]
        member_risks["delta_cosine"][member_index] = 1.0 - safe_rowwise_cosine(truth, member_prediction)
        member_risks["systema_centroid_accuracy"][member_index] = _systema_risk_chunked(member_prediction, truth)
        member_risks["absolute_effect_rank_agreement"][member_index] = 1.0 - absolute_effect_rank_agreement(truth, member_prediction)
    return mean_risk, member_risks


def _metric_risks_batch(prediction_members: np.ndarray, truths: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Evaluate all three metrics for every split seed without refitting."""

    prediction_members = np.asarray(prediction_members, dtype=np.float32)
    truths = np.asarray(truths, dtype=np.float32)
    mean_prediction = prediction_members.mean(axis=1)
    n_seeds, n_labels = truths.shape[:2]
    mean_risk = {metric: np.zeros((n_seeds, n_labels), dtype=np.float64) for metric in METRICS}
    member_risks = {metric: np.zeros((n_seeds, prediction_members.shape[1], n_labels), dtype=np.float64) for metric in METRICS}
    mean_versions = np.concatenate([mean_prediction[None, ...], np.transpose(prediction_members, (1, 0, 2))], axis=0)
    systema = _systema_risk_batch(mean_versions, truths)
    mean_risk["systema_centroid_accuracy"] = systema[:, 0]
    member_risks["systema_centroid_accuracy"] = np.transpose(systema[:, 1:], (0, 1, 2))
    # Rank each full seed chunk at once.  This is algebraically identical to
    # the per-seed/member calls but avoids thousands of repeated 8,229-gene
    # Python-level dispatches during the sensitivity audit.
    rank_seed_chunk = 1 if _cuda_rank_available() else 5
    for seed_start in range(0, n_seeds, rank_seed_chunk):
        seed_end = min(seed_start + rank_seed_chunk, n_seeds)
        truth_chunk = truths[seed_start:seed_end]
        truth_flat = truth_chunk.reshape(-1, truth_chunk.shape[-1])
        mean_flat = np.broadcast_to(mean_prediction, truth_chunk.shape).reshape(-1, truth_chunk.shape[-1])
        mean_risk["delta_cosine"][seed_start:seed_end] = (1.0 - safe_rowwise_cosine(truth_flat, mean_flat)).reshape(seed_end - seed_start, n_labels)
        true_abs = np.abs(truth_flat)
        true_constant = np.max(np.abs(true_abs - true_abs[:, :1]), axis=1) <= 1e-8
        rank_fn = _gpu_rank_rows_average if rank_seed_chunk == 1 else _rowwise_rankdata_average
        true_rank = rank_fn(true_abs)
        for member_index in range(prediction_members.shape[1]):
            member = prediction_members[:, member_index, :]
            member_flat = np.broadcast_to(member, truth_chunk.shape).reshape(-1, truth_chunk.shape[-1])
            member_risks["delta_cosine"][seed_start:seed_end, member_index] = (1.0 - safe_rowwise_cosine(truth_flat, member_flat)).reshape(seed_end - seed_start, n_labels)
            pred_abs = np.abs(member_flat)
            pred_constant = np.max(np.abs(pred_abs - pred_abs[:, :1]), axis=1) <= 1e-8
            rank_agreement = _rowwise_centered_pearson(true_rank, rank_fn(pred_abs))
            both_constant = true_constant & pred_constant
            one_constant = true_constant ^ pred_constant
            rank_agreement[one_constant] = 0.0
            rank_agreement[both_constant] = 0.0
            same_constant = both_constant & (np.max(np.abs(true_abs - pred_abs), axis=1) <= 1e-8)
            rank_agreement[same_constant] = 1.0
            member_risks["absolute_effect_rank_agreement"][seed_start:seed_end, member_index] = (1.0 - np.clip(rank_agreement, -1.0, 1.0)).reshape(seed_end - seed_start, n_labels)
        pred_abs = np.abs(mean_flat)
        pred_constant = np.max(np.abs(pred_abs - pred_abs[:, :1]), axis=1) <= 1e-8
        rank_agreement = _rowwise_centered_pearson(true_rank, rank_fn(pred_abs))
        both_constant = true_constant & pred_constant
        one_constant = true_constant ^ pred_constant
        rank_agreement[one_constant] = 0.0
        rank_agreement[both_constant] = 0.0
        same_constant = both_constant & (np.max(np.abs(true_abs - pred_abs), axis=1) <= 1e-8)
        rank_agreement[same_constant] = 1.0
        mean_risk["absolute_effect_rank_agreement"][seed_start:seed_end] = (1.0 - np.clip(rank_agreement, -1.0, 1.0)).reshape(seed_end - seed_start, n_labels)
    return mean_risk, member_risks


def _d(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.nanmean(normalized_rank_displacement(left, right)))


def _rank_rows_average(values: np.ndarray) -> np.ndarray:
    """Tie-aware row ranks equivalent to scipy.stats.rankdata(axis=1)."""

    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("row ranking expects a two-dimensional array")
    n_rows, n_columns = values.shape
    if n_columns == 0:
        return np.empty(values.shape, dtype=np.float64)
    # The rank value of an equal-valued group does not depend on the order
    # within that group, so quicksort is an exact faster replacement here.
    order = np.argsort(values, axis=1, kind="quicksort")
    sorted_values = np.take_along_axis(values, order, axis=1)
    positions = np.broadcast_to(np.arange(n_columns), (n_rows, n_columns))
    starts = np.ones((n_rows, n_columns), dtype=bool)
    if n_columns > 1:
        starts[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    group_starts = np.maximum.accumulate(np.where(starts, positions, 0), axis=1)
    ends = np.empty_like(starts)
    if n_columns > 1:
        ends[:, :-1] = starts[:, 1:]
    ends[:, -1] = True
    group_ends = np.minimum.accumulate(
        np.where(ends, positions, n_columns - 1)[:, ::-1], axis=1
    )[:, ::-1]
    sizes = group_ends - group_starts + 1.0
    sorted_ranks = group_starts + 1.0 + (sizes - 1.0) / 2.0
    ranks = np.empty((n_rows, n_columns), dtype=np.float64)
    np.put_along_axis(ranks, order, sorted_ranks, axis=1)
    return ranks


def _gpu_rank_rows_average(values: np.ndarray) -> np.ndarray:
    """Exact tie-aware ranks on CUDA for the float32 audit tensors."""
    values = np.asarray(values)
    if values.ndim != 2 or values.dtype != np.float32 or not np.isfinite(values).all():
        return _rank_rows_average(values)
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            return _rank_rows_average(values)
        n_rows, n_columns = values.shape
        if n_columns == 0:
            return np.empty(values.shape, dtype=np.float64)
        gpu_values = cp.asarray(values, dtype=cp.float32)
        # CuPy's default sort is exact for this purpose; ordering inside an
        # equal-valued tie group does not change its average rank.
        order = cp.argsort(gpu_values, axis=1)
        sorted_values = cp.take_along_axis(gpu_values, order, axis=1)
        positions = cp.broadcast_to(cp.arange(n_columns, dtype=cp.int32), (n_rows, n_columns))
        starts = cp.ones((n_rows, n_columns), dtype=cp.bool_)
        if n_columns > 1:
            starts[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
        group_starts = cp.maximum.accumulate(cp.where(starts, positions, 0), axis=1)
        ends = cp.empty_like(starts)
        if n_columns > 1:
            ends[:, :-1] = starts[:, 1:]
        ends[:, -1] = True
        group_ends = cp.minimum.accumulate(
            cp.where(ends, positions, n_columns - 1)[:, ::-1], axis=1
        )[:, ::-1]
        sizes = group_ends - group_starts + 1.0
        sorted_ranks = group_starts + 1.0 + (sizes - 1.0) / 2.0
        ranks = cp.empty((n_rows, n_columns), dtype=cp.float64)
        cp.put_along_axis(ranks, order, sorted_ranks, axis=1)
        result = cp.asnumpy(ranks)
        cp.get_default_memory_pool().free_all_blocks()
        return result
    except (ImportError, RuntimeError, MemoryError):
        return _rank_rows_average(values)


def _cuda_rank_available() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except (ImportError, RuntimeError):
        return False


def _bootstrap_floor(
    inputs: list[dict[str, Any]],
    *,
    seed: int,
    draws: int = 2000,
) -> dict[str, float]:
    if not inputs:
        return {"cross_d": float("nan"), "measurement_floor_d": float("nan"), "joint_floor_d": float("nan"), "delta_meas": float("nan"), "delta_joint": float("nan")}
    rng = np.random.default_rng(seed)
    n_draws = int(draws)
    selected_items = rng.integers(0, len(inputs), size=n_draws)
    member_choices = rng.integers(0, len(MODEL_PAIRS), size=n_draws)
    n = inputs[0]["cross_left_a"].shape[0]
    selected_labels = rng.integers(0, n, size=(n_draws, n))

    def sampled_values(field: str, member: int | None = None) -> np.ndarray:
        result = np.empty((n_draws, n), dtype=np.float64)
        for item_index, item in enumerate(inputs):
            mask = selected_items == item_index
            values = item[field] if member is None else item[field][member]
            result[mask] = np.asarray(values, dtype=np.float64)[selected_labels[mask]]
        return result

    def batch_d(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_rank = _rank_rows_average(left)
        right_rank = _rank_rows_average(right)
        return np.mean(np.abs(left_rank - right_rank), axis=1) / float(n - 1)

    point_cross = 0.5 * (
        batch_d(sampled_values("cross_left_a"), sampled_values("cross_right_a"))
        + batch_d(sampled_values("cross_left_b"), sampled_values("cross_right_b"))
    )
    point_meas = np.maximum(
        batch_d(sampled_values("meas_left_a"), sampled_values("meas_left_b")),
        batch_d(sampled_values("meas_right_a"), sampled_values("meas_right_b")),
    )
    point_joint = np.empty(n_draws, dtype=float)
    for model_choice, (member_a, member_b) in enumerate(MODEL_PAIRS):
        draw_mask = member_choices == model_choice
        joint_left = 0.5 * (
            batch_d(sampled_values("joint_left_a", member_a)[draw_mask], sampled_values("joint_left_b", member_b)[draw_mask])
            + batch_d(sampled_values("joint_left_a", member_b)[draw_mask], sampled_values("joint_left_b", member_a)[draw_mask])
        )
        joint_right = 0.5 * (
            batch_d(sampled_values("joint_right_a", member_a)[draw_mask], sampled_values("joint_right_b", member_b)[draw_mask])
            + batch_d(sampled_values("joint_right_a", member_b)[draw_mask], sampled_values("joint_right_b", member_a)[draw_mask])
        )
        point_joint[draw_mask] = np.maximum(joint_left, joint_right)
    delta_meas = point_cross - point_meas
    delta_joint = point_cross - point_joint

    def interval(values: list[float]) -> tuple[float, float]:
        return tuple(float(value) for value in np.quantile(np.asarray(values), [0.025, 0.975]))

    cross_low, cross_high = interval(point_cross)
    meas_low, meas_high = interval(point_meas)
    joint_low, joint_high = interval(point_joint)
    meas_delta_low, meas_delta_high = interval(delta_meas)
    joint_delta_low, joint_delta_high = interval(delta_joint)
    return {
        "cross_d": float(np.mean(point_cross)),
        "cross_d_ci_low": cross_low,
        "cross_d_ci_high": cross_high,
        "measurement_floor_d": float(np.mean(point_meas)),
        "measurement_floor_d_ci_low": meas_low,
        "measurement_floor_d_ci_high": meas_high,
        "joint_floor_d": float(np.mean(point_joint)),
        "joint_floor_d_ci_low": joint_low,
        "joint_floor_d_ci_high": joint_high,
        "delta_meas": float(np.mean(delta_meas)),
        "delta_meas_ci_low": meas_delta_low,
        "delta_meas_ci_high": meas_delta_high,
        "delta_joint": float(np.mean(delta_joint)),
        "delta_joint_ci_low": joint_delta_low,
        "delta_joint_ci_high": joint_delta_high,
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
    }


def _guide_coverage(metadata: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        for label in labels:
            group = metadata.loc[(metadata["environment_key"].eq(environment)) & (metadata["perturbation_label"].eq(label))].copy()
            guide = group["guide_id"].fillna("NA").astype(str)
            counts = guide.value_counts()
            eligible = counts[counts >= 2]
            rows.append({
                "environment_id": environment,
                "perturbation_label": label,
                "n_cells_qc": int(len(group)),
                "n_guides": int(len(counts)),
                "n_guides_with_two_or_more_cells": int(len(eligible)),
                "guide_cell_coverage_fraction": float(eligible.sum() / counts.sum()) if counts.sum() else float("nan"),
                "status": "audit_only_not_primary",
                "note": "guide-stratified split is reported for coverage; primary floors use environment×perturbation splits",
            })
    return pd.DataFrame(rows)


def run(
    root: Path,
    *,
    seed_chunk: int = 3,
    draws: int = 2000,
    minimum_cells: int = MIN_CELLS_PRIMARY,
    output_stem: str = "formal_v2_claim_lock_measurement",
) -> dict[str, Any]:
    root = root.resolve()
    if int(minimum_cells) < 1:
        raise ValueError("minimum_cells must be positive")
    if not output_stem or Path(output_stem).name != output_stem:
        raise ValueError("output_stem must be a simple file stem")
    payload = np.load(root / PREDICTION_PATH.relative_to(ROOT), allow_pickle=False)
    labels = payload["perturbation_label"].astype(str).tolist()
    predictions = payload["prediction"].astype(np.float32, copy=False)
    panel = payload["evaluation_gene_symbols"].astype(str).tolist()
    metadata = _load_metadata(root, labels)
    metadata_by_row = metadata.set_index("row_index")
    groups = _groups(metadata, labels)
    budgets = _cell_budgets(groups, labels, minimum=int(minimum_cells))
    if not budgets:
        raise ValueError("no primary matched-budget perturbation rows passed the minimum cell threshold")
    panel_positions = _load_panel_positions(root, panel)
    guide_frame = _guide_coverage(metadata, labels)
    output_dir = root / "artifacts/manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    guide_suffix = "" if output_stem == "formal_v2_claim_lock_measurement" else output_stem.removeprefix("formal_v2_claim_lock_measurement")
    guide_path = output_dir / f"formal_v2_claim_lock_guide_composition_coverage{guide_suffix}.csv"
    guide_frame.to_csv(guide_path, index=False)

    seed_outputs: dict[int, dict[tuple[str, str, int, int], np.ndarray]] = {}
    with h5py.File(root / RAW_PATH.relative_to(ROOT), "r") as handle:
        for start in range(0, len(SPLIT_SEEDS), max(1, int(seed_chunk))):
            selected_seeds = SPLIT_SEEDS[start:start + max(1, int(seed_chunk))]
            plans = [_seed_plan(groups, labels, budgets, seed) for seed in selected_seeds]
            outputs = _aggregate_seed_plans(
                handle,
                plans,
                metadata_by_row,
                panel_positions,
                panel_size=len(panel),
            )
            for seed, plan, output in zip(selected_seeds, plans, outputs):
                seed_outputs[int(seed)] = _profile_lookup(plan, output)
            print(f"[measurement-lock] completed seeds {selected_seeds[0]}-{selected_seeds[-1]}", flush=True)

    floor_rows: list[dict[str, Any]] = []
    bootstrap_inputs: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    label_to_index = {label: index for index, label in enumerate(labels)}
    for source_index, source in enumerate(ENVIRONMENTS):
        source_members = predictions[source_index]
        for pair_index, (left, right) in enumerate(PAIR_ORDER):
            eligible_labels = [
                label for label in labels
                if (left, right, label) in budgets and (left, right, "control") in budgets
            ]
            indices = np.asarray([label_to_index[label] for label in eligible_labels], dtype=np.int64)
            if len(indices) < 2:
                continue
            inputs_by_metric = {metric: [] for metric in METRICS}
            truths: dict[tuple[str, int], np.ndarray] = {}
            for environment in (left, right):
                for half in (0, 1):
                    per_seed: list[np.ndarray] = []
                    for seed in SPLIT_SEEDS:
                        profiles = seed_outputs[int(seed)]
                        budget = budgets[(left, right, eligible_labels[0])]
                        control_budget = budgets[(left, right, "control")]
                        per_seed.append(np.stack([
                            profiles[(environment, label, half, budgets[(left, right, label)])]
                            - profiles[(environment, "control", half, control_budget)]
                            for label in eligible_labels
                        ]).astype(np.float32))
                    truths[(environment, half)] = np.stack(per_seed).astype(np.float32)
            batched_risks: dict[tuple[str, int], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
            for environment in (left, right):
                for half in (0, 1):
                    batched_risks[(environment, half)] = _metric_risks_batch(
                        source_members[indices], truths[(environment, half)]
                    )
            for metric in METRICS:
                for seed_index, seed in enumerate(SPLIT_SEEDS):
                    mean_left_a = batched_risks[(left, 0)][0][metric][seed_index]
                    mean_left_b = batched_risks[(left, 1)][0][metric][seed_index]
                    mean_right_a = batched_risks[(right, 0)][0][metric][seed_index]
                    mean_right_b = batched_risks[(right, 1)][0][metric][seed_index]
                    member_left_a = batched_risks[(left, 0)][1][metric][seed_index]
                    member_left_b = batched_risks[(left, 1)][1][metric][seed_index]
                    member_right_a = batched_risks[(right, 0)][1][metric][seed_index]
                    member_right_b = batched_risks[(right, 1)][1][metric][seed_index]
                    cross_d = 0.5 * (_d(mean_left_a, mean_right_a) + _d(mean_left_b, mean_right_b))
                    meas_left = _d(mean_left_a, mean_left_b)
                    meas_right = _d(mean_right_a, mean_right_b)
                    inputs_by_metric[metric].append({
                        "cross_left_a": mean_left_a,
                        "cross_right_a": mean_right_a,
                        "cross_left_b": mean_left_b,
                        "cross_right_b": mean_right_b,
                        "meas_left_a": mean_left_a,
                        "meas_left_b": mean_left_b,
                        "meas_right_a": mean_right_a,
                        "meas_right_b": mean_right_b,
                        "joint_left_a": member_left_a,
                        "joint_left_b": member_left_b,
                        "joint_right_a": member_right_a,
                        "joint_right_b": member_right_b,
                    })
                    for member_a, member_b in MODEL_PAIRS:
                        joint_left = 0.5 * (
                            _d(member_left_a[member_a], member_left_b[member_b])
                            + _d(member_left_a[member_b], member_left_b[member_a])
                        )
                        joint_right = 0.5 * (
                            _d(member_right_a[member_a], member_right_b[member_b])
                            + _d(member_right_a[member_b], member_right_b[member_a])
                        )
                        floor_rows.append({
                            "estimand": "matched_budget_source_frozen",
                            "source_environment_id": source,
                            "left_target_environment_id": left,
                            "right_target_environment_id": right,
                            "metric": metric,
                            "split_seed": int(seed),
                            "member_pair": f"{member_a},{member_b}",
                            "n_perturbations": int(len(indices)),
                            "min_cells_primary": MIN_CELLS_PRIMARY,
                            "eligibility_min_cells": int(minimum_cells),
                            "sensitivity_min_cells": MIN_CELLS_SENSITIVITY,
                            "cross_d": cross_d,
                            "measurement_left_d": meas_left,
                            "measurement_right_d": meas_right,
                            "measurement_floor_max_d": max(meas_left, meas_right),
                            "joint_left_d": joint_left,
                            "joint_right_d": joint_right,
                            "joint_floor_max_d": max(joint_left, joint_right),
                            "delta_meas": cross_d - max(meas_left, meas_right),
                            "delta_joint": cross_d - max(joint_left, joint_right),
                            "raw_split_contract": "QC-passed raw cells; independent half A/B within environment×perturbation; controls split independently",
                            "budget_contract": "floor(min(n_left,n_right)/2) cells per half per ordered pair",
                            "refit_per_metric": 0,
                        })
            for metric in METRICS:
                key = (source, left, right, metric)
                bootstrap_inputs[key] = inputs_by_metric[metric]

    floor_path = output_dir / f"{output_stem}_floors.csv"
    pd.DataFrame(floor_rows).to_csv(floor_path, index=False)
    summary_rows: list[dict[str, Any]] = []
    for key, inputs in bootstrap_inputs.items():
        source, left, right, metric = key
        stats = _bootstrap_floor(
            inputs,
            seed=SPLIT_SEED + sum(ord(c) for c in f"{source}|{left}|{right}|{metric}"),
            draws=draws,
        )
        summary_rows.append({
            "estimand": "matched_budget_source_frozen",
            "source_environment_id": source,
            "left_target_environment_id": left,
            "right_target_environment_id": right,
            "metric": metric,
            "n_split_seeds": len(inputs),
            "n_perturbations_primary_min": int(min(item["cross_left_a"].shape[0] for item in inputs)),
            "min_cells_primary": MIN_CELLS_PRIMARY,
            "eligibility_min_cells": int(minimum_cells),
            "sensitivity_min_cells": MIN_CELLS_SENSITIVITY,
            **stats,
            "bootstrap_unit": "matched perturbation label; each draw also selects one of 30 measurement seeds and one unordered model-member pair",
            "measurement_definition": "fixed source-frozen mean prediction versus half A/B truths within each target context",
            "joint_definition": "source-frozen model member m1 with half A versus member m2 with half B, averaged over both orientations",
            "metric_entrypoint": "delta_cosine; third_party/systema/evaluation/centroid_accuracy.py-equivalent chunked distance; absolute_effect_rank_agreement",
            "refit_per_metric": 0,
        })
    summary_path = (
        output_dir / f"{output_stem}.csv"
        if output_stem.endswith("sensitivity40")
        else output_dir / f"{output_stem}_summary.csv"
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    report_path = output_dir / f"{output_stem}.json"
    report = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2_frangieh_claim_lock_measurement_v1",
        "status": (
            "matched_budget_measurement_and_joint_floors_sensitivity40_executed"
            if int(minimum_cells) == MIN_CELLS_SENSITIVITY
            else "matched_budget_measurement_and_joint_floors_executed"
        ),
        "analysis_label": "sensitivity_min40" if int(minimum_cells) == MIN_CELLS_SENSITIVITY else "primary_min20",
        "raw_source": "data/raw/scperturb_v1.4/FrangiehIzar2021_RNA.h5ad",
        "raw_shape": [218331, 23712],
        "qc_contract": "existing Frangieh preprocessing contract: target_sum=10000, log1p, min_genes=700, min_counts=1500, percent_mito<=25, perturbation_2 conditions, fixed 8229-gene panel",
        "split_seeds": list(SPLIT_SEEDS),
        "min_cells_primary": MIN_CELLS_PRIMARY,
        "eligibility_min_cells": int(minimum_cells),
        "sensitivity_min_cells": MIN_CELLS_SENSITIVITY,
        "sensitivity_executed": bool(int(minimum_cells) == MIN_CELLS_SENSITIVITY),
        "common_perturbation_count": len(labels),
        "primary_budget_definition": "floor(min(n_p_e1,n_p_e2)/2) per half and per ordered target pair; controls matched analogously",
        "measurement_floor_definition": "fixed prediction versus truth half A/B",
        "model_floor_definition": "existing formal_v2_controlled_shift_noise_floor.csv",
        "joint_floor_definition": "source-frozen member pair crossed with independent truth halves",
        "floor_rows": floor_path.relative_to(root).as_posix(),
        "summary_rows": summary_path.relative_to(root).as_posix(),
        "guide_coverage": guide_path.relative_to(root).as_posix(),
        "metric_policy": "three metrics only; same source-frozen prediction vectors, matched labels, truth halves, panel and bootstrap unit; no metric-specific refit",
        "status_note": (
            "n>=40 matched-budget sensitivity uses the same source-frozen predictions, split seeds, metrics and paired bootstrap; it is secondary."
            if int(minimum_cells) == MIN_CELLS_SENSITIVITY
            else "full-size optional sensitivity was not used as a headline and does not block the matched-budget primary lock"
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed-chunk", type=int, default=3)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--min-cells", type=int, default=MIN_CELLS_PRIMARY)
    parser.add_argument("--output-stem", default="formal_v2_claim_lock_measurement")
    args = parser.parse_args()
    result = run(
        args.root.resolve(),
        seed_chunk=args.seed_chunk,
        draws=args.draws,
        minimum_cells=args.min_cells,
        output_stem=args.output_stem,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
