"""Run one fixed source-frozen Claim Lock replication on the selected pair.

The candidate is read from the outcome-blind registry and must already be
frozen there.  This runner is intentionally limited to the selected Nadig
HepG2/Jurkat pair: it uses the same strong-linear predictor, global
perturbation-label OOF folds, three fixed metrics, 30 matched raw-cell
split-half seeds, paired perturbation bootstrap, and model/measurement/joint
floors.  It never evaluates a candidate before selection and never selects a
candidate from a risk result.

The public Nadig H5AD files are dense raw matrices.  Aggregation therefore
uses bounded row and gene blocks, keeping the full cell-by-gene matrix out of
memory while preserving the exact target-sum/log1p profile construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_claim_lock_measurement import (  # noqa: E402
    METRICS,
    MODEL_PAIRS,
    _bootstrap_floor,
    _d,
    _metric_risks_batch,
)
from scripts.run_formal_v2_predictors import (  # noqa: E402
    fit_ahlmann_eltze_bilinear_ridge,
    load_panel,
)


CANDIDATE_ID = "nadig_hepg2_vs_jurkat"
CONTEXTS = ("nadig_hepg2", "nadig_jurkat")
CONTEXT_LABELS = {
    "nadig_hepg2": "Hep-G2",
    "nadig_jurkat": "Jurkat",
}
RAW_PATHS = {
    "nadig_hepg2": "data/raw/scperturb_v1.4/NadigOConner2024_hepg2.h5ad",
    "nadig_jurkat": "data/raw/scperturb_v1.4/NadigOConner2024_jurkat.h5ad",
}
SPLIT_SEED = 20260908
SPLIT_SEEDS = tuple(SPLIT_SEED + index for index in range(30))
FOLDS = 5
MODEL_SEEDS = (0, 1, 2)
MIN_CELLS_PRIMARY = 20
MIN_CELLS_SENSITIVITY = 40
BOOTSTRAP_DRAWS = 2000
TARGET_SUM = 10000.0
ROW_CHUNK_SIZE = 8192
GENE_CHUNK_SIZE = 512
MAX_ROW_READ_SPAN = 2048
CONTROL_ALIASES = {"control", "ctrl", "non-targeting", "non-targeting control", "ntc"}


def _stable_seed(*values: object) -> int:
    return int(
        SPLIT_SEED
        + sum((index + 1) * sum(ord(char) for char in str(value)) for index, value in enumerate(values))
    )


def _is_control(value: str) -> bool:
    return str(value).strip().casefold() in CONTROL_ALIASES


def _fold_ids(labels: list[str]) -> np.ndarray:
    if len(labels) < FOLDS:
        raise ValueError("the common perturbation surface is smaller than the fold count")
    fold = np.empty(len(labels), dtype=np.int16)
    order = np.random.default_rng(SPLIT_SEED).permutation(len(labels))
    for fold_id, indices in enumerate(np.array_split(order, FOLDS)):
        fold[indices] = fold_id
    return fold


def _checksum(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _load_registry_decision(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    registry_path = root / "artifacts/manifests/reordering_replication_candidate_registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"the outcome-blind registry is missing: {registry_path}")
    report = json.loads(registry_path.read_text(encoding="utf-8"))
    if report.get("outcome_blind") is not True:
        raise ValueError("registry does not certify outcome-blind selection")
    if report.get("prediction_evaluated") or report.get("risk_evaluated"):
        raise ValueError("registry selection was not materialized before risk evaluation")
    if report.get("selected_candidate") != CANDIDATE_ID:
        raise ValueError(f"registry selected {report.get('selected_candidate')!r}, expected {CANDIDATE_ID!r}")
    selected = next(
        (row for row in report.get("candidate_summary", []) if row.get("candidate_id") == CANDIDATE_ID),
        None,
    )
    if not selected or selected.get("eligible") is not True:
        raise ValueError("selected candidate is not eligible in the frozen metadata registry")
    if selected.get("batch_context_status") != "supported_independent_context":
        raise ValueError("selected candidate does not have supported independent-context provenance")
    csv_path = root / str(report["registry_csv"])
    observed_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if observed_hash != report.get("registry_sha256"):
        raise ValueError("registry CSV hash does not match the materialized selection report")
    return report, selected


def _load_context(path: Path) -> dict[str, Any]:
    """Read outcome-blind obs and variable metadata for one raw H5AD."""

    dataset = ad.read_h5ad(path, backed="r")
    try:
        required = {"perturbation", "ncounts", "guide_id", "batch", "cell_line", "perturbation_type"}
        missing = sorted(required.difference(dataset.obs.columns))
        if missing:
            raise ValueError(f"{path.name} is missing metadata columns: {missing}")
        obs = dataset.obs[sorted(required)].copy()
        var_names = dataset.var_names.astype(str).tolist()
        shape = [int(dataset.n_obs), int(dataset.n_vars)]
    finally:
        dataset.file.close()
    obs = obs.copy()
    for column in ("perturbation", "guide_id", "batch", "cell_line", "perturbation_type"):
        obs[column] = obs[column].astype("string").fillna("").str.strip()
    obs["ncounts"] = pd.to_numeric(obs["ncounts"], errors="coerce").fillna(1.0).clip(lower=1.0).astype(np.float32)
    perturbations = obs["perturbation"].astype(str).to_numpy()
    groups: dict[str, np.ndarray] = {}
    for label in sorted(set(perturbations)):
        if label:
            groups[label] = np.flatnonzero(perturbations == label).astype(np.int64)
    target_counts = {label: int(len(rows)) for label, rows in groups.items() if not _is_control(label)}
    control_rows = np.concatenate([rows for label, rows in groups.items() if _is_control(label)]) if any(_is_control(label) for label in groups) else np.empty(0, dtype=np.int64)
    groups["control"] = np.unique(control_rows)
    return {
        "path": path,
        "shape": shape,
        "obs": obs,
        "var_names": var_names,
        "groups": groups,
        "target_counts": target_counts,
        "control_cell_count": int(len(control_rows)),
        "batch_values": sorted(value for value in obs["batch"].astype(str).unique() if value),
        "cell_line_values": sorted(value for value in obs["cell_line"].astype(str).unique() if value),
        "perturbation_type_values": sorted(value for value in obs["perturbation_type"].astype(str).unique() if value),
    }


def _read_dense_block(x_dataset: h5py.Dataset, row_start: int, row_end: int, columns: np.ndarray) -> np.ndarray:
    """Read a bounded dense H5AD block with a safe fallback for h5py indexing."""

    try:
        return np.asarray(x_dataset[row_start:row_end, columns], dtype=np.float32)
    except (TypeError, ValueError):
        full = np.asarray(x_dataset[row_start:row_end, :], dtype=np.float32)
        return full[:, columns]


def _read_dense_rows(
    x_dataset: h5py.Dataset,
    row_ids: np.ndarray,
    columns: np.ndarray,
    *,
    max_span: int = MAX_ROW_READ_SPAN,
) -> np.ndarray:
    """Read sorted sparse row IDs without repeatedly materializing huge spans."""

    row_ids = np.asarray(row_ids, dtype=np.int64)
    if row_ids.ndim != 1 or (len(row_ids) and np.any(np.diff(row_ids) < 0)):
        raise ValueError("bounded row reader requires sorted row IDs")
    output = np.empty((len(row_ids), len(columns)), dtype=np.float32)
    start_index = 0
    while start_index < len(row_ids):
        raw_start = int(row_ids[start_index])
        end_index = start_index + 1
        while end_index < len(row_ids) and int(row_ids[end_index]) - raw_start < int(max_span):
            end_index += 1
        raw_end = int(row_ids[end_index - 1]) + 1
        block = _read_dense_block(x_dataset, raw_start, raw_end, columns)
        output[start_index:end_index] = block[row_ids[start_index:end_index] - raw_start]
        start_index = end_index
    return output


def _panel_for_contexts(root: Path, contexts: dict[str, dict[str, Any]]) -> list[str]:
    frozen_panel = load_panel(root)
    available = [set(contexts[key]["var_names"]) for key in CONTEXTS]
    panel = [gene for gene in frozen_panel if all(gene in genes for genes in available)]
    if len(panel) < 2000:
        raise ValueError(f"common frozen evaluation panel is unexpectedly small: {len(panel)}")
    if len(set(panel)) != len(panel):
        raise ValueError("common evaluation panel contains duplicate genes")
    return panel


def _panel_cache_path(root: Path, context_id: str, panel: list[str]) -> Path:
    checksum = hashlib.sha256("\n".join(panel).encode("utf-8")).hexdigest()[:16]
    return root / "artifacts/source_data" / f"nadig_{context_id}_normalized_panel_{len(panel)}_{checksum}.npy"


def _attach_panel_caches(root: Path, contexts: dict[str, dict[str, Any]], panel: list[str]) -> None:
    """Attach exact read-only normalized panel caches when precomputed."""

    for context_id in CONTEXTS:
        path = _panel_cache_path(root, context_id, panel)
        if not path.is_file():
            continue
        cached = np.load(path, mmap_mode="r")
        expected_shape = (contexts[context_id]["shape"][0], len(panel))
        if tuple(cached.shape) != expected_shape or cached.dtype != np.dtype(np.float32):
            raise ValueError(f"normalized panel cache has unexpected shape/dtype: {path}")
        contexts[context_id]["normalized_panel"] = cached
        contexts[context_id]["normalized_panel_path"] = path


def _aggregate_full_profiles(context: dict[str, Any], labels: list[str], panel: list[str]) -> np.ndarray:
    """Aggregate each label and control once from all raw cells."""

    group_labels = [*labels, "control"]
    group_id_by_label = {label: index for index, label in enumerate(group_labels)}
    row_group = np.full(context["shape"][0], -1, dtype=np.int32)
    for label in group_labels:
        rows = context["groups"].get(label, np.empty(0, dtype=np.int64))
        row_group[rows] = group_id_by_label[label]
    # The selected pair is evaluated on the exact shared perturbation surface;
    # cells belonging to context-private labels are valid raw cells but must
    # not enter either the shared-label profiles or the control profile.
    panel_positions = np.asarray([context["var_names"].index(gene) for gene in panel], dtype=np.int64)
    sums = np.zeros((len(group_labels), len(panel)), dtype=np.float64)
    counts = np.zeros(len(group_labels), dtype=np.int64)
    cached_panel = context.get("normalized_panel")
    if cached_panel is not None:
        for start in range(0, context["shape"][0], ROW_CHUNK_SIZE):
            end = min(start + ROW_CHUNK_SIZE, context["shape"][0])
            block = np.asarray(cached_panel[start:end], dtype=np.float32)
            groups = row_group[start:end]
            valid = groups >= 0
            if valid.any():
                np.add.at(sums, groups[valid], block[valid])
                counts += np.bincount(groups[valid], minlength=len(group_labels))
    else:
        with h5py.File(context["path"], "r") as handle:
            x_dataset = handle["X"]
            if str(x_dataset.attrs.get("encoding-type", "")) != "array":
                raise ValueError("replication H5AD must retain a dense raw X array")
            for start in range(0, context["shape"][0], ROW_CHUNK_SIZE):
                end = min(start + ROW_CHUNK_SIZE, context["shape"][0])
                block = _read_dense_block(x_dataset, start, end, panel_positions)
                scales = TARGET_SUM / context["obs"].iloc[start:end]["ncounts"].to_numpy(dtype=np.float32)
                block *= scales[:, None]
                np.log1p(block, out=block)
                groups = row_group[start:end]
                valid = groups >= 0
                if valid.any():
                    np.add.at(sums, groups[valid], block[valid])
                    counts += np.bincount(groups[valid], minlength=len(group_labels))
    if (counts <= 0).any():
        raise ValueError("one or more perturbation/control groups has no raw cells")
    return (sums / counts[:, None]).astype(np.float32)


def _budgets(
    contexts: dict[str, dict[str, Any]],
    labels: list[str],
    minimum_cells: int,
    *,
    resampling_mode: str = "split_half",
) -> dict[str, int]:
    result: dict[str, int] = {}
    for label in [*labels, "control"]:
        paired_count = min(
            len(contexts[CONTEXTS[0]]["groups"].get(label, [])),
            len(contexts[CONTEXTS[1]]["groups"].get(label, [])),
        )
        if resampling_mode == "full_size_nonparametric":
            budget = paired_count
        else:
            budget = paired_count // 2
        if budget >= minimum_cells:
            result[label] = int(budget)
    return result


def _seed_plan(
    context: dict[str, Any],
    labels: list[str],
    budgets_by_minimum: dict[int, dict[str, int]],
    seed: int,
    context_id: str,
    *,
    resampling_mode: str = "split_half",
) -> dict[str, Any]:
    virtual_keys: list[tuple[int, str, int, int]] = []
    selected_rows: list[np.ndarray] = []
    counts: list[int] = []
    for minimum_cells in (MIN_CELLS_PRIMARY, MIN_CELLS_SENSITIVITY):
        for label in [*labels, "control"]:
            budget = budgets_by_minimum[minimum_cells].get(label)
            if budget is None:
                continue
            source_rows = context["groups"][label]
            if resampling_mode == "full_size_nonparametric":
                halves = [
                    np.random.default_rng(_stable_seed(seed, context_id, label, half, "full_size")).choice(
                        source_rows, size=budget, replace=True
                    )
                    for half in (0, 1)
                ]
            else:
                permutation = np.random.default_rng(_stable_seed(seed, context_id, label)).permutation(source_rows)
                halves = np.array_split(permutation, 2)
            for half in (0, 1):
                chosen = halves[half][:budget].astype(np.int64, copy=False)
                if len(chosen) != budget:
                    raise ValueError("matched replication budget exceeds an assigned split half")
                virtual_keys.append((minimum_cells, label, half, int(budget)))
                selected_rows.append(chosen)
                counts.append(int(budget))
    return {
        "seed": int(seed),
        "virtual_keys": virtual_keys,
        "selected_rows": selected_rows,
        "counts": np.asarray(counts, dtype=np.int32),
    }


def _aggregate_seed_plans(
    context: dict[str, Any],
    plans: list[dict[str, Any]],
    panel: list[str],
    *,
    row_chunk_size: int = ROW_CHUNK_SIZE,
    gene_chunk_size: int = GENE_CHUNK_SIZE,
) -> list[dict[tuple[int, str, int, int], np.ndarray]]:
    """Aggregate several split plans in bounded dense row/gene blocks."""

    plan_rows: list[np.ndarray] = []
    plan_groups: list[np.ndarray] = []
    offsets: list[int] = []
    group_offset = 0
    for plan in plans:
        rows = np.concatenate(plan["selected_rows"])
        groups = np.concatenate([
            np.full(len(selected), group_index, dtype=np.int32)
            for group_index, selected in enumerate(plan["selected_rows"])
        ])
        order = np.argsort(rows, kind="stable")
        plan_rows.append(rows[order])
        plan_groups.append(groups[order])
        offsets.append(group_offset)
        group_offset += len(plan["virtual_keys"])
    union_rows = np.unique(np.concatenate(plan_rows))
    design_row_parts: list[np.ndarray] = []
    design_column_parts: list[np.ndarray] = []
    for rows, groups, offset in zip(plan_rows, plan_groups, offsets):
        design_row_parts.append(np.searchsorted(union_rows, rows))
        design_column_parts.append(groups + offset)
    design_rows = np.concatenate(design_row_parts)
    design_columns = np.concatenate(design_column_parts)
    design = sparse.csr_matrix(
        (np.ones(len(design_rows), dtype=np.float32), (design_rows, design_columns)),
        shape=(len(union_rows), group_offset),
    )
    outputs = [np.zeros((len(plan["virtual_keys"]), len(panel)), dtype=np.float32) for plan in plans]
    panel_positions = np.asarray([context["var_names"].index(gene) for gene in panel], dtype=np.int64)
    sorted_order = np.argsort(panel_positions, kind="stable")
    sorted_positions = panel_positions[sorted_order]
    cached_panel = context.get("normalized_panel")
    torch = None
    use_gpu = False
    if cached_panel is not None:
        try:
            import torch as _torch
            use_gpu = bool(_torch.cuda.is_available())
            torch = _torch if use_gpu else None
        except (ImportError, OSError):
            torch = None
            use_gpu = False
    if cached_panel is not None:
        handle = None
    else:
        handle = h5py.File(context["path"], "r")
        x_dataset = handle["X"]
    try:
        for row_start_index in range(0, len(union_rows), int(row_chunk_size)):
            row_end_index = min(row_start_index + int(row_chunk_size), len(union_rows))
            row_ids = union_rows[row_start_index:row_end_index]
            if cached_panel is not None:
                selected = np.asarray(cached_panel[row_ids], dtype=np.float32)
            else:
                selected = _read_dense_rows(x_dataset, row_ids, sorted_positions)
                scales = TARGET_SUM / context["obs"].iloc[row_ids]["ncounts"].to_numpy(dtype=np.float32)
                selected *= scales[:, None]
                np.log1p(selected, out=selected)
            design_chunk = design[row_start_index:row_end_index]
            if use_gpu:
                coo = design_chunk.tocoo(copy=False)
                device = torch.device("cuda")
                row_index = torch.as_tensor(coo.row, dtype=torch.long, device=device)
                column_index = torch.as_tensor(coo.col, dtype=torch.long, device=device)
                weights = torch.as_tensor(np.asarray(coo.data, dtype=np.float32), dtype=torch.float32, device=device)
                selected_gpu = torch.as_tensor(np.ascontiguousarray(selected), dtype=torch.float32, device=device)
                for gene_start in range(0, len(panel), int(gene_chunk_size)):
                    gene_end = min(gene_start + int(gene_chunk_size), len(panel))
                    grouped_gpu = torch.zeros((group_offset, gene_end - gene_start), dtype=torch.float32, device=device)
                    values = selected_gpu[:, gene_start:gene_end]
                    if len(coo.data):
                        grouped_gpu.index_add_(0, column_index, values[row_index] * weights[:, None])
                    grouped = grouped_gpu.cpu().numpy()
                    output_start = 0
                    for plan_index, plan in enumerate(plans):
                        n_groups = len(plan["virtual_keys"])
                        outputs[plan_index][:, sorted_order[gene_start:gene_end]] = grouped[output_start:output_start + n_groups]
                        output_start += n_groups
                del selected_gpu
            else:
                for gene_start in range(0, len(panel), int(gene_chunk_size)):
                    gene_end = min(gene_start + int(gene_chunk_size), len(panel))
                    grouped = design_chunk.T @ selected[:, gene_start:gene_end]
                    if sparse.issparse(grouped):
                        grouped = grouped.toarray()
                    grouped = np.asarray(grouped, dtype=np.float32)
                    output_start = 0
                    for plan_index, plan in enumerate(plans):
                        n_groups = len(plan["virtual_keys"])
                        outputs[plan_index][:, sorted_order[gene_start:gene_end]] = grouped[output_start:output_start + n_groups]
                        output_start += n_groups
    finally:
        if handle is not None:
            handle.close()
    lookups: list[dict[tuple[int, str, int, int], np.ndarray]] = []
    for plan, output in zip(plans, outputs):
        output /= plan["counts"][:, None]
        lookups.append({key: output[index] for index, key in enumerate(plan["virtual_keys"])})
    return lookups


def _fit_source_predictions(
    labels: list[str],
    profiles: dict[str, np.ndarray],
    panel: list[str],
    fold_ids: np.ndarray,
) -> np.ndarray:
    """Fit the exact strong-linear model once per source/fold/member."""

    predictions = np.zeros((len(CONTEXTS), len(labels), len(MODEL_SEEDS), len(panel)), dtype=np.float32)
    label_array = np.asarray(labels, dtype=str)
    for source_index, source in enumerate(CONTEXTS):
        source_profile = profiles[source]
        post_values = source_profile[:-1]
        control = source_profile[-1]
        change_values = post_values - control
        for fold_id in range(FOLDS):
            test_indices = np.flatnonzero(fold_ids == fold_id)
            train_indices = np.flatnonzero(fold_ids != fold_id)
            for member_index, model_seed in enumerate(MODEL_SEEDS):
                rng = np.random.default_rng(
                    int(model_seed) + 1009 * (sum(ord(c) for c in source) + 1) + 100000 * fold_id
                )
                bootstrap = rng.choice(train_indices, size=len(train_indices), replace=True)
                prediction = fit_ahlmann_eltze_bilinear_ridge(
                    label_array[bootstrap],
                    change_values[bootstrap],
                    label_array[test_indices],
                    gene_names=panel,
                    post_train_values=post_values[bootstrap],
                    alpha=0.1,
                    n_components=10,
                )
                predictions[source_index, test_indices, member_index] = prediction
    if not np.isfinite(predictions).all():
        raise ValueError("non-finite source-frozen replication predictions")
    return predictions


def _truths_from_lookup(
    lookup: dict[tuple[int, str, int, int], np.ndarray],
    labels: list[str],
    budgets: dict[str, int],
    minimum_cells: int,
) -> dict[tuple[str, int], np.ndarray]:
    result: dict[tuple[str, int], np.ndarray] = {}
    for context in CONTEXTS:
        for half in (0, 1):
            result[(context, half)] = np.stack([
                lookup[(minimum_cells, label, half, budgets[label])]
                - lookup[(minimum_cells, "control", half, budgets["control"])]
                for label in labels
            ]).astype(np.float32)
    return result


def _write_bootstrap_input_chunk(
    output_path: Path,
    inputs: dict[tuple[int, str, str], list[dict[str, np.ndarray]]],
) -> list[str]:
    """Write one bounded bootstrap-input chunk and return its encoded keys."""

    input_fields = (
        "cross_left_a", "cross_right_a", "cross_left_b", "cross_right_b",
        "meas_left_a", "meas_left_b", "meas_right_a", "meas_right_b",
        "joint_left_a", "joint_left_b", "joint_right_a", "joint_right_b",
    )
    input_arrays: dict[str, np.ndarray] = {}
    input_keys: list[str] = []
    for (minimum_cells, source, metric), values in inputs.items():
        if not values:
            continue
        encoded = f"{minimum_cells}__{source}__{metric}"
        input_keys.append(encoded)
        for field in input_fields:
            input_arrays[f"{encoded}__{field}"] = np.stack([value[field] for value in values]).astype(np.float32)
    if not input_arrays:
        return []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **input_arrays)
    return sorted(input_keys)


def _run_thresholds(
    contexts: dict[str, dict[str, Any]],
    labels: list[str],
    panel: list[str],
    predictions: np.ndarray,
    budgets_by_minimum: dict[int, dict[str, int]],
    *,
    seed_chunk: int,
    resampling_mode: str = "split_half",
    split_seeds: tuple[int, ...] = SPLIT_SEEDS,
    bootstrap_input_dir: Path | None = None,
    bootstrap_input_stem: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, str, str], list[dict[str, np.ndarray]]], list[dict[str, Any]]]:
    floor_rows: list[dict[str, Any]] = []
    inputs: dict[tuple[int, str, str], list[dict[str, Any]]] = {
        (minimum, source, metric): []
        for minimum in (MIN_CELLS_PRIMARY, MIN_CELLS_SENSITIVITY)
        for source in CONTEXTS
        for metric in METRICS
    }
    stream_files: list[dict[str, Any]] = []
    for start in range(0, len(split_seeds), max(1, int(seed_chunk))):
        selected_seeds = split_seeds[start:start + max(1, int(seed_chunk))]
        active_inputs = inputs if bootstrap_input_dir is None else {
            (minimum, source, metric): []
            for minimum in (MIN_CELLS_PRIMARY, MIN_CELLS_SENSITIVITY)
            for source in CONTEXTS
            for metric in METRICS
        }
        plans_by_context: dict[str, list[dict[str, Any]]] = {}
        for context_id in CONTEXTS:
            plans_by_context[context_id] = [
                _seed_plan(
                    contexts[context_id], labels, budgets_by_minimum, seed, context_id,
                    resampling_mode=resampling_mode,
                )
                for seed in selected_seeds
            ]
        lookups_by_context: dict[str, list[dict[tuple[int, str, int, int], np.ndarray]]] = {}
        for context_id in CONTEXTS:
            lookups_by_context[context_id] = _aggregate_seed_plans(contexts[context_id], plans_by_context[context_id], panel)
        for local_seed_index, seed in enumerate(selected_seeds):
            for minimum_cells in (MIN_CELLS_PRIMARY, MIN_CELLS_SENSITIVITY):
                eligible_labels = [label for label in labels if label in budgets_by_minimum[minimum_cells] and "control" in budgets_by_minimum[minimum_cells]]
                if len(eligible_labels) < 2:
                    continue
                truth_by_context = {
                    context_id: _truths_from_lookup(
                        lookups_by_context[context_id][local_seed_index],
                        eligible_labels,
                        budgets_by_minimum[minimum_cells],
                        minimum_cells,
                    )
                    for context_id in CONTEXTS
                }
                indices = np.asarray([labels.index(label) for label in eligible_labels], dtype=np.int64)
                for source_index, source in enumerate(CONTEXTS):
                    metric_risks: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
                    for target in CONTEXTS:
                        # Score half A and half B separately; this keeps the
                        # model prediction fixed while only the raw-cell truth
                        # changes.
                        risks_by_half = []
                        for half in (0, 1):
                            risks_by_half.append(_metric_risks_batch(
                                predictions[source_index, indices],
                                truth_by_context[target][(target, half)][None, ...],
                            ))
                        metric_risks[target] = (
                            {metric: np.stack([risks_by_half[0][0][metric][0], risks_by_half[1][0][metric][0]]) for metric in METRICS},
                            {metric: np.stack([risks_by_half[0][1][metric][0], risks_by_half[1][1][metric][0]]) for metric in METRICS},
                        )
                    for metric in METRICS:
                        mean_left = metric_risks[CONTEXTS[0]][0][metric]
                        mean_right = metric_risks[CONTEXTS[1]][0][metric]
                        member_left = metric_risks[CONTEXTS[0]][1][metric]
                        member_right = metric_risks[CONTEXTS[1]][1][metric]
                        cross_d = 0.5 * (_d(mean_left[0], mean_right[0]) + _d(mean_left[1], mean_right[1]))
                        meas_left = _d(mean_left[0], mean_left[1])
                        meas_right = _d(mean_right[0], mean_right[1])
                        active_inputs[(minimum_cells, source, metric)].append({
                            "cross_left_a": mean_left[0],
                            "cross_right_a": mean_right[0],
                            "cross_left_b": mean_left[1],
                            "cross_right_b": mean_right[1],
                            "meas_left_a": mean_left[0],
                            "meas_left_b": mean_left[1],
                            "meas_right_a": mean_right[0],
                            "meas_right_b": mean_right[1],
                            "joint_left_a": member_left[0],
                            "joint_left_b": member_left[1],
                            "joint_right_a": member_right[0],
                            "joint_right_b": member_right[1],
                        })
                        for member_a, member_b in MODEL_PAIRS:
                            joint_left = 0.5 * (
                                _d(member_left[0, member_a], member_left[1, member_b])
                                + _d(member_left[0, member_b], member_left[1, member_a])
                            )
                            joint_right = 0.5 * (
                                _d(member_right[0, member_a], member_right[1, member_b])
                                + _d(member_right[0, member_b], member_right[1, member_a])
                            )
                            floor_rows.append({
                                "candidate_id": CANDIDATE_ID,
                                "estimand": "matched_budget_source_frozen_replication",
                                "analysis_label": "sensitivity_min40" if minimum_cells == MIN_CELLS_SENSITIVITY else "primary_min20",
                                "source_context_id": source,
                                "left_target_context_id": CONTEXTS[0],
                                "right_target_context_id": CONTEXTS[1],
                                "metric": metric,
                                "split_seed": int(seed),
                                "member_pair": f"{member_a},{member_b}",
                                "n_perturbations": int(len(indices)),
                                "eligibility_min_cells": int(minimum_cells),
                                "cross_d": cross_d,
                                "measurement_left_d": meas_left,
                                "measurement_right_d": meas_right,
                                "measurement_floor_max_d": max(meas_left, meas_right),
                                "joint_left_d": joint_left,
                                "joint_right_d": joint_right,
                                "joint_floor_max_d": max(joint_left, joint_right),
                                "delta_meas": cross_d - max(meas_left, meas_right),
                                "delta_joint": cross_d - max(joint_left, joint_right),
                                "raw_split_contract": (
                                    "raw cells; independent full-size with-replacement pseudo-replicates within "
                                    "context×perturbation; controls resampled independently"
                                    if resampling_mode == "full_size_nonparametric"
                                    else "raw cells; independent half A/B within context×perturbation; controls split independently"
                                ),
                                "budget_contract": (
                                    "min(n_left,n_right) cells per full-size pseudo-replicate"
                                    if resampling_mode == "full_size_nonparametric"
                                    else "floor(min(n_left,n_right)/2) cells per half"
                                ),
                                "refit_per_metric": 0,
                            })
        print(f"[replication-lock] completed seeds {selected_seeds[0]}-{selected_seeds[-1]}", flush=True)
        if bootstrap_input_dir is not None:
            chunk_stem = bootstrap_input_stem or "formal_v2_claim_lock_replication_bootstrap"
            chunk_path = bootstrap_input_dir / f"{chunk_stem}_bootstrap_chunk_{selected_seeds[0]}_{selected_seeds[-1]}.npz"
            encoded_keys = _write_bootstrap_input_chunk(chunk_path, active_inputs)
            if not encoded_keys:
                raise ValueError(f"no bootstrap inputs were written for seeds {selected_seeds}")
            stream_files.append({
                "path": chunk_path,
                "split_seeds": [int(seed) for seed in selected_seeds],
                "keys": encoded_keys,
            })
    if bootstrap_input_dir is not None:
        return [], floor_rows, {}, stream_files
    summary_rows: list[dict[str, Any]] = []
    for (minimum_cells, source, metric), values in inputs.items():
        if not values:
            continue
        stats = _bootstrap_floor(
            values,
            seed=SPLIT_SEED + sum(ord(c) for c in f"nadig|{minimum_cells}|{source}|{metric}"),
            draws=BOOTSTRAP_DRAWS,
        )
        summary_rows.append({
            "candidate_id": CANDIDATE_ID,
            "estimand": "matched_budget_source_frozen_replication",
            "analysis_label": "sensitivity_min40" if minimum_cells == MIN_CELLS_SENSITIVITY else "primary_min20",
            "source_context_id": source,
            "left_target_context_id": CONTEXTS[0],
            "right_target_context_id": CONTEXTS[1],
            "metric": metric,
            "n_split_seeds": len(values),
            "n_perturbations": int(values[0]["cross_left_a"].shape[0]),
            "eligibility_min_cells": int(minimum_cells),
            "split_seed_start": int(SPLIT_SEEDS[0]),
            "split_seed_end": int(SPLIT_SEEDS[-1]),
            "bootstrap_unit": "matched perturbation label; each draw selects one measurement seed and one unordered model-member pair",
            "measurement_definition": (
                "fixed source-frozen mean prediction versus independent full-size with-replacement raw-cell pseudo-replicates"
                if resampling_mode == "full_size_nonparametric"
                else "fixed source-frozen mean prediction versus independent raw-cell half A/B truths"
            ),
            "joint_definition": "source-frozen model member pair crossed with independent truth halves",
            "metric_entrypoint": "same delta cosine, pinned Systema centroid-accuracy, and absolute-effect-rank implementation as Frangieh Claim Lock",
            "refit_per_metric": 0,
            **stats,
        })
    return summary_rows, floor_rows, inputs, stream_files


def _write_executed_boundary(
    root: Path,
    registry_report: dict[str, Any],
    selected: dict[str, Any],
    report: dict[str, Any],
) -> Path:
    """Replace the pending boundary with the executed, still-bounded claim."""

    output_path = root / "artifacts/manifests/formal_v2_claim_lock_replication_boundary.json"
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2_frangieh_claim_lock_v1",
        "status": "independent_replication_claim_lock_executed",
        "boundary": (
            "the outcome-blind registry selected one supported independent-context pair and the exact "
            "Claim Lock was executed once; the result is bounded to the Nadig HepG2/Jurkat contexts and "
            "does not establish universality or a causal biological mechanism"
        ),
        "policy": "report the selected replication and its floors without promoting a universal transport claim",
        "registry": registry_report["registry_json"],
        "registry_sha256": registry_report["registry_sha256"],
        "selected_candidate": report["candidate_id"],
        "prediction_evaluated": True,
        "risk_evaluated": True,
        "selection_materialized_before_risk_evaluation": True,
        "provenance_status": selected["batch_context_status"],
        "provenance_evidence_source": selected["provenance_evidence_source"],
        "registry_selection_reasons": {
            "candidate_count": registry_report["candidate_count"],
            "eligible_candidate_count": registry_report["eligible_candidate_count"],
            "selected_exact_shared_perturbation_count": selected["exact_shared_perturbation_count"],
            "selected_median_min_cells_per_shared_perturbation": selected["median_min_cells_per_shared_perturbation"],
            "selected_batch_context_status": selected["batch_context_status"],
            "selected_reason": selected["reason"],
            "provenance_audits": registry_report.get("provenance_audits", {}),
        },
        "replication_report": report["summary_path"].removesuffix(".csv") + ".json",
        "replication_summary": report["summary_path"],
        "replication_floors": report["floor_path"],
        "negative_result_audit": report["negative_result_audit"],
        "optional_candidate": "GEARS remains optional only if a legal same-predictor/source-outcome contract is established; no new model training was introduced",
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return output_path


def run(
    root: Path,
    *,
    seed_chunk: int = 2,
    resampling_mode: str = "split_half",
    seed_start: int = 0,
    seed_stop: int | None = None,
    output_stem: str | None = None,
    stream_bootstrap_inputs: bool = False,
) -> dict[str, Any]:
    if resampling_mode not in {"split_half", "full_size_nonparametric"}:
        raise ValueError("resampling_mode must be split_half or full_size_nonparametric")
    if seed_start < 0 or seed_start >= len(SPLIT_SEEDS):
        raise ValueError("seed_start must index the declared 30-seed schedule")
    if seed_stop is None:
        seed_stop = len(SPLIT_SEEDS)
    if seed_stop <= seed_start or seed_stop > len(SPLIT_SEEDS):
        raise ValueError("seed_stop must be greater than seed_start and within the declared 30-seed schedule")
    selected_split_seeds = tuple(SPLIT_SEEDS[seed_start:seed_stop])
    root = root.resolve()
    registry_report, selected = _load_registry_decision(root)
    contexts = {context_id: _load_context(root / RAW_PATHS[context_id]) for context_id in CONTEXTS}
    print("[replication-lock] contexts loaded", flush=True)
    labels = sorted(set(contexts[CONTEXTS[0]]["target_counts"]) & set(contexts[CONTEXTS[1]]["target_counts"]))
    if len(labels) < 200:
        raise ValueError("selected replication pair has fewer than 200 exact shared perturbations")
    panel = _panel_for_contexts(root, contexts)
    _attach_panel_caches(root, contexts, panel)
    print(f"[replication-lock] panel ready genes={len(panel)} cache={all('normalized_panel' in contexts[key] for key in CONTEXTS)}", flush=True)
    output_dir = root / "artifacts/manifests"
    source_dir = root / "artifacts/source_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or (
        "formal_v2_claim_lock_replication_nadig_fullsize"
        if resampling_mode == "full_size_nonparametric"
        else "formal_v2_claim_lock_replication_nadig"
    )
    full_profiles = {
        context_id: _aggregate_full_profiles(contexts[context_id], labels, panel)
        for context_id in CONTEXTS
    }
    print("[replication-lock] full profiles ready", flush=True)
    fold_ids = _fold_ids(labels)
    predictions = _fit_source_predictions(labels, full_profiles, panel, fold_ids)
    print("[replication-lock] source-frozen predictions ready", flush=True)
    budgets_by_minimum = {
        minimum: _budgets(contexts, labels, minimum, resampling_mode=resampling_mode)
        for minimum in (MIN_CELLS_PRIMARY, MIN_CELLS_SENSITIVITY)
    }
    if resampling_mode == "full_size_nonparametric":
        # The existing split-half min-40 artifact is the prespecified
        # sensitivity. Do not recompute it inside the more expensive
        # full-size primary pass.
        budgets_by_minimum[MIN_CELLS_SENSITIVITY] = {}
    if "control" not in budgets_by_minimum[MIN_CELLS_PRIMARY]:
        raise ValueError("matched controls do not satisfy the primary split-half budget")
    print(f"[replication-lock] budgets ready primary={len(budgets_by_minimum[MIN_CELLS_PRIMARY]) - 1}", flush=True)
    stream_bootstrap_inputs = bool(stream_bootstrap_inputs)
    summary_rows, floor_rows, bootstrap_inputs, stream_files = _run_thresholds(
        contexts,
        labels,
        panel,
        predictions,
        budgets_by_minimum,
        seed_chunk=seed_chunk,
        resampling_mode=resampling_mode,
        split_seeds=selected_split_seeds,
        bootstrap_input_dir=output_dir if stream_bootstrap_inputs else None,
        bootstrap_input_stem=stem if stream_bootstrap_inputs else None,
    )
    if stream_bootstrap_inputs:
        if not floor_rows or len(set(int(row["split_seed"]) for row in floor_rows)) != len(selected_split_seeds):
            raise ValueError("streamed replication did not produce every requested split seed")
    elif not summary_rows or any(row["n_split_seeds"] != len(selected_split_seeds) for row in summary_rows):
        raise ValueError("replication did not produce every requested split seed for every threshold/source/metric")
    prediction_path = source_dir / f"{stem}_predictions.npz"
    np.savez_compressed(
        prediction_path,
        context_id=np.asarray(CONTEXTS),
        context_label=np.asarray([CONTEXT_LABELS[key] for key in CONTEXTS]),
        perturbation_label=np.asarray(labels),
        fold_id=fold_ids,
        model_seed=np.asarray(MODEL_SEEDS, dtype=np.int16),
        prediction=predictions,
        evaluation_gene_symbols=np.asarray(panel),
    )
    summary_path = output_dir / f"{stem}.csv"
    floor_path = output_dir / f"{stem}_floors.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(floor_rows).to_csv(floor_path, index=False)
    if stream_bootstrap_inputs:
        input_keys = sorted({key for item in stream_files for key in item["keys"]})
        serialized_stream_files = [
            {
                "path": Path(item["path"]).relative_to(root).as_posix(),
                "split_seeds": [int(seed) for seed in item["split_seeds"]],
                "keys": list(item["keys"]),
            }
            for item in stream_files
        ]
        input_path = output_dir / f"{stem}_bootstrap_inputs.json"
        input_path.write_text(json.dumps({"files": serialized_stream_files, "keys": input_keys}, indent=2) + "\n", encoding="utf-8")
    else:
        input_path = output_dir / f"{stem}_bootstrap_inputs.npz"
        input_arrays: dict[str, np.ndarray] = {}
        input_fields = (
            "cross_left_a", "cross_right_a", "cross_left_b", "cross_right_b",
            "meas_left_a", "meas_left_b", "meas_right_a", "meas_right_b",
            "joint_left_a", "joint_left_b", "joint_right_a", "joint_right_b",
        )
        input_keys = []
        for (minimum_cells, source, metric), values in bootstrap_inputs.items():
            if not values:
                continue
            encoded = f"{minimum_cells}__{source}__{metric}"
            input_keys.append(encoded)
            for field in input_fields:
                input_arrays[f"{encoded}__{field}"] = np.stack([value[field] for value in values]).astype(np.float32)
        np.savez_compressed(input_path, **input_arrays)
        serialized_stream_files = []
    if resampling_mode == "full_size_nonparametric":
        split_half_sensitivity = _budgets(contexts, labels, MIN_CELLS_SENSITIVITY, resampling_mode="split_half")
        sensitivity_count = len(split_half_sensitivity) - ("control" in split_half_sensitivity)
    else:
        sensitivity_count = len(budgets_by_minimum[MIN_CELLS_SENSITIVITY]) - ("control" in budgets_by_minimum[MIN_CELLS_SENSITIVITY])
    report = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2_independent_replication_claim_lock_v1",
        "status": "independent_replication_claim_lock_executed",
        "resampling_mode": resampling_mode,
        "candidate_id": CANDIDATE_ID,
        "study": "NadigOConner2024",
        "contexts": [CONTEXT_LABELS[key] for key in CONTEXTS],
        "raw_sources": [RAW_PATHS[key] for key in CONTEXTS],
        "raw_shapes": {CONTEXT_LABELS[key]: contexts[key]["shape"] for key in CONTEXTS},
        "registry_json": "artifacts/manifests/reordering_replication_candidate_registry.json",
        "registry_sha256": registry_report["registry_sha256"],
        "selection_materialized_before_risk_evaluation": True,
        "provenance_status": selected["batch_context_status"],
        "provenance_evidence_source": selected["provenance_evidence_source"],
        "outcome_blind_selection": True,
        "prediction_evaluated_after_selection": True,
        "risk_evaluated_after_selection": True,
        "common_perturbation_count": len(labels),
        "evaluation_gene_count": len(panel),
        "evaluation_gene_checksum_sha256": hashlib.sha256("\n".join(panel).encode("utf-8")).hexdigest(),
        "prediction_checksums_by_source": {
            CONTEXT_LABELS[source]: _checksum(predictions[index]) for index, source in enumerate(CONTEXTS)
        },
        "prediction_path": prediction_path.relative_to(root).as_posix(),
        "summary_path": summary_path.relative_to(root).as_posix(),
        "floor_path": floor_path.relative_to(root).as_posix(),
        "predictor": "Ahlmann-style strong linear bilinear ridge",
        "predictor_contract": "source context fit once per global 5-fold perturbation-label OOF assignment; the same prediction vectors are scored unchanged in both contexts",
        "folds": FOLDS,
        "fold_seed": SPLIT_SEED,
        "model_seeds": list(MODEL_SEEDS),
        "split_seeds": list(selected_split_seeds),
        "seed_range_indices": [int(seed_start), int(seed_stop)],
        "bootstrap_input_path": input_path.relative_to(root).as_posix(),
        "bootstrap_input_keys": input_keys,
        "bootstrap_input_files": serialized_stream_files,
        "streamed_bootstrap_inputs": stream_bootstrap_inputs,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "thresholds": {
            "primary_min20": int(len(budgets_by_minimum[MIN_CELLS_PRIMARY]) - ("control" in budgets_by_minimum[MIN_CELLS_PRIMARY])),
            "sensitivity_min40": int(sensitivity_count),
        },
        "sensitivity_report": "artifacts/manifests/formal_v2_claim_lock_replication_nadig.csv" if resampling_mode == "full_size_nonparametric" else None,
        "measurement_floor_definition": (
            "matched full-size nonparametric raw-cell truth with fixed source-frozen prediction"
            if resampling_mode == "full_size_nonparametric"
            else "matched raw-cell split-half truth with fixed source-frozen prediction"
        ),
        "joint_floor_definition": "fixed source-frozen model member pair crossed with matched independent truth halves",
        "metric_policy": "same labels, prediction vectors, panel, split seeds, and bootstrap unit across delta cosine, Systema, and effect-rank; no per-metric refit",
        "guide_semantics_policy": "guide_id remains an audit field; perturbations are scored at the exact perturbation-label level, not as single-sgRNA identities",
        "negative_result_audit": "any negative delta is interpreted only after checking source-frozen identity, held-out label folds, disjoint raw-cell halves, matched budgets, finite outputs, and metric equivalence",
    }
    report_path = output_dir / f"{stem}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _write_executed_boundary(root, registry_report, selected, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed-chunk", type=int, default=2)
    parser.add_argument("--resampling-mode", choices=("split_half", "full_size_nonparametric"), default="split_half")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop", type=int, default=None)
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--stream-bootstrap-inputs", action="store_true")
    args = parser.parse_args()
    result = run(
        args.root.resolve(),
        seed_chunk=args.seed_chunk,
        resampling_mode=args.resampling_mode,
        seed_start=args.seed_start,
        seed_stop=args.seed_stop,
        output_stem=args.output_stem,
        stream_bootstrap_inputs=args.stream_bootstrap_inputs,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
