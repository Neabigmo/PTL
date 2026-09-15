"""Run the fixed-cell-budget measurement-depth identifiability audit.

The source-frozen prediction artifact and the existing Frangieh raw-cell
preprocessing contract are reused unchanged.  Each requested cell budget gets
independent with-replacement pseudoreplicates within environment×perturbation
strata.  Seeds are processed in disjoint, deterministic chunks; outputs are
written as per-seed rows first and summarized only at the seed level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from scripts.run_formal_v2_claim_lock_measurement import (  # noqa: E402
    CONDITION_BY_ENVIRONMENT,
    ENVIRONMENTS,
    METRICS,
    MODEL_PAIRS,
    PAIR_ORDER,
    RAW_PATH,
    SPLIT_SEEDS,
    _aggregate_seed_plans,
    _load_metadata,
    _load_panel_positions,
    _metric_risks_batch,
    _profile_lookup,
    _seed_plan,
    _stable_seed,
)
from src.evaluation.decision_theory import DEFAULT_BUDGET_FRACTIONS, replicate_decision_curve  # noqa: E402
from src.evaluation.measurement_depth import (  # noqa: E402
    FULL_LABEL,
    coverage_curve,
    depth_resolution_table,
    matched_label_universe,
    summarize_depth_rows,
)
from src.evaluation.ordering_estimands import perturbation_reordering_burden, summarize_fixed_predictor_ordering  # noqa: E402


FIXED_CELL_BUDGETS = (10, 20, 40, 80, 160)
DEPTH_ORDER = {budget: index for index, budget in enumerate(FIXED_CELL_BUDGETS, start=1)}
FULL_ORDER = len(FIXED_CELL_BUDGETS) + 1


def _aggregate_seed_plans_direct(
    handle: h5py.File,
    plans: list[dict[str, Any]],
    metadata_by_row: pd.DataFrame,
    panel_positions: np.ndarray,
    *,
    panel_size: int,
    gene_chunk_size: int = 256,
) -> list[np.ndarray]:
    """Aggregate large full-depth plans without a huge sparse design product.

    Full-depth sampling can include nearly every QC-passed cell.  Building a
    sparse occurrence matrix for all sampled groups makes ``design.T @ X``
    disproportionately expensive.  This path normalizes one unique-row
    expression block, then reduces each group's explicit sampled row indices;
    duplicate indices remain duplicated and therefore preserve replacement
    sampling exactly.
    """

    x_group = handle["X"]
    shape = tuple(int(value) for value in x_group.attrs["shape"])
    if str(x_group.attrs.get("encoding-type", "")).lower() != "csc_matrix":
        raise ValueError("the raw Frangieh matrix must remain CSC for the low-memory audit")
    data_ds = x_group["data"]
    indices_ds = x_group["indices"]
    indptr_ds = x_group["indptr"]
    sorted_panel_order = np.argsort(panel_positions)
    sorted_panel = panel_positions[sorted_panel_order]
    outputs = [np.zeros((len(plan["virtual_keys"]), panel_size), dtype=np.float32) for plan in plans]
    counts_by_plan = [plan["counts"] for plan in plans]
    plan_rows: list[np.ndarray] = []
    plan_group_slices: list[list[np.ndarray]] = []
    for plan in plans:
        rows = np.concatenate(plan["selected_rows"])
        offsets = np.cumsum([0, *[len(selected) for selected in plan["selected_rows"]]])
        plan_rows.append(rows)
        plan_group_slices.append([slice(int(offsets[i]), int(offsets[i + 1])) for i in range(len(offsets) - 1)])
    union_rows = np.unique(np.concatenate(plan_rows))
    union_positions = [np.searchsorted(union_rows, rows) for rows in plan_rows]
    scales = 10000.0 / metadata_by_row.loc[union_rows, "ncounts"].to_numpy(dtype=np.float32)
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
        selected = selected.multiply(scales[:, None]).tocsr()
        selected.data = np.log1p(selected.data)
        for plan_index, (positions, slices) in enumerate(zip(union_positions, plan_group_slices)):
            for group_index, group_slice in enumerate(slices):
                group_positions = positions[group_slice]
                outputs[plan_index][group_index, output_columns] = np.asarray(
                    selected[group_positions].sum(axis=0)
                ).ravel()
    for plan_index, output in enumerate(outputs):
        output /= counts_by_plan[plan_index][:, None]
    return outputs


def _counts(metadata: pd.DataFrame) -> pd.DataFrame:
    return (
        metadata.groupby(["environment_key", "perturbation_label"], sort=True)
        .size()
        .rename("n_cells_qc")
        .reset_index()
        .rename(columns={"environment_key": "environment_id"})
    )


def _budget_map(
    counts: dict[tuple[str, str], np.ndarray],
    labels: list[str],
    budget: int,
    *,
    full: bool,
) -> dict[tuple[str, str, str], int]:
    result: dict[tuple[str, str, str], int] = {}
    for left, right in PAIR_ORDER:
        control_count = min(len(counts[(left, "control")]), len(counts[(right, "control")]))
        if not full and control_count < budget:
            continue
        result[(left, right, "control")] = int(control_count if full else budget)
        for label in labels:
            matched = min(len(counts[(left, label)]), len(counts[(right, label)]))
            if (full and matched > 0) or (not full and matched >= budget):
                result[(left, right, label)] = int(matched if full else budget)
    return result


def _pair_label_set(
    counts: dict[tuple[str, str], np.ndarray], labels: list[str], left: str, right: str, budget: int, *, full: bool,
    fixed_universe: set[str] | None = None,
) -> list[str]:
    if fixed_universe is not None:
        return [label for label in labels if label in fixed_universe]
    if full:
        control_ok = min(len(counts[(left, "control")]), len(counts[(right, "control")])) > 0
        return labels.copy() if control_ok else []
    control_ok = min(len(counts[(left, "control")]), len(counts[(right, "control")])) >= budget
    if not control_ok:
        return []
    return [label for label in labels if min(len(counts[(left, label)]), len(counts[(right, label)])) >= budget]


def _safe_mean(values: list[dict[str, Any]], key: str) -> float:
    array = np.asarray([row[key] for row in values], dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def _append_decision_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    left: str,
    right: str,
    metric: str,
    split_seed: int,
    cell_budget: int,
    cell_budget_label: str,
    left_risk: np.ndarray,
    right_risk: np.ndarray,
    left_members: np.ndarray,
    right_members: np.ndarray,
    universe_mode: str = "coverage",
) -> None:
    if source == left:
        source_risk, target_risk = left_risk, right_risk
        target_environment = right
        source_side = "left"
    elif source == right:
        source_risk, target_risk = right_risk, left_risk
        target_environment = left
        source_side = "right"
    else:
        raise ValueError("directed decision source must be one endpoint of the target pair")
    curve = replicate_decision_curve(source_risk, target_risk, budgets=DEFAULT_BUDGET_FRACTIONS)
    # The measurement floor must be target self-reproducibility: target A
    # selects and independent target B evaluates.  Source self-instability is
    # not a valid baseline for a source-to-target deployment quantity.
    measurement_floor = replicate_decision_curve(target_risk, target_risk, budgets=DEFAULT_BUDGET_FRACTIONS, independent_only=True)
    joint_left = np.asarray(left_members, dtype=float).reshape(-1, left_members.shape[-1])
    joint_right = np.asarray(right_members, dtype=float).reshape(-1, right_members.shape[-1])
    joint_source, joint_target = (joint_left, joint_right) if source_side == "left" else (joint_right, joint_left)
    joint_curve = replicate_decision_curve(joint_source, joint_target, budgets=DEFAULT_BUDGET_FRACTIONS)
    joint_floor = replicate_decision_curve(joint_target, joint_target, budgets=DEFAULT_BUDGET_FRACTIONS, independent_only=True)
    for item, measurement, joint, joint_floor_item in zip(curve, measurement_floor, joint_curve, joint_floor):
        rows.append({
            "source_environment_id": source,
            "target_environment_id": target_environment,
            "source_target_orientation": f"{source}->{target_environment}",
            "left_target_environment_id": left,
            "right_target_environment_id": right,
            "metric": metric,
            "split_seed": int(split_seed),
            "cell_budget": int(cell_budget),
            "cell_budget_label": cell_budget_label,
            "cell_budget_order": FULL_ORDER if cell_budget_label == FULL_LABEL else DEPTH_ORDER[cell_budget],
            "universe_mode": universe_mode,
            "decision_budget_fraction": item["budget_fraction"],
            "k": item["k"],
            "n_items": item["n_items"],
            "retention": item["retention"],
            "selected_target_risk": item["selected_target_risk"],
            "oracle_target_risk": item["oracle_target_risk"],
            "random_target_risk": item["random_target_risk"],
            "regret": item["regret"],
            "normalized_regret": item["normalized_regret"],
            "boundary_inversion": item["boundary_inversion"],
            "boundary_inversion_lower_bound": item["boundary_inversion_lower_bound"],
            "boundary_bound_holds": item["boundary_bound_holds"],
            "measurement_floor_regret": measurement["regret"],
            "joint_floor_regret": joint_floor_item["regret"],
            "excess_regret": item["regret"] - measurement["regret"],
            "joint_excess_regret": item["regret"] - joint_floor_item["regret"],
            "measurement_floor_status": "target_context_A_selects_target_context_B_evaluates_off_diagonal",
            "joint_floor_status": "target_context_pipeline_self_reproducibility_off_diagonal_member_pairs",
        })


def run(
    root: Path = ROOT,
    *,
    seed_chunk: int = 3,
    draws: int = 2000,
    budgets: tuple[int, ...] = FIXED_CELL_BUDGETS,
    split_seeds: tuple[int, ...] = SPLIT_SEEDS,
    max_labels: int | None = None,
    output_dir: Path | None = None,
    include_full: bool = True,
    include_fixed: bool = True,
    label_start_index: int = 0,
    label_stop_index: int | None = None,
    universe_mode: str = "coverage",
) -> dict[str, Any]:
    root = root.resolve()
    if universe_mode not in {"coverage", "matched_fixed"}:
        raise ValueError("universe_mode must be coverage or matched_fixed")
    budgets = tuple(sorted({int(value) for value in budgets}))
    if include_fixed and (not budgets or any(value < 2 for value in budgets)):
        raise ValueError("budgets must contain integers >= 2")
    if not split_seeds or any(seed not in SPLIT_SEEDS for seed in split_seeds):
        raise ValueError("split_seeds must be a non-empty subset of the declared schedule")
    payload = np.load(root / "artifacts/source_data/frangieh_source_frozen_predictions.npz", allow_pickle=False)
    all_labels = payload["perturbation_label"].astype(str).tolist()
    label_start_index = int(label_start_index)
    label_stop_index = len(all_labels) if label_stop_index is None else int(label_stop_index)
    if not 0 <= label_start_index < label_stop_index <= len(all_labels):
        raise ValueError("label range must be a non-empty interval within the frozen label order")
    labels = all_labels[label_start_index:label_stop_index]
    predictions = payload["prediction"].astype(np.float32, copy=False)
    panel = payload["evaluation_gene_symbols"].astype(str).tolist()
    metadata = _load_metadata(root, labels)
    metadata_by_row = metadata.set_index("row_index")
    groups: dict[tuple[str, str], np.ndarray] = {
        (str(environment), str(label)): group["row_index"].to_numpy(dtype=np.int64)
        for (environment, label), group in metadata.groupby(["environment_key", "perturbation_label"], sort=True)
    }
    required_groups = [(environment, label) for environment in ENVIRONMENTS for label in [*labels, "control"]]
    missing = [key for key in required_groups if key not in groups]
    if missing:
        raise ValueError(f"raw Frangieh groups are missing: {missing[:5]}")
    if max_labels is not None:
        labels = labels[: int(max_labels)]
    panel_positions = _load_panel_positions(root, panel)
    count_frame = _counts(metadata)
    fixed_budget = max(budgets) if budgets else FIXED_CELL_BUDGETS[-1]
    fixed_universe_metadata = _load_metadata(root, all_labels) if universe_mode == "matched_fixed" else metadata
    fixed_universe_groups: dict[tuple[str, str], np.ndarray] = {
        (str(environment), str(label)): group["row_index"].to_numpy(dtype=np.int64)
        for (environment, label), group in fixed_universe_metadata.groupby(["environment_key", "perturbation_label"], sort=True)
    }
    fixed_universes = {
        pair: set(_pair_label_set(fixed_universe_groups, all_labels, pair[0], pair[1], fixed_budget, full=False))
        for pair in PAIR_ORDER
    } if universe_mode == "matched_fixed" else {}
    output_dir = (output_dir or (root / "artifacts/manifests")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    item_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    label_index = {label: index for index, label in enumerate(labels)}
    depth_specs = ([(budget, str(budget), False) for budget in budgets] if include_fixed else [])
    if include_full:
        depth_specs.append((10**9, FULL_LABEL, True))
    if not depth_specs:
        raise ValueError("at least one fixed or full depth must be selected")
    for left, right in PAIR_ORDER:
        for budget, budget_label, full in depth_specs:
            if full:
                eligible = _pair_label_set(
                    groups, labels, left, right, budget, full=True,
                    fixed_universe=fixed_universes[(left, right)] if universe_mode == "matched_fixed" else None,
                )
                coverage_rows.append({
                    "left_target_environment_id": left,
                    "right_target_environment_id": right,
                    "cell_budget": int(budget),
                    "cell_budget_label": FULL_LABEL,
                    "cell_budget_order": FULL_ORDER,
                    "n_labels_all": len(labels),
                    "n_labels_matched": len(eligible),
                    "coverage_fraction": float(len(eligible) / len(labels)) if labels else float("nan"),
                    "matched_universe_sha256": hashlib.sha256("\n".join(eligible).encode()).hexdigest(),
                    "matched_universe_policy": "all labels with nonempty matched target strata; each label uses its own full matched cell count with replacement",
                    "universe_mode": universe_mode,
                })
            else:
                if universe_mode == "coverage":
                    curve = coverage_curve(count_frame, left_environment=left, right_environment=right, budgets=(budget,))
                    row = curve.iloc[0].to_dict()
                else:
                    eligible = _pair_label_set(
                        groups, all_labels, left, right, budget, full=False,
                        fixed_universe=fixed_universes[(left, right)],
                    )
                    row = {
                        "left_target_environment_id": left,
                        "right_target_environment_id": right,
                        "cell_budget": int(budget),
                        "n_labels_all": len(all_labels),
                        "n_labels_matched": len(eligible),
                        "coverage_fraction": float(len(eligible) / len(all_labels)) if all_labels else float("nan"),
                        "matched_universe_sha256": hashlib.sha256("\n".join(eligible).encode()).hexdigest(),
                        "matched_universe_policy": f"fixed universe frozen at {fixed_budget} cells and reused at {budget} cells; control must pass",
                    }
                row.update({"cell_budget_label": budget_label, "cell_budget_order": DEPTH_ORDER[budget]})
                row["universe_mode"] = universe_mode
                coverage_rows.append(row)
    with h5py.File(root / RAW_PATH.relative_to(ROOT), "r") as handle:
        for budget, budget_label, full in depth_specs:
            budget_values = _budget_map(groups, labels, budget, full=full)
            if not budget_values:
                continue
            for start in range(0, len(split_seeds), max(1, int(seed_chunk))):
                selected_seeds = split_seeds[start:start + max(1, int(seed_chunk))]
                plans = [_seed_plan(groups, labels, budget_values, seed, replacement=True) for seed in selected_seeds]
                # The sparse design product is faster for a small label shard;
                # the indexed reduction is retained for the all-label case,
                # where the design matrix becomes pathologically large.
                aggregate_fn = (
                    _aggregate_seed_plans_direct
                    if full and len(labels) > 40
                    else _aggregate_seed_plans
                )
                outputs = aggregate_fn(handle, plans, metadata_by_row, panel_positions, panel_size=len(panel))
                for seed, plan, output in zip(selected_seeds, plans, outputs):
                    profiles = _profile_lookup(plan, output)
                    fixed_risk_cache: dict[tuple[str, str], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
                    fixed_label_cache: dict[str, list[str]] = {}
                    if not full:
                        # For a fixed budget, a target-context truth profile is
                        # identical across all pairs containing that context.
                        # Score the union once, then slice each matched pair.
                        for environment in ENVIRONMENTS:
                            environment_labels = [
                                label for label in labels
                                if any(
                                    environment in (left_candidate, right_candidate)
                                    and (left_candidate, right_candidate, label) in budget_values
                                    for left_candidate, right_candidate in PAIR_ORDER
                                )
                            ]
                            if not environment_labels:
                                continue
                            control_keys = [
                                (left_candidate, right_candidate, "control")
                                for left_candidate, right_candidate in PAIR_ORDER
                                if environment in (left_candidate, right_candidate)
                                and (left_candidate, right_candidate, "control") in budget_values
                            ]
                            control_budget = budget_values[control_keys[0]]
                            truth = np.stack([
                                np.stack([
                                    profiles[(environment, label, half, budget)]
                                    - profiles[(environment, "control", half, control_budget)]
                                    for label in environment_labels
                                ])
                                for half in (0, 1)
                            ])
                            fixed_label_cache[environment] = environment_labels
                            for source_index, source in enumerate(ENVIRONMENTS):
                                fixed_risk_cache[(source, environment)] = _metric_risks_batch(
                                    predictions[source_index][np.asarray([label_index[label] for label in environment_labels])], truth
                                )
                    for source_index, source in enumerate(ENVIRONMENTS):
                        source_members = predictions[source_index]
                        for left, right in PAIR_ORDER:
                            if source not in (left, right):
                                continue
                            eligible = _pair_label_set(
                                groups, labels, left, right, budget, full=full,
                                fixed_universe=fixed_universes[(left, right)] if universe_mode == "matched_fixed" else None,
                            )
                            if not eligible:
                                continue
                            indices = np.asarray([labels.index(label) for label in eligible], dtype=np.int64)
                            control_budget = budget_values[(left, right, "control")]
                            truths: dict[str, np.ndarray] = {}
                            for environment in (left, right):
                                truths[environment] = np.stack([
                                    np.stack([
                                        profiles[(environment, label, half, budget_values[(left, right, label)])]
                                        - profiles[(environment, "control", half, control_budget)]
                                        for label in eligible
                                    ])
                                    for half in (0, 1)
                                ])
                            for metric in METRICS:
                                if full:
                                    risk_by_environment: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {
                                        environment: _metric_risks_batch(source_members[indices], truths[environment])
                                        for environment in (left, right)
                                    }
                                    left_mean = risk_by_environment[left][0][metric]
                                    right_mean = risk_by_environment[right][0][metric]
                                    left_members = risk_by_environment[left][1][metric]
                                    right_members = risk_by_environment[right][1][metric]
                                else:
                                    left_union = fixed_label_cache[left]
                                    right_union = fixed_label_cache[right]
                                    left_positions = [left_union.index(label) for label in eligible]
                                    right_positions = [right_union.index(label) for label in eligible]
                                    left_cached = fixed_risk_cache[(source, left)]
                                    right_cached = fixed_risk_cache[(source, right)]
                                    left_mean = left_cached[0][metric][:, left_positions]
                                    right_mean = right_cached[0][metric][:, right_positions]
                                    left_members = left_cached[1][metric][:, :, left_positions]
                                    right_members = right_cached[1][metric][:, :, right_positions]
                                burden = perturbation_reordering_burden(left_mean, right_mean, block_size=min(512, len(eligible)))
                                ordering = summarize_fixed_predictor_ordering(left_mean, right_mean)
                                for index, label in enumerate(eligible):
                                    item_rows.append({
                                        "source_environment_id": source,
                                        "left_target_environment_id": left,
                                        "right_target_environment_id": right,
                                        "metric": metric,
                                        "split_seed": int(seed),
                                        "perturbation_label": label,
                                        "cell_budget": int(budget),
                                        "cell_budget_label": budget_label,
                                        "cell_budget_order": FULL_ORDER if full else DEPTH_ORDER[budget],
                                        "universe_mode": universe_mode,
                                        "effective_cell_budget": int(budget_values[(left, right, label)]),
                                        "cross_disagreement": burden["cross_burden"][index],
                                        "within_disagreement_left": burden["within_burden_left"][index],
                                        "within_disagreement_right": burden["within_burden_right"][index],
                                        "identifiable_divergence": burden["identifiable_burden"][index],
                                        "stable_pair_fraction": ordering["stable_fraction_both"],
                                        "n_items": len(eligible),
                                        "n_pairs": len(eligible) * (len(eligible) - 1) // 2,
                                        "truth_contract": "QC-passed Frangieh raw cells, source-frozen prediction, independent with-replacement A/B pseudoreplicates, matched perturbation/control strata",
                                    })
                                _append_decision_rows(
                                    decision_rows,
                                    source=source,
                                    left=left,
                                    right=right,
                                    metric=metric,
                                    split_seed=int(seed),
                                    cell_budget=int(budget),
                                    cell_budget_label=budget_label,
                                    universe_mode=universe_mode,
                                    left_risk=left_mean,
                                    right_risk=right_mean,
                                    left_members=left_members,
                                    right_members=right_members,
                                )
                                # Keep member-level risks for every depth.  The
                                # directed hierarchical decision bootstrap needs
                                # synchronized label×seed resampling at fixed
                                # budgets as well as at full depth.
                                for index, label in enumerate(eligible):
                                    for replicate_index in range(left_mean.shape[0]):
                                        risk_row = {
                                            "source_environment_id": source,
                                            "left_target_environment_id": left,
                                            "right_target_environment_id": right,
                                            "metric": metric,
                                            "split_seed": int(seed),
                                            "perturbation_label": label,
                                            "measurement_replicate": int(replicate_index),
                                            "cell_budget": int(budget),
                                            "cell_budget_label": budget_label,
                                            "cell_budget_order": FULL_ORDER if full else DEPTH_ORDER[budget],
                                            "left_risk": float(left_mean[replicate_index, index]),
                                            "right_risk": float(right_mean[replicate_index, index]),
                                        }
                                        for member_index in range(left_members.shape[1]):
                                            risk_row[f"left_member_{member_index}"] = float(left_members[replicate_index, member_index, index])
                                            risk_row[f"right_member_{member_index}"] = float(right_members[replicate_index, member_index, index])
                                        risk_rows.append(risk_row)
                print(f"[measurement-depth] budget={budget_label} seeds={selected_seeds[0]}-{selected_seeds[-1]}", flush=True)
    items = pd.DataFrame(item_rows)
    decisions = pd.DataFrame(decision_rows)
    risks = pd.DataFrame(risk_rows)
    coverage = pd.DataFrame(coverage_rows)
    summary = summarize_depth_rows(items, draws=draws, ci_level=0.90)
    resolution = depth_resolution_table(summary)
    items_path = output_dir / "reliability_transport_measurement_depth_items.csv"
    summary_path = output_dir / "reliability_transport_measurement_depth_summary.csv"
    decision_path = output_dir / "reliability_transport_measurement_depth_decision.csv"
    coverage_path = output_dir / "reliability_transport_measurement_depth_coverage.csv"
    resolution_path = output_dir / "reliability_transport_measurement_depth_resolution.csv"
    risks_path = output_dir / "reliability_transport_measurement_depth_risks.csv"
    items.to_csv(items_path, index=False)
    summary.to_csv(summary_path, index=False)
    decisions.to_csv(decision_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    resolution.to_csv(resolution_path, index=False)
    risks.to_csv(risks_path, index=False)
    report = {
        "schema_version": 1,
        "status": "executed",
        "analysis": "measurement_depth_identifiability",
        "source_frozen_prediction": "artifacts/source_data/frangieh_source_frozen_predictions.npz",
        "raw_source": "data/raw/scperturb_v1.4/FrangiehIzar2021_RNA.h5ad",
        "split_seeds": list(split_seeds),
        "label_index_range": [label_start_index, label_stop_index],
        "n_labels": len(labels),
        "fixed_cell_budgets": list(budgets),
        "universe_mode": universe_mode,
        "matched_fixed_budget": int(fixed_budget) if universe_mode == "matched_fixed" else None,
        "full_depth_label": FULL_LABEL,
        "full_depth_contract": "per-label matched target cell count, independently sampled with replacement",
        "decision_budgets": list(DEFAULT_BUDGET_FRACTIONS),
        "bootstrap_draws": int(draws),
        "bootstrap_unit": "measurement seed after perturbation-level exact burden aggregation",
        "metrics": list(METRICS),
        "outputs": {
            "items": items_path.relative_to(root).as_posix(),
            "summary": summary_path.relative_to(root).as_posix(),
            "decision": decision_path.relative_to(root).as_posix(),
            "coverage": coverage_path.relative_to(root).as_posix(),
            "resolution": resolution_path.relative_to(root).as_posix(),
            "risks": risks_path.relative_to(root).as_posix(),
        },
        "checks": {
            "item_rows": int(len(items)),
            "summary_rows": int(len(summary)),
            "decision_rows": int(len(decisions)),
            "coverage_rows": int(len(coverage)),
            "resolution_rows": int(len(resolution)),
            "risk_rows": int(len(risks)),
            "no_prediction_refit": True,
            "target_outcomes_not_used": True,
        },
    }
    report_path = output_dir / "reliability_transport_measurement_depth.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed-chunk", type=int, default=3)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--budgets", default=",".join(str(value) for value in FIXED_CELL_BUDGETS))
    parser.add_argument("--seed-start-index", type=int, default=0)
    parser.add_argument("--seed-stop-index", type=int, default=None)
    parser.add_argument("--max-labels", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-full", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--label-start-index", type=int, default=0)
    parser.add_argument("--label-stop-index", type=int, default=None)
    parser.add_argument("--universe-mode", choices=("coverage", "matched_fixed"), default="coverage")
    args = parser.parse_args()
    stop = len(SPLIT_SEEDS) if args.seed_stop_index is None else int(args.seed_stop_index)
    if not 0 <= args.seed_start_index < stop <= len(SPLIT_SEEDS):
        raise ValueError("seed index range is outside the declared schedule")
    run(
        args.root,
        seed_chunk=args.seed_chunk,
        draws=args.draws,
        budgets=tuple(int(value) for value in args.budgets.split(",") if value.strip()),
        split_seeds=tuple(SPLIT_SEEDS[args.seed_start_index:stop]),
        max_labels=args.max_labels,
        output_dir=args.output_dir,
        include_full=not args.no_full,
        include_fixed=not args.full_only,
        label_start_index=args.label_start_index,
        label_stop_index=args.label_stop_index,
        universe_mode=args.universe_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
