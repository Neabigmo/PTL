"""Build and replay the leakage-free, nested-prefix Frangieh target audit.

The audit pool is generated independently from the canonical measurement-depth
artifacts.  For every audit seed, context, perturbation, and replicate, one
160-cell with-replacement draw is made and only its exact prefixes are exposed
at 10/20/40/80/160 cells.  The policy receives a per-candidate view of the
currently measured target risk only; reference and evaluation pools are used
only after allocation for offline reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_claim_lock_measurement import (  # noqa: E402
    ENVIRONMENTS,
    METRICS as CANONICAL_METRICS,
    PREDICTION_PATH,
    RAW_PATH,
    SPLIT_SEEDS,
    _aggregate_seed_plans,
    _load_metadata,
    _load_panel_positions,
    _metric_risks_batch,
    _profile_lookup,
    _stable_seed,
)
from src.evaluation.decision_theory import top_k_decision_metrics  # noqa: E402
from src.evaluation.shortlist_audit import (  # noqa: E402
    allocate_nested_prefixes,
    audit_shortlist,
    boundary_priority,
    empirical_target_interval,
)


METRICS = tuple(CANONICAL_METRICS)
DECISION_BUDGETS = (0.05, 0.10, 0.20, 0.50)
DEPTHS = (10, 20, 40, 80, 160)
METHODS = ("pilot_only", "uniform_depth", "random_extra_depth", "source_uncertainty", "ptl_boundary")
AUDIT_SEEDS = tuple(SPLIT_SEEDS[:10])
REFERENCE_SEEDS = tuple(SPLIT_SEEDS[10:20])
EVALUATION_SEEDS = tuple(SPLIT_SEEDS[20:30])
N_REPLICATES = 2
PILOT_DEPTH = DEPTHS[0]
MAX_DEPTH = DEPTHS[-1]
EPSILON = 0.05
CACHE_NAME = "target_audit_nested_prefix_risks.csv"
OUTPUT_NAME = "target_audit_simulation.csv"
REPORT_NAME = "target_audit_simulation.json"
CACHE_COLUMNS = (
    "source_environment_id",
    "target_environment_id",
    "metric",
    "split_seed",
    "measurement_replicate",
    "depth",
    "perturbation_label",
    "target_risk",
)


def _draw_plan(groups: dict[tuple[str, str], np.ndarray], labels: list[str], seed: int) -> dict[str, Any]:
    """Draw one maximum-depth sample and derive exact nested prefixes from it."""

    max_keys: list[tuple[str, str, int]] = []
    max_rows: list[np.ndarray] = []
    virtual_keys: list[tuple[str, str, int, int]] = []
    selected_rows: list[np.ndarray] = []
    for environment_index, environment in enumerate(ENVIRONMENTS):
        for label in [*labels, "control"]:
            source_rows = np.asarray(groups[(environment, label)], dtype=np.int64)
            if source_rows.ndim != 1 or source_rows.size == 0:
                raise ValueError(f"no raw cells available for {(environment, label)}")
            for replicate in range(N_REPLICATES):
                maximum = np.random.default_rng(
                    _stable_seed("target_audit_nested_prefix_v2", seed, environment_index, environment, label, replicate)
                ).choice(source_rows, MAX_DEPTH, replace=True).astype(np.int64, copy=False)
                max_keys.append((environment, label, replicate))
                max_rows.append(maximum.copy())
                for depth in DEPTHS:
                    prefix = maximum[:depth].copy()
                    if prefix.shape != (depth,):
                        raise AssertionError("nested-prefix draw was not preserved exactly")
                    virtual_keys.append((environment, label, replicate, depth))
                    selected_rows.append(prefix)
    if len(max_rows) != len(virtual_keys) // len(DEPTHS):
        raise AssertionError("one maximum-depth draw was not created per audit replicate")
    return {
        "seed": int(seed),
        "max_keys": max_keys,
        "max_rows": max_rows,
        "virtual_keys": virtual_keys,
        "selected_rows": selected_rows,
        "counts": np.asarray([len(value) for value in selected_rows], dtype=np.int32),
    }


def _validate_nested_cache(frame: pd.DataFrame, n_labels: int) -> None:
    required = set(CACHE_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"nested target-risk cache is missing columns: {missing}")
    if frame.empty:
        raise ValueError("nested target-risk cache is empty")
    if set(pd.to_numeric(frame["split_seed"], errors="raise").astype(int).unique()) != set(AUDIT_SEEDS):
        raise ValueError("nested target-risk cache has the wrong audit seed pool")
    if set(pd.to_numeric(frame["depth"], errors="raise").astype(int).unique()) != set(DEPTHS):
        raise ValueError("nested target-risk cache has the wrong depth schedule")
    key = [
        "source_environment_id",
        "target_environment_id",
        "metric",
        "split_seed",
        "measurement_replicate",
        "depth",
        "perturbation_label",
    ]
    if frame.duplicated(key).any():
        raise ValueError("nested target-risk cache contains duplicate observations")
    expected = len(ENVIRONMENTS) ** 2 * len(METRICS) * len(AUDIT_SEEDS) * N_REPLICATES * len(DEPTHS) * n_labels
    if len(frame) != expected:
        raise ValueError(f"nested target-risk cache has {len(frame)} rows; expected {expected}")
    if not np.isfinite(pd.to_numeric(frame["target_risk"], errors="raise").to_numpy(float)).all():
        raise ValueError("nested target-risk cache contains non-finite target risks")


def _cache_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        header = pd.read_csv(path, nrows=0)
        if set(CACHE_COLUMNS).difference(header.columns):
            return False
        frame = pd.read_csv(path, usecols=list(CACHE_COLUMNS), low_memory=False)
        n_labels = int(frame["perturbation_label"].astype(str).nunique())
        _validate_nested_cache(frame, n_labels)
        return True
    except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError):
        return False


def build_nested_prefix_cache(root: Path = ROOT, *, force: bool = False) -> Path:
    """Build a separate nested-prefix risk cache; canonical artifacts are read-only."""

    root = root.resolve()
    path = root / "artifacts/manifests" / CACHE_NAME
    if not force and _cache_is_valid(path):
        return path

    prediction_path = root / PREDICTION_PATH.relative_to(ROOT)
    payload = np.load(prediction_path, allow_pickle=False)
    labels = payload["perturbation_label"].astype(str).tolist()
    predictions = payload["prediction"].astype(np.float32, copy=False)
    panel = payload["evaluation_gene_symbols"].astype(str).tolist()
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError("source-frozen prediction labels must be unique and non-empty")
    if predictions.shape[:2] != (len(ENVIRONMENTS), len(labels)):
        raise ValueError("source-frozen prediction artifact has an unexpected label/environment shape")
    metadata = _load_metadata(root, labels)
    groups = {
        (str(environment), str(label)): group["row_index"].to_numpy(np.int64)
        for (environment, label), group in metadata.groupby(["environment_key", "perturbation_label"], sort=True)
    }
    missing = [(environment, label) for environment in ENVIRONMENTS for label in [*labels, "control"] if (environment, label) not in groups]
    if missing:
        raise ValueError(f"raw Frangieh groups missing: {missing[:5]}")
    positions = _load_panel_positions(root, panel)
    metadata_by_row = metadata.set_index("row_index")
    rows: list[dict[str, Any]] = []

    def append_plan_rows(seed: int, plan: dict[str, Any], output: np.ndarray) -> None:
        profiles = _profile_lookup(plan, output)
        for depth in DEPTHS:
            for target in ENVIRONMENTS:
                truth = np.stack(
                    [
                        np.stack(
                            [
                                profiles[(target, label, replicate, depth)]
                                - profiles[(target, "control", replicate, depth)]
                                for label in labels
                            ]
                        )
                        for replicate in range(N_REPLICATES)
                    ]
                )
                for source_index, source in enumerate(ENVIRONMENTS):
                    risks, _ = _metric_risks_batch(predictions[source_index], truth)
                    for metric in METRICS:
                        for replicate in range(N_REPLICATES):
                            rows.extend(
                                {
                                    "source_environment_id": source,
                                    "target_environment_id": target,
                                    "metric": metric,
                                    "split_seed": int(seed),
                                    "measurement_replicate": replicate,
                                    "depth": depth,
                                    "perturbation_label": label,
                                    "target_risk": float(risks[metric][replicate, label_index]),
                                }
                                for label_index, label in enumerate(labels)
                            )

    with h5py.File(root / RAW_PATH.relative_to(ROOT), "r") as handle:
        seed_chunk = 2
        for start in range(0, len(AUDIT_SEEDS), seed_chunk):
            selected_seeds = AUDIT_SEEDS[start : start + seed_chunk]
            plans = [_draw_plan(groups, labels, seed) for seed in selected_seeds]
            outputs = _aggregate_seed_plans(
                handle,
                plans,
                metadata_by_row,
                positions,
                panel_size=len(panel),
            )
            for seed, plan, output in zip(selected_seeds, plans, outputs):
                append_plan_rows(seed, plan, output)
            print(f"[target-audit-cache] seeds={selected_seeds}", flush=True)
    cache = pd.DataFrame(rows, columns=CACHE_COLUMNS)
    _validate_nested_cache(cache, len(labels))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    cache.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def _matrix(frame: pd.DataFrame, seeds: tuple[int, ...], value: str) -> tuple[list[str], np.ndarray]:
    """Materialize a complete seed×replicate×label matrix without silent deduplication."""

    frame = frame.copy()
    frame["split_seed"] = pd.to_numeric(frame["split_seed"], errors="raise").astype(int)
    frame["measurement_replicate"] = pd.to_numeric(frame["measurement_replicate"], errors="raise").astype(int)
    frame["perturbation_label"] = frame["perturbation_label"].astype(str)
    key = ["split_seed", "measurement_replicate", "perturbation_label"]
    if frame.duplicated(key).any():
        raise ValueError(f"duplicate rows in risk matrix for {value}")
    index = pd.MultiIndex.from_product((seeds, range(N_REPLICATES)), names=("split_seed", "measurement_replicate"))
    pivot = frame.loc[frame["split_seed"].isin(seeds)].pivot(index=key[:2], columns="perturbation_label", values=value).reindex(index)
    if pivot.empty or pivot.shape[0] != len(seeds) * N_REPLICATES or pivot.isna().any().any():
        raise ValueError(f"incomplete risk matrix for {value}")
    labels = sorted(str(label) for label in pivot.columns)
    pivot = pivot.reindex(columns=labels)
    matrix = pivot.to_numpy(float)
    if matrix.shape != (len(seeds) * N_REPLICATES, len(labels)) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid risk matrix for {value}")
    return labels, matrix


def _canonical_matrix(group: pd.DataFrame, context: str, seeds: tuple[int, ...]) -> tuple[list[str], np.ndarray]:
    """Select one side of a canonical pair without consulting target data."""

    if group.empty:
        raise ValueError("empty canonical transfer group")
    left = str(group["left_target_environment_id"].iloc[0])
    right = str(group["right_target_environment_id"].iloc[0])
    if context == left:
        column = "left_risk"
    elif context == right:
        column = "right_risk"
    else:
        raise ValueError(f"context {context!r} is not a side of canonical pair {left!r}, {right!r}")
    return _matrix(group.assign(value=group[column]), seeds, "value")


def _current_replicates(matrices: dict[int, np.ndarray], allocated: np.ndarray) -> np.ndarray:
    """Return only each item's currently observed risk replicate values."""

    allocated = np.asarray(allocated, dtype=int)
    if allocated.ndim != 1 or allocated.size < 1 or not np.isin(allocated, DEPTHS).all():
        raise ValueError("allocated depths must be one of the declared audit depths")
    first = matrices[PILOT_DEPTH]
    current = np.empty((first.shape[0], allocated.size), dtype=float)
    for depth in np.unique(allocated):
        indices = np.flatnonzero(allocated == depth)
        matrix = np.asarray(matrices[int(depth)], dtype=float)
        if matrix.shape[0] != current.shape[0] or matrix.shape[1] != allocated.size:
            raise ValueError("nested risk matrices have incompatible shapes")
        current[:, indices] = matrix[:, indices]
    if not np.isfinite(current).all():
        raise ValueError("current audit observations contain non-finite values")
    return current


def _policy_observation(matrices: dict[int, np.ndarray], allocated: np.ndarray) -> dict[str, np.ndarray]:
    """Build the information-firewalled input passed to a replay policy."""

    return {
        "allocated_depths": np.asarray(allocated, dtype=int).copy(),
        "target_risk_replicates": _current_replicates(matrices, allocated),
    }


def _intervals(matrices: dict[int, np.ndarray], allocated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return empirical_target_interval(_current_replicates(matrices, allocated))


def _priority(
    method: str,
    source: np.ndarray,
    observation: dict[str, np.ndarray],
    selected: np.ndarray,
    random_rank: np.ndarray,
) -> np.ndarray:
    """Compute a policy score from source data and current audit data only."""

    allocated = np.asarray(observation["allocated_depths"], dtype=int)
    current = np.asarray(observation["target_risk_replicates"], dtype=float)
    if allocated.ndim != 1 or current.ndim != 2 or current.shape[1] != allocated.size:
        raise ValueError("policy observation has incompatible allocation/data shapes")
    if method == "uniform_depth":
        return -allocated.astype(float)
    if method == "random_extra_depth":
        return np.asarray(random_rank, dtype=float) - allocated / 1e6
    if method == "source_uncertainty":
        return np.asarray(source, dtype=float).std(axis=0, ddof=1) - allocated / 1e6
    lower, upper = empirical_target_interval(current)
    return boundary_priority(lower, upper, selected)


def _actual_additional_cells(allocated: np.ndarray) -> int:
    """Compute target-plus-shared-control extra cells from the final allocation."""

    allocated = np.asarray(allocated, dtype=int)
    if allocated.ndim != 1 or allocated.size < 1 or np.any(allocated < PILOT_DEPTH):
        raise ValueError("allocated depths must be at least the pilot depth")
    control_depth = int(np.max(allocated))
    candidate_extra = int(np.sum(allocated - PILOT_DEPTH))
    control_extra = control_depth - PILOT_DEPTH
    return N_REPLICATES * (candidate_extra + control_extra)


def replay_target_audit(root: Path = ROOT, *, force_cache: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay every source shortlist under the declared audit information firewall."""

    root = root.resolve()
    manifests = root / "artifacts/manifests"
    cache_path = build_nested_prefix_cache(root, force=force_cache)
    nested = pd.read_csv(cache_path, usecols=list(CACHE_COLUMNS), low_memory=False)
    nested["source_environment_id"] = nested["source_environment_id"].astype(str)
    nested["target_environment_id"] = nested["target_environment_id"].astype(str)
    nested["metric"] = nested["metric"].astype(str)
    nested["perturbation_label"] = nested["perturbation_label"].astype(str)
    nested["split_seed"] = pd.to_numeric(nested["split_seed"], errors="raise").astype(int)
    nested["measurement_replicate"] = pd.to_numeric(nested["measurement_replicate"], errors="raise").astype(int)
    nested["depth"] = pd.to_numeric(nested["depth"], errors="raise").astype(int)
    _validate_nested_cache(nested, int(nested["perturbation_label"].nunique()))

    canonical_path = manifests / "reliability_transport_measurement_depth_matched_fixed_risks.csv"
    canonical_columns = [
        "source_environment_id",
        "left_target_environment_id",
        "right_target_environment_id",
        "metric",
        "split_seed",
        "perturbation_label",
        "measurement_replicate",
        "cell_budget_label",
        "left_risk",
        "right_risk",
    ]
    canonical = pd.read_csv(canonical_path, usecols=canonical_columns, low_memory=False)
    canonical["cell_budget_label"] = canonical["cell_budget_label"].astype(str).str.strip("'\" ")
    canonical = canonical.loc[canonical["cell_budget_label"].eq("full")].copy()
    rows: list[dict[str, Any]] = []
    group_columns = ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"]
    for key, group in canonical.groupby(group_columns, observed=True):
        source, left, right, metric = map(str, key)
        if metric not in METRICS or source not in (left, right):
            continue

        source_labels, source_matrix = _canonical_matrix(group, source, SPLIT_SEEDS)
        audit = nested.loc[
            nested["source_environment_id"].eq(source)
            & nested["target_environment_id"].isin((left, right))
            & nested["metric"].eq(metric)
        ]
        target = right if source == left else left
        audit = audit.loc[audit["target_environment_id"].eq(target)].copy()
        if audit.empty:
            raise ValueError(f"missing nested audit rows for {source}->{target}, {metric}")
        raw = {depth: _matrix(audit.loc[audit["depth"].eq(depth)], AUDIT_SEEDS, "target_risk") for depth in DEPTHS}
        common = sorted(set(source_labels).intersection(*(set(labels) for labels, _ in raw.values())))
        if not common:
            raise ValueError(f"no fixed source/audit candidate universe for {source}->{target}, {metric}")
        source_matrix = source_matrix[:, [source_labels.index(label) for label in common]]
        matrices = {
            depth: matrix[:, [labels.index(label) for label in common]]
            for depth, (labels, matrix) in raw.items()
        }
        source_mean = source_matrix.mean(axis=0)
        n = len(common)

        for fraction in DECISION_BUDGETS:
            selected = np.argsort(source_mean, kind="stable")[: max(1, int(np.floor(n * fraction)))]
            additional_budget = int(np.floor(fraction * N_REPLICATES * (MAX_DEPTH - PILOT_DEPTH) * (n + 1)))
            for method in METHODS:
                rng = np.random.default_rng(
                    _stable_seed("target_audit_policy_v2", source, target, metric, fraction, method)
                )
                random_rank = np.empty(n, dtype=float)
                random_rank[rng.permutation(n)] = np.arange(n, 0, -1, dtype=float)

                if method == "pilot_only":
                    allocated = np.full(n, PILOT_DEPTH, dtype=int)
                    control_depth, spent = PILOT_DEPTH, 0
                else:
                    allocated, control_depth, spent = allocate_nested_prefixes(
                        DEPTHS,
                        n,
                        additional_budget,
                        lambda observation, m=method: _priority(m, source_matrix, observation, selected, random_rank),
                        n_replicates=N_REPLICATES,
                        observe=lambda current, cache=matrices: _policy_observation(cache, current),
                    )
                expected_spent = _actual_additional_cells(allocated)
                if spent != expected_spent or control_depth != int(np.max(allocated)):
                    raise AssertionError("nested audit cost does not equal actual target-cell use")
                if spent > additional_budget:
                    raise AssertionError("nested audit exceeded its additional target-cell budget")

                lower, upper = _intervals(matrices, allocated)
                result = audit_shortlist(lower, upper, selected, epsilon=EPSILON)

                # Evaluation outcomes are loaded only after the policy has
                # completed.  They are never part of selection or allocation.
                eval_labels, evaluation = _canonical_matrix(group, target, EVALUATION_SEEDS)
                missing_eval = sorted(set(common).difference(eval_labels))
                if missing_eval:
                    raise ValueError(f"evaluation pool is missing fixed candidates: {missing_eval[:5]}")
                evaluation = evaluation[:, [eval_labels.index(label) for label in common]]
                truth = top_k_decision_metrics(source_mean, evaluation.mean(axis=0), budget_fraction=fraction)
                true_state = "REUSE" if truth["regret"] <= EPSILON else "REVISE"
                rows.append(
                    {
                        "source_environment_id": source,
                        "target_environment_id": target,
                        "transfer_id": f"{source}->{target}",
                        "metric": metric,
                        "decision_budget_fraction": fraction,
                        "audit_extra_budget_fraction": fraction,
                        "k": result["k"],
                        "audit_method": method,
                        "pilot_target_cells_per_label": PILOT_DEPTH,
                        "target_labels_piloted": n,
                        "target_labels_upgraded": int(np.sum(allocated > PILOT_DEPTH)),
                        "target_labels_at_full_depth": int(np.sum(allocated == MAX_DEPTH)),
                        "control_prefix_depth": control_depth,
                        "target_cells_spent": N_REPLICATES * PILOT_DEPTH * (n + 1) + spent,
                        "additional_target_cells": spent,
                        "additional_target_cell_budget": additional_budget,
                        "depth_profile": json.dumps(
                            {str(depth): int(np.sum(allocated == depth)) for depth in DEPTHS}, sort_keys=True
                        ),
                        "audit_state": result["state"],
                        "ub_regret": result["ub_regret"],
                        "lb_regret": result["lb_regret"],
                        "true_regret_eval_pool": truth["regret"],
                        "true_normalized_regret_eval_pool": truth["normalized_regret"],
                        "true_retention_eval_pool": truth["retention"],
                        "true_state": true_state,
                        "false_reuse": result["state"] == "REUSE" and true_state == "REVISE",
                        "correct_reuse": result["state"] == "REUSE" and true_state == "REUSE",
                        "correct_revise": result["state"] == "REVISE" and true_state == "REVISE",
                        "seed_pool_protocol": (
                            f"audit={AUDIT_SEEDS}; reference={REFERENCE_SEEDS} offline-only; evaluation={EVALUATION_SEEDS}"
                        ),
                        "reference_pool_used_for_policy": False,
                        "evaluation_pool_used_for_policy": False,
                        "target_outcomes_used_for_policy": True,
                        "future_depths_used_for_policy": False,
                        "actual_cell_cost_check": True,
                        "nested_prefix_contract": "one max-depth draw per seed/replicate/context/perturbation; exact prefixes",
                        "status": "executed_nested_prefix_audit_replay",
                    }
                )

    table = pd.DataFrame(rows)
    if table.empty or table["transfer_id"].nunique() != 6:
        raise ValueError("audit did not cover all six directed transfers")
    expected_rows = 6 * len(METRICS) * len(DECISION_BUDGETS) * len(METHODS)
    if len(table) != expected_rows:
        raise ValueError(f"audit produced {len(table)} rows; expected {expected_rows}")
    table = table.sort_values(
        ["source_environment_id", "target_environment_id", "metric", "decision_budget_fraction", "audit_method"]
    ).reset_index(drop=True)

    output_path = manifests / OUTPUT_NAME
    report_path = manifests / REPORT_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".tmp")
    table.to_csv(temporary_output, index=False)
    temporary_output.replace(output_path)
    report = {
        "schema_version": 4,
        "status": "executed_nested_prefix_audit_replay",
        "unit": "six directed transfers × three metrics × four decision budgets × five target-audit policies",
        "epsilon": EPSILON,
        "depth_schedule": list(DEPTHS),
        "nested_prefix_contract": "one 160-cell draw per seed/replicate/context/perturbation; exact prefixes at every scheduled depth",
        "budget_definition": "actual additional target cells across two target replicates, including the shared control-prefix extension exactly once per replicate",
        "information_policy": "source data plus each candidate's current audit observation only; reference, evaluation, and future-depth observations are unavailable to policy",
        "seed_pool_protocol": {
            "audit": list(AUDIT_SEEDS),
            "reference_offline_only": list(REFERENCE_SEEDS),
            "evaluation_realized_regret_only": list(EVALUATION_SEEDS),
        },
        "methods": list(METHODS),
        "rows": len(table),
        "checks": {
            "all_candidates_receive_pilot": bool((table["pilot_target_cells_per_label"] == PILOT_DEPTH).all()),
            "actual_cell_cost_matches_allocation": bool(table["actual_cell_cost_check"].all()),
            "reference_pool_used_for_policy": bool(table["reference_pool_used_for_policy"].any()),
            "evaluation_pool_used_for_policy": bool(table["evaluation_pool_used_for_policy"].any()),
            "future_depths_used_for_policy": bool(table["future_depths_used_for_policy"].any()),
            "nested_prefix_contract_rows": int(table["nested_prefix_contract"].nunique()),
        },
        "false_reuse_rate_by_method": table.groupby("audit_method")["false_reuse"].mean().to_dict(),
        "outputs": {
            "table": output_path.relative_to(root).as_posix(),
            "nested_risk_cache": cache_path.relative_to(root).as_posix(),
        },
        "nested_risk_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
    }
    temporary_report = report_path.with_name(report_path.name + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_report.replace(report_path)
    return table, report


def build(root: Path = ROOT, *, force_cache: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compatibility entry point used by the figure orchestrator."""

    return replay_target_audit(root, force_cache=force_cache)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    table, report = replay_target_audit(args.root, force_cache=args.force_cache)
    print(json.dumps(report, indent=2))
    print(table.groupby("audit_method")[["false_reuse", "additional_target_cells"]].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
