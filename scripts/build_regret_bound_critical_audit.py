"""Build the regret-bound-critical active target-audit comparison.

The source shortlist, measurement protocol, nested-prefix risk cache, and seed
pools are inherited from ``run_target_audit_simulation``.  This module adds
one allocation policy only.  At every allocation step it recomputes the
current empirical upper regret bound and its maximizing exchange count
``m*``.  Shortlist items with the largest upper bounds and outside items with
the smallest lower bounds form the critical set; interval width and a
schedule-only width-weighted UB-reduction proxy are used only below that
critical ordering.

The output is deliberately a comparison artifact rather than a claim of
improvement.  The source shortlist is frozen, so realized evaluation-pool
regret is a post-allocation quantity and is expected to be common to all
allocation policies for a case.  A kill-switch therefore demotes the result
when the active policy does not improve the prespecified decision outcomes.
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

from scripts.run_target_audit_simulation import (  # noqa: E402
    AUDIT_SEEDS,
    DECISION_BUDGETS,
    DEPTHS,
    EPSILON,
    EVALUATION_SEEDS,
    MAX_DEPTH,
    METRICS,
    N_REPLICATES,
    PILOT_DEPTH,
    REFERENCE_SEEDS,
    SPLIT_SEEDS,
    _canonical_matrix,
    _matrix,
    _policy_observation,
    _priority,
    _stable_seed,
    _validate_nested_cache,
)
from src.evaluation.shortlist_audit import (  # noqa: E402
    allocate_nested_prefixes,
    audit_shortlist,
    empirical_target_interval,
)


PROPOSED_METHOD = "regret_bound_critical"
COMPARATOR_METHODS = ("random_extra_depth", "uniform_depth", "source_uncertainty", "ptl_boundary")
METHODS = (PROPOSED_METHOD, *COMPARATOR_METHODS)
BASELINE_AUDIT_NAME = "target_audit_simulation.csv"
NESTED_CACHE_NAME = "target_audit_nested_prefix_risks.csv"
CANONICAL_RISK_NAME = "reliability_transport_measurement_depth_matched_fixed_risks.csv"
OUTPUT_NAME = "regret_bound_critical_audit.csv"
REPORT_NAME = "regret_bound_critical_audit.json"

NESTED_COLUMNS = (
    "source_environment_id",
    "target_environment_id",
    "metric",
    "split_seed",
    "measurement_replicate",
    "depth",
    "perturbation_label",
    "target_risk",
)
CANONICAL_COLUMNS = (
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
)
BASELINE_COLUMNS = (
    "source_environment_id",
    "target_environment_id",
    "metric",
    "decision_budget_fraction",
    "audit_method",
    "target_labels_piloted",
    "additional_target_cells",
    "additional_target_cell_budget",
    "audit_state",
    "ub_regret",
    "lb_regret",
    "true_regret_eval_pool",
    "true_normalized_regret_eval_pool",
    "true_retention_eval_pool",
    "true_state",
    "false_reuse",
    "correct_reuse",
    "correct_revise",
    "depth_profile",
    "nested_prefix_contract",
    "reference_pool_used_for_policy",
    "evaluation_pool_used_for_policy",
    "future_depths_used_for_policy",
    "actual_cell_cost_check",
)


def _as_float(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"expected a finite numeric value, got {value!r}")
    return result


def _normalise_nested(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(NESTED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"nested audit cache is missing columns: {missing}")
    frame = frame.loc[:, list(NESTED_COLUMNS)].copy()
    for column in ("source_environment_id", "target_environment_id", "metric", "perturbation_label"):
        frame[column] = frame[column].astype(str)
    for column in ("split_seed", "measurement_replicate", "depth"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame["target_risk"] = pd.to_numeric(frame["target_risk"], errors="raise").astype(float)
    _validate_nested_cache(frame, int(frame["perturbation_label"].nunique()))
    return frame


def _normalise_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(CANONICAL_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"canonical risk surface is missing columns: {missing}")
    frame = frame.loc[:, list(CANONICAL_COLUMNS)].copy()
    for column in (
        "source_environment_id",
        "left_target_environment_id",
        "right_target_environment_id",
        "metric",
        "perturbation_label",
    ):
        frame[column] = frame[column].astype(str)
    frame["split_seed"] = pd.to_numeric(frame["split_seed"], errors="raise").astype(int)
    frame["measurement_replicate"] = pd.to_numeric(frame["measurement_replicate"], errors="raise").astype(int)
    frame["cell_budget_label"] = frame["cell_budget_label"].astype(str).str.strip(" '\"")
    for column in ("left_risk", "right_risk"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if not np.isfinite(frame[["left_risk", "right_risk"]].to_numpy(float)).all():
        raise ValueError("canonical full-depth risk surface contains non-finite risks")
    frame = frame.loc[frame["cell_budget_label"].eq("full")].copy()
    if frame.empty:
        raise ValueError("canonical full-depth risk surface is empty")
    return frame


def _load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifests = root / "artifacts/manifests"
    nested_path = manifests / NESTED_CACHE_NAME
    canonical_path = manifests / CANONICAL_RISK_NAME
    baseline_path = manifests / BASELINE_AUDIT_NAME
    missing = [path.relative_to(root).as_posix() for path in (nested_path, canonical_path, baseline_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"regret-bound-critical audit requires existing inputs: {missing}")

    nested = _normalise_nested(pd.read_csv(nested_path, usecols=list(NESTED_COLUMNS), low_memory=False))
    canonical = _normalise_canonical(pd.read_csv(canonical_path, usecols=list(CANONICAL_COLUMNS), low_memory=False))

    baseline_missing = sorted(set(BASELINE_COLUMNS).difference(pd.read_csv(baseline_path, nrows=0).columns))
    if baseline_missing:
        raise ValueError(f"existing target audit output is missing baseline columns: {baseline_missing}")
    baseline = pd.read_csv(baseline_path, usecols=list(BASELINE_COLUMNS), low_memory=False)
    for column in ("source_environment_id", "target_environment_id", "metric", "audit_method", "true_state", "nested_prefix_contract"):
        baseline[column] = baseline[column].astype(str)
    baseline["decision_budget_fraction"] = pd.to_numeric(baseline["decision_budget_fraction"], errors="raise").astype(float)
    numeric = [
        "target_labels_piloted",
        "additional_target_cells",
        "additional_target_cell_budget",
        "ub_regret",
        "lb_regret",
        "true_regret_eval_pool",
        "true_normalized_regret_eval_pool",
        "true_retention_eval_pool",
    ]
    for column in numeric:
        baseline[column] = pd.to_numeric(baseline[column], errors="raise")
    expected_methods = set(COMPARATOR_METHODS)
    observed_methods = set(baseline["audit_method"].unique())
    if not expected_methods.issubset(observed_methods):
        raise ValueError(f"existing target audit output lacks one or more comparators: {sorted(expected_methods - observed_methods)}")
    return nested, canonical, baseline


def _critical_details(
    lower: np.ndarray,
    upper: np.ndarray,
    selected: np.ndarray,
    allocated: np.ndarray,
    *,
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    """Return the active UB critical set and tie-break quantities.

    ``m_star`` is taken from the current, clipped upper regret bound.  When
    the current upper bound is exactly zero there is no positive exchange
    threat to prioritize, so the critical set is empty and interval width is
    the first available signal.
    """

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    allocated = np.asarray(allocated, dtype=int)
    selected = np.asarray(selected, dtype=int)
    audit = audit_shortlist(lower, upper, selected, epsilon=epsilon)
    m_star = int(audit["ub_m"]) if float(audit["ub_regret"]) > 0.0 else 0
    outside = np.asarray(audit["outside_indices"], dtype=int)

    critical_selected = selected[np.argsort(-upper[selected], kind="stable")[:m_star]] if m_star else np.asarray([], dtype=int)
    critical_outside = outside[np.argsort(lower[outside], kind="stable")[:m_star]] if m_star else np.asarray([], dtype=int)
    critical_mask = np.zeros(lower.size, dtype=bool)
    critical_mask[critical_selected] = True
    critical_mask[critical_outside] = True

    critical_rank = np.zeros(lower.size, dtype=float)
    if m_star:
        critical_rank[critical_selected] = np.arange(m_star, 0, -1, dtype=float)
        critical_rank[critical_outside] = np.arange(m_star, 0, -1, dtype=float)

    width = np.maximum(upper - lower, 0.0)
    next_depth_contraction = np.zeros(lower.size, dtype=float)
    for index, depth in enumerate(allocated):
        position = int(np.searchsorted(DEPTHS, int(depth)))
        if position + 1 < len(DEPTHS):
            # This is a schedule-only proxy.  It uses no value at the next
            # depth and therefore cannot leak a future target observation.
            next_depth_contraction[index] = max(
                0.0,
                1.0 - np.sqrt(float(depth) / float(DEPTHS[position + 1])),
            )

    exposure = np.zeros(lower.size, dtype=float)
    if selected.size and outside.size:
        outside_floor = float(np.min(lower[outside]))
        selected_ceiling = float(np.max(upper[selected]))
        exposure[selected] = np.maximum(upper[selected] - outside_floor, 0.0)
        exposure[outside] = np.maximum(selected_ceiling - lower[outside], 0.0)

    # The proxy is deliberately secondary to the critical set and its bound
    # ordering.  It estimates the amount of active exchange exposure that a
    # width reduction could remove at the next scheduled prefix; it is not a
    # realized improvement and is never reported as one.
    expected_ub_reduction_proxy = exposure * width * next_depth_contraction / max(1, selected.size)
    score = critical_mask.astype(float) * 1_000_000_000.0
    score += critical_rank * 1_000_000.0
    score += width * 1_000.0
    score += expected_ub_reduction_proxy * 100.0
    score += exposure
    if not np.isfinite(score).all():
        raise ValueError("regret-bound-critical priority produced non-finite scores")

    return {
        "audit": audit,
        "m_star": m_star,
        "critical_selected": critical_selected,
        "critical_outside": critical_outside,
        "critical_mask": critical_mask,
        "width": width,
        "exposure": exposure,
        "next_depth_contraction": next_depth_contraction,
        "expected_ub_reduction_proxy": expected_ub_reduction_proxy,
        "score": score,
    }


def regret_bound_critical_priority(
    observation: dict[str, np.ndarray],
    selected: np.ndarray,
    *,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Return the leakage-safe priority for the proposed active policy."""

    allocated = np.asarray(observation["allocated_depths"], dtype=int)
    current = np.asarray(observation["target_risk_replicates"], dtype=float)
    if allocated.ndim != 1 or current.ndim != 2 or current.shape != (current.shape[0], allocated.size):
        raise ValueError("policy observation has incompatible allocation/data shapes")
    lower, upper = empirical_target_interval(current)
    return _critical_details(lower, upper, np.asarray(selected, dtype=int), allocated, epsilon=epsilon)["score"]


def _depth_profile(allocated: np.ndarray) -> str:
    return json.dumps({str(depth): int(np.sum(allocated == depth)) for depth in DEPTHS}, sort_keys=True)


def _random_rank(n_items: int, source: str, target: str, metric: str, fraction: float, method: str) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed("target_audit_policy_v2", source, target, metric, fraction, method))
    random_rank = np.empty(n_items, dtype=float)
    random_rank[rng.permutation(n_items)] = np.arange(n_items, 0, -1, dtype=float)
    return random_rank


def _run_allocation(
    method: str,
    source_matrix: np.ndarray,
    matrices: dict[int, np.ndarray],
    selected: np.ndarray,
    additional_budget: int,
    source: str,
    target: str,
    metric: str,
    fraction: float,
) -> dict[str, Any]:
    n_items = int(source_matrix.shape[1])
    random_rank = _random_rank(n_items, source, target, metric, fraction, method)

    if method == PROPOSED_METHOD:
        def priority(observation: dict[str, np.ndarray]) -> np.ndarray:
            return regret_bound_critical_priority(observation, selected)
    else:
        def priority(observation: dict[str, np.ndarray]) -> np.ndarray:
            return _priority(method, source_matrix, observation, selected, random_rank)

    allocated, control_depth, spent = allocate_nested_prefixes(
        DEPTHS,
        n_items,
        int(additional_budget),
        priority,
        n_replicates=N_REPLICATES,
        observe=lambda current, cache=matrices: _policy_observation(cache, current),
    )
    expected_spent = N_REPLICATES * (
        int(np.sum(allocated - PILOT_DEPTH)) + int(np.max(allocated)) - PILOT_DEPTH
    )
    if int(spent) != expected_spent:
        raise AssertionError("nested audit cost does not equal target-plus-shared-control cost")
    if int(spent) > int(additional_budget):
        raise AssertionError("allocation exceeded its actual additional-cell cap")
    if int(control_depth) != int(np.max(allocated)):
        raise AssertionError("shared control depth is inconsistent with allocation")

    observation = _policy_observation(matrices, allocated)
    lower, upper = empirical_target_interval(observation["target_risk_replicates"])
    details = _critical_details(lower, upper, selected, allocated)
    return {
        "method": method,
        "allocated": allocated,
        "control_depth": int(control_depth),
        "spent": int(spent),
        "lower": lower,
        "upper": upper,
        "details": details,
    }


def _matched_allocations(
    source_matrix: np.ndarray,
    matrices: dict[int, np.ndarray],
    selected: np.ndarray,
    nominal_budget: int,
    source: str,
    target: str,
    metric: str,
    fraction: float,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    """Run all methods at one realized additional-cell cost.

    The declared budget is first applied to every method.  If an indivisible
    nested-prefix step leaves one method below the others, the common cap is
    lowered to the smallest realized spend and the complete comparison is
    replayed.  The returned rows are only accepted when every method has
    exactly the same realized additional-cell cost.
    """

    cap = int(nominal_budget)
    iterations = 0
    while iterations < 8:
        iterations += 1
        results = {
            method: _run_allocation(
                method,
                source_matrix,
                matrices,
                selected,
                cap,
                source,
                target,
                metric,
                fraction,
            )
            for method in METHODS
        }
        costs = [int(result["spent"]) for result in results.values()]
        if len(set(costs)) == 1:
            matched_cost = costs[0]
            if matched_cost > nominal_budget:
                raise AssertionError("matched actual cost exceeded the declared audit budget")
            return results, matched_cost, iterations
        next_cap = min(costs)
        if next_cap >= cap:
            break
        cap = int(next_cap)
    raise RuntimeError(
        f"could not obtain an exact matched actual-cell cost for {source}->{target}/{metric}/{fraction}; "
        f"last costs={costs!r}"
    )


def _baseline_row(baseline: pd.DataFrame, source: str, target: str, metric: str, fraction: float, method: str) -> pd.Series:
    rows = baseline.loc[
        baseline["source_environment_id"].eq(source)
        & baseline["target_environment_id"].eq(target)
        & baseline["metric"].eq(metric)
        & baseline["audit_method"].eq(method)
        & np.isclose(baseline["decision_budget_fraction"].to_numpy(float), float(fraction))
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one existing baseline row for {source}->{target}/{metric}/{fraction}/{method}, found {len(rows)}")
    return rows.iloc[0]


def _assert_baseline_contract(row: pd.Series, method: str, n_items: int, nominal_budget: int) -> None:
    """Validate the existing baseline metadata without treating it as replay oracle.

    The existing table is retained as the audit-data/contract reference.  The
    comparator allocation is intentionally rerun below at the matched actual
    cost because a changed deterministic tie path can alter an indivisible
    final prefix while preserving the declared protocol.
    """

    if int(row["target_labels_piloted"]) != int(n_items):
        raise ValueError(f"existing {method} baseline has a different candidate universe")
    if int(row["additional_target_cell_budget"]) != int(nominal_budget):
        raise ValueError(f"existing {method} baseline has a different nominal budget")
    if not bool(row["actual_cell_cost_check"]):
        raise ValueError(f"existing {method} baseline did not pass its actual-cell cost check")
    if str(row["nested_prefix_contract"]) != "one max-depth draw per seed/replicate/context/perturbation; exact prefixes":
        raise ValueError(f"existing {method} baseline uses a different nested-prefix contract")
    if bool(row["reference_pool_used_for_policy"]) or bool(row["evaluation_pool_used_for_policy"]) or bool(row["future_depths_used_for_policy"]):
        raise ValueError(f"existing {method} baseline violates the target-audit information firewall")


def _initial_details(matrices: dict[int, np.ndarray], selected: np.ndarray) -> dict[str, Any]:
    n_items = int(next(iter(matrices.values())).shape[1])
    allocated = np.full(n_items, PILOT_DEPTH, dtype=int)
    observation = _policy_observation(matrices, allocated)
    lower, upper = empirical_target_interval(observation["target_risk_replicates"])
    return _critical_details(lower, upper, selected, allocated)


def _row(
    *,
    source: str,
    target: str,
    metric: str,
    fraction: float,
    n_items: int,
    selected: np.ndarray,
    nominal_budget: int,
    matched_cost: int,
    result: dict[str, Any],
    initial: dict[str, Any],
    truth: pd.Series,
    existing_baseline_match: bool | None,
    status: str,
) -> dict[str, Any]:
    details = result["details"]
    audit = details["audit"]
    true_regret = _as_float(truth["true_regret_eval_pool"])
    true_state = str(truth["true_state"])
    expected_true_state = "REUSE" if true_regret <= EPSILON else "REVISE"
    if true_state != expected_true_state:
        raise ValueError("existing evaluation-only truth state is inconsistent with epsilon")
    state = str(audit["state"])
    return {
        "source_environment_id": source,
        "target_environment_id": target,
        "transfer_id": f"{source}->{target}",
        "metric": metric,
        "decision_budget_fraction": float(fraction),
        "audit_extra_budget_fraction": float(fraction),
        "k": int(selected.size),
        "audit_method": result["method"],
        "proposed_method": PROPOSED_METHOD,
        "is_proposed_method": result["method"] == PROPOSED_METHOD,
        "pilot_target_cells_per_label": PILOT_DEPTH,
        "target_labels_piloted": n_items,
        "target_labels_upgraded": int(np.sum(result["allocated"] > PILOT_DEPTH)),
        "target_labels_at_full_depth": int(np.sum(result["allocated"] == MAX_DEPTH)),
        "control_prefix_depth": int(result["control_depth"]),
        "target_cells_spent": N_REPLICATES * PILOT_DEPTH * (n_items + 1) + int(result["spent"]),
        "additional_target_cells": int(result["spent"]),
        "additional_target_cell_budget": int(nominal_budget),
        "matched_additional_cell_budget": int(matched_cost),
        "matched_actual_additional_cells": int(result["spent"]),
        "actual_cell_cost_match": int(result["spent"]) == int(matched_cost),
        "depth_profile": _depth_profile(result["allocated"]),
        "initial_ub_regret": _as_float(initial["audit"]["ub_regret"]),
        "initial_ub_m_star": int(initial["m_star"]),
        "ub_regret": _as_float(audit["ub_regret"]),
        "ub_m_star": int(details["m_star"]),
        "lb_regret": _as_float(audit["lb_regret"]),
        "critical_shortlist_count": int(details["critical_selected"].size),
        "critical_outside_count": int(details["critical_outside"].size),
        "critical_items_count": int(np.sum(details["critical_mask"])),
        "mean_interval_width": _as_float(np.mean(details["width"])),
        "expected_ub_reduction_proxy": _as_float(np.sum(details["expected_ub_reduction_proxy"])),
        "audit_state": state,
        "true_regret_eval_pool": true_regret,
        "true_normalized_regret_eval_pool": _as_float(truth["true_normalized_regret_eval_pool"]),
        "true_retention_eval_pool": _as_float(truth["true_retention_eval_pool"]),
        "true_state": true_state,
        "false_reuse": state == "REUSE" and true_state == "REVISE",
        "correct_reuse": state == "REUSE" and true_state == "REUSE",
        "correct_revise": state == "REVISE" and true_state == "REVISE",
        "existing_baseline_contract_match": existing_baseline_match,
        "seed_pool_protocol": (
            f"audit={AUDIT_SEEDS}; reference={REFERENCE_SEEDS} offline-only; evaluation={EVALUATION_SEEDS}"
        ),
        "reference_pool_used_for_policy": False,
        "evaluation_pool_used_for_policy": False,
        "target_outcomes_used_for_policy": True,
        "source_shortlist_target_outcomes_used": False,
        "future_depths_used_for_policy": False,
        "actual_cell_cost_check": True,
        "nested_prefix_contract": "one max-depth draw per seed/replicate/context/perturbation; exact prefixes",
        "allocation_information_firewall": "current target prefixes only; evaluation and future depths unavailable",
        "audit_replay_status": "executed_regret_bound_critical_matched_cost_replay",
        "status": status,
        "result_status": status,
    }


def _truth_consistency(truth_rows: list[pd.Series]) -> pd.Series:
    if not truth_rows:
        raise ValueError("no evaluation-only truth row available")
    first = truth_rows[0]
    for row in truth_rows[1:]:
        for field in ("true_regret_eval_pool", "true_normalized_regret_eval_pool", "true_retention_eval_pool", "true_state"):
            if field == "true_state":
                equal = str(row[field]) == str(first[field])
            else:
                equal = np.isclose(float(row[field]), float(first[field]), rtol=0.0, atol=1e-12)
            if not equal:
                raise ValueError("existing baseline rows disagree on the fixed evaluation-only truth")
    return first


def _summary(table: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = table.loc[table["audit_method"].isin(COMPARATOR_METHODS)].copy()
    active = table.loc[table["audit_method"].eq(PROPOSED_METHOD)].copy()
    if len(active) != 6 * len(METRICS) * len(DECISION_BUDGETS):
        raise ValueError("active result does not cover the complete directed transfer/metric/budget surface")
    if len(baseline) != len(active) * len(COMPARATOR_METHODS):
        raise ValueError("comparator result does not cover the complete matched-cost surface")

    grouped = table.groupby("audit_method", sort=False)
    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        frame = grouped.get_group(method)
        summary_rows.append(
            {
                "audit_method": method,
                "rows": int(len(frame)),
                "mean_actual_additional_cells": _as_float(frame["matched_actual_additional_cells"].mean()),
                "mean_ub_regret": _as_float(frame["ub_regret"].mean()),
                "mean_lb_regret": _as_float(frame["lb_regret"].mean()),
                "false_reuse_rate": _as_float(frame["false_reuse"].mean()),
                "correct_reuse_rate": _as_float(frame["correct_reuse"].mean()),
                "correct_revise_rate": _as_float(frame["correct_revise"].mean()),
                "mean_true_regret_eval_pool": _as_float(frame["true_regret_eval_pool"].mean()),
                "mean_true_normalized_regret_eval_pool": _as_float(frame["true_normalized_regret_eval_pool"].mean()),
            }
        )
    by_method = {row["audit_method"]: row for row in summary_rows}

    def extrema(field: str, *, lower_is_better: bool) -> dict[str, Any]:
        values = {method: float(by_method[method][field]) for method in COMPARATOR_METHODS}
        best_value = min(values.values()) if lower_is_better else max(values.values())
        best_methods = [method for method in COMPARATOR_METHODS if np.isclose(values[method], best_value, rtol=0.0, atol=1e-12)]
        return {"value": best_value, "methods": best_methods, "all_values": values}

    active_summary = by_method[PROPOSED_METHOD]
    strongest = {
        "false_reuse": extrema("false_reuse_rate", lower_is_better=True),
        "correct_revise": extrema("correct_revise_rate", lower_is_better=False),
        "realized_regret": extrema("mean_true_regret_eval_pool", lower_is_better=True),
    }
    tolerance = 1e-12
    false_reuse_worse = active_summary["false_reuse_rate"] > strongest["false_reuse"]["value"] + tolerance
    correct_revise_beats = active_summary["correct_revise_rate"] > strongest["correct_revise"]["value"] + tolerance
    realized_regret_values = np.asarray(
        [by_method[method]["mean_true_regret_eval_pool"] for method in COMPARATOR_METHODS],
        dtype=float,
    )
    realized_regret_policy_invariant = bool(
        np.all(np.isclose(realized_regret_values, realized_regret_values[0], rtol=0.0, atol=tolerance))
    )
    # The evaluation pool is frozen in this replay, so realized regret is a
    # policy-invariant reference quantity.  It cannot be a promotion
    # criterion, even though it is retained in the summary for transparency.
    kill_switch = {
        "false_reuse_worse_than_strongest_simple_baseline": bool(false_reuse_worse),
        "correct_revise_beats_strongest_simple_baseline": bool(correct_revise_beats),
        "realized_regret_beats_strongest_simple_baseline": None,
        "realized_regret_policy_invariant_reference": realized_regret_policy_invariant,
        "triggered": bool(false_reuse_worse or not correct_revise_beats),
        "rule": "demote if false reuse is worse, or if correct revise does not strictly beat the strongest simple baseline; realized evaluation-pool regret is a policy-invariant reference under the frozen-shortlist replay and is not a promotion criterion",
        "tolerance": tolerance,
    }
    return {"by_method": summary_rows, "strongest_simple_baseline": strongest, "kill_switch": kill_switch}, by_method


def run(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    nested, canonical, baseline = _load_inputs(root)
    rows: list[dict[str, Any]] = []
    case_count = 0

    group_columns = ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"]
    for key, group in canonical.groupby(group_columns, sort=True):
        source, left, right, metric = map(str, key)
        if metric not in METRICS or source not in (left, right):
            continue
        target = right if source == left else left
        source_labels, source_matrix = _canonical_matrix(group, source, SPLIT_SEEDS)
        target_rows = nested.loc[
            nested["source_environment_id"].eq(source)
            & nested["target_environment_id"].eq(target)
            & nested["metric"].eq(metric)
        ].copy()
        if target_rows.empty:
            raise ValueError(f"missing nested audit rows for {source}->{target}/{metric}")
        raw = {depth: _matrix(target_rows.loc[target_rows["depth"].eq(depth)], AUDIT_SEEDS, "target_risk") for depth in DEPTHS}
        common = sorted(set(source_labels).intersection(*(set(labels) for labels, _ in raw.values())))
        if not common:
            raise ValueError(f"no fixed source/audit candidate universe for {source}->{target}/{metric}")
        source_matrix = source_matrix[:, [source_labels.index(label) for label in common]]
        matrices = {depth: matrix[:, [labels.index(label) for label in common]] for depth, (labels, matrix) in raw.items()}
        source_mean = source_matrix.mean(axis=0)
        n_items = len(common)
        case_count += 1

        for fraction in DECISION_BUDGETS:
            selected = np.argsort(source_mean, kind="stable")[: max(1, int(np.floor(n_items * fraction)))]
            nominal_budget = int(np.floor(fraction * N_REPLICATES * (MAX_DEPTH - PILOT_DEPTH) * (n_items + 1)))
            initial = _initial_details(matrices, selected)
            results, matched_cost, _ = _matched_allocations(
                source_matrix,
                matrices,
                selected,
                nominal_budget,
                source,
                target,
                metric,
                float(fraction),
            )

            references = [_baseline_row(baseline, source, target, metric, float(fraction), method) for method in COMPARATOR_METHODS]
            truth = _truth_consistency(references)
            for method in METHODS:
                existing_match: bool | None = None
                if method in COMPARATOR_METHODS:
                    reference = references[COMPARATOR_METHODS.index(method)]
                    _assert_baseline_contract(reference, method, n_items, nominal_budget)
                    existing_match = True
                rows.append(
                    _row(
                        source=source,
                        target=target,
                        metric=metric,
                        fraction=float(fraction),
                        n_items=n_items,
                        selected=selected,
                        nominal_budget=nominal_budget,
                        matched_cost=matched_cost,
                        result=results[method],
                        initial=initial,
                        truth=truth,
                        existing_baseline_match=existing_match,
                        status="pending_kill_switch",
                    )
                )

    table = pd.DataFrame(rows)
    expected_cases = 6 * len(METRICS)
    expected_rows = expected_cases * len(DECISION_BUDGETS) * len(METHODS)
    if case_count != expected_cases or len(table) != expected_rows:
        raise ValueError(f"active audit surface has {case_count} cases and {len(table)} rows; expected {expected_cases} and {expected_rows}")
    case_keys = ["source_environment_id", "target_environment_id", "metric", "decision_budget_fraction"]
    if table.groupby(case_keys, sort=False)["matched_actual_additional_cells"].nunique().max() != 1:
        raise AssertionError("matched actual additional-cell cost differs across methods within a case")
    if not bool(table["actual_cell_cost_match"].all()):
        raise AssertionError("one or more methods failed the exact matched actual-cell cost check")
    if not bool(table["pilot_target_cells_per_label"].eq(PILOT_DEPTH).all()):
        raise AssertionError("not every candidate received the declared pilot")
    if bool(table[["reference_pool_used_for_policy", "evaluation_pool_used_for_policy", "future_depths_used_for_policy"]].any().any()):
        raise AssertionError("target-audit information firewall was violated")

    summary, by_method = _summary(table)
    kill_switch = summary["kill_switch"]
    status = "demote_to_supplement_negative_result" if kill_switch["triggered"] else "promote_active_audit_result"
    table["status"] = status
    table["result_status"] = status
    table = table.sort_values(
        ["source_environment_id", "target_environment_id", "metric", "decision_budget_fraction", "audit_method"],
        key=lambda series: series.map({method: index for index, method in enumerate(METHODS)}) if series.name == "audit_method" else series,
        kind="stable",
    ).reset_index(drop=True)

    manifests = root / "artifacts/manifests"
    csv_path = manifests / OUTPUT_NAME
    json_path = manifests / REPORT_NAME
    table.to_csv(csv_path, index=False)
    report = {
        "schema_version": 1,
        "status": status,
        "analysis": "one active regret-bound-critical target-audit policy compared with four existing simple policies",
        "surface": "six directed transfers x three metrics x four decision budgets x five methods",
        "rows": int(len(table)),
        "case_count": int(case_count),
        "methods": list(METHODS),
        "inputs": {
            "nested_risk_cache": (manifests / NESTED_CACHE_NAME).relative_to(root).as_posix(),
            "canonical_full_depth_risk_surface": (manifests / CANONICAL_RISK_NAME).relative_to(root).as_posix(),
            "existing_baseline_audit": (manifests / BASELINE_AUDIT_NAME).relative_to(root).as_posix(),
        },
        "protocol_contract": {
            "depth_schedule": list(DEPTHS),
            "pilot_depth": PILOT_DEPTH,
            "maximum_depth": MAX_DEPTH,
            "replicates": N_REPLICATES,
            "epsilon": EPSILON,
            "budget_fractions": list(DECISION_BUDGETS),
            "budget_definition": "additional target cells across two target replicates, including the shared control-prefix extension exactly once per replicate",
            "nested_prefix_contract": "one 160-cell draw per seed/replicate/context/perturbation; exact prefixes at every scheduled depth",
            "seed_pool_protocol": {
                "audit": list(AUDIT_SEEDS),
                "reference_offline_only": list(REFERENCE_SEEDS),
                "evaluation_realized_regret_only": list(EVALUATION_SEEDS),
            },
            "source_shortlist": "fixed from the source-side canonical full-depth risk mean at each decision budget; target outcomes do not select the shortlist",
        },
        "allocation_policy": {
            "name": PROPOSED_METHOD,
            "active_signal": "current empirical audit_shortlist UB_regret and its maximizing exchange count m*",
            "critical_shortlist_rule": "the m* selected items with highest current upper bounds",
            "critical_outside_rule": "the m* unselected items with lowest current lower bounds",
            "secondary_rule": "current interval width, then schedule-only width-weighted expected UB-reduction proxy, then current exchange exposure",
            "expected_ub_reduction_proxy": "current exchange exposure x current interval width x schedule-only root-n contraction factor / shortlist size; no future-depth values are used",
            "information_firewall": "priority receives only allocated depths and currently observed target-risk replicates",
            "future_depths_used_for_policy": False,
        },
        "matched_cost_contract": {
            "method": "apply the declared nominal budget to every method; lower the common cap only when an indivisible nested-prefix step leaves a method below the others",
            "comparison_quantity": "matched_actual_additional_cells",
            "all_rows_exact_match": bool(table["actual_cell_cost_match"].all()),
            "nominal_budget_unchanged": True,
            "protocol_or_model_changed": False,
        },
        "summary": summary,
        "checks": {
            "all_candidates_receive_pilot": bool(table["pilot_target_cells_per_label"].eq(PILOT_DEPTH).all()),
            "matched_actual_cell_cost_all_methods": bool(table.groupby(case_keys, sort=False)["matched_actual_additional_cells"].nunique().max() == 1),
            "existing_comparator_contracts_match": bool(table.loc[table["audit_method"].isin(COMPARATOR_METHODS), "existing_baseline_contract_match"].all()),
            "reference_pool_used_for_policy": bool(table["reference_pool_used_for_policy"].any()),
            "evaluation_pool_used_for_policy": bool(table["evaluation_pool_used_for_policy"].any()),
            "future_depths_used_for_policy": bool(table["future_depths_used_for_policy"].any()),
            "realized_regret_invariant_across_methods": bool(table.groupby(case_keys, sort=False)["true_regret_eval_pool"].nunique().max() == 1),
            "no_new_data_or_model": True,
        },
        "kill_switch": kill_switch,
        "interpretation": (
            "The active policy is a supplementary negative result under the kill-switch because the frozen source shortlist makes realized evaluation-pool regret common across allocation methods; no main-text promotion is justified."
            if kill_switch["triggered"]
            else "The active policy passed the prespecified comparative kill-switch."
        ),
        "outputs": {
            "csv": csv_path.relative_to(root).as_posix(),
            "json": json_path.relative_to(root).as_posix(),
        },
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def build(root: Path = ROOT) -> dict[str, Any]:
    """Compatibility entry point for focused audit orchestration."""

    return run(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
