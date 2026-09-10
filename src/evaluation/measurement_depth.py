"""Measurement-depth utilities for reliability transportability.

The depth estimand is deliberately separated from the biological outcome: a
cell budget changes only the raw-cell pseudoreplicate used to measure the same
source-frozen prediction.  The helpers here summarize seed-level rows, retain
matched universes, and make the resolution/detectability rule explicit.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CELL_BUDGETS = (10, 20, 40, 80, 160)
FULL_LABEL = "full"


def validate_cell_budgets(budgets: Iterable[int] = DEFAULT_CELL_BUDGETS) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in budgets}))
    if not values or any(value < 2 for value in values):
        raise ValueError("cell budgets must contain integers >= 2")
    return values


def matched_label_universe(
    counts: pd.DataFrame,
    *,
    left_environment: str,
    right_environment: str,
    budget: int,
    label_column: str = "perturbation_label",
    environment_column: str = "environment_id",
    count_column: str = "n_cells_qc",
) -> list[str]:
    """Return labels meeting the budget in both target contexts and control.

    ``counts`` must contain one row per environment/label.  Controls are used
    for eligibility but never enter the returned perturbation universe.
    """

    required = {label_column, environment_column, count_column}
    missing = required.difference(counts.columns)
    if missing:
        raise ValueError(f"counts is missing columns: {sorted(missing)}")
    pivot = counts.pivot_table(index=label_column, columns=environment_column, values=count_column, aggfunc="min")
    for environment in (left_environment, right_environment):
        if environment not in pivot.columns:
            return []
    if "control" not in pivot.index:
        raise ValueError("counts must include the control row")
    eligible = (pivot.loc[:, [left_environment, right_environment]] >= int(budget)).all(axis=1)
    control_ok = bool((pivot.loc["control", [left_environment, right_environment]] >= int(budget)).all())
    if not control_ok:
        return []
    return sorted(str(label) for label in pivot.index[eligible] if str(label) != "control")


def coverage_curve(
    counts: pd.DataFrame,
    *,
    left_environment: str,
    right_environment: str,
    budgets: Iterable[int] = DEFAULT_CELL_BUDGETS,
    label_column: str = "perturbation_label",
) -> pd.DataFrame:
    """Describe coverage and matched-universe fingerprints at each depth."""

    import hashlib

    labels = sorted(str(value) for value in counts[label_column].unique() if str(value) != "control")
    rows: list[dict[str, Any]] = []
    for budget in validate_cell_budgets(budgets):
        matched = matched_label_universe(
            counts,
            left_environment=left_environment,
            right_environment=right_environment,
            budget=budget,
            label_column=label_column,
        )
        fingerprint = hashlib.sha256("\n".join(matched).encode("utf-8")).hexdigest()
        rows.append({
            "left_target_environment_id": left_environment,
            "right_target_environment_id": right_environment,
            "cell_budget": int(budget),
            "n_labels_all": int(len(labels)),
            "n_labels_matched": int(len(matched)),
            "coverage_fraction": float(len(matched) / len(labels)) if labels else float("nan"),
            "matched_universe_sha256": fingerprint,
            "matched_universe_policy": "same perturbation labels pass the fixed budget in both target contexts; control must also pass",
        })
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: Iterable[float], *, draws: int = 2000, seed: int = 20260910, level: float = 0.90) -> tuple[float, float, float]:
    """Return mean and a synchronized nonparametric seed-level interval."""

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    if int(draws) < 1 or not 0.0 < float(level) < 1.0:
        raise ValueError("draws must be positive and level must lie in (0,1)")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(draws), array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - float(level)) / 2.0
    return float(array.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def summarize_depth_rows(
    item_rows: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "cell_budget"),
    draws: int = 2000,
    ci_level: float = 0.90,
) -> pd.DataFrame:
    """Aggregate per-seed depth rows without treating perturbations as IID."""

    optional_group_columns = tuple(
        column for column in ("cell_budget_label", "cell_budget_order")
        if column in item_rows.columns and column not in group_columns
    )
    if "universe_mode" in item_rows.columns and "universe_mode" not in group_columns:
        optional_group_columns += ("universe_mode",)
    effective_group_columns = tuple(group_columns) + optional_group_columns
    required = set(effective_group_columns) | {"split_seed", "cross_disagreement", "within_disagreement_left", "within_disagreement_right", "identifiable_divergence", "stable_pair_fraction"}
    missing = required.difference(item_rows.columns)
    if missing:
        raise ValueError(f"depth rows are missing columns: {sorted(missing)}")
    frame = item_rows.copy()
    frame["measurement_floor"] = 0.5 * (frame["within_disagreement_left"] + frame["within_disagreement_right"])
    # Per-seed global values are means over perturbation burdens.  This keeps
    # the seed as the bootstrap unit and preserves the exact burden identity.
    seed_columns = list(effective_group_columns) + ["split_seed"]
    seed_level = frame.groupby(seed_columns, sort=True, observed=True)[
        ["cross_disagreement", "within_disagreement_left", "within_disagreement_right", "measurement_floor", "identifiable_divergence", "stable_pair_fraction"]
    ].mean(numeric_only=True).reset_index()
    rows: list[dict[str, Any]] = []
    value_columns = ("cross_disagreement", "within_disagreement_left", "within_disagreement_right", "measurement_floor", "identifiable_divergence", "stable_pair_fraction")
    for key, group in seed_level.groupby(list(effective_group_columns), sort=True, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(effective_group_columns, key))
        record["n_split_seeds"] = int(group["split_seed"].nunique())
        group_mask = np.ones(len(frame), dtype=bool)
        for column in effective_group_columns:
            group_mask &= frame[column].eq(record[column]).to_numpy()
        group_frame = frame.loc[group_mask]
        record["n_perturbations_mean"] = float(
            group_frame.groupby("split_seed", sort=True)["perturbation_label"].nunique().mean()
        ) if "perturbation_label" in frame.columns else float("nan")
        for column in value_columns:
            values = group[column].to_numpy(dtype=float)
            mean, low, high = bootstrap_mean_ci(
                values,
                draws=draws,
                seed=20260910 + sum((index + 1) * sum(ord(char) for char in str(value)) for index, value in enumerate(key)) + sum(ord(char) for char in column),
                level=ci_level,
            )
            record[column] = mean
            record[f"{column}_ci_low"] = low
            record[f"{column}_ci_high"] = high
        rows.append(record)
    return pd.DataFrame(rows)


def hierarchical_macro_bootstrap(
    item_rows: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = (
        "source_environment_id", "left_target_environment_id", "right_target_environment_id",
        "metric", "cell_budget", "cell_budget_label", "cell_budget_order", "universe_mode",
    ),
    value_columns: tuple[str, ...] = (
        "cross_disagreement", "within_disagreement_left", "within_disagreement_right",
        "identifiable_divergence", "stable_pair_fraction",
    ),
    draws: int = 2000,
    seed: int = 20260910,
) -> pd.DataFrame:
    """Create synchronized label×seed macro draws for depth curves.

    The scientific unit is a perturbation label and the measurement seed is a
    separate random layer.  For every group and draw, labels and seeds are
    sampled with replacement and the mean is computed from the resulting
    two-dimensional cell.  The returned table is intentionally draw-level so
    figures can take quantiles of the macro distribution rather than averaging
    row-level confidence endpoints.
    """

    required = set(group_columns) | {"split_seed", "perturbation_label"} | set(value_columns)
    missing = required.difference(item_rows.columns)
    if missing:
        raise ValueError(f"item rows are missing columns: {sorted(missing)}")
    if int(draws) < 1:
        raise ValueError("draws must be positive")
    rows: list[dict[str, Any]] = []
    for key, group in item_rows.groupby(list(group_columns), sort=True, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        labels = sorted(group["perturbation_label"].astype(str).unique())
        seeds = sorted(group["split_seed"].astype(int).unique())
        if len(labels) < 2 or not seeds:
            continue
        index = pd.MultiIndex.from_product([seeds, labels], names=["split_seed", "perturbation_label"])
        matrix_frame = group.assign(perturbation_label=group["perturbation_label"].astype(str)).set_index(["split_seed", "perturbation_label"])
        matrix_frame = matrix_frame.reindex(index)
        if matrix_frame[list(value_columns)].isna().any().any():
            raise ValueError(f"incomplete label×seed grid for {key}")
        rng = np.random.default_rng(int(seed) + sum((i + 1) * sum(ord(char) for char in str(value)) for i, value in enumerate(key)))
        arrays = {
            column: matrix_frame[column].to_numpy(dtype=float).reshape(len(seeds), len(labels))
            for column in value_columns
        }
        # Each draw resamples a label cohort and a seed layer.  The cartesian
        # product is the synchronized macro estimand for that draw.
        seed_indices = rng.integers(0, len(seeds), size=int(draws))
        label_indices = rng.integers(0, len(labels), size=int(draws))
        for draw_index, (seed_index, label_index) in enumerate(zip(seed_indices, label_indices)):
            label_bootstrap = rng.integers(0, len(labels), size=len(labels))
            seed_bootstrap = rng.integers(0, len(seeds), size=len(seeds))
            record = dict(zip(group_columns, key))
            record["bootstrap_draw"] = int(draw_index)
            record["bootstrap_unit"] = "perturbation_label_then_measurement_seed"
            for column, array in arrays.items():
                record[column] = float(array[np.ix_(seed_bootstrap, label_bootstrap)].mean())
            rows.append(record)
    return pd.DataFrame(rows)


def depth_resolution_table(summary: pd.DataFrame, *, value_column: str = "identifiable_divergence", tolerance_fraction: float = 0.10) -> pd.DataFrame:
    """Find the first budget within 10% of the full-depth estimate."""

    required = {"source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "cell_budget", value_column}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")
    group_columns = ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"]
    rows: list[dict[str, Any]] = []
    for key, group in summary.groupby(group_columns, sort=True, observed=True):
        order_column = "cell_budget_order" if "cell_budget_order" in group.columns else "cell_budget"
        full = float(group.loc[group[order_column].eq(group[order_column].max()), value_column].iloc[0])
        tolerance = max(abs(full) * float(tolerance_fraction), 1e-3)
        group = group.sort_values(order_column)
        resolved = group.loc[(group[value_column] - full).abs() <= tolerance, "cell_budget"]
        detectable = group.loc[group[f"{value_column}_ci_low"] > 0.0, "cell_budget"] if f"{value_column}_ci_low" in group else pd.Series(dtype=object)
        rows.append({
            **dict(zip(group_columns, key)),
            "full_depth_budget": group.loc[group[order_column].eq(group[order_column].max()), "cell_budget"].iloc[0],
            "full_depth_value": full,
            "resolution_tolerance": tolerance,
            "resolution_90pct_full_budget": resolved.iloc[0] if len(resolved) else None,
            "detectability_threshold_budget": detectable.iloc[0] if len(detectable) else None,
            "detectability_rule": "first fixed budget whose 90% seed bootstrap lower bound for identifiable divergence exceeds zero; null means not detected",
        })
    return pd.DataFrame(rows)
