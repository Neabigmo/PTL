"""Benchmark rank comparators on the corrected matched-fixed surface.

The rank diagnostics are deliberately kept separate from the canonical
measurement-identifiable ordering divergence.  Rank distances and stable
pair-state diagnostics are recomputed from the member-level risk grid, while
``d_adj`` is read only from the corrected matched-fixed summary.  The canonical
``measurement_floor`` is not a same-context ``d_adj`` null and is therefore
never used as one here.  The full-depth value is checked against the
decision-link artifact before any output is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.decision_analysis_common import CANONICAL_SUMMARY, load_canonical_d_adj  # noqa: E402
from src.evaluation.ordering_estimands import (  # noqa: E402
    crossfit_seed_stable_ordering_summary,
    pairwise_order_probabilities,
    stable_order_mask,
)


METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")
FIXED_DEPTHS = ("10", "20", "40", "80", "160")
DEPTHS = FIXED_DEPTHS + ("full",)
RAW_COMPARATORS = (
    "kendall_distance",
    "spearman_distance",
    "top10_jaccard_distance",
    "stable_pair_inversion_fraction",
    "d_adj",
)
FLOOR_ADJUSTED_COMPARATORS = (
    "kendall_distance_floor_adjusted",
    "spearman_distance_floor_adjusted",
    "top10_jaccard_distance_floor_adjusted",
)
NORMALIZATION_EPSILON = 1e-12


def _normalize_budget_labels(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["cell_budget_label"] = output["cell_budget_label"].astype(str).str.strip("'\" ")
    return output


def _read_risks(path: Path) -> pd.DataFrame:
    """Read the large matched-fixed risk grid in bounded parser chunks."""

    columns = [
        "source_environment_id",
        "left_target_environment_id",
        "right_target_environment_id",
        "metric",
        "split_seed",
        "perturbation_label",
        "measurement_replicate",
        "cell_budget_label",
        "universe_mode",
        "left_risk",
        "right_risk",
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000, low_memory=False):
        chunk = chunk.loc[chunk["metric"].isin(METRICS)].copy()
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"matched-fixed risk grid is empty: {path}")
    risks = _normalize_budget_labels(pd.concat(chunks, ignore_index=True, sort=False))
    risks["split_seed"] = risks["split_seed"].astype(int)
    risks["measurement_replicate"] = risks["measurement_replicate"].astype(int)
    modes = set(risks["universe_mode"].astype(str).unique())
    if modes != {"matched_fixed"}:
        raise ValueError(f"rank comparator input must be matched_fixed, found {sorted(modes)}")
    return risks


def _matrix(group: pd.DataFrame, context: str, seeds: list[int], budget: str) -> tuple[list[str], np.ndarray]:
    """Return ``[seed, measurement_replicate, perturbation]`` risks."""

    left = str(group["left_target_environment_id"].iloc[0])
    right = str(group["right_target_environment_id"].iloc[0])
    if context == left:
        column = "left_risk"
    elif context == right:
        column = "right_risk"
    else:
        raise ValueError(f"context {context} is not an endpoint of {left}<->{right}")
    budget_mask = group["cell_budget_label"].eq(str(budget))
    subset = group.loc[
        group["split_seed"].isin(seeds) & budget_mask,
        ["perturbation_label", "split_seed", "measurement_replicate", column],
    ].copy()
    key_columns = ["split_seed", "measurement_replicate", "perturbation_label"]
    if subset.duplicated(key_columns).any():
        raise ValueError(f"duplicate risk cells for {context} at budget {budget}")
    index = pd.MultiIndex.from_product(
        (seeds, (0, 1)), names=("split_seed", "measurement_replicate")
    )
    pivot = (
        subset.pivot(index=["split_seed", "measurement_replicate"], columns="perturbation_label", values=column)
        .reindex(index=index)
        .dropna(axis=1, how="any")
    )
    if pivot.shape[0] != len(seeds) * 2 or pivot.empty or pivot.isna().any().any():
        raise ValueError(
            f"incomplete {budget} grid for {context}: rows={pivot.shape[0]}, "
            f"expected={len(seeds) * 2}, labels={pivot.shape[1]}"
        )
    labels = [str(value) for value in pivot.columns]
    return labels, pivot.to_numpy(dtype=float).reshape(len(seeds), 2, -1)


def _top_k(values: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    k = max(1, int(np.floor(values.size * fraction)))
    # Lower risk is better.  The item index is a deterministic tie-breaker.
    return np.lexsort((np.arange(values.size), values))[:k]


def _tie_pair_count(values: np.ndarray) -> int:
    _, counts = np.unique(np.asarray(values), return_counts=True)
    return int(np.sum(counts * (counts - 1) // 2))


def _rank_values(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Compute raw deterministic rank distances and their tie diagnostics."""

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 1 or target.ndim != 1 or source.shape != target.shape:
        raise ValueError("rank vectors must be equal one-dimensional arrays")
    if source.size < 2 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("rank vectors must be finite and contain at least two items")

    tau = float(kendalltau(source, target, variant="b", method="auto").statistic)
    rho = float(spearmanr(source, target).statistic)
    tau = 1.0 if not np.isfinite(tau) else tau
    rho = 1.0 if not np.isfinite(rho) else rho

    source_top = set(map(int, _top_k(source)))
    target_top = set(map(int, _top_k(target)))
    intersection = len(source_top & target_top)
    union = len(source_top | target_top)

    source_difference = source[:, None] - source[None, :]
    target_difference = target[:, None] - target[None, :]
    upper = np.triu(np.ones(source_difference.shape, dtype=bool), k=1)
    discordant = int(np.sum(upper & (source_difference * target_difference < 0.0)))
    n_items = int(source.size)
    n_pairs = n_items * (n_items - 1) // 2
    source_ties = _tie_pair_count(source)
    target_ties = _tie_pair_count(target)
    no_tie = source_ties == 0 and target_ties == 0
    no_tie_distance = float(discordant / n_pairs) if no_tie else float("nan")
    relation_error = float(abs((1.0 - tau) / 2.0 - no_tie_distance)) if no_tie else float("nan")

    return {
        "n_items": n_items,
        "n_pairs": n_pairs,
        "n_top10": len(source_top),
        "kendall_tau": tau,
        "kendall_distance": float((1.0 - tau) / 2.0),
        "kendall_discordant_pairs": discordant,
        "kendall_no_tie": bool(no_tie),
        "kendall_no_tie_distance": no_tie_distance,
        "kendall_no_tie_relation_error": relation_error,
        "spearman_rho": rho,
        "spearman_distance": float((1.0 - rho) / 2.0),
        "top10_retention": float(intersection / len(source_top)),
        "top10_jaccard": float(intersection / union) if union else float("nan"),
        "top10_jaccard_distance": float(1.0 - intersection / union) if union else 0.0,
    }


def _stable_metrics(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Return cross-fitted stable inversion and both coverage notions."""

    summary = crossfit_seed_stable_ordering_summary(
        source,
        target,
        credible_level=0.95,
        minimum_strict_support=8,
    )
    n_items = int(source.shape[-1])
    n_pairs = n_items * (n_items - 1) // 2
    selected_coverage = float(summary["crossfit_selection_stable_both_fraction"])
    evaluable_coverage = float(summary["crossfit_heldout_evaluable_fraction"])
    return {
        "stable_pair_inversion_fraction": float(summary["crossfit_heldout_inversion_fraction"]),
        "stable_pair_count": int(round(selected_coverage * n_pairs)),
        "stable_pair_coverage": selected_coverage,
        "stable_pair_evaluable_count": int(summary["crossfit_heldout_evaluable_pairs"]),
        "stable_pair_evaluable_coverage": evaluable_coverage,
        "stable_pair_crossfit_splits": int(summary["crossfit_splits"]),
    }


def _same_context_stable_floor(values: np.ndarray) -> tuple[float, int, float, int]:
    """Compute a deterministic two-half stable-inversion floor diagnostic."""

    first = values[: values.shape[0] // 2].reshape(-1, values.shape[2])
    second = values[values.shape[0] // 2 :].reshape(-1, values.shape[2])
    stable_first, _, _ = stable_order_mask(first, minimum_strict_support=8)
    stable_second, _, _ = stable_order_mask(second, minimum_strict_support=8)
    _, pi_first = pairwise_order_probabilities(first)
    _, pi_second = pairwise_order_probabilities(second)
    stable = stable_first & stable_second
    sign_first = np.sign(pi_first[:, 2] - pi_first[:, 0])
    sign_second = np.sign(pi_second[:, 2] - pi_second[:, 0])
    evaluable = stable & (sign_first != 0) & (sign_second != 0)
    inversion = evaluable & (sign_first != sign_second)
    denominator = int(np.sum(evaluable))
    n_pairs = int(values.shape[2] * (values.shape[2] - 1) // 2)
    return (
        float(np.sum(inversion) / denominator) if denominator else float("nan"),
        denominator,
        float(np.sum(stable) / n_pairs) if n_pairs else float("nan"),
        int(np.sum(stable)),
    )


def _same_context_floors(values: np.ndarray) -> dict[str, tuple[float, int]]:
    """Estimate rank-only floors from independent 15-seed halves of one context.

    ``D_adj`` is intentionally absent.  Its canonical ``measurement_floor`` is
    a raw-cell measurement component, not a same-context pseudo-context
    estimate of the ``D_adj`` comparator.
    """

    first = values[: values.shape[0] // 2].reshape(-1, values.shape[2])
    second = values[values.shape[0] // 2 :].reshape(-1, values.shape[2])
    first_mean = first.mean(axis=0)
    second_mean = second.mean(axis=0)
    first_rank = _rank_values(first_mean, second_mean)
    stable_fraction, stable_evaluable, stable_coverage, stable_count = _same_context_stable_floor(values)
    n_items = int(values.shape[2])
    n_pairs = n_items * (n_items - 1) // 2
    return {
        "kendall_distance": (float(first_rank["kendall_distance"]), n_pairs),
        "spearman_distance": (float(first_rank["spearman_distance"]), n_items),
        "top10_jaccard_distance": (float(first_rank["top10_jaccard_distance"]), n_items),
        "stable_pair_inversion_fraction": (stable_fraction, stable_evaluable),
        "stable_pair_coverage": (stable_coverage, n_pairs),
        "stable_pair_count": (float(stable_count), n_pairs),
    }


def _canonical_surface(root: Path) -> tuple[dict[tuple[str, str, str, str], float], pd.DataFrame]:
    """Load all directed canonical depths and the full-depth decision-link check."""

    summary_path = root / CANONICAL_SUMMARY
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = _normalize_budget_labels(pd.read_csv(summary_path, dtype={"cell_budget_label": "string"}))
    required = {
        "source_environment_id",
        "left_target_environment_id",
        "right_target_environment_id",
        "metric",
        "cell_budget_label",
        "identifiable_divergence",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise RuntimeError(f"corrected canonical summary is missing columns: {missing}")
    summary = summary.loc[summary["metric"].isin(METRICS)].copy()
    endpoint = summary["source_environment_id"].eq(summary["left_target_environment_id"]) | summary[
        "source_environment_id"
    ].eq(summary["right_target_environment_id"])
    summary = summary.loc[endpoint].copy()
    summary["target_environment_id"] = np.where(
        summary["source_environment_id"].eq(summary["left_target_environment_id"]),
        summary["right_target_environment_id"],
        summary["left_target_environment_id"],
    )
    key_columns = ["source_environment_id", "target_environment_id", "metric", "cell_budget_label"]
    if summary.duplicated(key_columns).any():
        raise RuntimeError("corrected canonical summary has duplicate directed depth keys")
    expected_keys = 18 * len(DEPTHS)
    if len(summary) != expected_keys or set(summary["cell_budget_label"].unique()) != set(DEPTHS):
        raise RuntimeError(
            f"corrected canonical summary must have {expected_keys} endpoint depth rows, found {len(summary)}"
        )
    if not np.isfinite(summary[["identifiable_divergence"]].to_numpy(dtype=float)).all():
        raise RuntimeError("corrected canonical summary contains non-finite D_adj values")

    d_adj = {
        tuple(str(row._asdict()[column]) for column in key_columns): float(row.identifiable_divergence)
        for row in summary.itertuples(index=False)
    }
    canonical_full = load_canonical_d_adj(root, METRICS)
    decision_path = root / "artifacts/manifests/reliability_transport_decision_link.csv"
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    decision = pd.read_csv(decision_path, dtype={"cell_budget_label": "string"})
    decision["cell_budget_label"] = decision["cell_budget_label"].astype(str).str.strip("'\" ")
    decision = decision.loc[decision["cell_budget_label"].eq("full") & decision["metric"].isin(METRICS)].copy()
    decision_keys = ["source_environment_id", "target_environment_id", "metric"]
    if len(decision) != 18 or decision.duplicated(decision_keys).any() or "d_meas_id" not in decision.columns:
        raise RuntimeError("decision link must contain 18 unique full-depth rows with d_meas_id")
    comparison = canonical_full.merge(
        decision[decision_keys + ["d_meas_id"]],
        on=decision_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_canonical", "_decision"),
    )
    if len(comparison) != 18 or not comparison["_merge"].eq("both").all():
        raise RuntimeError("canonical summary and decision link do not cover the same 18 keys")
    canonical_values = comparison["d_meas_id_canonical"].to_numpy(dtype=float)
    decision_values = comparison["d_meas_id_decision"].to_numpy(dtype=float)
    # The canonical summary and decision link are required to be exactly equal
    # after CSV parsing, not merely numerically close.
    if not np.array_equal(canonical_values, decision_values):
        raise AssertionError("corrected canonical full-depth D_adj does not exactly equal decision-link d_meas_id")
    return d_adj, decision


def _normalized_error(value: float, full: float) -> float:
    if not np.isfinite(value) or not np.isfinite(full):
        return float("nan")
    return float(abs(value - full) / max(abs(full), NORMALIZATION_EPSILON))


def _surface_summary(depth_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = depth_table.groupby(["comparator", "cell_budget_label"], sort=True, observed=True)
    for (comparator, budget), group in grouping:
        record: dict[str, Any] = {
            "comparator": str(comparator),
            "cell_budget_label": str(budget),
            "cell_budget_order": int(group["cell_budget_order"].iloc[0]),
            "n_surfaces": int(len(group)),
        }
        for column in (
            "estimate",
            "same_context_floor",
            "floor_adjusted_estimate",
            "absolute_error_to_full",
            "normalized_depth_error",
            "absolute_floor_adjusted_error_to_full",
            "normalized_floor_adjusted_depth_error",
            "stable_pair_coverage",
            "stable_pair_evaluable_coverage",
        ):
            values = group[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                q25, median, q75 = np.quantile(values, (0.25, 0.50, 0.75))
                record[f"{column}_median"] = float(median)
                record[f"{column}_iqr_q25"] = float(q25)
                record[f"{column}_iqr_q75"] = float(q75)
                record[f"{column}_iqr"] = float(q75 - q25)
            else:
                record[f"{column}_median"] = float("nan")
                record[f"{column}_iqr_q25"] = float("nan")
                record[f"{column}_iqr_q75"] = float("nan")
                record[f"{column}_iqr"] = float("nan")
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["comparator", "cell_budget_order"], kind="stable").reset_index(drop=True)


def _safe_spearman(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    valid = frame[[column, "normalized_regret"]].dropna()
    return {
        "n": int(len(valid)),
        "spearman": float(spearmanr(valid[column], valid["normalized_regret"]).statistic)
        if len(valid) >= 3 and valid[column].nunique() > 1 and valid["normalized_regret"].nunique() > 1
        else float("nan"),
    }


def build(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    risk_path = manifests / "reliability_transport_measurement_depth_matched_fixed_risks.csv"
    d_adj_map, decision = _canonical_surface(root)
    risks = _read_risks(risk_path)
    seeds = sorted(int(value) for value in risks["split_seed"].unique())
    if len(seeds) != 30:
        raise ValueError(f"expected 30 seeds, found {len(seeds)}")

    decision_lookup = decision.set_index(["source_environment_id", "target_environment_id", "metric"])
    rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    for _, group in risks.groupby(
        ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"],
        sort=True,
        observed=True,
    ):
        metric = str(group["metric"].iloc[0])
        left = str(group["left_target_environment_id"].iloc[0])
        right = str(group["right_target_environment_id"].iloc[0])
        source_id = str(group["source_environment_id"].iloc[0])
        if metric not in METRICS or source_id not in {left, right}:
            continue
        available_budgets = set(group["cell_budget_label"].astype(str))
        if not set(DEPTHS).issubset(available_budgets):
            continue
        source = source_id
        target = right if source == left else left
        surface_key = (source, target, metric)
        if surface_key not in decision_lookup.index:
            raise RuntimeError(f"missing decision-link row for {source}->{target} {metric}")
        for budget in DEPTHS:
            source_labels, source_values = _matrix(group, source, seeds, budget)
            target_labels, target_values = _matrix(group, target, seeds, budget)
            common = sorted(set(source_labels) & set(target_labels))
            if not common:
                raise ValueError(f"no common perturbation labels for {source}->{target} {metric} at {budget}")
            source_idx = [source_labels.index(label) for label in common]
            target_idx = [target_labels.index(label) for label in common]
            source_values = source_values[:, :, source_idx]
            target_values = target_values[:, :, target_idx]
            source_mean = source_values.reshape(-1, len(common)).mean(axis=0)
            target_mean = target_values.reshape(-1, len(common)).mean(axis=0)
            rank = _rank_values(source_mean, target_mean)
            stable = _stable_metrics(source_values, target_values)
            floor_values = _same_context_floors(source_values)
            key = (source, target, metric, budget)
            if key not in d_adj_map:
                raise RuntimeError(f"missing corrected canonical depth row for {key}")
            values: dict[str, Any] = {
                **rank,
                **stable,
                "d_adj": d_adj_map[key],
            }
            same_context_floors = {
                "kendall_distance": floor_values["kendall_distance"][0],
                "spearman_distance": floor_values["spearman_distance"][0],
                "top10_jaccard_distance": floor_values["top10_jaccard_distance"][0],
                "stable_pair_inversion_fraction": floor_values["stable_pair_inversion_fraction"][0],
                # The canonical summary's measurement floor is not a
                # same-context D_adj null.  A matched pseudo-context D_adj is
                # computed by build_finite_measurement_rank_discrimination.py.
                "d_adj": float("nan"),
            }
            floor_adjusted = {
                comparator: float(values[comparator] - same_context_floors[comparator])
                for comparator in ("kendall_distance", "spearman_distance", "top10_jaccard_distance")
            }
            for comparator in RAW_COMPARATORS:
                depth_rows.append(
                    {
                        "source_environment_id": source,
                        "target_environment_id": target,
                        "transfer_id": f"{source}->{target}",
                        "metric": metric,
                        "n_items": int(len(common)),
                        "n_pairs": int(rank["n_pairs"]),
                        "comparator": comparator,
                        "cell_budget_label": budget,
                        "cell_budget_order": DEPTHS.index(budget) + 1,
                        "estimate": float(values[comparator]),
                        "same_context_floor": float(same_context_floors[comparator]),
                        "floor_adjusted_estimate": (
                            floor_adjusted[comparator] if comparator in floor_adjusted else float("nan")
                        ),
                        "floor_n": int(
                            floor_values[comparator][1] if comparator in floor_values else 0
                        ),
                        "stable_pair_coverage": float(stable["stable_pair_coverage"]),
                        "stable_pair_evaluable_coverage": float(stable["stable_pair_evaluable_coverage"]),
                        "stable_pair_count": int(stable["stable_pair_count"]),
                        "stable_pair_evaluable_count": int(stable["stable_pair_evaluable_count"]),
                        "full_estimate": float("nan"),
                        "full_same_context_floor": float("nan"),
                        "full_floor_adjusted_estimate": float("nan"),
                        "absolute_error_to_full": float("nan"),
                        "normalized_depth_error": float("nan"),
                        "absolute_floor_adjusted_error_to_full": float("nan"),
                        "normalized_floor_adjusted_depth_error": float("nan"),
                    }
                )
            if budget == "full":
                decision_row = decision_lookup.loc[surface_key]
                row: dict[str, Any] = {
                    "source_environment_id": source,
                    "target_environment_id": target,
                    "transfer_id": f"{source}->{target}",
                    "metric": metric,
                    "n_items": int(len(common)),
                    "n_pairs": int(rank["n_pairs"]),
                    **{key_name: values[key_name] for key_name in (
                        "kendall_tau",
                        "kendall_distance",
                        "kendall_discordant_pairs",
                        "kendall_no_tie",
                        "kendall_no_tie_distance",
                        "kendall_no_tie_relation_error",
                        "spearman_rho",
                        "spearman_distance",
                        "top10_retention",
                        "top10_jaccard",
                        "top10_jaccard_distance",
                        "stable_pair_inversion_fraction",
                        "stable_pair_coverage",
                        "stable_pair_evaluable_coverage",
                        "stable_pair_count",
                        "stable_pair_evaluable_count",
                        "d_adj",
                    )},
                    "normalized_regret": float(decision_row["normalized_regret"]),
                    "status": "executed",
                }
                for comparator in RAW_COMPARATORS:
                    row[f"same_context_floor_{comparator}"] = float(same_context_floors[comparator])
                row.update({
                    "kendall_distance_floor_adjusted": floor_adjusted["kendall_distance"],
                    "spearman_distance_floor_adjusted": floor_adjusted["spearman_distance"],
                    "top10_jaccard_distance_floor_adjusted": floor_adjusted["top10_jaccard_distance"],
                })
                rows.append(row)

    table = pd.DataFrame(rows).sort_values(
        ["source_environment_id", "target_environment_id", "metric"], kind="stable"
    ).reset_index(drop=True)
    depth_table = pd.DataFrame(depth_rows)
    if depth_table.empty:
        raise ValueError("no complete directed comparator surfaces were found")
    depth_table = depth_table.sort_values(
        ["source_environment_id", "target_environment_id", "metric", "comparator", "cell_budget_order"],
        kind="stable",
    ).reset_index(drop=True)

    full_values = depth_table.loc[
        depth_table["cell_budget_label"].eq("full"),
        [
            "source_environment_id",
            "target_environment_id",
            "metric",
            "comparator",
            "estimate",
            "same_context_floor",
            "floor_adjusted_estimate",
        ],
    ].rename(
        columns={
            "estimate": "full_estimate",
            "same_context_floor": "full_same_context_floor",
            "floor_adjusted_estimate": "full_floor_adjusted_estimate",
        }
    )
    depth_table = depth_table.drop(
        columns=["full_estimate", "full_same_context_floor", "full_floor_adjusted_estimate"]
    ).merge(
        full_values,
        on=["source_environment_id", "target_environment_id", "metric", "comparator"],
        how="left",
        validate="many_to_one",
    )
    depth_table["absolute_error_to_full"] = (
        depth_table["estimate"] - depth_table["full_estimate"]
    ).abs()
    depth_table["normalized_depth_error"] = [
        _normalized_error(value, full)
        for value, full in zip(depth_table["estimate"], depth_table["full_estimate"])
    ]
    depth_table["absolute_floor_adjusted_error_to_full"] = (
        depth_table["floor_adjusted_estimate"] - depth_table["full_floor_adjusted_estimate"]
    ).abs()
    depth_table["normalized_floor_adjusted_depth_error"] = [
        _normalized_error(value, full)
        for value, full in zip(
            depth_table["floor_adjusted_estimate"], depth_table["full_floor_adjusted_estimate"]
        )
    ]

    if len(table) != 18 or table.duplicated(["source_environment_id", "target_environment_id", "metric"]).any():
        raise ValueError(f"expected 18 unique comparator rows, found {len(table)}")
    if len(depth_table) != 18 * len(RAW_COMPARATORS) * len(DEPTHS):
        raise ValueError(f"unexpected depth table size: {len(depth_table)}")
    canonical_full_d_adj = (
        load_canonical_d_adj(root, METRICS)
        .set_index(["source_environment_id", "target_environment_id", "metric"])
        .loc[
            sorted(table[["source_environment_id", "target_environment_id", "metric"]]
                   .itertuples(index=False, name=None))
        ]["d_meas_id"].to_numpy(dtype=float)
    )
    table_keys = table[["source_environment_id", "target_environment_id", "metric"]].copy()
    table_ordered = table.set_index(["source_environment_id", "target_environment_id", "metric"]).loc[
        sorted(table_keys.itertuples(index=False, name=None))
    ]["d_adj"].to_numpy(dtype=float)
    if not np.array_equal(table_ordered, canonical_full_d_adj):
        raise AssertionError("benchmark full-depth D_adj is not the corrected canonical summary value")

    surface_summary = _surface_summary(depth_table)
    correlations = {
        comparator: _safe_spearman(table, comparator)
        for comparator in RAW_COMPARATORS
    }
    correlations.update({
        comparator: _safe_spearman(table, comparator)
        for comparator in FLOOR_ADJUSTED_COMPARATORS
    })

    no_tie_rows = table.loc[table["kendall_no_tie"]].copy()
    surface_metadata = {
        str(comparator): {
            str(row.cell_budget_label): {
                "n": int(row.n_surfaces),
                "estimate_median": float(row.estimate_median),
                "estimate_iqr": float(row.estimate_iqr),
                "same_context_floor_median": float(row.same_context_floor_median),
                "normalized_depth_error_median": float(row.normalized_depth_error_median),
                "normalized_depth_error_iqr": float(row.normalized_depth_error_iqr),
                "normalized_floor_adjusted_depth_error_median": float(row.normalized_floor_adjusted_depth_error_median),
                "normalized_floor_adjusted_depth_error_iqr": float(row.normalized_floor_adjusted_depth_error_iqr),
            }
            for row in group.itertuples(index=False)
        }
        for comparator, group in surface_summary.groupby("comparator", sort=True, observed=True)
    }
    report = {
        "schema_version": 3,
        "status": "executed",
        "rows": int(len(table)),
        "depth_rows": int(len(depth_table)),
        "surface_summary_rows": int(len(surface_summary)),
        "metrics": list(METRICS),
        "depths": {"fixed": list(FIXED_DEPTHS), "full": "full"},
        "comparators": {
            "raw": list(RAW_COMPARATORS),
            "floor_adjusted": list(FLOOR_ADJUSTED_COMPARATORS),
        },
        "estimands": {
            "kendall_distance": "(1-tau-b)/2 on deterministic source/target mean risk vectors",
            "spearman_distance": "(1-rho)/2 on deterministic source/target mean risk vectors",
            "top10_jaccard_distance": "1-Jaccard(source top-10%, target top-10%); lower risk ranks first",
            "stable_pair_inversion_fraction": "cross-fitted held-out inversion fraction among stable, evaluable pair states; minimum strict support >=8",
            "d_adj": "corrected canonical pair-state ordering divergence from the matched-fixed summary; no same-context floor is attached here",
            "floor_adjusted_rank_distance": "raw rank distance minus the independent same-context source-half floor; signed and not clipped",
        },
        "canonical_d_adj": {
            "source": CANONICAL_SUMMARY,
            "decision_link": "artifacts/manifests/reliability_transport_decision_link.csv",
            "decision_link_column": "d_meas_id",
            "rows_checked": 18,
            "exact_equal_all_rows": True,
            "max_abs_difference": 0.0,
            "depth_values_also_loaded_from_corrected_summary": True,
        },
        "report_metadata": {
            "deterministic_no_tie_kendall_relation": {
                "relation": "For tie-free deterministic vectors, (1-tau)/2 = discordant_pair_count / binomial(n_items, 2).",
                "tau_variant": "Kendall tau-b; the equality is asserted only when both vectors have no tied item values.",
                "full_depth_surfaces_checked": int(len(table)),
                "full_depth_tie_free_surfaces": int(len(no_tie_rows)),
                "max_relation_error": float(no_tie_rows["kendall_no_tie_relation_error"].max())
                if not no_tie_rows.empty
                else float("nan"),
            },
            "pair_state_mmd_interpretation": {
                "state_space": ["-1", "0", "+1"],
                "state_definition": "sign of the item-pair risk difference, with 0 denoting a tie",
                "population_relation": "D_adj = 1/2 * mean_pair ||pi_source(pair)-pi_target(pair)||_2^2 = 1/2 * mean_pair MMD_delta^2.",
                "finite_sample_note": "The canonical matched-fixed value is the U-corrected cross-minus-within estimate of this population quantity; it is not the uncorrected plug-in squared distance.",
                "interpretation": "D_adj is a measurement-corrected squared MMD between categorical pair-state laws under the delta kernel; it is not a substitute for a continuous rank distance.",
            },
            "stable_pair_coverage": {
                "selection_coverage": "cross-fitted mean fraction of item pairs stable in both contexts on the fit halves",
                "evaluable_coverage": "cross-fitted mean fraction of all item pairs with non-tie held-out directions after stable-pair selection",
                "minimum_strict_support": 8,
                "measurement_replicates_per_seed": 2,
                "split_seeds": 30,
            },
            "same_context_floor": "independent 15-seed source-context halves with both measurement replicates retained for rank diagnostics only; canonical measurement_floor is not a same-context D_adj null",
            "depth_error": "absolute error to the full-depth surface divided by max(abs(full-depth estimate), 1e-12); floor-adjusted errors use the corresponding floor-adjusted full value",
            "surface_aggregation": "median and IQR (q25, q75) across the 18 directed transfer-by-metric surfaces",
        },
        "correlation_with_normalized_regret": correlations,
        "surface_aggregation": surface_metadata,
        "outputs": {
            "full_depth": "artifacts/manifests/rank_comparator_benchmark.csv",
            "depth_convergence": "artifacts/manifests/rank_comparator_depth_convergence.csv",
            "surface_summary": "artifacts/manifests/rank_comparator_surface_summary.csv",
        },
    }
    table_path = manifests / "rank_comparator_benchmark.csv"
    depth_path = manifests / "rank_comparator_depth_convergence.csv"
    summary_path = manifests / "rank_comparator_surface_summary.csv"
    table.to_csv(table_path, index=False)
    depth_table.to_csv(depth_path, index=False)
    surface_summary.to_csv(summary_path, index=False)
    (manifests / "rank_comparator_benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return table, depth_table, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    table, depth, report = build(args.root)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
    print(table.to_string(index=False))
    print(depth.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
