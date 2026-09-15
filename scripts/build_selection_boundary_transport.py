"""Build source-only top-k boundary-pair transport diagnostics.

The boundary for a top-k shortlist is the set of unordered item pairs with
exactly one item in the source shortlist.  A shortlist is fitted from the
source risk vectors in one 15-seed half of the fixed 30-seed protocol and is
then evaluated on the other half.  The two half assignments are swapped.  In
particular, target risk values are never used to select the shortlist or its
boundary pair set; they are used only after the pair set has been frozen for
transport and decision evaluation.

This script deliberately keeps the existing matched-fixed, full-depth risk
surface and its 18 directed transfer-by-metric rows.  It writes only the two
requested manifest products:

``artifacts/manifests/selection_boundary_transport.csv``
``artifacts/manifests/selection_boundary_transport.json``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

METRICS = (
    "delta_cosine",
    "systema_centroid_accuracy",
    "absolute_effect_rank_agreement",
)
BUDGET_FRACTIONS = (0.05, 0.10, 0.20, 0.50)
FULL_DEPTH = "full"
FIXED_UNIVERSE = "matched_fixed"
N_SPLIT_SEEDS = 30
N_MEASUREMENT_REPLICATES = 2
CROSSFIT_HALF = 15
TIE_TOLERANCE = 1e-8
CI_LEVEL = 0.90
DEFAULT_BOOTSTRAP_DRAWS = 2000

RISK_INPUT = "artifacts/manifests/reliability_transport_measurement_depth_matched_fixed_risks.csv"
CANONICAL_SUMMARY = "artifacts/manifests/reliability_transport_measurement_depth_matched_fixed_summary.csv"
OUTPUT_CSV = "artifacts/manifests/selection_boundary_transport.csv"
OUTPUT_JSON = "artifacts/manifests/selection_boundary_transport.json"


@dataclass(frozen=True)
class SurfaceData:
    """One source-to-target directed surface on the fixed label universe."""

    source_environment_id: str
    target_environment_id: str
    left_target_environment_id: str
    right_target_environment_id: str
    metric: str
    labels: tuple[str, ...]
    split_seeds: tuple[int, ...]
    source_replicates: np.ndarray  # [seed, measurement_replicate, item]
    target_replicates: np.ndarray  # [seed, measurement_replicate, item]
    pair_left: np.ndarray  # [pair]
    pair_right: np.ndarray  # [pair]
    source_states: np.ndarray  # [seed, measurement_replicate, pair]
    target_states: np.ndarray  # [seed, measurement_replicate, pair]
    d_adj_by_seed_pair: np.ndarray  # [seed, pair]


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _stable_seed(*parts: object, base: int = 20260914) -> int:
    """Make a reproducible integer seed without using Python's hash randomization."""

    value = int(base)
    for index, part in enumerate(parts, start=1):
        value += index * sum(ord(char) for char in str(part))
    return value


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def _normalise_string_column(frame: pd.DataFrame, column: str) -> None:
    if column in frame.columns:
        frame[column] = frame[column].astype(str).str.strip("'\" ")


def _read_full_fixed_risks(root: Path) -> pd.DataFrame:
    """Load only the canonical full-depth matched-fixed risk rows."""

    path = root / RISK_INPUT
    if not path.is_file():
        raise FileNotFoundError(path)
    usecols = [
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
    dtypes = {
        "source_environment_id": "string",
        "left_target_environment_id": "string",
        "right_target_environment_id": "string",
        "metric": "string",
        "split_seed": "int64",
        "perturbation_label": "string",
        "measurement_replicate": "int8",
        "cell_budget_label": "string",
        "universe_mode": "string",
        "left_risk": "float64",
        "right_risk": "float64",
    }
    parts: list[pd.DataFrame] = []
    # The canonical risk file is large and also contains fixed-depth rows.
    # Filtering chunks keeps the focused build's peak memory bounded without
    # changing the source data or the fixed universe.
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=250_000, low_memory=False):
        _normalise_string_column(chunk, "cell_budget_label")
        _normalise_string_column(chunk, "universe_mode")
        selected = chunk.loc[
            chunk["cell_budget_label"].eq(FULL_DEPTH)
            & chunk["universe_mode"].eq(FIXED_UNIVERSE)
        ].copy()
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise ValueError("canonical risk input has no full matched-fixed rows")
    frame = pd.concat(parts, ignore_index=True, sort=False)
    for column in ("source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "perturbation_label"):
        _normalise_string_column(frame, column)
    if set(frame["metric"].dropna().unique()) != set(METRICS):
        raise ValueError("full matched-fixed risk input does not contain exactly the three frozen metrics")
    if frame["split_seed"].nunique() != N_SPLIT_SEEDS:
        raise ValueError(f"expected {N_SPLIT_SEEDS} fixed split seeds, found {frame['split_seed'].nunique()}")
    if frame["measurement_replicate"].nunique() != N_MEASUREMENT_REPLICATES or set(frame["measurement_replicate"].unique()) != {0, 1}:
        raise ValueError("fixed risk input must contain measurement replicates 0 and 1")
    if not np.isfinite(frame[["left_risk", "right_risk"]].to_numpy(dtype=float)).all():
        raise ValueError("full matched-fixed risk input contains non-finite risks")
    key_columns = [
        "source_environment_id", "left_target_environment_id", "right_target_environment_id",
        "metric", "split_seed", "measurement_replicate", "perturbation_label",
    ]
    if frame.duplicated(key_columns).any():
        raise ValueError("full matched-fixed risk input has duplicate seed/replicate/label keys")
    return frame


def _read_canonical_summary(root: Path) -> dict[tuple[str, str, str, str], dict[str, float]]:
    """Read the existing canonical global D_adj and its supported intervals."""

    path = root / CANONICAL_SUMMARY
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"cell_budget_label": "string", "universe_mode": "string"})
    for column in ("source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "cell_budget_label", "universe_mode"):
        _normalise_string_column(frame, column)
    required = {
        "source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric",
        "cell_budget_label", "universe_mode", "identifiable_divergence",
        "identifiable_divergence_ci_low", "identifiable_divergence_ci_high",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"canonical summary is missing columns: {missing}")
    frame = frame.loc[
        frame["cell_budget_label"].eq(FULL_DEPTH)
        & frame["universe_mode"].eq(FIXED_UNIVERSE)
        & frame["metric"].isin(METRICS)
    ].copy()
    keys = ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"]
    if frame.duplicated(keys).any():
        raise ValueError("canonical full matched-fixed summary has duplicate surface keys")
    if not np.isfinite(frame[["identifiable_divergence", "identifiable_divergence_ci_low", "identifiable_divergence_ci_high"]].to_numpy(dtype=float)).all():
        raise ValueError("canonical global D_adj summary contains non-finite values")
    return {
        tuple(str(record[column]) for column in keys): {
            "global_d_adj": float(record["identifiable_divergence"]),
            "global_d_adj_ci_low": float(record["identifiable_divergence_ci_low"]),
            "global_d_adj_ci_high": float(record["identifiable_divergence_ci_high"]),
        }
        for record in frame.to_dict("records")
    }


def _order_states(values: np.ndarray, pair_left: np.ndarray, pair_right: np.ndarray) -> np.ndarray:
    """Return tie-aware pair states in the existing [-1, 0, +1] convention."""

    differences = values[:, pair_left] - values[:, pair_right]
    return np.where(
        differences < -TIE_TOLERANCE,
        -1,
        np.where(differences > TIE_TOLERANCE, 1, 0),
    ).astype(np.int8)


def _prepare_surface(
    group: pd.DataFrame,
    *,
    source_environment_id: str,
    left_target_environment_id: str,
    right_target_environment_id: str,
    metric: str,
    split_seeds: tuple[int, ...],
) -> SurfaceData:
    """Materialise one complete [seed, replicate, label] directed surface."""

    source_is_left = source_environment_id == left_target_environment_id
    if not source_is_left and source_environment_id != right_target_environment_id:
        raise ValueError("source environment is not an endpoint of the context pair")
    target_environment_id = right_target_environment_id if source_is_left else left_target_environment_id
    source_column = "left_risk" if source_is_left else "right_risk"
    target_column = "right_risk" if source_is_left else "left_risk"
    labels = tuple(sorted(group["perturbation_label"].astype(str).unique().tolist()))
    if len(labels) < 2:
        raise ValueError(f"surface {source_environment_id}->{target_environment_id} has fewer than two labels")
    expected_rows = len(split_seeds) * N_MEASUREMENT_REPLICATES * len(labels)
    if len(group) != expected_rows:
        raise ValueError(
            f"incomplete fixed risk grid for {source_environment_id}->{target_environment_id} {metric}: "
            f"expected {expected_rows}, found {len(group)}"
        )
    index = pd.MultiIndex.from_product(
        [list(split_seeds), list(range(N_MEASUREMENT_REPLICATES))],
        names=["split_seed", "measurement_replicate"],
    )
    pivot = group.pivot(
        index=["split_seed", "measurement_replicate"],
        columns="perturbation_label",
        values=[source_column, target_column],
    )
    source = pivot[source_column].reindex(index=index, columns=list(labels)).to_numpy(dtype=float)
    target = pivot[target_column].reindex(index=index, columns=list(labels)).to_numpy(dtype=float)
    if source.shape != (len(split_seeds) * N_MEASUREMENT_REPLICATES, len(labels)) or target.shape != source.shape:
        raise ValueError("fixed risk grid has an unexpected pivot shape")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("surface contains non-finite source or target risks")
    source = source.reshape(len(split_seeds), N_MEASUREMENT_REPLICATES, len(labels))
    target = target.reshape(len(split_seeds), N_MEASUREMENT_REPLICATES, len(labels))
    pair_left, pair_right = np.triu_indices(len(labels), k=1)
    source_states = np.stack([
        _order_states(source[:, replicate, :], pair_left, pair_right)
        for replicate in range(N_MEASUREMENT_REPLICATES)
    ], axis=1)
    target_states = np.stack([
        _order_states(target[:, replicate, :], pair_left, pair_right)
        for replicate in range(N_MEASUREMENT_REPLICATES)
    ], axis=1)
    # The cross term averages the 2x2 independent source/target replicate
    # pairs.  The within terms are order-2 U-statistics because the fixed
    # protocol retains exactly two measurement replicates per seed.
    cross = np.mean(
        source_states[:, :, None, :] != target_states[:, None, :, :],
        axis=(1, 2),
    )
    within_source = np.not_equal(source_states[:, 0, :], source_states[:, 1, :]).astype(float)
    within_target = np.not_equal(target_states[:, 0, :], target_states[:, 1, :]).astype(float)
    d_adj_by_seed_pair = cross - 0.5 * (within_source + within_target)
    return SurfaceData(
        source_environment_id=source_environment_id,
        target_environment_id=target_environment_id,
        left_target_environment_id=left_target_environment_id,
        right_target_environment_id=right_target_environment_id,
        metric=metric,
        labels=labels,
        split_seeds=split_seeds,
        source_replicates=source,
        target_replicates=target,
        pair_left=pair_left.astype(np.int64),
        pair_right=pair_right.astype(np.int64),
        source_states=source_states,
        target_states=target_states,
        d_adj_by_seed_pair=d_adj_by_seed_pair,
    )


def deterministic_top_k(values: Iterable[float] | np.ndarray, k: int) -> np.ndarray:
    """Return the deterministic lowest-risk item indices, breaking ties by index."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("risk values must be a finite one-dimensional vector with at least two items")
    if not 1 <= int(k) <= array.size:
        raise ValueError("k must lie between one and the number of items")
    return np.lexsort((np.arange(array.size, dtype=np.int64), array))[: int(k)]


def top_k_size(n_items: int, budget_fraction: float) -> int:
    """Use the existing max(1, floor(budget*n)) rule."""

    if int(n_items) < 2 or not 0.0 < float(budget_fraction) <= 1.0:
        raise ValueError("invalid item count or budget fraction")
    return max(1, int(np.floor(float(budget_fraction) * int(n_items))))


def boundary_pair_mask(n_items: int, selected_indices: np.ndarray) -> np.ndarray:
    """Mark all unordered selected-versus-outside pairs."""

    selected = np.zeros(int(n_items), dtype=bool)
    selected[np.asarray(selected_indices, dtype=np.int64)] = True
    pair_left, pair_right = np.triu_indices(int(n_items), k=1)
    mask = np.logical_xor(selected[pair_left], selected[pair_right])
    expected = int(np.sum(selected)) * (int(n_items) - int(np.sum(selected)))
    if int(mask.sum()) != expected:
        raise AssertionError("boundary pair count does not equal k*(n-k)")
    return mask


def _shortlist_hash(labels: tuple[str, ...], selected_indices: np.ndarray) -> str:
    selected_labels = [labels[int(index)] for index in selected_indices]
    return hashlib.sha256("\n".join(selected_labels).encode("utf-8")).hexdigest()


def _boundary_hash(
    labels: tuple[str, ...],
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    boundary_mask: np.ndarray,
) -> str:
    pairs = [
        f"{labels[int(left)]}|{labels[int(right)]}"
        for left, right in zip(pair_left[boundary_mask], pair_right[boundary_mask])
    ]
    return hashlib.sha256("\n".join(pairs).encode("utf-8")).hexdigest()


def _fixed_shortlist_decision(target_risk: np.ndarray, selected_indices: np.ndarray) -> dict[str, float | int]:
    """Evaluate a frozen source shortlist on one target risk vector."""

    target = np.asarray(target_risk, dtype=float)
    selected = np.asarray(selected_indices, dtype=np.int64)
    n_items = int(target.size)
    k = int(selected.size)
    selected_mask = np.zeros(n_items, dtype=bool)
    selected_mask[selected] = True
    outside = np.flatnonzero(~selected_mask)
    oracle = deterministic_top_k(target, k)
    oracle_mask = np.zeros(n_items, dtype=bool)
    oracle_mask[oracle] = True
    selected_mean = float(np.mean(target[selected]))
    oracle_mean = float(np.mean(target[oracle]))
    random_mean = float(np.mean(target))
    denominator = random_mean - oracle_mean
    regret = selected_mean - oracle_mean
    boundary_inversion = (
        float(np.mean(target[selected, None] > target[outside][None, :] + TIE_TOLERANCE))
        if outside.size
        else float("nan")
    )
    mistakes = int(np.sum(selected_mask & ~oracle_mask))
    return {
        "deterministic_boundary_inversion": boundary_inversion,
        "shortlist_retention": float(np.sum(selected_mask & oracle_mask) / k),
        "normalized_regret": float(regret / denominator) if denominator > 0.0 else float("nan"),
        "mis_selection_fraction": float(mistakes / k),
        "mis_selection_count": mistakes,
    }


def _paired_seed_bootstrap(
    evaluation: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Return paired held-out-seed q05/q95 intervals for the new diagnostics.

    The source shortlist is conditional/frozen within each cross-fit split.
    Resampling is therefore over the 30 held-out evaluation-seed records,
    preserving the paired global/boundary/nonboundary and decision values.
    This is the seed-level uncertainty supported by the fixed measurement
    protocol; it is not a simultaneous selection-uncertainty guarantee.
    """

    columns = [
        "global_d_adj_eval",
        "boundary_d_adj_eval",
        "nonboundary_d_adj_eval",
        "deterministic_boundary_inversion_eval",
        "shortlist_retention_eval",
        "normalized_regret_eval",
        "mis_selection_fraction_eval",
    ]
    values = evaluation[columns].to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[0] != N_SPLIT_SEEDS:
        raise ValueError(f"expected {N_SPLIT_SEEDS} paired evaluation-seed records, found {len(evaluation)}")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.shape[0], size=(int(draws), values.shape[0]))
    sampled = values[indices]
    means = np.nanmean(sampled, axis=1)
    draw_values: dict[str, np.ndarray] = {
        "global_d_adj": means[:, 0],
        "boundary_d_adj": means[:, 1],
        "nonboundary_d_adj": means[:, 2],
        "deterministic_boundary_inversion": means[:, 3],
        "shortlist_retention": means[:, 4],
        "normalized_regret": means[:, 5],
        "mis_selection_fraction": means[:, 6],
    }
    draw_values["boundary_excess_d_adj"] = means[:, 1] - means[:, 2]
    draw_values["boundary_enrichment"] = np.divide(
        means[:, 1],
        means[:, 0],
        out=np.full(means.shape[0], np.nan, dtype=float),
        where=means[:, 0] != 0.0,
    )
    draw_values["boundary_to_nonboundary_ratio"] = np.divide(
        means[:, 1],
        means[:, 2],
        out=np.full(means.shape[0], np.nan, dtype=float),
        where=means[:, 2] != 0.0,
    )
    output: dict[str, tuple[float, float]] = {}
    alpha = (1.0 - CI_LEVEL) / 2.0
    for name, array in draw_values.items():
        finite = array[np.isfinite(array)]
        output[name] = (
            float(np.quantile(finite, alpha)),
            float(np.quantile(finite, 1.0 - alpha)),
        ) if finite.size else (float("nan"), float("nan"))
    return output


def _evaluation_records(surface: SurfaceData, budget_fraction: float) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Cross-fit source selection and return one row per evaluation seed."""

    seeds = np.asarray(surface.split_seeds, dtype=np.int64)
    if len(seeds) != N_SPLIT_SEEDS or N_SPLIT_SEEDS != 2 * CROSSFIT_HALF:
        raise ValueError("fixed source-boundary cross-fit requires 30 seeds split into two 15-seed halves")
    n_items = len(surface.labels)
    k = top_k_size(n_items, budget_fraction)
    split_specs = (
        ("fit_first15_eval_last15", np.arange(0, CROSSFIT_HALF), np.arange(CROSSFIT_HALF, N_SPLIT_SEEDS)),
        ("fit_last15_eval_first15", np.arange(CROSSFIT_HALF, N_SPLIT_SEEDS), np.arange(0, CROSSFIT_HALF)),
    )
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    pair_count = int(len(surface.pair_left))
    for split_id, fit_indices, eval_indices in split_specs:
        # This is the only object passed to shortlist selection: no target
        # vector, target state, or target-derived statistic is available here.
        source_fit_score = np.mean(surface.source_replicates[fit_indices], axis=(0, 1))
        selected = deterministic_top_k(source_fit_score, k)
        boundary = boundary_pair_mask(n_items, selected)
        nonboundary = ~boundary
        if int(boundary.sum()) != k * (n_items - k) or int(nonboundary.sum()) + int(boundary.sum()) != pair_count:
            raise AssertionError("boundary/nonboundary pair partition is invalid")
        audit.append({
            "split_id": split_id,
            "fit_split_seeds": [int(value) for value in seeds[fit_indices]],
            "evaluation_split_seeds": [int(value) for value in seeds[eval_indices]],
            "selected_labels": [surface.labels[int(index)] for index in selected],
            "selected_label_count": int(k),
            "boundary_pair_count": int(boundary.sum()),
            "nonboundary_pair_count": int(nonboundary.sum()),
            "shortlist_sha256": _shortlist_hash(surface.labels, selected),
            "boundary_pair_sha256": _boundary_hash(surface.labels, surface.pair_left, surface.pair_right, boundary),
        })
        for eval_index in eval_indices:
            d_adj = surface.d_adj_by_seed_pair[int(eval_index)]
            decision_rows = [
                _fixed_shortlist_decision(surface.target_replicates[int(eval_index), replicate], selected)
                for replicate in range(N_MEASUREMENT_REPLICATES)
            ]
            record = {
                "split_id": split_id,
                "evaluation_seed": int(seeds[int(eval_index)]),
                "global_d_adj_eval": float(np.mean(d_adj)),
                "boundary_d_adj_eval": float(np.mean(d_adj[boundary])),
                "nonboundary_d_adj_eval": float(np.mean(d_adj[nonboundary])),
                "deterministic_boundary_inversion_eval": float(np.mean([row["deterministic_boundary_inversion"] for row in decision_rows])),
                "shortlist_retention_eval": float(np.mean([row["shortlist_retention"] for row in decision_rows])),
                "normalized_regret_eval": float(np.nanmean([row["normalized_regret"] for row in decision_rows])),
                "mis_selection_fraction_eval": float(np.mean([row["mis_selection_fraction"] for row in decision_rows])),
                "mis_selection_count_eval": float(np.mean([row["mis_selection_count"] for row in decision_rows])),
            }
            rows.append(record)
    frame = pd.DataFrame(rows).sort_values(["evaluation_seed", "split_id"], kind="stable").reset_index(drop=True)
    if len(frame) != N_SPLIT_SEEDS or frame["evaluation_seed"].nunique() != N_SPLIT_SEEDS:
        raise AssertionError("cross-fit evaluation must cover each fixed seed exactly once")
    return frame, audit


def _row_for_surface_budget(
    surface: SurfaceData,
    budget_fraction: float,
    canonical: dict[str, float],
    *,
    draws: int,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    evaluation, audit = _evaluation_records(surface, budget_fraction)
    k = top_k_size(len(surface.labels), budget_fraction)
    n_pairs = int(len(surface.pair_left))
    boundary_pair_count = int(k * (len(surface.labels) - k))
    nonboundary_pair_count = n_pairs - boundary_pair_count
    global_point = float(evaluation["global_d_adj_eval"].mean())
    boundary_point = float(evaluation["boundary_d_adj_eval"].mean())
    nonboundary_point = float(evaluation["nonboundary_d_adj_eval"].mean())
    pair_fraction = float(boundary_pair_count / n_pairs)
    weighted_global = pair_fraction * boundary_point + (1.0 - pair_fraction) * nonboundary_point
    if abs(weighted_global - global_point) > 1e-12:
        raise AssertionError("global D_adj is not reconstructed by boundary/nonboundary pair means")
    if abs(global_point - canonical["global_d_adj"]) > 1e-10:
        raise AssertionError(
            f"cross-fit global D_adj disagrees with canonical summary for "
            f"{surface.source_environment_id}->{surface.target_environment_id} {surface.metric}: "
            f"{global_point} vs {canonical['global_d_adj']}"
        )
    interval = _paired_seed_bootstrap(
        evaluation,
        draws=draws,
        seed=_stable_seed(surface.source_environment_id, surface.target_environment_id, surface.metric, budget_fraction),
    )
    boundary_enrichment = _safe_ratio(boundary_point, global_point)
    boundary_to_nonboundary = _safe_ratio(boundary_point, nonboundary_point)
    boundary_contribution = _safe_ratio(pair_fraction * boundary_point, global_point)
    row: dict[str, Any] = {
        "source_environment_id": surface.source_environment_id,
        "target_environment_id": surface.target_environment_id,
        "transfer_id": f"{surface.source_environment_id}->{surface.target_environment_id}",
        "source_context_id": surface.source_environment_id,
        "unordered_context_pair_id": "<->".join(sorted((surface.source_environment_id, surface.target_environment_id))),
        "left_target_environment_id": surface.left_target_environment_id,
        "right_target_environment_id": surface.right_target_environment_id,
        "metric": surface.metric,
        "cell_budget_label": FULL_DEPTH,
        "universe_mode": FIXED_UNIVERSE,
        "budget_fraction": float(budget_fraction),
        "k": int(k),
        "n_items": int(len(surface.labels)),
        "n_pairs_global": n_pairs,
        "n_pairs_boundary": boundary_pair_count,
        "n_pairs_nonboundary": nonboundary_pair_count,
        "boundary_pair_fraction": pair_fraction,
        "global_d_adj": float(canonical["global_d_adj"]),
        "global_d_adj_ci_low": float(canonical["global_d_adj_ci_low"]),
        "global_d_adj_ci_high": float(canonical["global_d_adj_ci_high"]),
        "global_d_adj_seed_q05": interval["global_d_adj"][0],
        "global_d_adj_seed_q95": interval["global_d_adj"][1],
        "boundary_d_adj": boundary_point,
        "boundary_d_adj_ci_low": interval["boundary_d_adj"][0],
        "boundary_d_adj_ci_high": interval["boundary_d_adj"][1],
        "boundary_d_adj_seed_q05": interval["boundary_d_adj"][0],
        "boundary_d_adj_seed_q95": interval["boundary_d_adj"][1],
        "nonboundary_d_adj": nonboundary_point,
        "nonboundary_d_adj_ci_low": interval["nonboundary_d_adj"][0],
        "nonboundary_d_adj_ci_high": interval["nonboundary_d_adj"][1],
        "nonboundary_d_adj_seed_q05": interval["nonboundary_d_adj"][0],
        "nonboundary_d_adj_seed_q95": interval["nonboundary_d_adj"][1],
        "boundary_excess_d_adj": float(boundary_point - nonboundary_point),
        "boundary_excess_d_adj_ci_low": interval["boundary_excess_d_adj"][0],
        "boundary_excess_d_adj_ci_high": interval["boundary_excess_d_adj"][1],
        "boundary_enrichment": boundary_enrichment,
        "boundary_enrichment_ci_low": interval["boundary_enrichment"][0],
        "boundary_enrichment_ci_high": interval["boundary_enrichment"][1],
        "boundary_to_nonboundary_ratio": boundary_to_nonboundary,
        "boundary_to_nonboundary_ratio_ci_low": interval["boundary_to_nonboundary_ratio"][0],
        "boundary_to_nonboundary_ratio_ci_high": interval["boundary_to_nonboundary_ratio"][1],
        "boundary_contribution_fraction": boundary_contribution,
        "deterministic_boundary_inversion": float(evaluation["deterministic_boundary_inversion_eval"].mean()),
        "boundary_inversion": float(evaluation["deterministic_boundary_inversion_eval"].mean()),
        "deterministic_boundary_inversion_ci_low": interval["deterministic_boundary_inversion"][0],
        "deterministic_boundary_inversion_ci_high": interval["deterministic_boundary_inversion"][1],
        "boundary_inversion_ci_low": interval["deterministic_boundary_inversion"][0],
        "boundary_inversion_ci_high": interval["deterministic_boundary_inversion"][1],
        "shortlist_retention": float(evaluation["shortlist_retention_eval"].mean()),
        "retention": float(evaluation["shortlist_retention_eval"].mean()),
        "shortlist_retention_ci_low": interval["shortlist_retention"][0],
        "shortlist_retention_ci_high": interval["shortlist_retention"][1],
        "retention_ci_low": interval["shortlist_retention"][0],
        "retention_ci_high": interval["shortlist_retention"][1],
        "normalized_regret": float(evaluation["normalized_regret_eval"].mean()),
        "normalized_regret_ci_low": interval["normalized_regret"][0],
        "normalized_regret_ci_high": interval["normalized_regret"][1],
        "mis_selection_fraction": float(evaluation["mis_selection_fraction_eval"].mean()),
        "mis_selection_fraction_ci_low": interval["mis_selection_fraction"][0],
        "mis_selection_fraction_ci_high": interval["mis_selection_fraction"][1],
        "crossfit_split_count": 2,
        "crossfit_fit_seed_count": CROSSFIT_HALF,
        "crossfit_eval_seed_count": CROSSFIT_HALF,
        "measurement_replicate_count": N_MEASUREMENT_REPLICATES,
        "uncertainty_draws": int(draws),
        "uncertainty_level": CI_LEVEL,
        "global_d_adj_interval_source": "canonical full-depth matched-fixed summary; existing per-seed 90% interval",
        "conditional_interval_source": "paired 30-seed q05/q95 bootstrap conditional on cross-fit source shortlist",
        "source_shortlist_only": True,
        "target_outcomes_used_for_pair_set": False,
        "boundary_mechanism_supported": bool(interval["boundary_enrichment"][0] > 1.0 and interval["boundary_excess_d_adj"][0] > 0.0),
        "status": "executed",
    }
    # Attach source-only audit fingerprints to the row without materialising
    # target-derived labels or pair values in the CSV.
    row["source_shortlist_sha256"] = hashlib.sha256(
        "\n".join(item["shortlist_sha256"] for item in audit).encode("utf-8")
    ).hexdigest()
    row["boundary_pair_sha256"] = hashlib.sha256(
        "\n".join(item["boundary_pair_sha256"] for item in audit).encode("utf-8")
    ).hexdigest()
    row["source_shortlist_split_audit"] = "fit_first15_eval_last15;fit_last15_eval_first15"
    return row, evaluation, audit


def _jsonable(value: Any) -> Any:
    """Convert NumPy/Pandas values and non-finite floats to JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def build(root: Path = ROOT, *, draws: int = DEFAULT_BOOTSTRAP_DRAWS) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build and write the source-boundary transport products."""

    root = root.resolve()
    if int(draws) < 1:
        raise ValueError("bootstrap draws must be positive")
    manifests = root / "artifacts/manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    risks = _read_full_fixed_risks(root)
    canonical = _read_canonical_summary(root)
    split_seeds = tuple(sorted(int(value) for value in risks["split_seed"].unique()))
    if len(split_seeds) != N_SPLIT_SEEDS:
        raise ValueError("fixed risk input does not have the declared 30-seed schedule")

    surfaces: list[SurfaceData] = []
    surface_keys: set[tuple[str, str, str, str]] = set()
    group_columns = [
        "source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric",
    ]
    for key, group in risks.groupby(group_columns, sort=True, observed=True):
        source, left, right, metric = map(str, key)
        if source not in (left, right):
            continue
        surface_key = (source, left, right, metric)
        if surface_key not in canonical:
            raise ValueError(f"missing canonical full-depth summary for {surface_key}")
        surfaces.append(_prepare_surface(
            group,
            source_environment_id=source,
            left_target_environment_id=left,
            right_target_environment_id=right,
            metric=metric,
            split_seeds=split_seeds,
        ))
        surface_keys.add(surface_key)
    if len(surfaces) != 18 or len(surface_keys) != 18:
        raise ValueError(f"expected exactly 18 directed transfer-by-metric surfaces, found {len(surfaces)}")
    transfers = {(surface.source_environment_id, surface.target_environment_id) for surface in surfaces}
    if len(transfers) != 6:
        raise ValueError(f"expected six directed transfers, found {len(transfers)}")
    for source, target in transfers:
        if (target, source) not in transfers:
            raise ValueError("directed surface does not contain both directions for every context pair")
    if {surface.metric for surface in surfaces} != set(METRICS):
        raise ValueError("directed surface does not contain exactly the frozen metric set")

    rows: list[dict[str, Any]] = []
    selection_audit: list[dict[str, Any]] = []
    for surface in sorted(surfaces, key=lambda item: (item.source_environment_id, item.target_environment_id, METRICS.index(item.metric))):
        canonical_row = canonical[(
            surface.source_environment_id,
            surface.left_target_environment_id,
            surface.right_target_environment_id,
            surface.metric,
        )]
        for budget_fraction in BUDGET_FRACTIONS:
            row, _evaluation, audit = _row_for_surface_budget(
                surface,
                budget_fraction,
                canonical_row,
                draws=int(draws),
            )
            rows.append(row)
            for record in audit:
                selection_audit.append({
                    "source_environment_id": surface.source_environment_id,
                    "target_environment_id": surface.target_environment_id,
                    "transfer_id": f"{surface.source_environment_id}->{surface.target_environment_id}",
                    "metric": surface.metric,
                    "budget_fraction": float(budget_fraction),
                    "k": int(row["k"]),
                    **record,
                })
    table = pd.DataFrame(rows)
    key_columns = ["source_environment_id", "target_environment_id", "metric", "budget_fraction"]
    if len(table) != 72 or table.duplicated(key_columns).any():
        raise AssertionError(f"selection-boundary output must have 72 unique surface-budget rows, found {len(table)}")
    if set(table["budget_fraction"].astype(float)) != set(BUDGET_FRACTIONS):
        raise AssertionError("selection-boundary output does not contain the four frozen budgets")
    if not np.isfinite(table[["global_d_adj", "boundary_d_adj", "nonboundary_d_adj", "boundary_enrichment", "deterministic_boundary_inversion", "shortlist_retention", "normalized_regret", "mis_selection_fraction"]].to_numpy(dtype=float)).all():
        raise AssertionError("selection-boundary output contains non-finite primary estimates")
    max_reconstruction_error = float(
        np.max(np.abs(
            table["global_d_adj"].to_numpy(dtype=float)
            - table["boundary_pair_fraction"].to_numpy(dtype=float) * table["boundary_d_adj"].to_numpy(dtype=float)
            - (1.0 - table["boundary_pair_fraction"].to_numpy(dtype=float)) * table["nonboundary_d_adj"].to_numpy(dtype=float)
        ))
    )
    if max_reconstruction_error > 1e-12:
        raise AssertionError(f"boundary/nonboundary reconstruction error is {max_reconstruction_error}")
    canonical_errors = [
        abs(float(row["global_d_adj"]) - canonical[(
            str(row["source_environment_id"]),
            str(row["left_target_environment_id"]),
            str(row["right_target_environment_id"]),
            str(row["metric"]),
        )]["global_d_adj"])
        for row in rows
    ]
    output_csv = root / OUTPUT_CSV
    output_json = root / OUTPUT_JSON
    table.to_csv(output_csv, index=False)

    point_positive = table["boundary_enrichment"].astype(float) > 1.0
    interval_supported = table["boundary_mechanism_supported"].astype(bool)
    support_by_budget = {
        f"{budget_fraction:g}": {
            "rows": int(np.sum(np.isclose(table["budget_fraction"].astype(float), budget_fraction))),
            "point_boundary_enrichment_gt_1": int(np.sum(point_positive & np.isclose(table["budget_fraction"].astype(float), budget_fraction))),
            "conditional_90pct_boundary_support": int(np.sum(interval_supported & np.isclose(table["budget_fraction"].astype(float), budget_fraction))),
        }
        for budget_fraction in BUDGET_FRACTIONS
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "executed",
        "analysis": "source_side_top_k_boundary_pair_transport",
        "mechanism_supported": bool(interval_supported.all()),
        "mechanism_support_status": "supported_as_surface_wide_mechanism" if bool(interval_supported.all()) else "not_supported_as_surface_wide_mechanism",
        "mechanism_support_definition": "A row is conditionally supported only when the paired held-out-seed 90% interval has boundary enrichment > 1 and positive boundary-minus-nonboundary D_adj; the surface-wide flag requires all 72 prespecified rows to pass. This is descriptive, not a new release gate or simultaneous claim test.",
        "mechanism_support_counts": {
            "total_rows": int(len(table)),
            "point_boundary_enrichment_gt_1": int(point_positive.sum()),
            "conditional_90pct_boundary_support": int(interval_supported.sum()),
            "by_budget": support_by_budget,
        },
        "surface": {
            "directed_transfer_metric_rows": 18,
            "directed_transfer_count": 6,
            "metric_count": 3,
            "semantics": "three unordered context pairs x two directions x three frozen metrics; each row is repeated at four top-k budgets",
            "output_rows": int(len(table)),
            "budgets": list(BUDGET_FRACTIONS),
        },
        "fixed_data_protocol": {
            "risk_input": RISK_INPUT,
            "canonical_summary": CANONICAL_SUMMARY,
            "cell_budget_label": FULL_DEPTH,
            "universe_mode": FIXED_UNIVERSE,
            "split_seeds": list(split_seeds),
            "split_seed_count": len(split_seeds),
            "measurement_replicates": [0, 1],
            "measurement_replicate_count": N_MEASUREMENT_REPLICATES,
            "crossfit": "fit on one source-only 15-seed half with both measurement replicates; evaluate on the disjoint 15-seed half; repeat after swapping halves",
            "tie_tolerance": TIE_TOLERANCE,
            "top_k_rule": "max(1, floor(budget_fraction*n_items)); lowest source risk; ties by fixed label order",
            "label_order": "lexicographic perturbation_label order, matching the existing fixed decision-table construction",
        },
        "estimands": {
            "global_d_adj": "canonical per-seed U-corrected pairwise order disagreement on all unordered pairs",
            "boundary_pair_set": "all unordered pairs with exactly one item in the source-only top-k shortlist",
            "boundary_d_adj": "mean per-pair U-corrected D_adj over source-selected versus source-outside pairs, evaluated on held-out source/target seeds",
            "nonboundary_d_adj": "mean per-pair U-corrected D_adj over the complement of the frozen boundary pair set",
            "boundary_enrichment": "boundary_d_adj/global_d_adj; equivalently boundary D_adj contribution share divided by boundary pair share",
            "boundary_to_nonboundary_ratio": "boundary_d_adj/nonboundary_d_adj, retained as a separate ratio",
            "deterministic_boundary_inversion": "strict target-risk inversions among selected-versus-outside pairs, with ties within the fixed tolerance not counted",
            "shortlist_retention": "fraction of the target top-k oracle retained by the frozen source shortlist",
            "normalized_regret": "(target mean risk of frozen shortlist - target oracle mean risk)/(target all-item mean risk - target oracle mean risk)",
            "mis_selection_fraction": "source-shortlist items absent from the target oracle divided by k",
        },
        "information_firewall": {
            "source_shortlist_selection_uses_target_outcomes": False,
            "boundary_pair_set_uses_target_outcomes": False,
            "target_outcomes_used_only_for": ["held-out target order states in D_adj", "target-side oracle and decision evaluation"],
            "source_fit_target_leakage": False,
        },
        "uncertainty": {
            "confidence_level": CI_LEVEL,
            "draws": int(draws),
            "global_d_adj_intervals": "copied from the existing canonical full-depth matched-fixed summary; no new global estimand",
            "boundary_and_decision_intervals": "paired 30-seed q05/q95 bootstrap, conditional on each cross-fit source shortlist; both split assignments contribute one held-out record per seed",
            "selection_uncertainty": "not included in the conditional intervals; the source shortlist is intentionally frozen within each cross-fit split",
            "simultaneous_claim": False,
        },
        "checks": {
            "rows": int(len(table)),
            "unique_surface_budget_keys": bool(not table.duplicated(key_columns).any()),
            "global_boundary_nonboundary_reconstruction_max_abs_error": max_reconstruction_error,
            "max_global_d_adj_difference_from_canonical": float(max(canonical_errors)),
            "boundary_pair_formula_holds": bool((table["n_pairs_boundary"].astype(int) == table["k"].astype(int) * (table["n_items"].astype(int) - table["k"].astype(int))).all()),
            "pair_partition_holds": bool((table["n_pairs_boundary"].astype(int) + table["n_pairs_nonboundary"].astype(int) == table["n_pairs_global"].astype(int)).all()),
            "target_outcomes_not_used_for_pair_set": True,
            "source_shortlists_cross_fitted": True,
            "primary_estimates_finite": True,
        },
        "source_shortlist_audit": selection_audit,
        "outputs": {
            "csv": _relative(output_csv, root),
            "json": _relative(output_json, root),
        },
    }
    output_json.write_text(
        json.dumps(_jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return table, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    table, report = build(args.root, draws=args.draws)
    print(json.dumps(_jsonable(report), indent=2, ensure_ascii=False, allow_nan=False))
    print(table[[
        "transfer_id", "metric", "budget_fraction", "k", "n_items",
        "global_d_adj", "boundary_d_adj", "nonboundary_d_adj", "boundary_enrichment",
        "deterministic_boundary_inversion", "shortlist_retention", "normalized_regret", "mis_selection_fraction",
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
