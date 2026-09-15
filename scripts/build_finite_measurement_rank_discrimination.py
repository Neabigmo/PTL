"""Compare rank comparators with a matched same-context pseudo-context null.

The canonical ``measurement_floor`` is a raw-cell measurement component of the
canonical ordering analysis.  It is not a same-context value for ``D_adj`` and
is never used as one here.

For every directed transfer, metric, and depth, the 30 canonical split seeds
are divided into two disjoint 15-seed partitions.  Both measurement replicates
within a seed remain together.  The cross estimate is the mean of
``source-A -> target-B`` and ``source-B -> target-A``.  The same-context null is
the mean of ``source-A -> source-B`` and ``target-A -> target-B``.  This keeps
the cross and null estimates matched in seed/replicate sample size and uses no
target refit or raw-data recomputation.

The CSV is a 30-row macro table (18 directed transfer-by-metric surfaces for
each of five comparators at six depths). The JSON keeps the surface-level
diagnostics, including the per-surface interval used for the lower-CI fraction.
A separate 540-row matched-unit audit covers all five comparator families at
the same surface-by-depth resolution without replacing the primary estimator.
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

from scripts.build_rank_comparator_benchmark import (  # noqa: E402
    DEPTHS,
    METRICS,
    _matrix,
    _normalize_budget_labels,
    _rank_values,
    _read_risks,
)
from scripts.decision_analysis_common import CANONICAL_SUMMARY  # noqa: E402
from src.evaluation.ordering_estimands import (  # noqa: E402
    measurement_identifiable_ordering_divergence,
    pairwise_order_probabilities,
    stable_order_mask,
)


COMPARATORS = (
    "kendall_distance",
    "spearman_distance",
    "top10_jaccard_distance",
    "stable_pair_inversion_fraction",
    "d_adj",
)
RANK_COMPARATORS = COMPARATORS[:3]
STABLE_COMPARATOR = "stable_pair_inversion_fraction"
N_SEEDS = 30
N_PARTITION_SEEDS = N_SEEDS // 2
MEASUREMENT_REPLICATES = (0, 1)
MINIMUM_STRICT_SUPPORT = 8
STABLE_CREDIBLE_LEVEL = 0.95
BOOTSTRAP_CI_LEVEL = 0.90
DEFAULT_BOOTSTRAP_DRAWS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260914
NORMALIZATION_EPSILON = 1e-12


def _mean_or_nan(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_mean(values: pd.Series | np.ndarray | list[float]) -> float:
    return _mean_or_nan(np.asarray(values, dtype=float))


def _direction(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    return np.where(
        values[:, 2] > values[:, 0],
        1,
        np.where(values[:, 0] > values[:, 2], -1, 0),
    ).astype(np.int8)


def _d_adj(left: np.ndarray, right: np.ndarray) -> float:
    """Return the U-corrected pair-state ``D_adj`` for two pseudo-contexts."""

    _, probabilities_left = pairwise_order_probabilities(left)
    _, probabilities_right = pairwise_order_probabilities(right)
    value, _, _ = measurement_identifiable_ordering_divergence(
        probabilities_left,
        probabilities_right,
        n_replicates_left=int(left.shape[0]),
        n_replicates_right=int(right.shape[0]),
    )
    # Do not clip finite-sample U estimates.  A negative value is a valid
    # finite-sample result and is part of the diagnostic requested here.
    return float(value)


def _seed_block_mean_d_adj(left: np.ndarray, right: np.ndarray) -> float:
    """Average per-seed U estimates without flattening seed blocks.

    This is an estimator-unit audit only.  Each aligned seed contributes one
    two-replicate U estimate, and the 15 seed-block estimates are averaged.
    It is deliberately not substituted for the primary 15/15 pseudo-context
    construction, because the two constructions target different finite-sample
    summaries.
    """

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    expected = (N_PARTITION_SEEDS, len(MEASUREMENT_REPLICATES), left.shape[-1] if left.ndim else 0)
    if left.ndim != 3 or right.ndim != 3 or left.shape != right.shape:
        raise ValueError("seed-block audit inputs must have equal [15, 2, item] shape")
    if left.shape[:2] != expected[:2]:
        raise ValueError("seed-block audit requires 15 seeds and two measurement replicates")
    values = [_d_adj(left[index], right[index]) for index in range(N_PARTITION_SEEDS)]
    return _mean_or_nan(values)


def _pair_components(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Compute all non-stable comparator values for one independent pair."""

    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("pair comparator inputs must have equal [replicate, item] shape")
    rank = _rank_values(left.mean(axis=0), right.mean(axis=0))
    return {
        "kendall_distance": float(rank["kendall_distance"]),
        "spearman_distance": float(rank["spearman_distance"]),
        "top10_jaccard_distance": float(rank["top10_jaccard_distance"]),
        "d_adj": _d_adj(left, right),
    }


def _matched_unit_components(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Compute the all-comparator audit on 15 aligned seed-block units.

    The two measurement replicates within each seed are first reduced to one
    block-level risk vector for rank comparators. ``D_adj`` keeps the
    measurement unit explicit by applying the order-2 U correction within
    every aligned two-replicate block and averaging those 15 estimates. For
    the stable comparator, the 15 block vectors are the replicate units and
    therefore retain the frozen minimum-support rule.

    This is a sensitivity artifact, not a replacement for the primary
    flattened 15-by-2 partition estimator.
    """

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    expected = (N_PARTITION_SEEDS, len(MEASUREMENT_REPLICATES))
    if left.ndim != 3 or right.ndim != 3 or left.shape != right.shape:
        raise ValueError("matched-unit inputs must have equal [15, 2, item] shape")
    if left.shape[:2] != expected:
        raise ValueError("matched-unit audit requires 15 seeds and two measurement replicates")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("matched-unit audit inputs contain non-finite values")

    left_blocks = left.mean(axis=1)
    right_blocks = right.mean(axis=1)
    rank = _rank_values(left_blocks.mean(axis=0), right_blocks.mean(axis=0))
    _, left_probabilities = pairwise_order_probabilities(left_blocks)
    _, right_probabilities = pairwise_order_probabilities(right_blocks)
    stable_left, _, pairs = stable_order_mask(
        left_blocks,
        credible_level=STABLE_CREDIBLE_LEVEL,
        minimum_strict_support=MINIMUM_STRICT_SUPPORT,
    )
    stable_right, _, right_pairs = stable_order_mask(
        right_blocks,
        credible_level=STABLE_CREDIBLE_LEVEL,
        minimum_strict_support=MINIMUM_STRICT_SUPPORT,
    )
    if not np.array_equal(pairs, right_pairs):
        raise AssertionError("matched-unit stable pair indexing differs across contexts")
    stable_both = np.asarray(stable_left, dtype=bool) & np.asarray(stable_right, dtype=bool)
    left_sign = _direction(left_probabilities)
    right_sign = _direction(right_probabilities)
    evaluable = stable_both & (left_sign != 0) & (right_sign != 0)
    inversion = evaluable & (left_sign != right_sign)
    n_pairs = max(len(pairs), 1)
    return {
        "kendall_distance": float(rank["kendall_distance"]),
        "spearman_distance": float(rank["spearman_distance"]),
        "top10_jaccard_distance": float(rank["top10_jaccard_distance"]),
        "stable_pair_inversion_fraction": float(np.sum(inversion) / np.sum(evaluable)) if np.any(evaluable) else float("nan"),
        "d_adj": _seed_block_mean_d_adj(left, right),
        "evaluable_coverage": float(np.sum(evaluable) / n_pairs),
        "stable_pair_coverage": float(np.sum(stable_both) / n_pairs),
    }


def _prepare_stable(values: np.ndarray) -> list[dict[str, np.ndarray | int]]:
    """Prepare seed-level cross-fit pieces for one 15-seed pseudo-context.

    The two fit/evaluation directions split seed blocks, rather than individual
    pair rows.  With 15 seeds this gives 7 fit seeds versus 8 evaluation seeds
    and then the reverse.  Both measurement replicates stay within their seed
    block, so selection and evaluation do not share a seed.
    """

    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[0] != N_PARTITION_SEEDS or values.shape[1] != len(MEASUREMENT_REPLICATES):
        raise ValueError(
            "stable pseudo-context must have shape [15, 2, item] after the seed partition"
        )
    if not np.isfinite(values).all():
        raise ValueError("stable pseudo-context contains non-finite values")
    n_items = int(values.shape[2])
    n_pairs = n_items * (n_items - 1) // 2
    split = N_PARTITION_SEEDS // 2
    pieces: list[dict[str, np.ndarray | int]] = []
    for fit_slice, eval_slice in (
        (slice(0, split), slice(split, None)),
        (slice(split, None), slice(0, split)),
    ):
        fit = values[fit_slice].reshape(-1, n_items)
        evaluation = values[eval_slice].reshape(-1, n_items)
        stable, _, pairs = stable_order_mask(
            fit,
            credible_level=STABLE_CREDIBLE_LEVEL,
            minimum_strict_support=MINIMUM_STRICT_SUPPORT,
        )
        _, fit_probabilities = pairwise_order_probabilities(fit)
        _, eval_probabilities = pairwise_order_probabilities(evaluation)
        pieces.append({
            "pairs": pairs,
            "stable": stable,
            "fit_sign": _direction(fit_probabilities),
            "eval_sign": _direction(eval_probabilities),
            "n_pairs": n_pairs,
        })
    return pieces


def _stable_compare(
    left: list[dict[str, np.ndarray | int]],
    right: list[dict[str, np.ndarray | int]],
) -> dict[str, float | int]:
    """Combine two prepared pseudo-contexts with held-out stable inversion."""

    if len(left) != 2 or len(right) != 2:
        raise ValueError("stable cross-fit requires two fit/evaluation directions")
    evaluable_counts: list[int] = []
    inversion_counts: list[int] = []
    stable_counts: list[int] = []
    n_pairs = int(left[0]["n_pairs"])
    for left_piece, right_piece in zip(left, right):
        if not np.array_equal(left_piece["pairs"], right_piece["pairs"]):
            raise AssertionError("stable pair indexing differs across pseudo-contexts")
        stable_both = np.asarray(left_piece["stable"], dtype=bool) & np.asarray(right_piece["stable"], dtype=bool)
        eval_left = np.asarray(left_piece["eval_sign"], dtype=np.int8)
        eval_right = np.asarray(right_piece["eval_sign"], dtype=np.int8)
        evaluable = stable_both & (eval_left != 0) & (eval_right != 0)
        inversion = evaluable & (eval_left != eval_right)
        stable_counts.append(int(np.sum(stable_both)))
        evaluable_counts.append(int(np.sum(evaluable)))
        inversion_counts.append(int(np.sum(inversion)))
    evaluable_total = int(np.sum(evaluable_counts))
    return {
        "stable_pair_inversion_fraction": (
            float(np.sum(inversion_counts) / evaluable_total) if evaluable_total else float("nan")
        ),
        "evaluable_coverage": float(np.mean(np.asarray(evaluable_counts, dtype=float) / n_pairs)),
        "stable_pair_coverage": float(np.mean(np.asarray(stable_counts, dtype=float) / n_pairs)),
        "evaluable_pairs": evaluable_total,
        "stable_pairs": int(np.sum(stable_counts)),
    }


def _stable_seed(*parts: str | int) -> int:
    """Derive a reproducible local RNG seed without adding a hash contract."""

    value = int(DEFAULT_BOOTSTRAP_SEED)
    for part in parts:
        for character in str(part):
            value = (value * 131 + ord(character)) % (2**32)
    return int(value)


def _orientation_bootstrap(
    cross_values: list[float],
    null_values: list[float],
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap the two independent cross orientations and two null terms.

    The fixed A/B split produces two independent cross orientations and two
    same-context terms.  Resampling those terms is deliberately reported as a
    partition bootstrap; it is not mislabeled as the canonical raw-cell
    measurement-floor bootstrap.
    """

    cross = np.asarray(cross_values, dtype=float)
    null = np.asarray(null_values, dtype=float)
    if cross.shape != (2,) or null.shape != (2,) or not np.isfinite(cross).all() or not np.isfinite(null).all():
        return {
            "cross_ci_low": float("nan"),
            "cross_ci_high": float("nan"),
            "null_ci_low": float("nan"),
            "null_ci_high": float("nan"),
            "margin_ci_low": float("nan"),
            "margin_ci_high": float("nan"),
            "bootstrap_valid_draws": 0,
        }
    if int(draws) < 100:
        raise ValueError("bootstrap draws must be at least 100")
    rng = np.random.default_rng(int(seed))
    cross_index = rng.integers(0, 2, size=(int(draws), 2))
    null_index = rng.integers(0, 2, size=(int(draws), 2))
    cross_draws = cross[cross_index].mean(axis=1)
    null_draws = null[null_index].mean(axis=1)
    margins = cross_draws - null_draws
    low = (1.0 - BOOTSTRAP_CI_LEVEL) / 2.0
    high = 1.0 - low
    return {
        "cross_ci_low": float(np.quantile(cross_draws, low)),
        "cross_ci_high": float(np.quantile(cross_draws, high)),
        "null_ci_low": float(np.quantile(null_draws, low)),
        "null_ci_high": float(np.quantile(null_draws, high)),
        "margin_ci_low": float(np.quantile(margins, low)),
        "margin_ci_high": float(np.quantile(margins, high)),
        "bootstrap_valid_draws": int(len(margins)),
    }


def _surface_pair_id(source: str, target: str) -> str:
    return "<->".join(sorted((str(source), str(target))))


def _macro_block_bootstrap(
    group: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap the macro surface mean by unordered context-pair blocks."""

    blocks: list[pd.DataFrame] = []
    for _, block in group.groupby("unordered_context_pair_id", sort=True, observed=True):
        block = block.loc[np.isfinite(block["cross_minus_null_margin"].to_numpy(dtype=float))].copy()
        if block.empty:
            continue
        blocks.append(block)
    if len(blocks) < 2 or int(draws) < 100:
        return {
            "cross_ci_low": float("nan"),
            "cross_ci_high": float("nan"),
            "null_ci_low": float("nan"),
            "null_ci_high": float("nan"),
            "margin_ci_low": float("nan"),
            "margin_ci_high": float("nan"),
            "bootstrap_valid_draws": 0,
            "bootstrap_blocks": int(len(blocks)),
        }
    # The canonical surface has six rows in every unordered-pair block for a
    # comparator/depth key.  Equal block sizes make the block mean an exact
    # surface mean after block resampling.
    if len({len(block) for block in blocks}) != 1:
        raise RuntimeError("unordered context-pair blocks are not matched in size")
    block_cross = np.asarray([block["cross_estimate"].mean() for block in blocks], dtype=float)
    block_null = np.asarray([block["same_context_null"].mean() for block in blocks], dtype=float)
    block_margin = np.asarray([block["cross_minus_null_margin"].mean() for block in blocks], dtype=float)
    if not np.isfinite(np.column_stack((block_cross, block_null, block_margin))).all():
        return {
            "cross_ci_low": float("nan"),
            "cross_ci_high": float("nan"),
            "null_ci_low": float("nan"),
            "null_ci_high": float("nan"),
            "margin_ci_low": float("nan"),
            "margin_ci_high": float("nan"),
            "bootstrap_valid_draws": 0,
            "bootstrap_blocks": int(len(blocks)),
        }
    rng = np.random.default_rng(int(seed))
    index = rng.integers(0, len(blocks), size=(int(draws), len(blocks)))
    cross_draws = block_cross[index].mean(axis=1)
    null_draws = block_null[index].mean(axis=1)
    margin_draws = block_margin[index].mean(axis=1)
    low = (1.0 - BOOTSTRAP_CI_LEVEL) / 2.0
    high = 1.0 - low
    return {
        "cross_ci_low": float(np.quantile(cross_draws, low)),
        "cross_ci_high": float(np.quantile(cross_draws, high)),
        "null_ci_low": float(np.quantile(null_draws, low)),
        "null_ci_high": float(np.quantile(null_draws, high)),
        "margin_ci_low": float(np.quantile(margin_draws, low)),
        "margin_ci_high": float(np.quantile(margin_draws, high)),
        "bootstrap_valid_draws": int(len(margin_draws)),
        "bootstrap_blocks": int(len(blocks)),
    }


def _read_canonical_surface(root: Path) -> tuple[dict[tuple[str, str, str, str], float], dict[str, Any]]:
    """Load the canonical D_adj depth surface and audit floor provenance."""

    path = root / CANONICAL_SUMMARY
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = _normalize_budget_labels(pd.read_csv(path, dtype={"cell_budget_label": "string"}))
    required = {
        "source_environment_id",
        "left_target_environment_id",
        "right_target_environment_id",
        "metric",
        "cell_budget_label",
        "universe_mode",
        "identifiable_divergence",
        "measurement_floor",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"canonical summary is missing columns: {missing}")
    frame = frame.loc[frame["metric"].isin(METRICS)].copy()
    frame = frame.loc[
        frame["source_environment_id"].eq(frame["left_target_environment_id"])
        | frame["source_environment_id"].eq(frame["right_target_environment_id"])
    ].copy()
    frame["target_environment_id"] = np.where(
        frame["source_environment_id"].eq(frame["left_target_environment_id"]),
        frame["right_target_environment_id"],
        frame["left_target_environment_id"],
    )
    key_columns = ["source_environment_id", "target_environment_id", "metric", "cell_budget_label"]
    expected = 18 * len(DEPTHS)
    if len(frame) != expected or frame.duplicated(key_columns).any():
        raise RuntimeError(f"canonical endpoint depth surface must have {expected} unique rows")
    if set(frame["cell_budget_label"].astype(str)) != set(DEPTHS):
        raise RuntimeError("canonical endpoint depth surface has an unexpected depth set")
    if set(frame["universe_mode"].astype(str)) != {"matched_fixed"}:
        raise RuntimeError("canonical endpoint depth surface is not matched_fixed")
    numeric = frame[["identifiable_divergence", "measurement_floor"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("canonical endpoint D_adj/floor values must be finite")
    d_adj = {
        tuple(str(row._asdict()[column]) for column in key_columns): float(row.identifiable_divergence)
        for row in frame.itertuples(index=False)
    }
    audit = {
        "source": path.relative_to(root).as_posix(),
        "d_adj_column": "identifiable_divergence",
        "measurement_floor_column": "measurement_floor",
        "measurement_floor_loaded_for_provenance_only": True,
        "measurement_floor_used_as_same_context_d_adj_null": False,
        "same_context_d_adj_source": "matched A/B pseudo-context U-corrected pair-state estimates from the risk grid",
        "rows_checked": int(len(frame)),
    }
    return d_adj, audit


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def build(
    root: Path = ROOT,
    *,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    risk_path = manifests / "reliability_transport_measurement_depth_matched_fixed_risks.csv"
    d_adj_map, canonical_audit = _read_canonical_surface(root)
    risks = _read_risks(risk_path)
    seeds = sorted(int(value) for value in risks["split_seed"].unique())
    if len(seeds) != N_SEEDS:
        raise ValueError(f"expected {N_SEEDS} split seeds, found {len(seeds)}")
    if sorted(risks["measurement_replicate"].astype(int).unique()) != list(MEASUREMENT_REPLICATES):
        raise ValueError("risk grid must contain exactly measurement replicates 0 and 1")
    if set(risks["universe_mode"].astype(str)) != {"matched_fixed"}:
        raise ValueError("finite rank discrimination requires matched_fixed risks")

    seed_a = seeds[:N_PARTITION_SEEDS]
    seed_b = seeds[N_PARTITION_SEEDS:]
    surface_rows: list[dict[str, Any]] = []
    estimator_unit_audit_rows: list[dict[str, Any]] = []
    matched_unit_audit_rows: list[dict[str, Any]] = []
    surface_keys: set[tuple[str, str, str]] = set()
    for _, group in risks.groupby(
        ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"],
        sort=True,
        observed=True,
    ):
        source = str(group["source_environment_id"].iloc[0])
        left = str(group["left_target_environment_id"].iloc[0])
        right = str(group["right_target_environment_id"].iloc[0])
        metric = str(group["metric"].iloc[0])
        if source not in {left, right}:
            continue
        target = right if source == left else left
        surface_key = (source, target, metric)
        if surface_key in surface_keys:
            raise RuntimeError(f"duplicate directed surface {surface_key}")
        surface_keys.add(surface_key)
        for depth in DEPTHS:
            source_labels, source_values = _matrix(group, source, seeds, depth)
            target_labels, target_values = _matrix(group, target, seeds, depth)
            if set(source_labels) != set(target_labels):
                raise RuntimeError(f"source/target labels are not matched at {surface_key} depth={depth}")
            common = sorted(set(source_labels) & set(target_labels))
            source_values = source_values[:, :, [source_labels.index(label) for label in common]]
            target_values = target_values[:, :, [target_labels.index(label) for label in common]]
            source_a, source_b = source_values[:N_PARTITION_SEEDS], source_values[N_PARTITION_SEEDS:]
            target_a, target_b = target_values[:N_PARTITION_SEEDS], target_values[N_PARTITION_SEEDS:]
            # The block preparation is shared by all four stable comparisons
            # for this surface/depth, avoiding repeated selection work.
            stable_source_a = _prepare_stable(source_a)
            stable_source_b = _prepare_stable(source_b)
            stable_target_a = _prepare_stable(target_a)
            stable_target_b = _prepare_stable(target_b)

            pair_inputs = {
                "cross_ab": (source_a.reshape(-1, len(common)), target_b.reshape(-1, len(common)), stable_source_a, stable_target_b),
                "cross_ba": (source_b.reshape(-1, len(common)), target_a.reshape(-1, len(common)), stable_source_b, stable_target_a),
                "null_source": (source_a.reshape(-1, len(common)), source_b.reshape(-1, len(common)), stable_source_a, stable_source_b),
                "null_target": (target_a.reshape(-1, len(common)), target_b.reshape(-1, len(common)), stable_target_a, stable_target_b),
            }
            pair_values: dict[str, dict[str, float]] = {}
            pair_coverage: dict[str, float] = {}
            for pair_name, (pair_left, pair_right, stable_left, stable_right) in pair_inputs.items():
                components = _pair_components(pair_left, pair_right)
                stable = _stable_compare(stable_left, stable_right)
                components[STABLE_COMPARATOR] = float(stable[STABLE_COMPARATOR])
                pair_values[pair_name] = components
                pair_coverage[pair_name] = float(stable["evaluable_coverage"])

            # Matched-unit sensitivity: retain the aligned 15 seed blocks and
            # run every comparator on the same block-level construction. This
            # is stored as a separate audit artifact so the canonical primary
            # surface remains unchanged.
            matched_inputs = {
                "cross_ab": (source_a, target_b),
                "cross_ba": (source_b, target_a),
                "null_source": (source_a, source_b),
                "null_target": (target_a, target_b),
            }
            matched_values = {
                pair_name: _matched_unit_components(pair_left, pair_right)
                for pair_name, (pair_left, pair_right) in matched_inputs.items()
            }
            for comparator in COMPARATORS:
                current_cross_terms = [pair_values["cross_ab"][comparator], pair_values["cross_ba"][comparator]]
                current_null_terms = [pair_values["null_source"][comparator], pair_values["null_target"][comparator]]
                matched_cross_terms = [matched_values["cross_ab"][comparator], matched_values["cross_ba"][comparator]]
                matched_null_terms = [matched_values["null_source"][comparator], matched_values["null_target"][comparator]]
                current_cross = _mean_or_nan(current_cross_terms)
                current_null = _mean_or_nan(current_null_terms)
                matched_cross = _mean_or_nan(matched_cross_terms)
                matched_null = _mean_or_nan(matched_null_terms)
                matched_unit_audit_rows.append({
                    "source_environment_id": source,
                    "target_environment_id": target,
                    "transfer_id": f"{source}->{target}",
                    "unordered_context_pair_id": _surface_pair_id(source, target),
                    "metric": metric,
                    "cell_budget_label": str(depth),
                    "cell_budget_order": int(DEPTHS.index(depth) + 1),
                    "comparator": comparator,
                    "current_cross_estimate": float(current_cross),
                    "current_same_context_null": float(current_null),
                    "current_cross_minus_null_margin": float(current_cross - current_null) if np.isfinite(current_cross) and np.isfinite(current_null) else float("nan"),
                    "matched_cross_ab": float(matched_values["cross_ab"][comparator]),
                    "matched_cross_ba": float(matched_values["cross_ba"][comparator]),
                    "matched_null_source": float(matched_values["null_source"][comparator]),
                    "matched_null_target": float(matched_values["null_target"][comparator]),
                    "matched_cross_estimate": float(matched_cross),
                    "matched_same_context_null": float(matched_null),
                    "matched_cross_minus_null_margin": float(matched_cross - matched_null) if np.isfinite(matched_cross) and np.isfinite(matched_null) else float("nan"),
                    "matched_cross_evaluable_coverage": float(_mean_or_nan([matched_values["cross_ab"]["evaluable_coverage"], matched_values["cross_ba"]["evaluable_coverage"]])),
                    "matched_null_evaluable_coverage": float(_mean_or_nan([matched_values["null_source"]["evaluable_coverage"], matched_values["null_target"]["evaluable_coverage"]])),
                    "matched_stable_pair_coverage_cross": float(_mean_or_nan([matched_values["cross_ab"]["stable_pair_coverage"], matched_values["cross_ba"]["stable_pair_coverage"]])),
                    "matched_stable_pair_coverage_null": float(_mean_or_nan([matched_values["null_source"]["stable_pair_coverage"], matched_values["null_target"]["stable_pair_coverage"]])),
                    "matched_unit": "15 aligned seed-block vectors; 2 measurement replicates retained within each block",
                })

            # Audit the finite-sample unit used for the same-context D_adj
            # null.  The primary estimate intentionally flattens the 15 seed
            # blocks (30 measurement replicates per side).  The companion
            # estimate preserves each aligned seed block, applies the same
            # U-correction to its two measurement replicates, and averages the
            # resulting 15 values.  Keep this as a diagnostic artifact rather
            # than changing the declared primary estimator.
            current_null_terms = [
                _d_adj(source_a.reshape(-1, len(common)), source_b.reshape(-1, len(common))),
                _d_adj(target_a.reshape(-1, len(common)), target_b.reshape(-1, len(common))),
            ]
            seed_block_null_terms = [
                _seed_block_mean_d_adj(source_a, source_b),
                _seed_block_mean_d_adj(target_a, target_b),
            ]
            estimator_unit_audit_rows.append({
                "source_environment_id": source,
                "target_environment_id": target,
                "transfer_id": f"{source}->{target}",
                "unordered_context_pair_id": _surface_pair_id(source, target),
                "metric": metric,
                "cell_budget_label": str(depth),
                "cell_budget_order": int(DEPTHS.index(depth) + 1),
                "current_flattened_null": float(_mean_or_nan(current_null_terms)),
                "seed_block_preserving_null": float(_mean_or_nan(seed_block_null_terms)),
                "difference_seed_block_minus_current": float(
                    _mean_or_nan(seed_block_null_terms) - _mean_or_nan(current_null_terms)
                ),
                "current_source_null": float(current_null_terms[0]),
                "current_target_null": float(current_null_terms[1]),
                "seed_block_source_null": float(seed_block_null_terms[0]),
                "seed_block_target_null": float(seed_block_null_terms[1]),
                "seed_pairing": "aligned index within sorted A/B 15-seed partitions",
            })

            key_prefix = (source, target, metric)
            canonical_key = (*key_prefix, str(depth))
            if canonical_key not in d_adj_map:
                raise RuntimeError(f"missing canonical D_adj depth row for {canonical_key}")
            for comparator in COMPARATORS:
                cross_orientation = [pair_values["cross_ab"][comparator], pair_values["cross_ba"][comparator]]
                null_terms = [pair_values["null_source"][comparator], pair_values["null_target"][comparator]]
                cross_estimate = _mean_or_nan(cross_orientation)
                same_context_null = _mean_or_nan(null_terms)
                margin = (
                    float(cross_estimate - same_context_null)
                    if np.isfinite(cross_estimate) and np.isfinite(same_context_null)
                    else float("nan")
                )
                if comparator == STABLE_COMPARATOR:
                    cross_coverage = _mean_or_nan([pair_coverage["cross_ab"], pair_coverage["cross_ba"]])
                    null_coverage = _mean_or_nan([pair_coverage["null_source"], pair_coverage["null_target"]])
                else:
                    cross_coverage = 1.0
                    null_coverage = 1.0
                interval = _orientation_bootstrap(
                    cross_orientation,
                    null_terms,
                    draws=int(bootstrap_draws),
                    seed=_stable_seed(source, target, metric, depth, comparator),
                )
                surface_rows.append({
                    "aggregation_level": "surface",
                    "source_environment_id": source,
                    "target_environment_id": target,
                    "transfer_id": f"{source}->{target}",
                    "unordered_context_pair_id": _surface_pair_id(source, target),
                    "metric": metric,
                    "surface_id": f"{source}->{target}::{metric}",
                    "cell_budget_label": str(depth),
                    "cell_budget_order": int(DEPTHS.index(depth) + 1),
                    "comparator": comparator,
                    "n_items": int(len(common)),
                    "n_pairs": int(len(common) * (len(common) - 1) // 2),
                    "cross_estimate": float(cross_estimate),
                    "same_context_null": float(same_context_null),
                    "cross_minus_null_margin": float(margin),
                    "margin": float(margin),
                    "cross_evaluable_coverage": float(cross_coverage),
                    "same_context_evaluable_coverage": float(null_coverage),
                    "evaluable_coverage": float(cross_coverage),
                    "surface_cross_ci_low": float(interval["cross_ci_low"]),
                    "surface_cross_ci_high": float(interval["cross_ci_high"]),
                    "surface_null_ci_low": float(interval["null_ci_low"]),
                    "surface_null_ci_high": float(interval["null_ci_high"]),
                    "surface_margin_ci_low": float(interval["margin_ci_low"]),
                    "surface_margin_ci_high": float(interval["margin_ci_high"]),
                    "surface_lower_ci_gt_zero": bool(
                        np.isfinite(interval["margin_ci_low"]) and float(interval["margin_ci_low"]) > 0.0
                    ),
                    "canonical_d_adj_reference": float(d_adj_map[canonical_key]) if comparator == "d_adj" else float("nan"),
                    "partition_cross_minus_canonical_d_adj": (
                        float(cross_estimate - d_adj_map[canonical_key]) if comparator == "d_adj" else float("nan")
                    ),
                    "bootstrap_draws": int(interval["bootstrap_valid_draws"]),
                })

    expected_surface_keys = 18
    if len(surface_keys) != expected_surface_keys:
        raise RuntimeError(f"expected {expected_surface_keys} directed surfaces, found {len(surface_keys)}")
    surface = pd.DataFrame(surface_rows)
    expected_surface_rows = expected_surface_keys * len(DEPTHS) * len(COMPARATORS)
    if len(surface) != expected_surface_rows:
        raise RuntimeError(f"expected {expected_surface_rows} surface comparator rows, found {len(surface)}")
    if surface.duplicated(["surface_id", "cell_budget_label", "comparator"]).any():
        raise RuntimeError("finite rank discrimination has duplicate surface/depth/comparator rows")

    macro_rows: list[dict[str, Any]] = []
    for comparator in COMPARATORS:
        for depth in DEPTHS:
            group = surface.loc[
                surface["comparator"].eq(comparator) & surface["cell_budget_label"].eq(depth)
            ].copy()
            if len(group) != expected_surface_keys:
                raise RuntimeError(f"macro surface is incomplete for {comparator} at depth={depth}")
            cross_estimate = _finite_mean(group["cross_estimate"])
            same_context_null = _finite_mean(group["same_context_null"])
            margin = _finite_mean(group["cross_minus_null_margin"])
            valid_margin = group["cross_minus_null_margin"].notna()
            valid_lower = group["surface_margin_ci_low"].notna()
            interval = _macro_block_bootstrap(
                group,
                draws=int(bootstrap_draws),
                seed=_stable_seed("macro", comparator, depth),
            )
            macro_rows.append({
                "aggregation_level": "macro_surface_mean",
                "comparator": comparator,
                "cell_budget_label": depth,
                "cell_budget_order": int(DEPTHS.index(depth) + 1),
                "n_surfaces": int(len(group)),
                "n_evaluable_surfaces": int(valid_margin.sum()),
                "cross_estimate": float(cross_estimate),
                "same_context_null": float(same_context_null),
                "cross_minus_null_margin": float(margin),
                "margin": float(margin),
                "bootstrap_ci_low": float(interval["margin_ci_low"]),
                "bootstrap_ci_high": float(interval["margin_ci_high"]),
                "margin_ci_low": float(interval["margin_ci_low"]),
                "margin_ci_high": float(interval["margin_ci_high"]),
                "cross_bootstrap_ci_low": float(interval["cross_ci_low"]),
                "cross_bootstrap_ci_high": float(interval["cross_ci_high"]),
                "same_context_null_bootstrap_ci_low": float(interval["null_ci_low"]),
                "same_context_null_bootstrap_ci_high": float(interval["null_ci_high"]),
                "fraction_surfaces_lower_ci_gt0": float(
                    group.loc[valid_lower, "surface_lower_ci_gt_zero"].mean() if valid_lower.any() else float("nan")
                ),
                "fraction_surfaces_with_lower_ci_gt_zero": float(
                    group.loc[valid_lower, "surface_lower_ci_gt_zero"].mean() if valid_lower.any() else float("nan")
                ),
                "surface_positive_margin_fraction": float(
                    (group.loc[valid_margin, "cross_minus_null_margin"] > 0.0).mean() if valid_margin.any() else float("nan")
                ),
                "negative_cross_surface_count": int(
                    (group["cross_estimate"].to_numpy(dtype=float) < 0.0).sum()
                ),
                "negative_same_context_null_surface_count": int(
                    (group["same_context_null"].to_numpy(dtype=float) < 0.0).sum()
                ),
                "negative_margin_surface_count": int(
                    (group["cross_minus_null_margin"].to_numpy(dtype=float) < 0.0).sum()
                ),
                "canonical_d_adj_negative_reference_count": int(
                    (group["canonical_d_adj_reference"].to_numpy(dtype=float) < 0.0).sum()
                ),
                "evaluable_coverage": _finite_mean(group["evaluable_coverage"]),
                "same_context_evaluable_coverage": _finite_mean(group["same_context_evaluable_coverage"]),
                "bootstrap_draws": int(interval["bootstrap_valid_draws"]),
                "bootstrap_blocks": int(interval["bootstrap_blocks"]),
                "bootstrap_ci_level": float(BOOTSTRAP_CI_LEVEL),
                "bootstrap_unit": "unordered context-pair block over fixed matched partition estimates",
                "surface_interval_unit": "two independent A/B partition orientations and two same-context terms",
            })
    macro = pd.DataFrame(macro_rows).sort_values(["comparator", "cell_budget_order"], kind="stable").reset_index(drop=True)

    # Signed recovery is relative to the full-depth macro margin.  The
    # denominator is its absolute value so a negative finite-depth margin stays
    # negative instead of being hidden by clipping or sign changes.
    full_margin = macro.loc[macro["cell_budget_label"].eq("full")].set_index("comparator")["cross_minus_null_margin"].to_dict()
    macro["full_depth_margin"] = macro["comparator"].map(full_margin).astype(float)
    macro["normalized_recovery"] = [
        float(value / max(abs(reference), NORMALIZATION_EPSILON))
        if np.isfinite(value) and np.isfinite(reference) and abs(reference) > NORMALIZATION_EPSILON
        else float("nan")
        for value, reference in zip(macro["cross_minus_null_margin"], macro["full_depth_margin"])
    ]
    macro["normalized_signal_fraction"] = [
        float(value / max(abs(cross), NORMALIZATION_EPSILON))
        if np.isfinite(value) and np.isfinite(cross) and abs(cross) > NORMALIZATION_EPSILON
        else float("nan")
        for value, cross in zip(macro["cross_minus_null_margin"], macro["cross_estimate"])
    ]

    surface_full_margin = surface.loc[surface["cell_budget_label"].eq("full")].set_index(
        ["surface_id", "comparator"]
    )["cross_minus_null_margin"].to_dict()
    surface["full_depth_margin"] = [
        float(surface_full_margin.get((surface_id, comparator), float("nan")))
        for surface_id, comparator in zip(surface["surface_id"], surface["comparator"])
    ]
    surface["normalized_recovery"] = [
        float(value / max(abs(reference), NORMALIZATION_EPSILON))
        if np.isfinite(value) and np.isfinite(reference) and abs(reference) > NORMALIZATION_EPSILON
        else float("nan")
        for value, reference in zip(surface["cross_minus_null_margin"], surface["full_depth_margin"])
    ]
    surface["normalized_signal_fraction"] = [
        float(value / max(abs(cross), NORMALIZATION_EPSILON))
        if np.isfinite(value) and np.isfinite(cross) and abs(cross) > NORMALIZATION_EPSILON
        else float("nan")
        for value, cross in zip(surface["cross_minus_null_margin"], surface["cross_estimate"])
    ]

    csv_columns = [
        "aggregation_level",
        "comparator",
        "cell_budget_label",
        "cell_budget_order",
        "n_surfaces",
        "n_evaluable_surfaces",
        "cross_estimate",
        "same_context_null",
        "cross_minus_null_margin",
        "margin",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "margin_ci_low",
        "margin_ci_high",
        "cross_bootstrap_ci_low",
        "cross_bootstrap_ci_high",
        "same_context_null_bootstrap_ci_low",
        "same_context_null_bootstrap_ci_high",
        "fraction_surfaces_lower_ci_gt0",
        "fraction_surfaces_with_lower_ci_gt_zero",
        "surface_positive_margin_fraction",
        "negative_cross_surface_count",
        "negative_same_context_null_surface_count",
        "negative_margin_surface_count",
        "canonical_d_adj_negative_reference_count",
        "normalized_recovery",
        "normalized_signal_fraction",
        "full_depth_margin",
        "evaluable_coverage",
        "same_context_evaluable_coverage",
        "bootstrap_draws",
        "bootstrap_blocks",
        "bootstrap_ci_level",
        "bootstrap_unit",
        "surface_interval_unit",
    ]
    output = macro[csv_columns].copy()
    csv_path = manifests / "finite_measurement_rank_discrimination.csv"
    json_path = manifests / "finite_measurement_rank_discrimination.json"
    estimator_unit_audit = pd.DataFrame(estimator_unit_audit_rows)
    if len(estimator_unit_audit) != expected_surface_keys * len(DEPTHS):
        raise RuntimeError("finite estimator-unit audit is incomplete")
    audit_csv_path = manifests / "finite_measurement_estimator_unit_audit.csv"
    estimator_unit_audit.to_csv(audit_csv_path, index=False)
    matched_unit_audit = pd.DataFrame(matched_unit_audit_rows)
    expected_matched_rows = expected_surface_keys * len(DEPTHS) * len(COMPARATORS)
    if len(matched_unit_audit) != expected_matched_rows:
        raise RuntimeError(f"matched estimator-unit audit is incomplete: expected {expected_matched_rows}, found {len(matched_unit_audit)}")
    if matched_unit_audit.duplicated(
        ["source_environment_id", "target_environment_id", "metric", "cell_budget_label", "comparator"]
    ).any():
        raise RuntimeError("matched estimator-unit audit has duplicate surface/depth/comparator rows")
    matched_audit_csv_path = manifests / "finite_measurement_matched_unit_audit.csv"
    matched_unit_audit.to_csv(matched_audit_csv_path, index=False)
    output.to_csv(csv_path, index=False)

    current_null = estimator_unit_audit["current_flattened_null"].to_numpy(dtype=float)
    seed_block_null = estimator_unit_audit["seed_block_preserving_null"].to_numpy(dtype=float)
    if not np.isfinite(np.column_stack((current_null, seed_block_null))).all():
        raise RuntimeError("finite estimator-unit audit contains non-finite D_adj values")
    current_order = pd.Series(current_null).rank(method="average").to_numpy(dtype=float)
    seed_block_order = pd.Series(seed_block_null).rank(method="average").to_numpy(dtype=float)
    estimator_unit_audit_summary = {
        "purpose": "diagnose finite-measurement D_adj null dependence on estimator unit; not a primary-result replacement",
        "current_unit": "flatten 15 seeds x 2 measurement replicates per pseudo-context, then one U-corrected estimate",
        "alternative_unit": "U-correct each aligned two-replicate seed block, then average 15 seed-level estimates",
        "seed_pairing": "same sorted index between partition A and partition B; deterministic and non-overlapping",
        "rows": int(len(estimator_unit_audit)),
        "current_negative_rows": int((current_null < 0.0).sum()),
        "seed_block_negative_rows": int((seed_block_null < 0.0).sum()),
        "current_mean": float(np.mean(current_null)),
        "seed_block_mean": float(np.mean(seed_block_null)),
        "difference_mean": float(np.mean(seed_block_null - current_null)),
        "max_absolute_difference": float(np.max(np.abs(seed_block_null - current_null))),
        "rank_correlation": float(np.corrcoef(current_order, seed_block_order)[0, 1]),
        "interpretation": "signed finite-sample null estimates remain diagnostic; no clipping or result selection is applied",
        "csv": "artifacts/manifests/finite_measurement_estimator_unit_audit.csv",
    }

    matched_unit_summary: dict[str, Any] = {
        "purpose": "compare the primary flattened partition with one common aligned seed-block construction for every comparator",
        "unit": "15 aligned seed blocks; each block retains its two measurement replicates",
        "rank_comparator_unit": "collapse the two replicates within each seed block, then compare the 15 block-level mean vectors",
        "d_adj_unit": "U-correct each aligned two-replicate block, then average the 15 block-level estimates",
        "stable_unit": "evaluate the frozen stable rule on the 15 block-level vectors; support threshold remains >=8",
        "rows": int(len(matched_unit_audit)),
        "comparators": list(COMPARATORS),
        "per_comparator": {},
        "csv": "artifacts/manifests/finite_measurement_matched_unit_audit.csv",
    }
    for comparator, group in matched_unit_audit.groupby("comparator", sort=True, observed=True):
        current_margin = group["current_cross_minus_null_margin"].to_numpy(dtype=float)
        matched_margin = group["matched_cross_minus_null_margin"].to_numpy(dtype=float)
        finite = np.isfinite(current_margin) & np.isfinite(matched_margin)
        if not np.any(finite):
            correlation = float("nan")
        elif np.std(current_margin[finite]) == 0.0 or np.std(matched_margin[finite]) == 0.0:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(current_margin[finite], matched_margin[finite])[0, 1])
        matched_unit_summary["per_comparator"][str(comparator)] = {
            "rows": int(len(group)),
            "matched_negative_cross_rows": int((group["matched_cross_estimate"].to_numpy(dtype=float) < 0.0).sum()),
            "matched_negative_null_rows": int((group["matched_same_context_null"].to_numpy(dtype=float) < 0.0).sum()),
            "matched_negative_margin_rows": int((matched_margin < 0.0).sum()),
            "matched_mean_cross": float(np.nanmean(group["matched_cross_estimate"].to_numpy(dtype=float))),
            "matched_mean_null": float(np.nanmean(group["matched_same_context_null"].to_numpy(dtype=float))),
            "matched_mean_margin": float(np.nanmean(matched_margin)),
            "matched_positive_margin_fraction": float(np.nanmean(matched_margin > 0.0)),
            "margin_rank_correlation_with_primary": correlation,
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "executed",
        "analysis": "finite_measurement_rank_discrimination",
        "surface": "3 unordered context pairs x 2 directions x 3 metrics = 18 directed surfaces",
        "metrics": list(METRICS),
        "comparators": list(COMPARATORS),
        "depths": {"fixed": list(DEPTHS[:-1]), "full": "full"},
        "risk_artifact": "artifacts/manifests/reliability_transport_measurement_depth_matched_fixed_risks.csv",
        "canonical_summary": canonical_audit,
        "estimator_unit_audit": estimator_unit_audit_summary,
        "matched_unit_audit": matched_unit_summary,
        "protocol": {
            "partition_policy": "sorted 30 split seeds into disjoint A=first 15 and B=last 15; both measurement replicates remain within each seed block",
            "seed_partition_a": [int(value) for value in seed_a],
            "seed_partition_b": [int(value) for value in seed_b],
            "measurement_replicates": list(MEASUREMENT_REPLICATES),
            "cross_terms": ["source_A_vs_target_B", "source_B_vs_target_A"],
            "same_context_null_terms": ["source_A_vs_source_B", "target_A_vs_target_B"],
            "target_refit": False,
            "raw_data_recomputation": False,
        },
        "estimands": {
            "kendall_distance": "(1-tau-b)/2 between mean risk vectors of independent pseudo-context blocks",
            "spearman_distance": "(1-rho)/2 between mean risk vectors of independent pseudo-context blocks",
            "top10_jaccard_distance": "1-Jaccard of deterministic lower-risk top-10% sets; item-index tie break",
            "stable_pair_inversion_fraction": "held-out inversion among pairs selected stable in both pseudo-contexts; 7/8 seed cross-fit, minimum strict support >=8",
            "d_adj": "U-corrected pair-state divergence between independent pseudo-context replicate arrays; finite negative estimates are retained",
            "cross_minus_null_margin": "cross_estimate - same_context_null, signed and unclipped",
            "normalized_recovery": "signed macro margin divided by the absolute full-depth macro margin for the same comparator; no clipping",
            "normalized_signal_fraction": "signed macro margin divided by the absolute macro cross estimate",
            "evaluable_coverage": "1 for full-vector comparators; for stable inversion, mean held-out non-tie evaluable pair fraction",
        },
        "bootstrap": {
            "ci_level": float(BOOTSTRAP_CI_LEVEL),
            "draws_requested": int(bootstrap_draws),
            "macro_interval": "percentile interval over unordered context-pair blocks applied to the fixed matched-partition surface estimates",
            "surface_interval": "percentile interval over the two independent A/B cross orientations and two same-context terms",
            "lower_ci_fraction": "fraction of finite surface-level partition-bootstrap lower bounds above zero",
            "negative_results_preserved": True,
        },
        "rows": {
            "macro": int(len(output)),
            "surface": int(len(surface)),
            "evaluable_surface_rows": int(surface["cross_minus_null_margin"].notna().sum()),
        },
        "negative_result_audit": {
            "surface_negative_cross_rows": int((surface["cross_estimate"] < 0.0).sum()),
            "surface_negative_same_context_null_rows": int((surface["same_context_null"] < 0.0).sum()),
            "surface_negative_margin_rows": int((surface["cross_minus_null_margin"] < 0.0).sum()),
            "canonical_d_adj_negative_reference_rows": int(
                (surface["canonical_d_adj_reference"] < 0.0).sum()
            ),
            "d_adj_is_not_ranked_as_winner": True,
        },
        "summary_rows": _json_safe(output.to_dict("records")),
        "surface_rows": _json_safe(
            surface.sort_values(
                ["source_environment_id", "target_environment_id", "metric", "comparator", "cell_budget_order"],
                kind="stable",
            ).to_dict("records")
        ),
        "outputs": {
            "csv": "artifacts/manifests/finite_measurement_rank_discrimination.csv",
            "json": "artifacts/manifests/finite_measurement_rank_discrimination.json",
            "estimator_unit_audit_csv": "artifacts/manifests/finite_measurement_estimator_unit_audit.csv",
            "matched_unit_audit_csv": "artifacts/manifests/finite_measurement_matched_unit_audit.csv",
        },
    }
    json_path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return output, surface, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    output, _, report = build(args.root, bootstrap_draws=args.bootstrap_draws)
    print(json.dumps({"status": report["status"], "rows": report["rows"], "outputs": report["outputs"]}, indent=2))
    print(output.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
