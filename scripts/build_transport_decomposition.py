"""Build the finite-measurement, cross-fitted ordering decomposition.

The population identity is an exact squared-distance identity for a three-state
pair-order law.  The finite-sample report below uses an independent
cross-product of the two seed halves and retains both measurement replicates;
it therefore does not turn a plug-in quadratic into the canonical
measurement-corrected ``D_adj``.  The nonlinear strength/flip split is kept as
population algebra only; the empirical table reports tie and directional-bias
components plus a separate stable-inversion diagnostic.
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
from src.evaluation.ordering_estimands import (
    crossfit_seed_stable_ordering_summary,
    perturbation_reordering_burden,
)
from src.evaluation.transport_decomposition import crossfit_pair_state_components, crossfit_pairwise_components, decompose_pair_states

METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")


def _seed_replicates(group: pd.DataFrame, context: str, seeds: list[int]) -> tuple[list[str], np.ndarray]:
    """Return ``[seed, measurement_replicate, perturbation]`` risk arrays."""

    left = str(group["left_target_environment_id"].iloc[0])
    column = "left_risk" if context == left else "right_risk"
    subset = group.loc[group["split_seed"].isin(seeds) & group["cell_budget_label"].eq("full"), ["perturbation_label", "split_seed", "measurement_replicate", column]].copy()
    subset["split_seed"] = subset["split_seed"].astype(int)
    subset["measurement_replicate"] = subset["measurement_replicate"].astype(int)
    index = pd.MultiIndex.from_product((seeds, (0, 1)), names=("split_seed", "measurement_replicate"))
    pivot = subset.pivot_table(index=["split_seed", "measurement_replicate"], columns="perturbation_label", values=column, aggfunc="first").reindex(index=index)
    pivot = pivot.dropna(axis=1, how="any")
    if pivot.shape[0] != len(seeds) * 2 or pivot.empty or not np.isfinite(pivot.to_numpy(dtype=float)).all():
        raise ValueError(f"incomplete measurement-replicate grid for {context}")
    labels = [str(value) for value in pivot.columns]
    return labels, pivot.to_numpy(dtype=float).reshape(len(seeds), 2, len(labels))


def _canonical_d_adj(source: np.ndarray, target: np.ndarray) -> float:
    """Recompute the canonical per-seed U-corrected quantity."""

    values = []
    for seed_index in range(source.shape[0]):
        burden = perturbation_reordering_burden(source[seed_index], target[seed_index], block_size=min(512, source.shape[1]))
        values.append(float(burden["identifiable_divergence"]))
    return float(np.mean(values))


def _surface_row(source: np.ndarray, target: np.ndarray, *, source_id: str, target_id: str, metric: str, labels: list[str]) -> dict[str, Any]:
    flat_source = source.reshape(source.shape[0] * source.shape[1], source.shape[2])
    flat_target = target.reshape(target.shape[0] * target.shape[1], target.shape[2])
    components = crossfit_pairwise_components(flat_source, flat_target)
    d_tie = np.asarray(components["d_tie_cf"], dtype=float)
    d_direction = np.asarray(components["d_direction_cf"], dtype=float)
    d_total = np.asarray(components["d_total_cf"], dtype=float)
    stable = crossfit_seed_stable_ordering_summary(source, target, minimum_strict_support=8)
    canonical = _canonical_d_adj(source, target)
    return {
        "source_environment_id": source_id,
        "target_environment_id": target_id,
        "transfer_id": f"{source_id}->{target_id}",
        "metric": metric,
        "n_items": int(len(labels)),
        "pair_count": int(components["pair_count"]),
        "d_tie_cf": float(np.mean(d_tie)),
        "d_direction_cf": float(np.mean(d_direction)),
        "corrected_total_cf": float(np.mean(d_total)),
        "d_tie_cf_q05": float(np.quantile(d_tie, 0.05)),
        "d_tie_cf_q95": float(np.quantile(d_tie, 0.95)),
        "d_direction_cf_q05": float(np.quantile(d_direction, 0.05)),
        "d_direction_cf_q95": float(np.quantile(d_direction, 0.95)),
        "corrected_total_cf_q05": float(np.quantile(d_total, 0.05)),
        "corrected_total_cf_q95": float(np.quantile(d_total, 0.95)),
        "canonical_d_adj": canonical,
        "cf_minus_canonical": float(np.mean(d_total) - canonical),
        "stable_inversion_fraction": float(stable["crossfit_heldout_inversion_fraction"]),
        "stable_evaluable_pairs": int(stable["crossfit_heldout_evaluable_pairs"]),
        "stable_inversion_pairs": int(stable["crossfit_heldout_inversion_pairs"]),
        "measurement_contract": "30 independent split seeds × 2 retained measurement replicates; fixed matched universe",
        "crossfit_contract": "15 seed source half × 15 seed target half, both measurement replicates retained; swapped halves",
        "status": "executed",
    }


def _synthetic_validation(seed: int = 20260914, trials: int = 300) -> pd.DataFrame:
    """Validate the cross-product estimator on independent categorical draws."""

    regimes = {
        "same_direction_same_uncertainty": (np.array([0.10, 0.20, 0.70]), np.array([0.10, 0.20, 0.70])),
        "same_direction_different_uncertainty": (np.array([0.10, 0.20, 0.70]), np.array([0.20, 0.20, 0.60])),
        "tie_shift": (np.array([0.10, 0.10, 0.80]), np.array([0.10, 0.50, 0.40])),
        "genuine_reversal": (np.array([0.10, 0.20, 0.70]), np.array([0.70, 0.20, 0.10])),
        "near_tie": (np.array([0.45, 0.10, 0.45]), np.array([0.40, 0.20, 0.40])),
        "shortlist_boundary_flips": (np.array([0.10, 0.10, 0.80]), np.array([0.42, 0.16, 0.42])),
    }
    rng = np.random.default_rng(seed)
    states = np.asarray((-1, 0, 1), dtype=np.int8)
    rows: list[dict[str, Any]] = []
    pair_count = 256
    for regime, (source_prob, target_prob) in regimes.items():
        truth = decompose_pair_states(np.tile(source_prob, (pair_count, 1)), np.tile(target_prob, (pair_count, 1)))
        truth_total = float(np.mean(truth["d_squared"]))
        estimates: list[float] = []
        coverages: list[bool] = []
        for _ in range(int(trials)):
            source_draws = rng.choice(states, size=(60, pair_count), p=source_prob)
            target_draws = rng.choice(states, size=(60, pair_count), p=target_prob)
            values = crossfit_pair_state_components(source_draws, target_draws)
            per_pair = np.asarray(values["d_total_cf"], dtype=float)
            estimates.append(float(np.mean(per_pair)))
            bootstrap = np.empty(300, dtype=float)
            for draw in range(300):
                bootstrap[draw] = float(np.mean(rng.choice(per_pair, size=pair_count, replace=True)))
            coverages.append(bool(np.quantile(bootstrap, 0.05) <= truth_total <= np.quantile(bootstrap, 0.95)))
        rows.append({
            "regime": regime,
            "trials": int(trials),
            "pair_count": pair_count,
            "truth_d_total": truth_total,
            "estimate_d_total_mean": float(np.mean(estimates)),
            "estimate_d_total_bias": float(np.mean(estimates) - truth_total),
            "estimate_d_total_q05": float(np.quantile(estimates, 0.05)),
            "estimate_d_total_q95": float(np.quantile(estimates, 0.95)),
            "bootstrap_90pct_coverage": float(np.mean(coverages)),
            "identity_check": "population algebra retained separately; empirical report uses cross-product tie + direction",
        })
    return pd.DataFrame(rows)


def build(root: Path = ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    risk_path = manifests / "reliability_transport_measurement_depth_matched_fixed_risks.csv"
    if not risk_path.is_file():
        raise FileNotFoundError(risk_path)
    usecols = ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "perturbation_label", "measurement_replicate", "cell_budget_label", "left_risk", "right_risk"]
    risks = pd.read_csv(risk_path, usecols=usecols, low_memory=False)
    risks["cell_budget_label"] = risks["cell_budget_label"].astype(str).str.strip("'\" ")
    seeds = sorted(int(value) for value in risks["split_seed"].dropna().unique())
    if len(seeds) != 30:
        raise ValueError(f"expected 30 split seeds, found {len(seeds)}")
    canonical_summary = pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_summary.csv")
    canonical_summary = canonical_summary.loc[canonical_summary["cell_budget_label"].eq("full")].copy()
    rows: list[dict[str, Any]] = []
    for key, group in risks.groupby(["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"], sort=True, observed=True):
        source_id, left, right, metric = map(str, key)
        if metric not in METRICS or source_id not in (left, right):
            continue
        for source, target in ((left, right), (right, left)):
            if source != source_id:
                continue
            source_labels, source_values = _seed_replicates(group, source, seeds)
            target_labels, target_values = _seed_replicates(group, target, seeds)
            common = sorted(set(source_labels) & set(target_labels))
            if len(common) < 2:
                continue
            source_idx = [source_labels.index(label) for label in common]
            target_idx = [target_labels.index(label) for label in common]
            row = _surface_row(source_values[:, :, source_idx], target_values[:, :, target_idx], source_id=source, target_id=target, metric=metric, labels=common)
            summary_row = canonical_summary.loc[
                canonical_summary["source_environment_id"].eq(source)
                & canonical_summary["left_target_environment_id"].eq(left)
                & canonical_summary["right_target_environment_id"].eq(right)
                & canonical_summary["metric"].eq(metric)
            ]
            if len(summary_row) != 1:
                raise ValueError(f"missing canonical summary row for {source}->{target} {metric}")
            row["canonical_summary_d_adj"] = float(summary_row.iloc[0]["identifiable_divergence"])
            row["canonical_reconstruction_error"] = float(row["canonical_d_adj"] - row["canonical_summary_d_adj"])
            rows.append(row)
    table = pd.DataFrame(rows).sort_values(["source_environment_id", "target_environment_id", "metric"], kind="stable").reset_index(drop=True)
    if table.empty or table["transfer_id"].nunique() != 6 or len(table) != 18:
        raise ValueError(f"expected 18 cross-fitted surfaces, found {len(table)}")
    validation = _synthetic_validation()
    table_path = manifests / "reliability_transport_decomposition.csv"
    validation_path = manifests / "transport_decomposition_synthetic_validation.csv"
    table.to_csv(table_path, index=False)
    validation.to_csv(validation_path, index=False)
    report = {
        "schema_version": 2,
        "status": "executed",
        "estimand": "finite-measurement cross-product estimator for the population pair-law squared separation",
        "population_identity": "D_tie + D_strength + D_flip = 1/2 ||pi_source-pi_target||^2",
        "finite_sample_report": "D_tie^CF + D_direction^CF; no unbiased empirical strength/flip split claimed",
        "canonical_comparison": "per-seed two-replicate U-corrected D_adj on the identical matched-fixed universe",
        "stable_diagnostic": "cross-fitted held-out stable inversion fraction; minimum strict support = 8",
        "directed_transfer_metric_rows": int(len(table)),
        "pair_count_by_surface": sorted(table["pair_count"].unique().tolist()),
        "max_canonical_reconstruction_error": float(table["canonical_reconstruction_error"].abs().max()),
        "max_cf_minus_canonical": float(table["cf_minus_canonical"].abs().max()),
        "synthetic_regimes": validation["regime"].tolist(),
        "outputs": {"summary": table_path.relative_to(root).as_posix(), "synthetic_validation": validation_path.relative_to(root).as_posix()},
    }
    (manifests / "reliability_transport_decomposition.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    return table, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    table, report = build(args.root)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
    print(table[["transfer_id", "metric", "n_items", "pair_count", "d_tie_cf", "d_direction_cf", "corrected_total_cf", "canonical_d_adj", "cf_minus_canonical", "stable_inversion_fraction"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
