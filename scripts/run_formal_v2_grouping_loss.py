"""Quantify hidden risk heterogeneity after conditioning on confidence.

Confidence-bin edges are fitted only on calibration-side scores.  Test rows
are then assigned to those fixed bins and summarized by biological environment.
The resulting ``contextual_grouping_loss`` is a descriptive measure of how
much realized risk varies across environments at nominally similar confidence;
it is not introduced as a new generic ML concept.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRIMARY_SEED = 20260907
METHODS = ("raw_normalized_uq", "u_only_rf", "ptl_rf")
METHOD_LABELS = {
    "raw_normalized_uq": "raw/native UQ",
    "u_only_rf": "U-only RF",
    "ptl_rf": "PTL",
}


def declared_seeds(root: Path) -> list[int]:
    config = yaml.safe_load((root / "configs/biological_instance_split.yaml").read_text(encoding="utf-8"))
    return [int(config["split_seed"]), *(int(value) for value in config.get("stability_seeds", []))]


def artifact_paths(root: Path, seed: int) -> tuple[Path, Path]:
    if seed == PRIMARY_SEED:
        base = root / "artifacts/source_data"
    else:
        base = root / "results/formal_v2/multisplit" / str(seed) / "reliability/artifacts/source_data"
    return base / "formal_v2_reliability_predictions.csv", base / "formal_v2_reliability_calibration_scores.csv"


def _bin_edges(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot define confidence bins from an empty calibration distribution")
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.unique(edges)


def assign_bins(values: np.ndarray, calibration_values: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    edges = _bin_edges(calibration_values, n_bins=n_bins)
    labels = np.searchsorted(edges[1:-1], np.asarray(values, dtype=float), side="right")
    return labels.astype(int), edges


def _pair_gap(group: pd.DataFrame, *, same_environment: bool) -> tuple[float, int]:
    """Return a mean absolute risk gap without quadratic pair enumeration."""

    def pair_sum(values: np.ndarray) -> tuple[float, int]:
        values = np.sort(np.asarray(values, dtype=float))
        count = int(len(values) * (len(values) - 1) // 2)
        if count == 0:
            return 0.0, 0
        coefficients = 2 * np.arange(len(values), dtype=float) - len(values) + 1
        return float(np.dot(coefficients, values)), count

    total_sum, total_count = pair_sum(group["continuous_risk"].to_numpy(dtype=float))
    same_sum = 0.0
    same_count = 0
    for _, environment_group in group.groupby("environment_id", sort=False):
        value, count = pair_sum(environment_group["continuous_risk"].to_numpy(dtype=float))
        same_sum += value
        same_count += count
    if same_environment:
        return (same_sum / same_count if same_count else float("nan"), same_count)
    cross_sum = total_sum - same_sum
    cross_count = total_count - same_count
    return (cross_sum / cross_count if cross_count else float("nan"), cross_count)


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    finite = [(float(value), int(weight)) for value, weight in zip(values, weights) if np.isfinite(value) and weight > 0]
    if not finite:
        return float("nan")
    return float(np.average(
        np.asarray([value for value, _ in finite], dtype=float),
        weights=np.asarray([weight for _, weight in finite], dtype=float),
    ))


def _summarize_grouped(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"contextual_grouping_loss": float("nan"), "between_environment_risk_variance": float("nan"), "risk_variance_reduction": float("nan")}
    global_mean = float(frame["continuous_risk"].mean())
    env_stats = frame.groupby("environment_id", sort=True)["continuous_risk"].agg(["mean", "count"])
    loss = float(np.average(np.abs(env_stats["mean"] - global_mean), weights=env_stats["count"]))
    between = float(np.average((env_stats["mean"] - global_mean) ** 2, weights=env_stats["count"]))
    within = float(
        sum(((group["continuous_risk"] - group["continuous_risk"].mean()) ** 2).sum() for _, group in frame.groupby("environment_id", sort=True))
        / max(len(frame) - len(env_stats), 1)
    )
    global_variance = float(frame["continuous_risk"].var(ddof=1)) if len(frame) > 1 else 0.0
    return {
        "contextual_grouping_loss": loss,
        "between_environment_risk_variance": between,
        "within_environment_risk_variance": within,
        "global_risk_variance": global_variance,
        "risk_variance_reduction": global_variance - within,
        "global_risk_mean": global_mean,
    }


def _summarize_conditional_grouped(frame: pd.DataFrame) -> dict[str, float]:
    """Summarize heterogeneity after conditioning on fixed confidence bins.

    The bins are created from calibration-side scores before this function is
    called.  A summary over the complete frame would silently remove that
    conditioning and measure only ordinary environment difficulty.
    """

    if frame.empty:
        return {
            "contextual_grouping_loss": float("nan"),
            "between_environment_risk_variance": float("nan"),
            "within_environment_risk_variance": float("nan"),
            "global_risk_variance": float("nan"),
            "risk_variance_reduction": float("nan"),
            "global_risk_mean": float("nan"),
            "matched_confidence_same_environment_risk_gap": float("nan"),
            "matched_confidence_cross_environment_risk_gap": float("nan"),
            "matched_confidence_cross_minus_same_risk_gap": float("nan"),
            "confidence_bin_weighted_max_environment_risk_gap": float("nan"),
        }
    if "confidence_bin" not in frame.columns:
        return _summarize_grouped(frame)

    metrics: list[dict[str, float]] = []
    bin_weights: list[int] = []
    same_values: list[float] = []
    same_weights: list[int] = []
    cross_values: list[float] = []
    cross_weights: list[int] = []
    max_gap_values: list[float] = []
    for _, group in frame.groupby("confidence_bin", sort=True, dropna=False):
        stats = _summarize_grouped(group)
        env_stats = group.groupby("environment_id", sort=True)["continuous_risk"].agg(["mean", "count"])
        same_gap, same_count = _pair_gap(group, same_environment=True)
        cross_gap, cross_count = _pair_gap(group, same_environment=False)
        metrics.append(stats)
        bin_weights.append(int(len(group)))
        max_gap_values.append(float(env_stats["mean"].max() - env_stats["mean"].min()) if len(env_stats) else float("nan"))
        if np.isfinite(same_gap) and same_count:
            same_values.append(same_gap)
            same_weights.append(same_count)
        if np.isfinite(cross_gap) and cross_count:
            cross_values.append(cross_gap)
            cross_weights.append(cross_count)

    values = {name: _weighted_mean([item[name] for item in metrics], bin_weights) for name in (
        "contextual_grouping_loss", "between_environment_risk_variance",
        "within_environment_risk_variance", "global_risk_variance", "global_risk_mean",
    )}
    same = _weighted_mean(same_values, same_weights)
    cross = _weighted_mean(cross_values, cross_weights)
    values.update({
        "risk_variance_reduction": values["global_risk_variance"] - values["within_environment_risk_variance"],
        "matched_confidence_same_environment_risk_gap": same,
        "matched_confidence_cross_environment_risk_gap": cross,
        "matched_confidence_cross_minus_same_risk_gap": cross - same if np.isfinite(cross) and np.isfinite(same) else float("nan"),
        "confidence_bin_weighted_max_environment_risk_gap": _weighted_mean(max_gap_values, bin_weights),
    })
    return values


def _summarize_arrays(risk: np.ndarray, environments: np.ndarray, confidence_bins: np.ndarray | None = None) -> dict[str, float]:
    """Summarize a bootstrap draw while preserving confidence bins."""

    risk = np.asarray(risk, dtype=float)
    environments = np.asarray(environments, dtype=str)
    if risk.size == 0:
        return _summarize_conditional_grouped(pd.DataFrame({"continuous_risk": [], "environment_id": [], "confidence_bin": []}))
    if confidence_bins is not None:
        # Bootstrap calls this function tens of thousands of times.  Keep the
        # same estimands as _summarize_conditional_grouped, but avoid creating
        # a DataFrame/groupby object for every draw.
        bins = np.asarray(confidence_bins, dtype=int)
        environment_codes, environment_levels = pd.factorize(environments, sort=True)
        bin_levels = np.unique(bins)
        metric_values: list[dict[str, float]] = []
        metric_weights: list[int] = []
        same_values: list[float] = []
        same_weights: list[int] = []
        cross_values: list[float] = []
        cross_weights: list[int] = []
        max_values: list[float] = []
        for bin_value in bin_levels:
            mask = bins == bin_value
            values = risk[mask]
            codes = environment_codes[mask]
            if not len(values):
                continue
            counts = np.bincount(codes, minlength=len(environment_levels)).astype(float)
            sums = np.bincount(codes, weights=values, minlength=len(environment_levels))
            present = counts > 0
            means = sums[present] / counts[present]
            weights = counts[present]
            global_mean = float(values.mean())
            global_variance = float(values.var(ddof=1)) if len(values) > 1 else 0.0
            residuals = values - (sums / np.maximum(counts, 1.0))[codes]
            within = float(np.sum(residuals ** 2) / max(len(values) - int(present.sum()), 1))
            metric_values.append({
                "contextual_grouping_loss": float(np.average(np.abs(means - global_mean), weights=weights)),
                "between_environment_risk_variance": float(np.average((means - global_mean) ** 2, weights=weights)),
                "within_environment_risk_variance": within,
                "global_risk_variance": global_variance,
                "global_risk_mean": global_mean,
            })
            metric_weights.append(int(len(values)))

            def sorted_pair_sum(array: np.ndarray) -> tuple[float, int]:
                ordered = np.sort(array)
                count = int(len(ordered) * (len(ordered) - 1) // 2)
                if count == 0:
                    return 0.0, 0
                coefficients = 2 * np.arange(len(ordered), dtype=float) - len(ordered) + 1
                return float(np.dot(coefficients, ordered)), count

            total_sum, total_count = sorted_pair_sum(values)
            same_sum = 0.0
            same_count = 0
            for code in np.flatnonzero(present):
                pair_sum, pair_count = sorted_pair_sum(values[codes == code])
                same_sum += pair_sum
                same_count += pair_count
            cross_count = total_count - same_count
            if same_count:
                same_values.append(same_sum / same_count)
                same_weights.append(same_count)
            if cross_count:
                cross_values.append((total_sum - same_sum) / cross_count)
                cross_weights.append(cross_count)
            max_values.append(float(means.max() - means.min()))
        values = {name: _weighted_mean([item[name] for item in metric_values], metric_weights) for name in (
            "contextual_grouping_loss", "between_environment_risk_variance",
            "within_environment_risk_variance", "global_risk_variance", "global_risk_mean",
        )}
        same = _weighted_mean(same_values, same_weights)
        cross = _weighted_mean(cross_values, cross_weights)
        values.update({
            "risk_variance_reduction": values["global_risk_variance"] - values["within_environment_risk_variance"],
            "matched_confidence_same_environment_risk_gap": same,
            "matched_confidence_cross_environment_risk_gap": cross,
            "matched_confidence_cross_minus_same_risk_gap": cross - same if np.isfinite(cross) and np.isfinite(same) else float("nan"),
            "confidence_bin_weighted_max_environment_risk_gap": _weighted_mean(max_values, metric_weights),
        })
        return values
    _, inverse = np.unique(environments, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    sums = np.bincount(inverse, weights=risk)
    means = sums / counts
    global_mean = float(risk.mean())
    loss = float(np.average(np.abs(means - global_mean), weights=counts))
    between = float(np.average((means - global_mean) ** 2, weights=counts))
    residuals = risk - means[inverse]
    within = float(np.sum(residuals ** 2) / max(len(risk) - len(means), 1))
    global_variance = float(risk.var(ddof=1)) if len(risk) > 1 else 0.0
    return {
        "contextual_grouping_loss": loss,
        "between_environment_risk_variance": between,
        "within_environment_risk_variance": within,
        "global_risk_variance": global_variance,
        "risk_variance_reduction": global_variance - within,
        "global_risk_mean": global_mean,
    }


def _hierarchical_bootstrap_sample(
    group_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample environment clusters and nested biological instances.

    The environment labels returned here intentionally include the outer draw
    number.  Two draws of the same named environment are separate bootstrap
    clusters and must not be merged by a downstream groupby operation.
    """

    if not group_arrays:
        return np.empty(0, dtype=float), np.empty(0, dtype=object), np.empty(0, dtype=int)
    sampled_risk: list[np.ndarray] = []
    sampled_environment: list[np.ndarray] = []
    sampled_bins: list[np.ndarray] = []
    for draw_number, group_index in enumerate(rng.integers(0, len(group_arrays), size=len(group_arrays))):
        risk, env_labels, confidence_bins, instances, positions = group_arrays[int(group_index)]
        sampled_ids = rng.choice(instances, size=len(instances), replace=True)
        sampled_positions = np.concatenate([positions[str(instance)] for instance in sampled_ids])
        sampled_risk.append(risk[sampled_positions])
        environment = str(env_labels[0]) if len(env_labels) else "unknown"
        sampled_environment.append(
            np.full(len(sampled_positions), f"{environment}__bootstrap_cluster_{draw_number}", dtype=object)
        )
        sampled_bins.append(confidence_bins[sampled_positions])
    return np.concatenate(sampled_risk), np.concatenate(sampled_environment), np.concatenate(sampled_bins)


def summarize_split(test: pd.DataFrame, calibration: pd.DataFrame, seed: int, n_bins: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    test = test.loc[test["scenario"].eq("in_domain")].copy()
    calibration = calibration.loc[calibration["scenario"].eq("in_domain")].copy()
    if test.empty or calibration.empty:
        raise ValueError(f"grouping-loss split {seed} needs in-domain test and calibration rows")
    for predictor in sorted(test["predictor"].astype(str).unique()):
        test_predictor = test.loc[test["predictor"].astype(str).eq(predictor)].copy()
        for method in METHODS:
            cal = calibration.loc[
                calibration["predictor"].astype(str).eq(predictor) & calibration["method"].eq(method), "confidence"
            ].to_numpy(dtype=float)
            if cal.size == 0:
                raise ValueError(f"missing calibration score distribution for {predictor}/{method}/{seed}")
            bins, edges = assign_bins(test_predictor[method].to_numpy(dtype=float), cal, n_bins=n_bins)
            test_predictor["confidence_bin"] = bins
            test_predictor["method"] = method
            for confidence_bin, group in test_predictor.groupby("confidence_bin", sort=True):
                stats = _summarize_grouped(group)
                env_stats = group.groupby("environment_id", sort=True)["continuous_risk"].agg(["mean", "count"])
                same_gap, same_count = _pair_gap(group, same_environment=True)
                different_gap, different_count = _pair_gap(group, same_environment=False)
                row = {
                    "split_seed": seed,
                    "predictor": predictor,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "confidence_bin": int(confidence_bin),
                    "confidence_bin_lower": float(edges[min(int(confidence_bin), len(edges) - 1)]),
                    "confidence_bin_upper": float(edges[min(int(confidence_bin) + 1, len(edges) - 1)]),
                    "n_rows": int(len(group)),
                    "n_environments": int(group["environment_id"].nunique()),
                    "global_mean_risk": stats["global_risk_mean"],
                    "between_environment_risk_variance": stats["between_environment_risk_variance"],
                    "maximum_environment_risk_gap": float(env_stats["mean"].max() - env_stats["mean"].min()) if len(env_stats) else float("nan"),
                    "contextual_grouping_loss": stats["contextual_grouping_loss"],
                    "within_environment_risk_variance": stats["within_environment_risk_variance"],
                    "risk_variance_reduction": stats["risk_variance_reduction"],
                    "matched_same_environment_risk_gap": same_gap,
                    "matched_same_environment_pair_count": same_count,
                    "matched_different_environment_risk_gap": different_gap,
                    "matched_different_environment_pair_count": different_count,
                    "bin_edges_fit_on": "calibration_rows_only",
                    "outcomes_used_for_bin_construction": 0,
                }
                rows.append(row)
            overall = _summarize_conditional_grouped(test_predictor)
            summary.append({
                "split_seed": seed,
                "predictor": predictor,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_test_rows": int(len(test_predictor)),
                "n_environments": int(test_predictor["environment_id"].nunique()),
                "contextual_grouping_loss": overall["contextual_grouping_loss"],
                "between_environment_risk_variance": overall["between_environment_risk_variance"],
                "within_environment_risk_variance": overall["within_environment_risk_variance"],
                "risk_variance_reduction": overall["risk_variance_reduction"],
                "matched_confidence_same_environment_gap": overall["matched_confidence_same_environment_risk_gap"],
                "matched_confidence_different_environment_gap": overall["matched_confidence_cross_environment_risk_gap"],
                "matched_confidence_same_environment_risk_gap": overall["matched_confidence_same_environment_risk_gap"],
                "matched_confidence_cross_environment_risk_gap": overall["matched_confidence_cross_environment_risk_gap"],
                "matched_confidence_cross_minus_same_risk_gap": overall["matched_confidence_cross_minus_same_risk_gap"],
                "confidence_bin_weighted_max_environment_risk_gap": overall["confidence_bin_weighted_max_environment_risk_gap"],
                "confidence_bins": int(len(edges) - 1),
                "bin_edges_fit_on": "calibration_rows_only",
            })
    return pd.DataFrame(rows), {"summary": summary}


def hierarchical_bootstrap(test: pd.DataFrame, split_seed: int, *, n_bins: int, replicates: int, random_state: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    for (predictor, method), frame in test.groupby(["predictor", "method"], sort=True):
        environments = sorted(frame["environment_id"].astype(str).unique())
        group_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]] = []
        for environment in environments:
            env = frame.loc[frame["environment_id"].astype(str).eq(str(environment))]
            risk = env["continuous_risk"].to_numpy(dtype=float)
            instance_ids = env["biological_instance_id"].astype(str).to_numpy()
            instance_order = list(pd.unique(instance_ids))
            positions = {instance: np.flatnonzero(instance_ids == instance) for instance in instance_order}
            confidence_bins = env["confidence_bin"].to_numpy(dtype=int)
            group_arrays.append((risk, np.full(len(risk), str(environment), dtype=object), confidence_bins, instance_order, positions))
        for replicate in range(replicates):
            sampled_risk, sampled_environment, sampled_bins = _hierarchical_bootstrap_sample(group_arrays, rng)
            stats = _summarize_arrays(
                sampled_risk,
                sampled_environment,
                sampled_bins,
            )
            rows.append({"split_seed": split_seed, "predictor": predictor, "method": method, "replicate": replicate, **stats})
    return pd.DataFrame(rows)


def run(root: Path, *, n_bins: int = 10, bootstrap_replicates: int = 1000, seeds: list[int] | None = None) -> dict[str, Any]:
    selected_seeds = seeds or declared_seeds(root)
    grouped_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    for seed in selected_seeds:
        prediction_path, calibration_path = artifact_paths(root, seed)
        if not prediction_path.is_file() or not calibration_path.is_file():
            raise FileNotFoundError(f"missing reliability outputs for grouping-loss split {seed}")
        test = pd.read_csv(prediction_path)
        calibration = pd.read_csv(calibration_path)
        grouped, detail = summarize_split(test, calibration, seed, n_bins)
        grouped_frames.append(grouped)
        summaries.extend(detail["summary"])
        boot_input = grouped.copy()
        # Bootstrap uses the row-level test frame with method-specific scores;
        # fixed bins are only a reporting partition and are not re-fit here.
        boot_rows = []
        for predictor in sorted(test.loc[test["scenario"].eq("in_domain"), "predictor"].astype(str).unique()):
            subset = test.loc[test["scenario"].eq("in_domain") & test["predictor"].astype(str).eq(predictor)].copy()
            for method in METHODS:
                subset_method = subset[["biological_instance_id", "environment_id", "continuous_risk", method]].copy()
                subset_method["predictor"] = predictor
                subset_method["method"] = method
                # Attach bins from the already materialized grouping table by
                # reusing calibration-defined edges, without using outcomes.
                cal = calibration.loc[calibration["predictor"].astype(str).eq(predictor) & calibration["method"].eq(method), "confidence"].to_numpy(dtype=float)
                subset_method["confidence_bin"], _ = assign_bins(subset_method[method].to_numpy(dtype=float), cal, n_bins=n_bins)
                boot_rows.append(subset_method)
        boot_frame = pd.concat(boot_rows, ignore_index=True)
        bootstrap_frames.append(hierarchical_bootstrap(boot_frame, seed, n_bins=n_bins, replicates=bootstrap_replicates, random_state=seed + 17))

    grouped_frame = pd.concat(grouped_frames, ignore_index=True).sort_values(["predictor", "method", "split_seed", "confidence_bin"], kind="stable")
    summary_frame = pd.DataFrame(summaries).sort_values(["predictor", "method", "split_seed"], kind="stable")
    bootstrap_frame = pd.concat(bootstrap_frames, ignore_index=True)
    summary_with_ci: list[dict[str, Any]] = []
    for (predictor, method), group in summary_frame.groupby(["predictor", "method"], sort=True):
        boot = bootstrap_frame.loc[(bootstrap_frame["predictor"].eq(predictor)) & (bootstrap_frame["method"].eq(method))]
        row = {"predictor": predictor, "method": method, "method_label": METHOD_LABELS[method], "n_splits": int(group["split_seed"].nunique())}
        for metric in (
            "contextual_grouping_loss",
            "between_environment_risk_variance",
            "risk_variance_reduction",
            "matched_confidence_same_environment_risk_gap",
            "matched_confidence_cross_environment_risk_gap",
            "matched_confidence_cross_minus_same_risk_gap",
            "confidence_bin_weighted_max_environment_risk_gap",
        ):
            row[f"mean_{metric}"] = float(group[metric].mean())
            values = boot[metric].to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            row[f"{metric}_ci_lower"] = float(np.quantile(finite_values, 0.025)) if len(finite_values) else float("nan")
            row[f"{metric}_ci_upper"] = float(np.quantile(finite_values, 0.975)) if len(finite_values) else float("nan")
        summary_with_ci.append(row)

    manifest_dir = root / "artifacts/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    grouped_path = manifest_dir / "formal_v2_grouping_loss.csv"
    bootstrap_path = manifest_dir / "formal_v2_grouping_loss_bootstrap.csv"
    summary_path = manifest_dir / "formal_v2_grouping_loss_summary.json"
    grouped_frame.to_csv(grouped_path, index=False)
    bootstrap_frame.to_csv(bootstrap_path, index=False)
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "split_seeds": selected_seeds,
        "methods": list(METHODS),
        "confidence_bins": n_bins,
        "binning": "predictor-relative quantile bins fit on calibration-side scores; applied unchanged to test rows",
        "matching": "same confidence bin; risk is measured after matching and never used to construct the match",
        "conditional_summary": "all summary heterogeneity metrics are weighted averages over calibration-defined confidence bins; no summary call is made on the unconditioned test frame",
        "bootstrap": "hierarchical environment-with-replacement then biological_instance_id-within-environment resampling; duplicate environment draws retain unique bootstrap cluster labels",
        "summary": summary_with_ci,
        "grouped_path": grouped_path.relative_to(root).as_posix(),
        "bootstrap_path": bootstrap_path.relative_to(root).as_posix(),
        "status": "formal_v2_grouping_loss_executed",
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--split-seed", action="append", type=int, dest="seeds", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), n_bins=args.n_bins, bootstrap_replicates=args.bootstrap_replicates, seeds=args.seeds), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
