"""Paired hierarchical bootstrap utilities for environment-first metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from .metrics import evaluate_scores


MetricFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]


def paired_hierarchical_bootstrap(
    frame: pd.DataFrame,
    methods: Iterable[str],
    *,
    n_resamples: int = 200,
    seed: int = 17,
    stratum_columns: tuple[str, ...] = ("predictor", "environment_id", "split_family"),
    environment_column: str = "environment_id",
    group_column: str = "biological_instance_id",
    metric_function: MetricFunction = evaluate_scores,
) -> pd.DataFrame:
    """Return paired macro metric draws with environment-then-instance resampling.

    Each replicate first samples environments within the non-environment
    strata, then samples biological-instance groups within each sampled
    environment. The same draws are reused for every method, preserving paired
    comparisons while matching the intended environment-first estimand.
    """

    methods = tuple(methods)
    required = set(stratum_columns) | {environment_column, group_column, "reliable_label", "continuous_risk", *methods}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"bootstrap frame is missing columns: {missing}")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    rng = np.random.default_rng(seed)
    outer_columns = tuple(column for column in stratum_columns if column != environment_column)
    if not outer_columns:
        outer_groups = [("all", frame)]
    else:
        outer_groups = list(frame.groupby(list(outer_columns), sort=True, dropna=False))
    hierarchy: list[tuple[tuple[object, ...], dict[object, dict[object, np.ndarray]]]] = []
    for key, outer in outer_groups:
        normalized_key = key if isinstance(key, tuple) else (key,)
        environments = {}
        for environment, env_frame in outer.groupby(environment_column, sort=True, dropna=False):
            group_rows = {
                group: values.to_numpy(dtype=int)
                for group, values in env_frame.groupby(group_column, sort=True).groups.items()
            }
            if group_rows:
                environments[environment] = group_rows
        if environments:
            hierarchy.append((normalized_key, environments))

    draws: list[dict[str, object]] = []
    metric_names: tuple[str, ...] | None = None
    for replicate in range(n_resamples):
        per_method: dict[str, list[dict[str, float]]] = {method: [] for method in methods}
        for _, environments in hierarchy:
            environment_ids = np.asarray(list(environments), dtype=object)
            sampled_environments = rng.choice(environment_ids, size=len(environment_ids), replace=True)
            for environment in sampled_environments:
                group_rows = environments[environment]
                groups = np.asarray(list(group_rows), dtype=object)
                sampled_groups = rng.choice(groups, size=len(groups), replace=True)
                rows = np.concatenate([group_rows[group] for group in sampled_groups])
                sample = frame.iloc[rows]
                for method in methods:
                    metrics = metric_function(
                        sample["reliable_label"].to_numpy(),
                        sample[method].to_numpy(),
                        sample["continuous_risk"].to_numpy(),
                    )
                    if metric_names is None:
                        metric_names = tuple(sorted(metrics))
                    per_method[method].append(metrics)
        for method, stratum_metrics in per_method.items():
            for metric in metric_names or ():
                values = [metrics[metric] for metrics in stratum_metrics if np.isfinite(metrics[metric])]
                draws.append(
                    {
                        "replicate": replicate,
                        "method": method,
                        "metric": metric,
                        "value": float(np.mean(values)) if values else np.nan,
                        "resample_level": "environment_then_biological_instance",
                    }
                )
    return pd.DataFrame(draws)


def summarize_bootstrap_ci(
    draws: pd.DataFrame,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Summarize paired bootstrap draws as point estimate and percentile CI."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    alpha = (1.0 - confidence) / 2.0
    rows = []
    for (method, metric), group in draws.groupby(["method", "metric"], sort=True):
        values = group["value"].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "metric": metric,
                "estimate": float(np.mean(values)) if len(values) else np.nan,
                "ci_lower": float(np.quantile(values, alpha)) if len(values) else np.nan,
                "ci_upper": float(np.quantile(values, 1.0 - alpha)) if len(values) else np.nan,
                "n_resamples": int(group["replicate"].nunique()),
                "n_valid": int(len(values)),
                "confidence": confidence,
            }
        )
    return pd.DataFrame(rows)


def summarize_paired_deltas(
    draws: pd.DataFrame,
    baseline_method: str,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Summarize paired method-minus-baseline bootstrap deltas."""

    if baseline_method not in set(draws["method"]):
        raise ValueError(f"baseline method is absent from bootstrap draws: {baseline_method}")
    alpha = (1.0 - confidence) / 2.0
    pivot = draws.pivot_table(index=["replicate", "metric"], columns="method", values="value")
    rows = []
    for method in sorted(set(draws["method"]) - {baseline_method}):
        if method not in pivot:
            continue
        for metric in sorted(set(draws["metric"].unique()) - {"n"}):
            if (metric not in pivot.index.get_level_values("metric")) or (baseline_method not in pivot):
                continue
            values = (pivot.loc[pivot.index.get_level_values("metric") == metric, method]
                      - pivot.loc[pivot.index.get_level_values("metric") == metric, baseline_method]).dropna().to_numpy()
            rows.append({
                "method": method,
                "baseline_method": baseline_method,
                "metric": metric,
                "estimate": float(np.mean(values)) if len(values) else np.nan,
                "ci_lower": float(np.quantile(values, alpha)) if len(values) else np.nan,
                "ci_upper": float(np.quantile(values, 1.0 - alpha)) if len(values) else np.nan,
                "n_resamples": int(draws["replicate"].nunique()),
                "n_valid": int(len(values)),
                "confidence": confidence,
            })
    return pd.DataFrame(rows)
