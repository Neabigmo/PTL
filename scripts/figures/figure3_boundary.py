"""Figure 3: measurement depth and the identifiability boundary."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figures.common import (
    CONTEXT_ORDER, CONTRAST_ORDER, DEPTH_ORDER, METRICS, PRIMARY_METRIC,
    PRIMARY_SOURCE, PRIMARY_TARGET, add_panel_label, clean_axes, context_label,
    metric_color, metric_label, num, primary_depth, set_title, save_figure,
)


def _resolution_state(lower: float, label: str, resolution_budget: float) -> str:
    if not np.isfinite(lower) or lower <= 0: return "unresolved"
    if np.isfinite(resolution_budget) and {"10": 10, "20": 20, "40": 40, "80": 80, "160": 160, "full": 1e9}[label] >= resolution_budget: return "resolved"
    return "detectable"


def figure3(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    depth = pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_summary.csv"); depth["cell_budget_label"] = depth["cell_budget_label"].astype(str).str.lower()
    resolution = pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_resolution.csv"); rows: list[dict] = []
    fig = plt.figure(figsize=(14, 8.1), constrained_layout=False); grid = fig.add_gridspec(2, 12, height_ratios=(1.62, .92), hspace=.55, wspace=.52)

    ax = fig.add_subplot(grid[0, 0:8]); add_panel_label(ax, "A"); set_title(ax, "Measurement depth determines when transport becomes identifiable", "hero: signed D_meas-ID trajectories for primary Ctrl → IFNγ; 90% intervals")
    sub = primary_depth(manifests); x = np.arange(len(DEPTH_ORDER))
    for metric in METRICS:
        metric_data = sub.loc[sub["metric"].eq(metric)].set_index("cell_budget_label").reindex(DEPTH_ORDER)
        y = pd.to_numeric(metric_data["identifiable_divergence"], errors="coerce"); low = pd.to_numeric(metric_data["identifiable_divergence_ci_low"], errors="coerce"); high = pd.to_numeric(metric_data["identifiable_divergence_ci_high"], errors="coerce")
        ax.plot(x, y, marker="o", ms=4.0, color=metric_color(metric), lw=1.9, label=metric_label(metric)); ax.fill_between(x, low, high, color=metric_color(metric), alpha=.13)
        for depth_label, value, lo, hi in zip(DEPTH_ORDER, y, low, high):
            rows.append({"panel": "A", "metric": metric, "source_environment_id": PRIMARY_SOURCE, "contrast": "Ctrl ↔ IFNγ", "depth": depth_label, "identifiable_divergence": value, "ci_low": lo, "ci_high": hi, "state": "signed; interval shown", "provenance": "reliability_transport_measurement_depth_matched_fixed_summary.csv"})
    ax.axhline(0, color="#303840", lw=.8); ax.set_xticks(x, DEPTH_ORDER); ax.set_xlabel("cells per pseudoreplicate"); ax.set_ylabel(r"signed $D_{\mathrm{meas-ID}}$"); ax.legend(frameon=False, fontsize=6.5); clean_axes(ax, grid=True)

    ax = fig.add_subplot(grid[0, 8:12]); add_panel_label(ax, "B"); set_title(ax, "Threshold strip", "cell-specific detectability and 90% resolution; no universal cutoff")
    cells = [(source, contrast) for source in CONTEXT_ORDER for contrast in CONTRAST_ORDER]; y_positions = np.arange(len(cells));
    for yi, (source, contrast) in enumerate(cells):
        for mi, metric in enumerate(METRICS):
            item = resolution.loc[resolution["source_environment_id"].eq(source) & resolution["left_target_environment_id"].eq(contrast[0]) & resolution["right_target_environment_id"].eq(contrast[1]) & resolution["metric"].eq(metric)]
            if item.empty: continue
            item = item.iloc[0]
            detect = num(item["detectability_threshold_budget"]); resolve = num(item["resolution_90pct_full_budget"])
            for kind, value in (("detect", detect), ("resolve", resolve)):
                label = "NR" if not np.isfinite(value) or value >= 1e8 else ("full" if value >= 1e8 else str(int(value)))
                xx = mi * 2 + (0 if kind == "detect" else .72)
                color = "#009E73" if kind == "resolve" and np.isfinite(value) else ("#E69F00" if kind == "detect" and np.isfinite(value) else "#C7CDD2")
                ax.scatter(xx, yi, s=45, marker="o" if kind == "detect" else "s", color=color, edgecolor="#303840", linewidth=.3); ax.text(xx, yi + .22, label, ha="center", va="bottom", fontsize=4.7)
                rows.append({"panel": "C", "source_environment_id": source, "contrast": f"{context_label(contrast[0])} ↔ {context_label(contrast[1])}", "metric": metric, "threshold_kind": kind, "budget": value, "budget_label": label, "resolution_90pct_full_budget": resolve, "provenance": "reliability_transport_measurement_depth_matched_fixed_resolution.csv"})
    ax.set_xlim(-.5, 5.8); ax.set_ylim(-1, len(cells) - .1); ax.set_xticks([.36, 2.36, 4.36], [metric_label(m) for m in METRICS], fontsize=5.6); ax.set_yticks(y_positions, [f"{context_label(s)} | {context_label(c[0])}↔{context_label(c[1])}" for s, c in cells], fontsize=4.6); ax.grid(axis="y", alpha=.14); ax.text(.02, -.10, "● detect   ■ resolve   NR = not reached", transform=ax.transAxes, fontsize=5.6, color="#46515C"); clean_axes(ax)

    ax = fig.add_subplot(grid[1, 0:7]); add_panel_label(ax, "D"); set_title(ax, "Six distribution ridges across the frozen depth boundary", "three metrics × low/full depth; raw seed-level values")
    items = pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_items.csv", dtype={"cell_budget_label": "string"}); items = items.loc[items["source_environment_id"].eq(PRIMARY_SOURCE) & items["left_target_environment_id"].eq(PRIMARY_SOURCE) & items["right_target_environment_id"].eq(PRIMARY_TARGET) & items["cell_budget_label"].astype(str).isin(["10", "full"])].copy()
    ridge_rows = []
    for ri, (metric, budget) in enumerate([(metric, budget) for metric in METRICS for budget in ("10", "full")]):
        vals = pd.to_numeric(items.loc[items["metric"].eq(metric) & items["cell_budget_label"].astype(str).eq(budget), "identifiable_divergence"], errors="coerce").dropna()
        if vals.empty: continue
        kde_x = np.linspace(vals.min() - .01, vals.max() + .01, 100); bw = max(float(vals.std()) * .35, .001); density = np.exp(-.5 * ((kde_x[:, None] - vals.to_numpy()[None, :]) / bw) ** 2).sum(axis=1); density = density / max(density.max(), 1)
        ax.fill_between(kde_x, ri, ri + density * .72, color=metric_color(metric), alpha=.28); ax.plot(kde_x, ri + density * .72, color=metric_color(metric), lw=.75); ax.scatter(vals, np.full(len(vals), ri + .07), s=3.5, color=metric_color(metric), alpha=.25)
        ax.plot([vals.median(), vals.median()], [ri, ri + .78], color="#303840", lw=1.0); ridge_rows.append({"panel": "D", "metric": metric, "budget_label": budget, "n": len(vals), "median": vals.median(), "q25": vals.quantile(.25), "q75": vals.quantile(.75), "distribution": "seed-level ridge; raw values", "provenance": "reliability_transport_measurement_depth_matched_fixed_items.csv"})
    ax.axvline(0, color="#303840", lw=.7); ax.set_yticks(range(6), [f"{metric_label(metric)} · {budget}" for metric in METRICS for budget in ("10", "full")], fontsize=5.2); ax.set_xlabel(r"signed $D_{\mathrm{meas-ID}}$"); clean_axes(ax, grid=True); rows.extend(ridge_rows)

    ax = fig.add_subplot(grid[1, 7:12]); add_panel_label(ax, "E"); set_title(ax, "Nested disagreement components", "primary contrast; metric colors only for the identifiable component")
    for mi, metric in enumerate(METRICS):
        item = sub.loc[sub["metric"].eq(metric) & sub["cell_budget_label"].eq("full")].iloc[0]
        values = [num(item["cross_disagreement"]), num(item["measurement_floor"]), num(item["identifiable_divergence"])]
        left = 0.
        for value, color, label in zip(values, ["#303840", "#B8C0C7", metric_color(metric)], ["cross", "measurement floor", "meas-ID"]):
            ax.barh(mi, value, left=left, color=color, alpha=.88, height=.42, label=label if mi == 0 else None); left += value
        rows.extend({"panel": "E", "metric": metric, "component": label, "value": value, "depth": "full", "provenance": "reliability_transport_measurement_depth_matched_fixed_summary.csv"} for value, label in zip(values, ["cross", "measurement floor", "identifiable_divergence"]))
        rows.append({"panel": "E", "metric": metric, "uncertainty_type": "descriptive IQR across source×contrast cells", "n_source_contrast_cells": int(len(sub.loc[sub["metric"].eq(metric)])), "resolution_90pct_full_budget": np.nan, "budget": np.nan, "budget_label": "full", "provenance": "reliability_transport_measurement_depth_matched_fixed_summary.csv"})
    ax.axvline(0, color="#303840", lw=.7); ax.set_yticks(range(3), [metric_label(m) for m in METRICS]); ax.set_xlabel("observed ordering disagreement"); ax.legend(frameon=False, fontsize=5.5, ncol=3); clean_axes(ax, grid=True)
    fig.suptitle("Figure 3 | Measurement depth defines the identifiability boundary", fontsize=12, fontweight="bold"); fig.text(.5, .012, "Signed values, cell-specific thresholds, and six raw distribution ridges make the boundary visible without clipping negative divergence or imposing a universal cutoff.", ha="center", fontsize=7.0, color="#46515C")
    return save_figure(fig, out_dir, "reliability_transportability_fig3_measurement_boundary"), pd.DataFrame(rows)


__all__ = ["figure3"]
