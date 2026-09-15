"""Supplementary Fig. S1: the prospective source-only boundary.

This is deliberately the sole paper-facing supplementary figure.  It consumes
the registered feature-ladder artifact rather than the older exploratory
failure-anatomy and deployment-gate products.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from scripts.figures.common import add_panel_label, clean_axes, save_figure
from scripts.ptl_figure_style import figure_size


LADDERS = ("U", "U+G", "U+G+S", "U+G+S+N")
LADDER_LABELS = ("U", "U+G", "U+G+S", "U+G+S\n+N")
TASKS = {
    "continuous_burden": ("Held-out Spearman $\\rho$", "spearman"),
    "high_burden_top_tercile": ("Held-out AUPRC", "auprc"),
}


def _point_summary(ax: plt.Axes, data: pd.DataFrame, task: str, color: str) -> list[dict]:
    label, field = TASKS[task]
    rows: list[dict] = []
    rng = np.random.default_rng(20260913 if task == "continuous_burden" else 20260914)
    for index, ladder in enumerate(LADDERS):
        values = pd.to_numeric(data.loc[(data["task"] == task) & (data["feature_ladder"] == ladder), field], errors="coerce").dropna()
        if values.empty:
            continue
        jitter = rng.uniform(-0.105, 0.105, len(values))
        ax.scatter(index + jitter, values, s=13, color=color, alpha=.45, edgecolor="white", linewidth=.25, zorder=2)
        q25, med, q75 = values.quantile([.25, .5, .75])
        ax.plot([index, index], [q25, q75], color=color, lw=2.0, zorder=3)
        ax.scatter(index, med, s=35, color=color, edgecolor="white", linewidth=.55, zorder=4)
        baseline = pd.to_numeric(data.loc[(data["task"] == task) & (data["feature_ladder"] == ladder), "positive_prevalence"], errors="coerce").dropna()
        baseline_med = float(baseline.median()) if not baseline.empty else float("nan")
        if task == "high_burden_top_tercile" and np.isfinite(baseline_med):
            ax.scatter(index, baseline_med, marker="_", s=90, color="#6B7280", linewidths=1.0, zorder=5)
        rows.extend({"panel": "B" if task == "continuous_burden" else "C", "task": task, "feature_ladder": ladder,
                     "value": float(value), "median": float(med), "q25": float(q25), "q75": float(q75),
                     "positive_prevalence": baseline_med,
                     "auprc_lift": float(value - baseline_med) if task == "high_burden_top_tercile" and np.isfinite(baseline_med) else float("nan"),
                     "provenance": "reviewer_prospective_feature_ladder.csv"} for value in values)
    ax.set_xticks(range(len(LADDERS)), LADDER_LABELS, fontsize=5.4)
    ax.set_ylabel(label, fontsize=6.6)
    ax.tick_params(axis="y", labelsize=5.4, pad=1.5)
    ax.set_xlim(-.48, len(LADDERS) - .52)
    clean_axes(ax, grid=True)
    return rows


def figure_s1(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    data = pd.read_csv(manifests / "reviewer_prospective_feature_ladder.csv")
    data["target_feature_leakage"] = data["target_feature_leakage"].astype(str).str.lower()
    if len(data) != 144 or set(data["feature_ladder"]) != set(LADDERS) or not data["target_feature_leakage"].eq("false").all():
        raise ValueError("Prospective ladder contract failed: expected 144 source-only rows across four ladders.")
    if data["outer_protocol"].nunique() != 1:
        raise ValueError("Prospective ladder contract failed: multiple outer protocols.")

    width = figure_size("fig6")[0]
    fig = plt.figure(figsize=(width, width * .62), constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=(.31, .345, .345), left=.07, right=.98, top=.77, bottom=.19, wspace=.40)
    rows: list[dict] = []

    ax = fig.add_subplot(grid[0]); add_panel_label(ax, "A")
    ax.text(0.0, 1.06, "Information firewall", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.0, fontweight="bold", color="#263238")
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # Explicitly clear ticks as well: hidden axes can otherwise retain
    # display-space tick extents and trigger false/real collisions in the
    # final raster audit.
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(axis="both", which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    source = FancyBboxPatch((.14, .63), .72, .20, boxstyle="round,pad=.018,rounding_size=.025", facecolor="#EAF1F7", edgecolor="#C4D3E2", linewidth=.75)
    barrier = FancyBboxPatch((.36, .40), .28, .12, boxstyle="round,pad=.014,rounding_size=.02", facecolor="#F2E8E2", edgecolor="#D8B9AA", linewidth=.75)
    target = FancyBboxPatch((.14, .10), .72, .20, boxstyle="round,pad=.018,rounding_size=.025", facecolor="#F8ECEA", edgecolor="#E4C0BB", linewidth=.75)
    for patch in (source, barrier, target): ax.add_patch(patch)
    ax.text(.50, .735, "Source-observable features", ha="center", va="center", fontsize=6.2, fontweight="bold", color="#263238")
    ax.text(.50, .675, "U · G · S · N", ha="center", va="center", fontsize=6.0, color="#46515C")
    ax.text(.50, .46, "INFORMATION FIREWALL", ha="center", va="center", fontsize=5.2, fontweight="bold", color="#8A5949")
    ax.text(.50, .205, "Target-only quantities", ha="center", va="center", fontsize=6.2, fontweight="bold", color="#263238")
    ax.text(.50, .145, "outcomes · burden · risk · counts", ha="center", va="center", fontsize=5.4, color="#46515C")
    ax.annotate("", (.50, .60), (.50, .635), arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#6B7280"})
    ax.annotate("", (.50, .35), (.50, .39), arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#6B7280"})
    rows.append({"panel": "A", "n_rows": 144, "ladders": ";".join(LADDERS), "target_feature_leakage": False,
                 "outer_protocol": data["outer_protocol"].iloc[0], "provenance": "reviewer_prospective_feature_ladder.csv"})

    ax = fig.add_subplot(grid[1]); add_panel_label(ax, "B"); ax.set_title("Continuous burden", loc="left", pad=8, fontweight="bold")
    rows.extend(_point_summary(ax, data, "continuous_burden", "#4C78A8"))
    ax.axhline(0, color="#B8BEC8", lw=.7, ls=(0, (3, 2)), zorder=1)
    ax.set_ylim(-.34, .34)

    ax = fig.add_subplot(grid[2]); ax.text(-.23, 1.06, "C", transform=ax.transAxes, ha="right", va="bottom", fontsize=9.0, fontweight="bold", color="#263238"); ax.set_title("High-burden", loc="left", pad=8, fontweight="bold")
    rows.extend(_point_summary(ax, data, "high_burden_top_tercile", "#D9825B"))
    ax.set_ylim(.32, .73)
    ax.text(.03, .03, "gray dash = empirical prevalence", transform=ax.transAxes, fontsize=5.8, color="#6B7280", va="bottom")

    fig.suptitle("Supplementary Fig. S1   Source-only boundary", x=.07, ha="left", fontsize=11.2, fontweight="bold")
    return save_figure(fig, out_dir, "reliability_transportability_supp_fig1_source_only_forecasting"), pd.DataFrame(rows)


__all__ = ["figure_s1"]
