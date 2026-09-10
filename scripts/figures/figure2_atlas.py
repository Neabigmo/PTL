"""Figure 2: context-transfer atlas and burden distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figures.common import (
    ATLAS_DATASET, CONTEXT_ORDER, CONTRAST_ORDER, METRICS, add_panel_label,
    clean_axes, context_label, metric_color, metric_label, num, rank_flow,
    set_title, state_counts, save_figure,
)
from scripts.figures.glyphs import atlas_cell
from scripts.figures.ribbons import rank_braid


def figure2(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    atlas = pd.read_csv(manifests / "reliability_transport_atlas.csv"); frangieh = atlas.loc[atlas["dataset"].eq(ATLAS_DATASET) & atlas["metric"].isin(METRICS)].copy(); rows: list[dict] = []
    if len(frangieh) != 27: raise ValueError("Figure 2 requires the 27-row Frangieh atlas")
    flow, labels = rank_flow(manifests); failure = pd.read_csv(manifests / "reliability_transport_failure_anatomy.csv"); complete = failure.loc[failure["status"].eq("executed")].copy()
    fig = plt.figure(figsize=(14, 8.1), constrained_layout=False); grid = fig.add_gridspec(2, 12, height_ratios=(1.45, .92), hspace=.55, wspace=.55)

    ax = fig.add_subplot(grid[0, 0:3]); add_panel_label(ax, "A"); set_title(ax, "Rank flow", "ten source-only quantile exemplars")
    rank_braid(ax, flow, labels, [context_label(v) for v in CONTEXT_ORDER], palette=["#0072B2", "#C44E52", "#009E73", "#D55E00"]); ax.set_title("Rank flow", loc="left", fontsize=9, pad=8, fontweight="bold")
    rows.extend({"panel": "A", **item.to_dict(), "provenance": "formal_v2_reliability_reordering_perturbations.csv + formal_v2_reliability_predictions.csv"} for _, item in flow.iterrows())

    ax = fig.add_subplot(grid[0, 3:12]); add_panel_label(ax, "B"); set_title(ax, "Context-transfer atlas", "effect-size geometry and rank-shift identifiability; color is the metric-specific signed component")
    metric_data = frangieh.copy(); metric_data["measurement_identifiable"] = pd.to_numeric(metric_data["measurement_identifiable"], errors="coerce"); vmax = max(abs(metric_data["measurement_identifiable"].min()), abs(metric_data["measurement_identifiable"].max()), .01)
    for mi, metric in enumerate(METRICS):
        mini = ax.inset_axes([.02 + mi * .33, .10, .29, .80]); mini.set_facecolor("#F4F5F6"); mini.set_xlim(-.65, 2.65); mini.set_ylim(2.65, -.65); mini.set_xticks(range(3), ["Ctrl↔Co", "Ctrl↔IFN", "Co↔IFN"], rotation=25, ha="right", fontsize=5.4); mini.set_yticks(range(3), [context_label(v) for v in CONTEXT_ORDER], fontsize=5.4); mini.set_title(metric_label(metric), loc="left", fontsize=6.7, color=metric_color(metric), fontweight="bold"); mini.grid(False)
        data = metric_data.loc[metric_data["metric"].eq(metric)]
        for ri, source in enumerate(CONTEXT_ORDER):
            for ci, contrast in enumerate(CONTRAST_ORDER):
                cell = data.loc[data["source_environment_id"].eq(source) & data["left_target_environment_id"].eq(contrast[0]) & data["right_target_environment_id"].eq(contrast[1])]
                if cell.empty: continue
                item = cell.iloc[0]; value = num(item["measurement_identifiable"]); atlas_cell(mini, ci, ri, value, num(item["measurement_identifiable_ci_low"]), num(item["joint_identifiable_ci_low"]), vmax)
                rows.append({"panel": "B", "metric": metric, "source_environment_id": source, "contrast": f"{context_label(contrast[0])} ↔ {context_label(contrast[1])}", "measurement_identifiable": value, "measurement_ci_low": item["measurement_identifiable_ci_low"], "measurement_ci_high": item["measurement_identifiable_ci_high"], "joint_identifiable": item["joint_identifiable"], "joint_ci_low": item["joint_identifiable_ci_low"], "joint_ci_high": item["joint_identifiable_ci_high"], "provenance": "reliability_transport_atlas.csv"})
    ax.text(.50, .02, "filled dot = measurement lower interval > 0 · ring = joint lower interval > 0 · size = |signed value|", transform=ax.transAxes, ha="center", fontsize=6.2, color="#46515C"); ax.axis("off")

    ax = fig.add_subplot(grid[1, 0:8]); add_panel_label(ax, "C"); set_title(ax, "Reordering burden is distributed across perturbations", "complete canonical rows; metric colors are kept stable")
    for yi, metric in enumerate(METRICS):
        vals = pd.to_numeric(complete.loc[complete["metric"].eq(metric), "reordering_burden"], errors="coerce").dropna(); vals = vals[np.isfinite(vals)]
        if len(vals):
            parts = ax.violinplot(vals, positions=[yi], vert=False, widths=.65, showmeans=False, showmedians=True, showextrema=False); parts["bodies"][0].set_facecolor(metric_color(metric)); parts["bodies"][0].set_alpha(.28); parts["bodies"][0].set_edgecolor(metric_color(metric)); ax.scatter(vals.sample(min(110, len(vals)), random_state=20260910), np.full(min(110, len(vals)), yi) + np.linspace(-.13, .13, min(110, len(vals))), s=4.5, alpha=.32, color=metric_color(metric)); ax.text(.98, yi, f"n={len(vals)} · median={vals.median():.3f}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=5.8)
            rows.append({"panel": "C", "metric": metric, "n_complete": len(vals), "median": vals.median(), "q25": vals.quantile(.25), "q75": vals.quantile(.75), "axis_scale": "linear", "provenance": "reliability_transport_failure_anatomy.csv"})
    ax.axvline(0, color="#303840", lw=.7); ax.set_yticks(range(3), [metric_label(m) for m in METRICS]); ax.set_xlabel("reordering burden"); clean_axes(ax, grid=True)

    ax = fig.add_subplot(grid[1, 8:12]); add_panel_label(ax, "D"); set_title(ax, "Independent context replication", "Nadig aggregate-only: registered limitation, no invented per-label number")
    ax.axis("off"); ax.add_patch(plt.Rectangle((.05, .22), .90, .60, facecolor="#F4F5F6", edgecolor="#AAB4BC", lw=.7)); ax.text(.50, .65, "BLOCKED / NOT AVAILABLE", ha="center", fontsize=9, fontweight="bold", color="#46515C"); ax.text(.50, .49, "Nadig has no shared per-label target\noutcome for leave-dataset-out transport.", ha="center", va="center", fontsize=7.0); ax.text(.50, .29, "No proxy · no zero imputation · no oracle", ha="center", fontsize=6.4, color="#9E2A2B", fontweight="bold")
    rows.append({"panel": "D", "status": "unavailable", "dataset": "Nadig", "reason": "aggregate-only; no shared per-label target", "numeric_result": False, "provenance": "Nadig Claim Lock"})
    fig.suptitle("Figure 2 | Reliability transport is an atlas of context-specific ordering shifts", fontsize=12, fontweight="bold"); fig.text(.5, .012, "The atlas makes context and metric distinct: glyph position carries biological context, while color and marker layers carry metric and identifiability evidence.", ha="center", fontsize=7.0, color="#46515C")
    return save_figure(fig, out_dir, "reliability_transportability_fig2_transport_atlas"), pd.DataFrame(rows)


__all__ = ["figure2"]
