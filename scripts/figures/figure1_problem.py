"""Figure 1: graphical statement of source-frozen reliability transport."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figures.common import (
    CONTEXT_ORDER, DEPTH_ORDER, METRICS, PRIMARY_METRIC, PRIMARY_SOURCE,
    PRIMARY_TARGET, add_panel_label, clean_axes, context_label, metric_color,
    metric_label, num, primary_depth, rank_flow, rank_table, set_title,
    primary_risks, save_figure,
)
from scripts.figures.ribbons import rank_braid


def figure1(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    flow, labels = rank_flow(manifests); depth = primary_depth(manifests); risks = primary_risks(manifests); ranks = rank_table(manifests); rows: list[dict] = []
    fig = plt.figure(figsize=(14, 8.1), constrained_layout=False); grid = fig.add_gridspec(3, 12, height_ratios=(.50, 1.55, .85), hspace=.55, wspace=.52)

    ax = fig.add_subplot(grid[0, 0:4]); add_panel_label(ax, "A"); ax.axis("off"); set_title(ax, "Frozen evaluation", "same predictor after context shift")
    ax.text(.30, .46, r"$f_{\theta}^{\mathrm{frozen}}$", ha="center", va="center", fontsize=14, fontweight="bold", bbox={"boxstyle": "round,pad=.42", "facecolor": "#EAF2F8", "edgecolor": "#7A8996"})
    for x, context in zip((.62, .77, .92), CONTEXT_ORDER):
        ax.text(x, .46, context_label(context), ha="center", va="center", fontsize=7, bbox={"boxstyle": "round,pad=.25", "facecolor": "#F4F5F6", "edgecolor": "#AAB4BC"})
        ax.annotate("", xy=(x - .035, .46), xytext=(.42, .46), arrowprops={"arrowstyle": "-|>", "color": "#7A8996", "lw": .8})
    rows.append({"panel": "A", "description": "source-frozen predictor evaluated across three canonical contexts", "status": "conceptual contract", "provenance": "frozen Frangieh protocol"})

    ax = fig.add_subplot(grid[0, 4:8]); add_panel_label(ax, "B"); set_title(ax, "Ordering moves", "rank is the transported object")
    ax.axis("off"); ax.text(.04, .57, "source\nrank", fontsize=7, fontweight="bold", ha="center"); ax.text(.50, .57, "context\nshift", fontsize=7, fontweight="bold", ha="center"); ax.text(.92, .57, "target\nrank", fontsize=7, fontweight="bold", ha="center")
    ax.annotate("", xy=(.84, .43), xytext=(.16, .43), arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "#46515C"}); ax.text(.50, .18, "reliability is not assumed invariant across biology", ha="center", fontsize=6.6, color="#46515C")
    rows.append({"panel": "B", "selection": "source-only rank quantiles 5%, 15%, ..., 95%", "n_labels": len(labels), "provenance": "figure_exemplar_registry.json"})

    ax = fig.add_subplot(grid[0, 8:12]); add_panel_label(ax, "C"); set_title(ax, "State semantics", "direction, tie, and unstable support are separate")
    for y, label, color in [(3, "target higher (+1)", "#C44E52"), (2, "tie", "#7A8996"), (1, "target lower (−1)", "#0072B2"), (0, "unstable (<8 strict)", "#C7CDD2")]:
        ax.scatter(.10, y, s=45, color=color, edgecolor="#303840", linewidth=.4); ax.text(.17, y, label, va="center", fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(-.7, 3.7); ax.axis("off")
    rows.append({"panel": "C", "strict_support_floor": 8, "provenance": "reliability_transport_metric_pair_states.csv"})

    ax = fig.add_subplot(grid[1, 0:12]); add_panel_label(ax, "D"); set_title(ax, "Reliability transport across contexts", "hero: real perturbation rank trajectories; 10 labels from the registered source-only quantile rule")
    rank_braid(ax, rank_table(manifests), labels, [context_label(v) for v in CONTEXT_ORDER], palette=["#0072B2", "#C44E52", "#009E73", "#D55E00"])
    rows.extend({"panel": "D", **row.to_dict(), "selected_exemplar": str(row["perturbation_label"]) in labels, "selection": "source-only rank quantile registry", "provenance": "formal_v2_reliability_reordering_perturbations.csv + formal_v2_reliability_predictions.csv"} for _, row in rank_table(manifests).loc[rank_table(manifests)["perturbation_label"].isin(labels)].iterrows())

    ax = fig.add_subplot(grid[2, 0:6]); add_panel_label(ax, "E"); set_title(ax, "Observed disagreement is nested", "full-depth primary contrast; cross + measurement floor + identifiable component")
    vals = []
    for mi, metric in enumerate(METRICS):
        item = depth.loc[depth["metric"].eq(metric) & depth["cell_budget_label"].eq("full")].iloc[0]; components = [num(item["cross_disagreement"]), num(item["measurement_floor"]), num(item["identifiable_divergence"])]
        left = 0.
        for value, color, label in zip(components, ["#303840", "#B8C0C7", metric_color(metric)], ["cross", "measurement floor", "meas-ID"]):
            ax.barh(mi, value, left=left, color=color, height=.45, alpha=.9, label=label if metric == METRICS[0] else None); left += value
        rows.extend({"panel": "E", "metric": metric, "component": label, "value": value, "depth": "full", "provenance": "reliability_transport_measurement_depth_matched_fixed_summary.csv"} for value, label in zip(components, ["cross", "measurement floor", "identifiable_divergence"]))
    ax.axvline(0, color="#303840", lw=.7); ax.set_yticks(range(3), [metric_label(v) for v in METRICS]); ax.set_xlabel("ordering disagreement"); ax.legend(frameon=False, fontsize=5.6, ncol=3); clean_axes(ax, grid=True)

    ax = fig.add_subplot(grid[2, 6:12]); add_panel_label(ax, "F"); set_title(ax, "The transport changes an experimental shortlist", "actual source/target top-10% sets; budget-dependent decision consequence")
    n, k = len(ranks), max(1, int(np.ceil(.10 * len(ranks)))); source = set(ranks.nsmallest(k, "source_risk")["perturbation_label"]); target = set(ranks.nsmallest(k, "ifng_risk")["perturbation_label"])
    for yi, label in enumerate(sorted(source | target)):
        row = ranks.loc[ranks["perturbation_label"].eq(label)].iloc[0]; color = "#56B4E9" if label in source & target else ("#0072B2" if label in source else "#C44E52")
        ax.plot([row["source_rank"], row["ifng_rank"]], [yi, yi], color=color, lw=1.8); ax.scatter([row["source_rank"], row["ifng_rank"]], [yi, yi], color=color, s=18); ax.text(.01, yi, label, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.8); rows.append({"panel": "F", "perturbation_label": label, "source_rank": row["source_rank"], "target_rank": row["ifng_rank"], "source_selected": label in source, "target_selected": label in target, "budget_fraction": .10, "provenance": "formal_v2_reliability_reordering_perturbations.csv"})
    ax.set_yticks([]); ax.set_xlabel("canonical rank (source → target)"); ax.invert_yaxis(); ax.text(.98, .96, f"retention = {len(source & target)}/{k} = {len(source & target)/k:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.8, fontweight="bold"); clean_axes(ax, grid=True)

    fig.suptitle("Figure 1 | A source-frozen reliability ordering can move under biological context shift", fontsize=12, fontweight="bold"); fig.text(.5, .012, "The study follows one object—reliability ordering—from frozen prediction through identifiable disagreement and downstream top-k choice.", ha="center", fontsize=7.0, color="#46515C")
    return save_figure(fig, out_dir, "reliability_transportability_fig1_graphical_abstract"), pd.DataFrame(rows)


__all__ = ["figure1"]
