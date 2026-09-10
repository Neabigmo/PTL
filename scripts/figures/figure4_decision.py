"""Figure 4: decision consequence of identifiable reliability reordering."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.figures.common import (
    CONTRAST_ORDER, METRICS, PRIMARY_SOURCE, PRIMARY_TARGET, add_panel_label,
    clean_axes, context_label, decision_summary, metric_color, metric_label,
    num, rank_table, set_title, write_source, save_figure,
)
from scripts.figures.ribbons import alluvial


def _topk(ax: plt.Axes, ranks: pd.DataFrame, rows: list[dict]) -> None:
    n, k = len(ranks), max(1, int(np.ceil(.10 * len(ranks))))
    source = set(ranks.nsmallest(k, "source_risk")["perturbation_label"])
    target = set(ranks.nsmallest(k, "ifng_risk")["perturbation_label"])
    labels = sorted(source | target, key=lambda x: (ranks.set_index("perturbation_label").loc[x, "source_rank"], x))
    source_counts = [("retained", len(source & target)), ("dropped", len(source - target)), ("new", len(target - source))]
    target_counts = [("retained", len(source & target)), ("dropped", 0), ("new", len(target - source))]
    alluvial(ax, source_counts, target_counts, {"retained": "#56B4E9", "dropped": "#C44E52", "new": "#009E73"}, "Top-k replacement: source → target")
    source_labels = ", ".join(sorted(source)); target_labels = ", ".join(sorted(target))
    ax.text(.50, .91, f"source top-k: {source_labels}\ntarget top-k: {target_labels}", transform=ax.transAxes, ha="center", va="top", fontsize=4.7, color="#46515C")
    for label in labels:
        row = ranks.loc[ranks["perturbation_label"].eq(label)].iloc[0]
        rows.append({"panel": "A", "perturbation_label": label, "source_rank": row["source_rank"], "target_rank": row["ifng_rank"], "source_selected": label in source, "target_selected": label in target, "budget_fraction": .10, "provenance": "formal_v2_reliability_reordering_perturbations.csv"})


def figure4(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    summary, link = decision_summary(manifests)
    rows: list[dict] = []
    fig = plt.figure(figsize=(14, 8.1), constrained_layout=False)
    grid = fig.add_gridspec(2, 12, height_ratios=(1.55, 1.0), hspace=.52, wspace=.55)

    ax = fig.add_subplot(grid[0, 0:3]); add_panel_label(ax, "A")
    _topk(ax, rank_table(manifests), rows); clean_axes(ax)

    ax = fig.add_subplot(grid[0, 3:9]); add_panel_label(ax, "B")
    set_title(ax, r"Identifiability predicts decision regret", r"full depth · 10% decision budget · 18 directed transfers · $D_{\mathrm{meas-ID}} \rightarrow$ normalized regret")
    link = link.copy(); link["d_meas_id"] = pd.to_numeric(link["d_meas_id"], errors="coerce"); link["normalized_regret"] = pd.to_numeric(link["normalized_regret"], errors="coerce")
    rho = float(spearmanr(link["d_meas_id"], link["normalized_regret"]).statistic)
    for metric in METRICS:
        sub = link.loc[link["metric"].eq(metric)].copy()
        xerr = [sub["d_meas_id"] - sub["d_meas_id_ci_low"], sub["d_meas_id_ci_high"] - sub["d_meas_id"]]
        yerr = [sub["normalized_regret"] - sub["normalized_regret_ci_low"], sub["normalized_regret_ci_high"] - sub["normalized_regret"]]
        ax.errorbar(sub["d_meas_id"], sub["normalized_regret"], xerr=xerr, yerr=yerr, fmt="o", ms=4.8, capsize=2, lw=.65, color=metric_color(metric), ecolor=metric_color(metric), alpha=.88, label=metric_label(metric))
        for _, item in sub.iterrows():
            rows.append({"panel": "B", **item.to_dict(), "uncertainty_type": "90% synchronized macro-bootstrap interval", "provenance": "reliability_transport_decision_link.csv"})
    z = np.polyfit(link["d_meas_id"], link["normalized_regret"], 1); xs = np.linspace(link["d_meas_id"].min(), link["d_meas_id"].max(), 80)
    ax.plot(xs, np.polyval(z, xs), color="#303840", ls=(0, (3, 2)), lw=1.0, zorder=1)
    ax.scatter(link["d_meas_id"], link["normalized_regret"], s=9, color="#303840", alpha=.18, zorder=0)
    ax.text(.03, .97, f"Spearman ρ={rho:.4f}\n95% bootstrap association: q05=.7388, q95=.9003", transform=ax.transAxes, va="top", fontsize=7.1, fontweight="bold", color="#303840")
    ax.text(.03, .08, "Higher identifiable reordering burden is associated with larger regret; descriptive association, not causality.", transform=ax.transAxes, fontsize=6.0, color="#46515C")
    ax.set_xlabel(r"$D_{\mathrm{meas-ID}}$ (signed, full depth)"); ax.set_ylabel("normalized regret (lower = better)"); ax.legend(frameon=False, fontsize=6.0, loc="lower right"); clean_axes(ax, grid=True)

    ax = fig.add_subplot(grid[0, 9:12]); add_panel_label(ax, "C")
    set_title(ax, "Retention and regret across budget", "four frozen decision budgets; transfer-level IQR")
    curve = summary.groupby(["metric", "decision_budget_fraction"], as_index=False).agg(
        retention_point=("retention_point", "median"), normalized_regret_point=("normalized_regret_point", "median"),
        retention_q25=("retention_point", lambda value: value.quantile(.25)), retention_q75=("retention_point", lambda value: value.quantile(.75)),
        regret_q25=("normalized_regret_point", lambda value: value.quantile(.25)), regret_q75=("normalized_regret_point", lambda value: value.quantile(.75)),
    )
    for metric in METRICS:
        sub = curve.loc[curve["metric"].eq(metric)].sort_values("decision_budget_fraction")
        ax.plot(sub["retention_point"], sub["normalized_regret_point"], marker="o", ms=3.7, lw=1.5, color=metric_color(metric), label=metric_label(metric))
        ax.fill_between(sub["retention_point"], sub["regret_q25"], sub["regret_q75"], color=metric_color(metric), alpha=.10)
        for _, item in sub.iterrows():
            ax.annotate(f"{float(item['decision_budget_fraction']):.2g}", (item["retention_point"], item["normalized_regret_point"]), textcoords="offset points", xytext=(2, 2), fontsize=5.0, color=metric_color(metric))
            rows.append({"panel": "C", "metric": metric, "budget_fraction": item["decision_budget_fraction"], "retention": item["retention_point"], "normalized_regret": item["normalized_regret_point"], "retention_q25": item["retention_q25"], "retention_q75": item["retention_q75"], "regret_q25": item["regret_q25"], "regret_q75": item["regret_q75"], "uncertainty_type": "descriptive IQR across directed transfers", "provenance": "reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv"})
    ax.set_xlabel("retention →"); ax.set_ylabel("normalized regret ↓"); ax.legend(frameon=False, fontsize=5.3); clean_axes(ax, grid=True)

    ax = fig.add_subplot(grid[1, 0:6]); add_panel_label(ax, "D")
    set_title(ax, "Retention–regret Pareto trajectories", "arrows show increasing measurement budget")
    for metric in METRICS:
        sub = curve.loc[curve["metric"].eq(metric)].sort_values("decision_budget_fraction")
        ax.plot(sub["normalized_regret_point"], sub["retention_point"], marker="o", ms=4.0, lw=1.6, color=metric_color(metric), label=metric_label(metric))
        for i in range(len(sub) - 1):
            ax.annotate("", xy=(sub.iloc[i + 1]["normalized_regret_point"], sub.iloc[i + 1]["retention_point"]), xytext=(sub.iloc[i]["normalized_regret_point"], sub.iloc[i]["retention_point"]), arrowprops={"arrowstyle": "-|>", "color": metric_color(metric), "lw": .7})
        rows.extend({"panel": "D", "metric": metric, "budget_fraction": item["decision_budget_fraction"], "normalized_regret": item["normalized_regret_point"], "retention": item["retention_point"], "retention_q25": item["retention_q25"], "retention_q75": item["retention_q75"], "regret_q25": item["regret_q25"], "regret_q75": item["regret_q75"], "uncertainty_type": "descriptive IQR across directed transfers", "provenance": "reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv"} for _, item in sub.iterrows())
    ax.set_xlabel("normalized regret (lower = better)"); ax.set_ylabel("retention (higher = better)"); ax.legend(frameon=False, fontsize=6.0); clean_axes(ax, grid=True)

    ax = fig.add_subplot(grid[1, 6:12]); add_panel_label(ax, "E")
    set_title(ax, "Decision-regret decomposition", "10% budget; point estimates, four frozen terms")
    components = [("measurement_floor_regret_point", "measurement floor", "#B8C0C7"), ("excess_regret_point", "excess", "#C44E52"), ("joint_floor_regret_point", "joint floor", "#6F7D88"), ("joint_excess_regret_point", "joint excess", "#009E73")]
    dec = summary.loc[summary["decision_budget_fraction"].eq(.1)].groupby("metric", as_index=False)[[c[0] for c in components]].median()
    for yi, (_, item) in enumerate(dec.iterrows()):
        for ci, (column, label, color) in enumerate(components):
            ax.scatter(item[column], yi + (ci - 1.5) * .11, s=28, color=color, edgecolor="white", linewidth=.3, label=label if yi == 0 else None)
            rows.append({"panel": "E", "metric": item["metric"], "component": label, "value": item[column], "budget_fraction": .1, "provenance": "reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv"})
    ax.axvline(0, color="#303840", lw=.7); ax.set_yticks(range(len(dec)), [metric_label(value) for value in dec["metric"]]); ax.set_xlabel("point regret component"); ax.legend(frameon=False, fontsize=5.4, ncol=2); clean_axes(ax, grid=True)

    fig.suptitle("Figure 4 | Reliability reordering propagates into experimental regret", fontsize=12, fontweight="bold")
    fig.text(.5, .012, "The hero association is fixed to full-depth D_meas-ID and normalized regret; budget curves and decomposition retain transfer-level uncertainty without pooling interval endpoints.", ha="center", fontsize=7.0, color="#46515C")
    return save_figure(fig, out_dir, "reliability_transportability_fig4_decision_consequence"), pd.DataFrame(rows)


__all__ = ["figure4"]
