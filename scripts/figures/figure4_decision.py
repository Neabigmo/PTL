"""Figure 4: from reordering to experimental regret."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr

from scripts.figures.common import METRICS, add_panel_label, clean_axes, decision_summary, full_rank_table, metric_color, metric_label, num, save_figure
from scripts.figures.ribbons import shortlist_alluvial
from scripts.ptl_figure_style import figure_size


FORWARD_TRANSFERS = (
    ("frangieh_melanoma_control", "frangieh_melanoma_coculture", "Ctrl → Co"),
    ("frangieh_melanoma_control", "frangieh_melanoma_ifng", "Ctrl → IFNγ"),
    ("frangieh_melanoma_coculture", "frangieh_melanoma_ifng", "Co → IFNγ"),
)
COMPONENT_COLORS = ("#DCE7F0", "#9FB9CC", "#4C78A8")


def _shortlist(ax: plt.Axes, ranks: pd.DataFrame, rows: list[dict]) -> None:
    """Draw the unique detailed delta-cosine shortlist mechanism."""
    shortlist_alluvial(
        ax,
        ranks,
        rows,
        budget=.10,
        label_fontsize=5.8,
        summary_fontsize=6.0,
        show_summary=False,
        tick_labels=("source", "target"),
        panel="A",
        label_limit=1,
        include_all_labels=True,
        provenance="frangieh_source_frozen_predictions.npz; canonical delta-cosine risk across all 243 shared labels",
    )


def figure4(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    _, link = decision_summary(manifests)
    ranks = full_rank_table(manifests)
    rows: list[dict] = []
    link = link.copy()
    for col in ["d_meas_id", "d_meas_id_ci_low", "d_meas_id_ci_high", "normalized_regret", "normalized_regret_ci_low", "normalized_regret_ci_high"]:
        link[col] = pd.to_numeric(link[col], errors="coerce")
    rho = float(spearmanr(link["d_meas_id"], link["normalized_regret"]).statistic)

    fig = plt.figure(figsize=figure_size("fig4"), constrained_layout=False)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.18, .92), hspace=.52, left=.17, right=.985, top=.88, bottom=.15)
    top = grid[0].subgridspec(1, 2, width_ratios=(.44, .56), wspace=.40)

    ax = fig.add_subplot(top[0]); add_panel_label(ax, "A"); ax.set_title("Delta cosine · Ctrl → IFNγ · top 10%", loc="left", pad=5, fontweight="bold"); _shortlist(ax, ranks, rows)

    ax = fig.add_subplot(top[1]); ax.text(-.08, 1.05, "B", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="right"); ax.set_title("Reordering magnitude ↔ decision regret", loc="left", pad=5, fontweight="bold")
    marker_map = {METRICS[0]: "o", METRICS[1]: "s", METRICS[2]: "^"}
    for _, sub in link.groupby("transfer_id", sort=False):
        sub = sub.sort_values("metric"); ax.plot(sub["d_meas_id"], sub["normalized_regret"], color="#B8BEC8", lw=.65, alpha=.65, zorder=1)
    for metric in METRICS:
        sub = link.loc[link["metric"].eq(metric)].copy()
        for _, item in sub.iterrows():
            x, y = num(item["d_meas_id"]), num(item["normalized_regret"]); xlo, xhi = num(item["d_meas_id_ci_low"]), num(item["d_meas_id_ci_high"]); ylo, yhi = num(item["normalized_regret_ci_low"]), num(item["normalized_regret_ci_high"])
            if np.isfinite([x, y, xlo, xhi, ylo, yhi]).all():
                ax.errorbar(x, y, xerr=[[max(x - xlo, 0)], [max(xhi - x, 0)]], yerr=[[max(y - ylo, 0)], [max(yhi - y, 0)]], fmt="none", ecolor=metric_color(metric), alpha=.55, lw=.58, capsize=1.8, zorder=2)
        ax.scatter(sub["d_meas_id"], sub["normalized_regret"], s=31, marker=marker_map[metric], color=metric_color(metric), edgecolor="white", linewidth=.42, zorder=3, label=metric_label(metric))
        rows.extend({"panel": "B", **item.to_dict(), "rho": rho, "provenance": "reliability_transport_decision_link.csv"} for _, item in sub.iterrows())
    ax.text(.04, .94, rf"$\rho_s={rho:.2f}$" + "\n18 transfer$\\times$metric rows", transform=ax.transAxes, fontsize=5.6, fontweight="bold", linespacing=.9, va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": .45})
    # The observed values occupy a narrow range; linear axes expose the
    # transfer geometry without the distracting minor labels of the old log-x
    # presentation.  The y label carries the exact regret normalization.
    x_values = pd.to_numeric(link["d_meas_id"], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(link["normalized_regret"], errors="coerce").to_numpy(dtype=float)
    x_margin = max((np.nanmax(x_values) - np.nanmin(x_values)) * .12, .005)
    y_margin = max((np.nanmax(y_values) - np.nanmin(y_values)) * .14, .12)
    ax.set_xlim(np.nanmin(x_values) - x_margin, np.nanmax(x_values) + x_margin)
    ax.set_ylim(max(0, np.nanmin(y_values) - y_margin), np.nanmax(y_values) + y_margin)
    ax.set_xlabel(r"measurement-adjusted reordering $D_{\rm adj}$", fontsize=6.6)
    # Keep the long normalization label inside the B-panel gutter rather than
    # pushing it toward the alluvial panel; the panel title already carries
    # the short semantic cue, while this axis label supplies the exact scale.
    ax.set_ylabel(
        "Random-reference\nnormalized regret\n(10% budget)",
        fontsize=6.1,
        labelpad=9,
        linespacing=.9,
    )
    ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis="both", labelsize=5.8, pad=1.0)
    clean_axes(ax, grid=True)
    # Direct labels use the reserved upper data band and keep the 18-point
    # geometry completely free of a legend box.
    for x_pos, metric, marker, label in zip((.56, .76, .95), METRICS, ("●", "■", "▲"), ("Δ cosine", "Systema", "Abs.-effect")):
        ax.text(x_pos, .995, f"{marker} {label}", transform=ax.transAxes,
                ha="center", va="top", fontsize=4.2, color=metric_color(metric),
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88, "pad": .55})

    # C: the new main-text decision panel is the prespecified incremental
    # value analysis. The four-budget association curve remains a compact
    # Supplement diagnostic, so the main figure does not overstate the pooled
    # rho as a unique predictive law.
    ax = fig.add_subplot(grid[1, :]); add_panel_label(ax, "C"); ax.set_title("Decision value beyond metric identity", loc="left", pad=5, fontweight="bold")
    summary = pd.read_csv(manifests / "reviewer_decision_incremental_value.csv")
    model_order = ["M1_metric_plus_mean", "M3_metric_plus_mean_plus_d_adj"]
    model_labels = ["Mean shift\nbeyond metric", "$D_{\\rm adj}$ beyond\nmetric + mean"]
    summary = summary.set_index("model").reindex(model_order).reset_index()
    y_pos = np.arange(len(summary))
    values = np.asarray([summary.iloc[0]["delta_mae_vs_M0"], summary.iloc[1]["delta_mae_vs_M1"]], dtype=float)
    lower = np.asarray([summary.iloc[0]["bootstrap_delta_mae_q05"], summary.iloc[1]["bootstrap_delta_mae_vs_M1_q05"]], dtype=float)
    upper = np.asarray([summary.iloc[0]["bootstrap_delta_mae_q95"], summary.iloc[1]["bootstrap_delta_mae_vs_M1_q95"]], dtype=float)
    xerr = np.vstack((values - lower, upper - values))
    ax.axvline(0, color="#6B7280", lw=.7, ls="--", zorder=0)
    ax.errorbar(values, y_pos, xerr=xerr, fmt="none", ecolor="#4C78A8", elinewidth=1.0, capsize=2.4, zorder=2)
    ax.scatter(values, y_pos, s=28, color="#4C78A8", edgecolor="white", linewidth=.45, zorder=3)
    ax.set_yticks(y_pos, model_labels)
    ax.invert_yaxis()
    ax.set_xlabel("$\\Delta$ MAE (positive = better)", fontsize=6.6)
    ax.set_ylabel("Nested contrast", fontsize=6.2)
    ax.text(.02, .98, "18 OOF observations · 3 context-pair folds · 90% block-bootstrap CI", transform=ax.transAxes, va="top", fontsize=5.2, color="#46515C")
    for index, (y, value, model) in enumerate(zip(y_pos, values, summary["model"])):
        rows.append({"panel": "C", "model": model, "delta_mae": float(value), "bootstrap_q05": float(lower[index]), "bootstrap_q95": float(upper[index]), "contrast_reference": "M0_metric" if index == 0 else "M1_metric_plus_mean", "provenance": "reviewer_decision_incremental_value.csv"})
    xmin = min(np.nanmin(lower) - .008, -.02)
    xmax = max(np.nanmax(upper) + .02, .04)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-.6, len(summary) - .4)
    clean_axes(ax, grid=True); ax.tick_params(axis="both", labelsize=5.3, pad=1)

    fig.suptitle("Fig. 4   From reordering to experimental regret", fontsize=12.5, fontweight="bold", x=.02, ha="left")
    return save_figure(fig, out_dir, "reliability_transportability_fig4_decision_consequence"), pd.DataFrame(rows)


__all__ = ["figure4"]
