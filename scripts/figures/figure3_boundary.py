"""Figure 3: measurement depth and the resolution boundary."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.figures.common import (
    CONTEXT_ORDER, CONTRAST_ORDER, DEPTH_ORDER, METRICS, PRIMARY_SOURCE,
    PRIMARY_TARGET, add_panel_label, clean_axes, context_label, metric_color,
    metric_label, num, primary_depth, save_figure,
)
from scripts.ptl_figure_style import figure_size


def _directional_rows() -> list[tuple[str, tuple[str, str], str]]:
    rows = []
    for source in CONTEXT_ORDER:
        for contrast in CONTRAST_ORDER:
            if source in contrast:
                target = contrast[1] if source == contrast[0] else contrast[0]
                rows.append((source, contrast, f"{context_label(source)} → {context_label(target)}"))
    return rows


def _threshold_index(value: float) -> float:
    if not np.isfinite(value) or value >= 1e8:
        return 5.5
    fixed = np.array([10, 20, 40, 80, 160], dtype=float)
    return float(np.argmin(np.abs(fixed - value)))


def _reached(value: float) -> bool:
    """Manifest sentinel values (1e9) encode an unreached threshold."""
    return bool(np.isfinite(value) and value < 1e8)


def _distribution(ax: plt.Axes, values: np.ndarray, y: float, color: str, alpha: float, seed: int) -> None:
    values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    parts = ax.violinplot(values, positions=[y], vert=False, widths=.58, showextrema=False, points=100)
    body = parts["bodies"][0]; body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(alpha); body.set_linewidth(.6)
    rng = np.random.default_rng(seed); jitter = rng.uniform(-.12, .12, len(values))
    ax.scatter(values, np.full(len(values), y) + jitter, s=3.0, color=color, alpha=.28, edgecolor="none", rasterized=True, zorder=2)
    q25, med, q75 = np.quantile(values, [.25, .5, .75])
    ax.plot([q25, q75], [y, y], color=color, lw=2.0, solid_capstyle="round", zorder=4)
    ax.scatter(med, y, s=22, facecolor="white", edgecolor=color, lw=.85, zorder=5)


def figure3(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    depth = pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_summary.csv")
    depth["cell_budget_label"] = depth["cell_budget_label"].astype(str).str.lower()
    resolution = pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_resolution.csv")
    seed = pd.read_csv(manifests / "measurement_seed_summary.csv")
    rows: list[dict] = []

    width = figure_size("fig3")[0]
    fig = plt.figure(figsize=(width, width * .66), constrained_layout=False)
    grid = fig.add_gridspec(2, 1, height_ratios=(.91, .72), hspace=.35, left=.10, right=.985, top=.87, bottom=.15)

    # A: phase-like depth trajectories with a single primary path and faint
    # transfer-level paths behind it.
    ax = fig.add_subplot(grid[0]); add_panel_label(ax, "A", x=-.07); ax.set_title("Depth-dependent trajectories", loc="left", pad=5, fontweight="bold")
    ax.axis("off"); sub = primary_depth(manifests); x = np.arange(len(DEPTH_ORDER), dtype=float)
    for mi, metric in enumerate(METRICS):
        small = ax.inset_axes([.045 + mi * .315, .13, .265, .76])
        for (source, left, right), bg in depth.groupby(["source_environment_id", "left_target_environment_id", "right_target_environment_id"], dropna=False):
            if source == PRIMARY_SOURCE and left == PRIMARY_SOURCE and right == PRIMARY_TARGET:
                continue
            bg = bg.loc[bg["metric"].eq(metric)].set_index("cell_budget_label").reindex(DEPTH_ORDER)
            yy = pd.to_numeric(bg["identifiable_divergence"], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(yy).sum() >= 3:
                small.plot(x, yy, color="#AEB7BF", lw=.5, alpha=.28, zorder=1)
        data = sub.loc[sub["metric"].eq(metric)].set_index("cell_budget_label").reindex(DEPTH_ORDER)
        yy = pd.to_numeric(data["identifiable_divergence"], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(data["identifiable_divergence_ci_low"], errors="coerce").to_numpy(dtype=float); hi = pd.to_numeric(data["identifiable_divergence_ci_high"], errors="coerce").to_numpy(dtype=float)
        small.axhspan(-.01, 0, color="#F4E7E5", alpha=.30, zorder=0); small.axhline(0, color="#8B98A4", lw=.55, ls=(0, (3, 2)), zorder=2)
        small.fill_between(x, lo, hi, color=metric_color(metric), alpha=.15, lw=0, zorder=2); small.plot(x, yy, color=metric_color(metric), lw=1.3, marker="o", ms=3.0, zorder=3)
        small.set_title(metric_label(metric), fontsize=6.2, color=metric_color(metric), pad=2)
        small.set_xticks(x, ["10", "20", "40", "80", "160", "full"], fontsize=5.8); small.set_xlim(-.15, 5.15); small.tick_params(axis="x", pad=1)
        clean_axes(small, grid=True)
        if mi == 0:
            small.set_ylabel(r"signed $D_{\rm adj}$", fontsize=5.8); small.tick_params(axis="y", labelsize=5.8, pad=1)
        else:
            small.tick_params(axis="y", labelleft=False)
        for d, val, lval, hval in zip(DEPTH_ORDER, yy, lo, hi):
            rows.append({"panel": "A", "metric": metric, "contrast": "Ctrl ↔ IFNγ", "depth": d, "identifiable_divergence": val, "ci_low": lval, "ci_high": hval, "provenance": "reliability_transport_measurement_depth_matched_fixed_summary.csv"})

    lower = grid[1].subgridspec(1, 2, width_ratios=(.41, .59), wspace=.28)
    # B: threshold glyph matrix; circles and squares are the two distinct
    # thresholds, empty markers encode not reached.
    ax = fig.add_subplot(lower[0]); add_panel_label(ax, "B"); ax.set_title("Resolution map", loc="left", pad=5, fontweight="bold"); transfers = _directional_rows(); ybase = np.arange(len(transfers))[::-1]
    for yi, (source, contrast, label) in enumerate(transfers):
        y = ybase[yi]; ax.text(-.05, y, label.replace("Co-culture", "Co"), ha="right", va="center", fontsize=5.8)
        for mi, metric in enumerate(METRICS):
            item = resolution.loc[resolution["source_environment_id"].eq(source) & resolution["left_target_environment_id"].eq(contrast[0]) & resolution["right_target_environment_id"].eq(contrast[1]) & resolution["metric"].eq(metric)]
            if item.empty:
                continue
            item = item.iloc[0]; detect = num(item["detectability_threshold_budget"]); resolve = num(item["resolution_90pct_full_budget"]); off = {METRICS[0]: .17, METRICS[1]: 0, METRICS[2]: -.17}[metric]
            if _reached(detect): ax.scatter(_threshold_index(detect), y + off, s=18, marker="o", facecolor=metric_color(metric), edgecolor="white", lw=.35, zorder=3)
            else: ax.scatter(5.5, y + off, s=18, marker="o", facecolor="white", edgecolor=metric_color(metric), lw=.65, zorder=3)
            if _reached(resolve): ax.scatter(_threshold_index(resolve), y + off, s=20, marker="s", facecolor=metric_color(metric), edgecolor="white", lw=.35, zorder=4)
            else: ax.scatter(5.5, y + off, s=20, marker="s", facecolor="white", edgecolor=metric_color(metric), lw=.65, zorder=4)
            rows.append({"panel": "B", "source_environment_id": source, "direction": label, "metric": metric, "detectability_budget": detect, "detectability_status": "reached" if _reached(detect) else "NR", "resolution_budget": resolve, "resolution_status": "reached" if _reached(resolve) else "NR", "provenance": "reliability_transport_measurement_depth_matched_fixed_resolution.csv"})
    ax.set_xlim(-1.7, 5.85); ax.set_ylim(-.7, len(transfers) - .3); ax.set_xticks(range(6), ["10", "20", "40", "80", "160", "NR"]); ax.tick_params(axis="x", labelsize=5.8, pad=1); ax.tick_params(axis="y", left=False, labelleft=False); ax.set_xlabel("cells per pseudoreplicate", fontsize=5.8, labelpad=2); clean_axes(ax, grid=True)
    ax.legend(handles=[Line2D([], [], marker="o", color="#6B7280", linestyle="None", ms=3.2, label="detect"), Line2D([], [], marker="s", color="#6B7280", linestyle="None", ms=3.2, label="resolve")], frameon=False, fontsize=5.1, loc="upper right", bbox_to_anchor=(.98, .98), handletextpad=.15, borderpad=.1)

    # C: finite-depth discrimination against the declared split-half pair-law
    # diagnostic.  This is not the canonical per-seed D_adj summary.
    # The prior seed-level distribution panel is retained in Supplementary
    # Fig. S4; this panel now answers the reviewer-facing question directly.
    ax = fig.add_subplot(lower[1]); add_panel_label(ax, "C", x=-.07, y=1.17); ax.set_title("Split-half pair-law diagnostic", loc="left", pad=5, fontweight="bold")
    ax.axis("off")
    finite = pd.read_csv(manifests / "finite_measurement_rank_discrimination.csv")
    finite["cell_budget_label"] = finite["cell_budget_label"].astype(str).str.lower()
    finite = finite.loc[finite["aggregation_level"].eq("macro_surface_mean")].copy()
    finite["cell_budget_order"] = pd.to_numeric(finite["cell_budget_order"], errors="coerce")
    finite = finite.sort_values(["comparator", "cell_budget_order"], kind="stable")
    xfinite = np.arange(6, dtype=float)
    left = ax.inset_axes([.055, .16, .43, .68])
    d_adj = finite.loc[finite["comparator"].eq("d_adj")].sort_values("cell_budget_order")
    cross = pd.to_numeric(d_adj["cross_estimate"], errors="coerce").to_numpy(dtype=float)
    null = pd.to_numeric(d_adj["same_context_null"], errors="coerce").to_numpy(dtype=float)
    cross_lo = pd.to_numeric(d_adj["cross_bootstrap_ci_low"], errors="coerce").to_numpy(dtype=float)
    cross_hi = pd.to_numeric(d_adj["cross_bootstrap_ci_high"], errors="coerce").to_numpy(dtype=float)
    null_lo = pd.to_numeric(d_adj["same_context_null_bootstrap_ci_low"], errors="coerce").to_numpy(dtype=float)
    null_hi = pd.to_numeric(d_adj["same_context_null_bootstrap_ci_high"], errors="coerce").to_numpy(dtype=float)
    left.fill_between(xfinite, cross_lo, cross_hi, color="#263238", alpha=.10, lw=0, zorder=1)
    left.fill_between(xfinite, null_lo, null_hi, color="#B8BEC8", alpha=.25, lw=0, zorder=1)
    left.plot(xfinite, cross, color="#263238", marker="o", ms=2.8, lw=1.15, zorder=3)
    left.plot(xfinite, null, color="#8B98A4", marker="o", ms=2.4, lw=.9, ls=(0, (2, 1.5)), zorder=3)
    left.axhline(0, color="#B8BEC8", lw=.55, ls=(0, (3, 2)), zorder=0)
    left.set_title(r"$D_{\rm adj}$", fontsize=6.0, pad=2, loc="left", fontweight="bold")
    left.set_xticks(xfinite, ["10", "20", "40", "80", "160", "full"], fontsize=5.0)
    left.set_ylabel("estimate", fontsize=5.4, labelpad=1)
    left.tick_params(axis="y", labelsize=5.2, pad=1)
    left.set_xlabel("cells", fontsize=5.4, labelpad=1)
    left.legend(handles=[Line2D([], [], color="#263238", marker="o", ms=2.6, lw=1.0, label="cross"), Line2D([], [], color="#8B98A4", marker="o", ms=2.4, lw=.9, ls=(0, (2, 1.5)), label="split-half null")], frameon=False, fontsize=4.8, loc="upper left", handletextpad=.15, borderpad=.1)
    clean_axes(left, grid=True)

    right = ax.inset_axes([.56, .16, .40, .68])
    comparator_styles = {
        "kendall_distance": ("#263238", "-", "Kendall"),
        "spearman_distance": ("#6B7280", "--", "Spearman"),
        "top10_jaccard_distance": ("#8B98A4", ":", "Jaccard"),
        "stable_pair_inversion_fraction": ("#4C78A8", "-.", "stable"),
        "d_adj": ("#D9825B", "-", r"$D_{\rm adj}$"),
    }
    for comparator, (color, linestyle, label) in comparator_styles.items():
        group = finite.loc[finite["comparator"].eq(comparator)].sort_values("cell_budget_order")
        if group.empty:
            continue
        margin = pd.to_numeric(group["cross_minus_null_margin"], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(group["margin_ci_low"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(group["margin_ci_high"], errors="coerce").to_numpy(dtype=float)
        right.fill_between(xfinite, lo, hi, color=color, alpha=.06, lw=0, zorder=1)
        right.plot(xfinite, margin, color=color, lw=1.0 if comparator == "d_adj" else .75, marker="o", ms=2.3, linestyle=linestyle, label=label, zorder=3)
        for row in group.itertuples(index=False):
            rows.append({"panel": "C", "comparator": comparator, "budget": str(row.cell_budget_label), "cross_estimate": float(row.cross_estimate), "same_context_null": float(row.same_context_null), "cross_minus_null_margin": float(row.cross_minus_null_margin), "margin_ci_low": float(row.margin_ci_low), "margin_ci_high": float(row.margin_ci_high), "fraction_surfaces_lower_ci_gt0": float(row.fraction_surfaces_lower_ci_gt0), "normalized_recovery": float(row.normalized_recovery), "provenance": "finite_measurement_rank_discrimination.csv"})
    right.axhline(0, color="#B8BEC8", lw=.55, ls=(0, (3, 2)), zorder=0)
    right.set_title("cross − split-half null", fontsize=6.0, pad=2, loc="left", fontweight="bold")
    right.set_xticks(xfinite, ["10", "20", "40", "80", "160", "full"], fontsize=5.0)
    right.set_ylabel("margin", fontsize=5.4, labelpad=1)
    right.tick_params(axis="y", labelsize=5.2, pad=1)
    right.set_xlabel("cells", fontsize=5.4, labelpad=1)
    clean_axes(right, grid=True)

    # Keep the five-comparator key in the title band. The previous in-axis
    # legend covered the low-budget margins in the compact right inset.
    comparator_handles = [
        Line2D([], [], color=color, lw=1.0 if comparator == "d_adj" else .8,
               linestyle=linestyle, marker="o", ms=2.2, label=label)
        for comparator, (color, linestyle, label) in comparator_styles.items()
    ]
    fig.legend(
        handles=comparator_handles,
        frameon=False,
        fontsize=4.7,
        loc="upper right",
        bbox_to_anchor=(.985, .925),
        ncol=5,
        handlelength=1.25,
        handletextpad=.15,
        columnspacing=.55,
        borderpad=.1,
    )
    fig.suptitle("Fig. 3   Measurement depth determines when reordering is resolvable", fontsize=10.8, fontweight="bold", x=.02, y=.985, ha="left")
    return save_figure(fig, out_dir, "reliability_transportability_fig3_measurement_boundary"), pd.DataFrame(rows)


__all__ = ["figure3"]
