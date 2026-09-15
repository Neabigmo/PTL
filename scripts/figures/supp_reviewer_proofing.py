"""Supplementary reviewer-proofing figures S2--S4.

All three builders consume canonical, already materialized artifacts.  They
are deliberately separate from the four main figures: these panels answer
reviewer objections without changing the prespecified headline surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.figures.common import add_panel_label, clean_axes, metric_color, metric_label, save_figure
from scripts.ptl_figure_style import figure_size


METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")
METRIC_MARKERS = {"delta_cosine": "o", "systema_centroid_accuracy": "s", "absolute_effect_rank_agreement": "^"}
TRANSFER_MARKERS = ("o", "s", "^", "D", "P", "X")


def _decision_link(manifests: Path) -> pd.DataFrame:
    # Use the canonical 18-row decision-link artifact, which carries the
    # matched-fixed D_adj uncertainty used for the headline link. The broader
    # four-budget surface is a separate reviewer robustness artifact.
    frame = pd.read_csv(manifests / "reliability_transport_decision_link.csv")
    frame = frame.loc[np.isclose(frame["decision_budget_fraction"].astype(float), 0.10) & frame["metric"].isin(METRICS)].copy()
    if len(frame) != 18 or frame.duplicated(["source_environment_id", "target_environment_id", "metric"]).any():
        raise ValueError("Supplementary Fig. S2 requires the fixed 18-row 10% decision surface")
    return frame.sort_values(["metric", "transfer_id"], kind="stable").reset_index(drop=True)


def figure_s2(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    """Metric-stratified decision association: raw and within-metric ranks."""

    link = _decision_link(manifests)
    stratified_report = json.loads((manifests / "reviewer_decision_metric_stratified.json").read_text(encoding="utf-8"))
    report_rho = float(stratified_report["point_spearman"])
    report_ci = [float(value) for value in stratified_report["within_metric_unordered_pair_bootstrap_ci"]]
    report_bootstrap = stratified_report["bootstrap"]
    if not np.isclose(report_rho, float(spearmanr(
        link.assign(
            d_rank_within_metric=link.groupby("metric")["d_meas_id"].rank(method="average"),
            regret_rank_within_metric=link.groupby("metric")["normalized_regret"].rank(method="average"),
        )["d_rank_within_metric"],
        link.assign(
            d_rank_within_metric=link.groupby("metric")["d_meas_id"].rank(method="average"),
            regret_rank_within_metric=link.groupby("metric")["normalized_regret"].rank(method="average"),
        )["regret_rank_within_metric"],
    ).statistic)):
        raise ValueError("Supplementary Fig. S2 report does not match the plotted 18-row surface")
    if len(report_ci) != 2 or not all(np.isfinite(report_ci)) or int(report_bootstrap["draws"]) <= 0:
        raise ValueError("Supplementary Fig. S2 requires a finite unordered-pair block-bootstrap interval")
    width = figure_size("fig4")[0]
    fig = plt.figure(figsize=(width, width * 0.48), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, left=.10, right=.97, bottom=.19, top=.82, wspace=.46, width_ratios=(1.08, .92))
    rows: list[dict] = []

    ax = fig.add_subplot(grid[0])
    # The unusually long y-label needs a little more left clearance than the
    # shared panel-label helper provides at this aspect ratio.
    add_panel_label(ax, "A", x=-.07, y=1.18)
    ax.set_title("Raw decision surface", loc="left", pad=8, fontweight="bold")
    for metric in METRICS:
        sub = link.loc[link["metric"].eq(metric)]
        x = sub["d_meas_id"].to_numpy(float); y = sub["normalized_regret"].to_numpy(float)
        xerr = np.vstack((x - sub["d_meas_id_ci_low"].to_numpy(float), sub["d_meas_id_ci_high"].to_numpy(float) - x))
        yerr = np.vstack((y - sub["normalized_regret_ci_low"].to_numpy(float), sub["normalized_regret_ci_high"].to_numpy(float) - y))
        ax.errorbar(x, y, xerr=np.maximum(xerr, 0), yerr=np.maximum(yerr, 0), fmt=METRIC_MARKERS[metric], ms=4.4,
                    color=metric_color(metric), ecolor=metric_color(metric), elinewidth=.65, capsize=1.4,
                    markeredgecolor="white", markeredgewidth=.35, label=metric_label(metric), alpha=.92, zorder=3)
    # Keep the diagnostic on the same linear geometry as Fig.4B.  The two
    # panels show the identical 18-row primary surface; changing scales here
    # would make a presentation choice look like a scientific difference.
    x_values = pd.to_numeric(link["d_meas_id"], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(link["normalized_regret"], errors="coerce").to_numpy(dtype=float)
    x_margin = max((np.nanmax(x_values) - np.nanmin(x_values)) * .12, .005)
    y_margin = max((np.nanmax(y_values) - np.nanmin(y_values)) * .14, .12)
    ax.set_xlim(np.nanmin(x_values) - x_margin, np.nanmax(x_values) + x_margin)
    ax.set_ylim(max(0, np.nanmin(y_values) - y_margin), np.nanmax(y_values) + y_margin)
    ax.set_xticks([.05, .10, .15])
    ax.set_xlabel(r"$D_{\mathrm{adj}}$ (full depth)")
    ax.set_ylabel("Normalized regret")
    clean_axes(ax, grid=True)

    ranked = link.copy()
    ranked["d_rank_within_metric"] = ranked.groupby("metric")["d_meas_id"].rank(method="average")
    ranked["regret_rank_within_metric"] = ranked.groupby("metric")["normalized_regret"].rank(method="average")
    ax = fig.add_subplot(grid[1]); add_panel_label(ax, "B", y=1.18)
    ax.set_title("Within-metric ranks", loc="left", pad=8, fontweight="bold")
    ax.plot([.7, 6.3], [.7, 6.3], color="#B8BEC8", lw=.8, ls=(0, (3, 2)), zorder=1)
    summary: list[dict] = []
    for metric in METRICS:
        sub = ranked.loc[ranked["metric"].eq(metric)].sort_values("d_rank_within_metric")
        rho = float(spearmanr(sub["d_rank_within_metric"], sub["regret_rank_within_metric"]).statistic)
        ax.plot(sub["d_rank_within_metric"], sub["regret_rank_within_metric"], color=metric_color(metric), lw=.85, alpha=.58, zorder=2)
        ax.scatter(sub["d_rank_within_metric"], sub["regret_rank_within_metric"], color=metric_color(metric), marker=METRIC_MARKERS[metric], s=25, edgecolor="white", linewidth=.35, zorder=3)
        summary.append({"panel": "B", "metric": metric, "n": int(len(sub)), "spearman": rho, "provenance": "reviewer_decision_metric_stratified.csv"})
    pooled = ranked[["d_rank_within_metric", "regret_rank_within_metric"]].dropna()
    pooled_rho = float(spearmanr(pooled["d_rank_within_metric"], pooled["regret_rank_within_metric"]).statistic)
    ax.set_xlim(.7, 6.3); ax.set_ylim(.7, 6.3)
    ax.set_xlabel("Rank of $D_{\\mathrm{adj}}$ within metric")
    ax.set_ylabel("Rank of regret within metric")
    ax.set_xticks(range(1, 7)); ax.set_yticks(range(1, 7))
    clean_axes(ax, grid=True)

    rows.extend([
        {"panel": "A", "metric": row.metric, "transfer_id": row.transfer_id, "d_meas_id": row.d_meas_id,
         "normalized_regret": row.normalized_regret, "d_meas_id_ci_low": row.d_meas_id_ci_low,
         "d_meas_id_ci_high": row.d_meas_id_ci_high, "normalized_regret_ci_low": row.normalized_regret_ci_low,
         "normalized_regret_ci_high": row.normalized_regret_ci_high, "provenance": "reliability_transport_decision_link.csv"}
        for row in link.itertuples()
    ])
    rows.extend(summary)
    rows.append({
        "panel": "B", "metric": "all_metrics_ranked_within_metric", "n": int(len(pooled)),
        "spearman": pooled_rho, "ci_low": report_ci[0], "ci_high": report_ci[1],
        "bootstrap_draws": int(report_bootstrap["draws"]),
        "bootstrap_unit": report_bootstrap["unit"],
        "interval": report_bootstrap["interval"],
        "provenance": "reviewer_decision_metric_stratified.json",
    })
    metric_handles = [Line2D([], [], marker=METRIC_MARKERS[m], color=metric_color(m), linestyle="None", ms=3.8, label=metric_label(m)) for m in METRICS]
    # Reserve the key for the title band above Panel A only.  Spanning both
    # axes makes the final legend footprint enter Panel B's title region.
    fig.legend(handles=metric_handles, frameon=False, loc="upper center", bbox_to_anchor=(.32, .94), ncol=3, fontsize=5.4, handletextpad=.22, columnspacing=.55, borderpad=.05)
    fig.suptitle("Supplementary Fig. S2   Metric-stratified association", x=.10, ha="left", fontsize=10.8, fontweight="bold")
    return save_figure(fig, out_dir, "reliability_transportability_supp_fig2_metric_stratified_decision"), pd.DataFrame(rows)


def figure_s3(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    """Metric-matched mean-fidelity shift versus ordering transport and regret."""

    data = pd.read_csv(manifests / "metric_matched_mean_fidelity_control.csv")
    if len(data) != 18 or data.duplicated(["source_environment_id", "target_environment_id", "metric"]).any():
        raise ValueError("Supplementary Fig. S3 requires 18 directed metric-matched rows")
    data["transfer_id"] = data["transfer_id"].astype(str)
    transfers = sorted(data["transfer_id"].unique())
    marker_map = {transfer: TRANSFER_MARKERS[i % len(TRANSFER_MARKERS)] for i, transfer in enumerate(transfers)}
    width = figure_size("fig4")[0]
    fig = plt.figure(figsize=(width, width * 0.46), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, left=.10, right=.97, bottom=.20, top=.73, wspace=.52)
    rows: list[dict] = []

    ax = fig.add_subplot(grid[0]); add_panel_label(ax, "A")
    ax.set_title("Mean fidelity vs. order transport", loc="left", pad=8, fontweight="bold")
    for metric in METRICS:
        sub = data.loc[data["metric"].eq(metric)]
        for transfer in transfers:
            item = sub.loc[sub["transfer_id"].eq(transfer)]
            if item.empty:
                continue
            ax.scatter(item["absolute_mean_risk_shift"], item["d_meas_id"], color=metric_color(metric), marker=marker_map[transfer], s=32, edgecolor="white", linewidth=.35, label=metric_label(metric) if transfer == transfers[0] else "_nolegend_", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$|\Delta\bar R_m|$ (matched labels)")
    ax.set_ylabel(r"$D_{\mathrm{adj}}$")
    # Keep the metric key in the reserved title band instead of allowing it
    # to extend into panel B's y-label gutter.
    metric_handles = [Line2D([], [], marker="o", color=metric_color(m), linestyle="None", ms=4.0, label=metric_label(m)) for m in METRICS]
    fig.legend(handles=metric_handles, frameon=False, loc="upper center", bbox_to_anchor=(.62, .915), ncol=3, fontsize=5.7, handletextpad=.25, columnspacing=.75)
    ax.text(.97, .05, "Color = metric\nMarker = directed transfer", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="#6B7280")
    ax.set_xticks([1e-3, 1e-2, 1e-1])
    ax.set_xticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"], fontsize=5.6)
    clean_axes(ax, grid=True)

    ax = fig.add_subplot(grid[1])
    ax.text(-.12, 1.24, "B", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="right")
    ax.set_title("Mean fidelity vs. decision regret", loc="left", pad=8, fontweight="bold")
    for metric in METRICS:
        sub = data.loc[data["metric"].eq(metric)]
        for transfer in transfers:
            item = sub.loc[sub["transfer_id"].eq(transfer)]
            if item.empty:
                continue
            ax.scatter(item["absolute_mean_risk_shift"], item["normalized_regret"], color=metric_color(metric), marker=marker_map[transfer], s=32, edgecolor="white", linewidth=.35, zorder=3)
    rho_shift_d = float(spearmanr(data["absolute_mean_risk_shift"], data["d_meas_id"]).statistic)
    rho_shift_r = float(spearmanr(data["absolute_mean_risk_shift"], data["normalized_regret"]).statistic)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$|\Delta\bar R_m|$ (matched labels)")
    ax.set_ylabel("Random-reference normalized regret")
    ax.set_xticks([1e-3, 1e-2, 1e-1])
    ax.set_xticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"], fontsize=5.6)
    clean_axes(ax, grid=True)

    for row in data.to_dict("records"):
        rows.append({"panel": "A/B", **row, "marker": marker_map[row["transfer_id"]], "provenance": "metric_matched_mean_fidelity_control.csv"})
    fig.suptitle("Supplementary Fig. S3   Mean fidelity and reliability-order transport", x=.10, ha="left", fontsize=11.2, fontweight="bold")
    return save_figure(fig, out_dir, "reliability_transportability_supp_fig3_mean_fidelity_control"), pd.DataFrame(rows)


def figure_s4(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    """Finite-replicate estimator validation with real simulation output."""

    trials_path = manifests / "ordering_identifiability_synthetic_trials.csv"
    summary_path = manifests / "ordering_identifiability_synthetic.csv"
    trials = pd.read_csv(trials_path)
    summary = pd.read_csv(summary_path).set_index("scenario")
    if not {"finite_null", "known_alternative"}.issubset(set(trials["scenario"])):
        raise ValueError("Synthetic validation trials are incomplete")
    width = figure_size("fig4")[0]
    fig = plt.figure(figsize=(width, width * 0.40), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, left=.09, right=.98, bottom=.20, top=.80, wspace=.30)
    rows: list[dict] = []
    colors = {"plugin_delta": "#B8BEC8", "u_delta": "#3A9D8F"}
    labels = {"plugin_delta": "Plug-in / V", "u_delta": "U-corrected"}
    for axis_index, scenario in enumerate(("finite_null", "known_alternative")):
        ax = fig.add_subplot(grid[axis_index]); add_panel_label(ax, "AB"[axis_index])
        ax.set_title("Finite null" if scenario == "finite_null" else "Known alternative", loc="left", pad=8, fontweight="bold")
        sub = trials.loc[trials["scenario"].eq(scenario)]
        values = [sub["plugin_delta"].to_numpy(float), sub["u_delta"].to_numpy(float)]
        violin = ax.violinplot(values, positions=(1, 2), widths=.52, showextrema=False, showmedians=False)
        for body, key in zip(violin["bodies"], ("plugin_delta", "u_delta")):
            body.set_facecolor(colors[key]); body.set_edgecolor(colors[key]); body.set_alpha(.50); body.set_linewidth(.65)
            mean = float(sub[key].mean())
            ax.scatter(1 if key == "plugin_delta" else 2, mean, s=20, color="#263238", zorder=4)
            rows.append({"panel": "AB"[axis_index], "scenario": scenario, "estimator": key, "mean": mean, "provenance": "ordering_identifiability_synthetic_trials.csv"})
        truth = float(summary.loc[scenario, "population_delta"])
        ax.axhline(truth, color="#263238", lw=.9, ls=(0, (3, 2)), zorder=3)
        ax.text(.97, .93, f"truth = {truth:.4f}", transform=ax.transAxes, ha="right", va="top", fontsize=5.8)
        ax.set_xticks((1, 2), ("Plug-in / V", "U-corrected"), fontsize=6.1)
        ax.set_ylabel(r"$\widehat D_{\mathrm{adj}}$" if axis_index == 0 else "")
        clean_axes(ax, grid=True)
    fig.suptitle("Supplementary Fig. S4   Finite-replicate validation of the ordering estimand", x=.09, ha="left", fontsize=11.2, fontweight="bold")
    return save_figure(fig, out_dir, "reliability_transportability_supp_fig4_estimand_validation"), pd.DataFrame(rows)


__all__ = ["figure_s2", "figure_s3", "figure_s4"]
