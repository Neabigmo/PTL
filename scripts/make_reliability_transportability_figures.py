"""Generate the five narrative-first reliability-transportability figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ptl_figure_style import (  # noqa: E402
    CONTEXT_ORDER, CONTRAST_ORDER, DEPTH_ORDER, add_metric_legend,
    add_panel_label, apply_style, clean_axes, context_label, contrast_label,
    metric_color, metric_label, save_figure,
)

METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")
ATLAS_DATASET = "FrangiehIzar2021_RNA"


def _source_path(root: Path, name: str) -> Path:
    path = root / "results/figures/reliability_transportability/source_tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_source(root: Path, name: str, frame: pd.DataFrame) -> str:
    path = _source_path(root, name)
    frame.to_csv(path, index=False)
    return path.relative_to(root).as_posix()


def _rowwise_endpoint(frame: pd.DataFrame) -> pd.Series:
    return frame["source_environment_id"].eq(frame["left_target_environment_id"]) | frame["source_environment_id"].eq(frame["right_target_environment_id"])


def _depth_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "cell_budget_label" not in frame.columns:
        frame["cell_budget_label"] = frame["cell_budget"].map(lambda x: "full" if float(x) >= 1e8 else str(int(float(x))))
    frame["cell_budget_label"] = frame["cell_budget_label"].astype(str)
    return frame


def _draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], **kwargs: Any) -> None:
    defaults = {"arrowstyle": "-|>", "lw": 1.2, "color": "#46515C", "mutation_scale": 12}
    defaults.update(kwargs)
    ax.annotate("", xy=end, xytext=start, arrowprops=defaults)


def figure1(out_dir: Path) -> tuple[list[str], pd.DataFrame]:
    fig = plt.figure(figsize=(14, 4.6))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.5, 1.1), wspace=0.12)
    ax_a, ax_b, ax_c = [fig.add_subplot(grid[0, i]) for i in range(3)]
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1); ax_a.axis("off"); add_panel_label(ax_a, "A")
    ax_a.text(0.04, 0.92, "Biological shift", fontsize=10, fontweight="bold", va="top")
    ax_a.text(0.04, 0.84, "the predictor is held fixed", color="#46515C", va="top")
    ax_a.text(0.5, 0.54, r"$f_{\theta}$", fontsize=18, ha="center", va="center", fontweight="bold", bbox={"boxstyle": "circle,pad=0.35", "facecolor": "#F4F7F9", "edgecolor": "#46515C", "lw": 1.2})
    positions = [(0.17, 0.28), (0.5, 0.16), (0.83, 0.28)]
    for pos, context in zip(positions, ("frangieh_melanoma_control", "frangieh_melanoma_coculture", "frangieh_melanoma_ifng")):
        ax_a.text(*pos, context_label(context), ha="center", va="center", fontsize=9, bbox={"boxstyle": "round,pad=0.28", "facecolor": "#EAF2F8", "edgecolor": "#7A8996", "lw": 0.8})
        _draw_arrow(ax_a, (0.5, 0.43), (pos[0], pos[1] + 0.09), color="#7A8996", lw=0.9)
    ax_a.text(0.5, 0.03, "same source-frozen predictor; three biological outcomes", ha="center", fontsize=7.8, color="#46515C")
    clean_axes(ax_a)

    ax_b.set_xlim(-0.25, 8.25); ax_b.set_ylim(-0.45, 8.65); ax_b.set_xticks([0, 8], ["source", "target"]); ax_b.set_yticks([]); add_panel_label(ax_b, "B")
    ax_b.set_title("Reliability ordering moves", loc="left", pad=14)
    ax_b.axhspan(3.55, 4.45, color="#F2F4F5", zorder=0)
    source_rank = np.arange(8); target_rank = np.array([1.0, 6.2, 2.4, 4.1, 3.8, 0.8, 6.9, 5.1])
    for i, (left, right) in enumerate(zip(source_rank, target_rank)):
        color = "#B6C0C8" if i not in (1, 6) else "#C44E52"; lw = 1.0 if i not in (1, 6) else 2.4
        ax_b.plot([0, 8], [left, right], color=color, lw=lw, alpha=0.92, zorder=1); ax_b.scatter([0, 8], [left, right], s=17, color=color, zorder=2)
    ax_b.scatter([8], [3.8], s=58, facecolors="white", edgecolors="#D55E00", linewidths=1.5, zorder=3)
    ax_b.text(0.0, 8.35, "ranked confidence", ha="left", fontsize=7.5, color="#46515C"); ax_b.text(8.0, 8.35, "same labels, target outcome", ha="right", fontsize=7.5, color="#46515C")
    ax_b.text(4.0, -0.26, r"$D_{\mathrm{cross}}-D_{\mathrm{noise}}\;\longrightarrow D_{\mathrm{identifiable}}$", ha="center", fontsize=10, fontweight="bold")
    ax_b.text(4.0, 7.9, "crossing trajectories = observed reordering", ha="center", fontsize=7.7, color="#C44E52")
    clean_axes(ax_b, grid=True)

    ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1); ax_c.axis("off"); add_panel_label(ax_c, "C")
    ax_c.text(0.03, 0.92, "Measurement / decision boundary", fontsize=10, fontweight="bold", va="top")
    ax_c.text(0.03, 0.84, "depth resolves some, but not all, reordering", fontsize=7.8, color="#46515C", va="top")
    x = np.linspace(0.07, 0.92, 7); y = 0.61 + 0.18 * (1 - np.exp(-4 * (x - x.min())))
    ax_c.plot(x, y, color="#0072B2", lw=2); ax_c.fill_between(x, y - 0.035, y + 0.035, color="#0072B2", alpha=0.13); ax_c.axhline(y[-1] * 0.95, color="#7A8996", lw=0.8, ls="--")
    ax_c.text(0.08, 0.49, "10", fontsize=7, color="#46515C"); ax_c.text(0.88, 0.49, "full", fontsize=7, color="#46515C", ha="right")
    ax_c.text(0.5, 0.41, "depth ↑ · uncertainty narrows · detectability is metric-specific", ha="center", fontsize=7.3)
    ax_c.plot([0.08, 0.08, 0.92, 0.92], [0.24, 0.34, 0.34, 0.24], color="#46515C", lw=1.0)
    ax_c.fill_between([0.08, 0.38], [0.24, 0.24], [0.34, 0.34], color="#56B4E9", alpha=0.8); ax_c.fill_between([0.38, 0.67], [0.24, 0.24], [0.34, 0.34], color="#E69F00", alpha=0.8); ax_c.fill_between([0.67, 0.92], [0.24, 0.24], [0.34, 0.34], color="#CC79A7", alpha=0.8)
    ax_c.text(0.08, 0.18, "source top-k", fontsize=7.2, ha="left"); ax_c.text(0.92, 0.18, "target top-k", fontsize=7.2, ha="right"); ax_c.text(0.5, 0.08, "retention ↓   regret ↑ when ordering is not identifiable", ha="center", fontsize=7.7, fontweight="bold")
    clean_axes(ax_c)
    fig.suptitle("Reliability transportability: from observed reordering to a measurable decision boundary", fontsize=12, fontweight="bold")
    source = pd.DataFrame([
        {"panel": "A", "element": "biological_shift", "description": "One frozen predictor evaluated across Ctrl, Co-culture, and IFNγ outcomes", "status": "conceptual schematic"},
        {"panel": "B", "element": "ordering", "description": "Crossing trajectories illustrate rank displacement and a tie", "status": "conceptual schematic"},
        {"panel": "C", "element": "boundary", "description": "Depth, noise, selection retention, and regret are linked", "status": "conceptual schematic"},
    ])
    return save_figure(fig, out_dir, "reliability_transportability_fig1_graphical_abstract"), source


def figure2(out_dir: Path, manifests: Path) -> tuple[list[str], pd.DataFrame]:
    atlas = pd.read_csv(manifests / "reliability_transport_atlas.csv")
    atlas = atlas.loc[atlas["dataset"].eq(ATLAS_DATASET) & atlas["metric"].isin(METRICS)].copy()
    atlas["contrast"] = list(zip(atlas["left_target_environment_id"], atlas["right_target_environment_id"]))
    atlas["contrast_order"] = atlas["contrast"].map({pair: i for i, pair in enumerate(CONTRAST_ORDER)})
    source_rows: list[dict[str, Any]] = []
    fig = plt.figure(figsize=(13, 7.3)); grid = fig.add_gridspec(2, 3, height_ratios=(1.55, 0.9), hspace=0.35, wspace=0.25)
    positive = pd.to_numeric(atlas["measurement_identifiable"], errors="coerce"); vmax = max(float(np.nanpercentile(positive.dropna(), 98)) if positive.notna().any() else 0.05, 0.05); norm = Normalize(vmin=0, vmax=vmax); cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#D9DDE2")
    for col, metric in enumerate(METRICS):
        ax = fig.add_subplot(grid[0, col]); add_panel_label(ax, chr(ord("A") + col)); sub = atlas.loc[atlas["metric"].eq(metric)].copy(); matrix = np.full((3, 3), np.nan)
        for ri, source_id in enumerate(CONTEXT_ORDER):
            for ci, contrast in enumerate(CONTRAST_ORDER):
                row = sub.loc[sub["source_environment_id"].eq(source_id) & sub["contrast"].map(lambda value: value == contrast)]
                if row.empty: continue
                item = row.iloc[0]; matrix[ri, ci] = float(item["measurement_identifiable"])
                source_rows.append({"panel": "A", "metric": metric, "source_environment_id": source_id, "contrast": contrast_label(*contrast), "measurement_identifiable": item["measurement_identifiable"], "measurement_identifiable_ci_low": item.get("measurement_identifiable_ci_low", np.nan), "measurement_identifiable_ci_high": item.get("measurement_identifiable_ci_high", np.nan), "joint_identifiable": item.get("joint_identifiable", np.nan), "joint_identifiable_ci_low": item.get("joint_identifiable_ci_low", np.nan), "evidence_tier": item.get("evidence_tier", np.nan), "availability": item.get("availability", "available"), "provenance": item.get("provenance", "reliability_transport_atlas.csv")})
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="auto")
        for ri, source_id in enumerate(CONTEXT_ORDER):
            for ci, contrast in enumerate(CONTRAST_ORDER):
                row = sub.loc[sub["source_environment_id"].eq(source_id) & sub["contrast"].map(lambda value: value == contrast)]; tier = str(row.iloc[0].get("evidence_tier", "")) if not row.empty else ""; ax.add_patch(Rectangle((ci - 0.48, ri - 0.48), 0.96, 0.96, fill=False, ec="#303840", lw=0.8, ls="-" if tier == "1" else "--"))
                if row.empty:
                    ax.add_patch(Rectangle((ci - 0.48, ri - 0.48), 0.96, 0.96, fill=False, ec="#7A8996", lw=0.8, hatch="///")); continue
                item = row.iloc[0]; value = float(item["measurement_identifiable"]); ax.text(ci, ri, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value > vmax * 0.55 else "#111111")
                if float(item.get("measurement_identifiable_ci_low", np.nan)) > 0: ax.scatter(ci + 0.31, ri - 0.31, s=18, marker=".", color="#111111", zorder=4)
                if float(item.get("joint_identifiable_ci_low", np.nan)) > 0: ax.scatter(ci + 0.29, ri + 0.29, s=22, marker="o", facecolors="none", edgecolors="#111111", linewidths=1, zorder=4)
        ax.set_title(metric_label(metric), loc="left", pad=10); ax.set_xticks(range(3), [contrast_label(*p) for p in CONTRAST_ORDER], rotation=25, ha="right"); ax.set_yticks(range(3), [context_label(v) for v in CONTEXT_ORDER]); ax.set_xlabel("biological contrast"); ax.set_ylabel("source context") if col == 0 else None; clean_axes(ax)
    cbar = fig.colorbar(im, ax=fig.axes[:3], shrink=0.85, pad=0.02); cbar.set_label(r"$D_{\mathrm{meas-ID}}$")

    nadig = pd.read_csv(manifests / "reliability_transport_atlas.csv"); nadig = nadig.loc[nadig["dataset"].astype(str).str.contains("Nadig", case=False, na=False) & nadig["metric"].isin(METRICS)].copy(); nadig = nadig.loc[_rowwise_endpoint(nadig)].copy()
    for col, metric in enumerate(METRICS):
        ax = fig.add_subplot(grid[1, col]); add_panel_label(ax, chr(ord("D") + col)); sub = nadig.loc[nadig["metric"].eq(metric)].copy()
        if sub.empty: ax.text(0.5, 0.5, "Nadig unavailable", ha="center", va="center", color="#7A8996")
        else:
            sub["target"] = np.where(sub["source_environment_id"].eq(sub["left_target_environment_id"]), sub["right_target_environment_id"], sub["left_target_environment_id"])
            for yi, (_, row) in enumerate(sub.iterrows()):
                value = float(row.get("measurement_identifiable", row.get("observed_D", np.nan))); low = float(row.get("measurement_identifiable_ci_low", np.nan)); high = float(row.get("measurement_identifiable_ci_high", np.nan)); xerr = [[value - low], [high - value]] if np.isfinite(low + high) else None
                ax.errorbar(value, yi, xerr=xerr, fmt="o", color=metric_color(metric), capsize=2); ax.text(0, yi, f"{context_label(row['source_environment_id'])} → {context_label(row['target'])}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7); source_rows.append({"panel": "B", "dataset": row.get("dataset", "Nadig"), "metric": metric, "source_environment_id": row["source_environment_id"], "target_environment_id": row["target"], "value": value, "provenance": "reliability_transport_atlas.csv"})
            ax.set_ylim(-0.8, max(len(sub) - 0.2, 0.8)); ax.set_yticks([])
        ax.set_title(f"Independent replication · {metric_label(metric)}", fontsize=8.5, loc="left"); ax.set_xlabel(r"$D_{\mathrm{meas-ID}}$"); clean_axes(ax, grid=True)
    fig.suptitle("Figure 2 | Biological context reshapes reliability ordering", fontsize=12, fontweight="bold"); fig.text(0.5, 0.01, "Frangieh: 3 source contexts × 3 contrasts; dot = measurement-identifiable lower CI > 0; ring = joint-identifiable lower CI > 0; border = Tier 1/2. Nadig is separate.", ha="center", fontsize=7.2, color="#46515C")
    return save_figure(fig, out_dir, "reliability_transportability_fig2_transport_atlas"), pd.DataFrame(source_rows)


def figure3(out_dir: Path, manifests: Path) -> tuple[list[str], pd.DataFrame]:
    depth = _depth_labels(pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_summary.csv", dtype={"cell_budget_label": "string"})); depth = depth.loc[depth["metric"].isin(METRICS)].copy(); resolution_path = manifests / "reliability_transport_measurement_depth_matched_fixed_resolution.csv"; resolution = pd.read_csv(resolution_path) if resolution_path.is_file() else pd.DataFrame(); resolution = resolution.loc[resolution["metric"].isin(METRICS)].copy() if not resolution.empty else resolution
    rows: list[dict[str, Any]] = []; fig = plt.figure(figsize=(13, 8)); grid = fig.add_gridspec(2, 3, height_ratios=(1.7, 1.0), hspace=0.35, wspace=0.22)
    for col, metric in enumerate(METRICS):
        ax = fig.add_subplot(grid[0, col]); add_panel_label(ax, chr(ord("A") + col)); sub = depth.loc[depth["metric"].eq(metric)].copy(); matrix = np.full((9, 6), np.nan); row_labels: list[str] = []
        for ri, source_id in enumerate(CONTEXT_ORDER):
            for ci, contrast in enumerate(CONTRAST_ORDER):
                rr = ri * 3 + ci; row_labels.append(f"{context_label(source_id)} | {contrast_label(*contrast)}"); item = sub.loc[sub["source_environment_id"].eq(source_id) & sub["left_target_environment_id"].eq(contrast[0]) & sub["right_target_environment_id"].eq(contrast[1])]; full = item.loc[item["cell_budget_label"].eq("full"), "identifiable_divergence"]; denom = float(full.iloc[0]) if not full.empty else np.nan; res = resolution.loc[resolution["source_environment_id"].eq(source_id) & resolution["left_target_environment_id"].eq(contrast[0]) & resolution["right_target_environment_id"].eq(contrast[1]) & resolution["metric"].eq(metric)] if not resolution.empty else pd.DataFrame(); resolution_budget = float(res.iloc[0]["resolution_90pct_full_budget"]) if not res.empty else np.nan; detect_budget = float(res.iloc[0]["detectability_threshold_budget"]) if not res.empty else np.nan
                for di, label in enumerate(DEPTH_ORDER):
                    current = item.loc[item["cell_budget_label"].eq(label)]
                    if current.empty: continue
                    value = float(current.iloc[0]["identifiable_divergence"]); ratio = value / denom if np.isfinite(denom) and denom != 0 else np.nan; matrix[rr, di] = ratio; rows.append({"panel": "A", "metric": metric, "source_environment_id": source_id, "contrast": contrast_label(*contrast), "depth": label, "identifiable_divergence": value, "full_depth_value": denom, "ratio_to_full": ratio, "detectability_threshold_budget": detect_budget, "resolution_90pct_full_budget": resolution_budget, "provenance": "reliability_transport_measurement_depth_matched_fixed_summary.csv + ..._resolution.csv"})
        cmap = plt.get_cmap("magma").copy(); cmap.set_bad("#D9DDE2"); vmax = max(1.0, float(np.nanpercentile(matrix[np.isfinite(matrix)], 97)) if np.isfinite(matrix).any() else 1.0); ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for ri in range(9):
            for di in range(6):
                if np.isfinite(matrix[ri, di]): ax.text(di, ri, f"{matrix[ri, di]:.2f}", ha="center", va="center", fontsize=5.8, color="white" if matrix[ri, di] > 0.62 else "#111111")
        ax.set_title(metric_label(metric), loc="left", pad=10); ax.set_xticks(range(6), DEPTH_ORDER); ax.set_yticks(range(9), row_labels if col == 0 else [""] * 9, fontsize=5.5); ax.set_xlabel("measurement depth"); ax.set_ylabel("source × contrast") if col == 0 else None; clean_axes(ax)
    ax_curve = fig.add_subplot(grid[1, :]); add_panel_label(ax_curve, "D"); curve_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        sub = depth.loc[depth["metric"].eq(metric)].copy(); summary = sub.groupby("cell_budget_label", sort=False, observed=True).agg(value=("identifiable_divergence", "mean"), low=("identifiable_divergence_ci_low", "mean"), high=("identifiable_divergence_ci_high", "mean")).reindex(DEPTH_ORDER).reset_index(); x = np.arange(6); ax_curve.plot(x, summary["value"], marker="o", lw=1.8, color=metric_color(metric), label=metric_label(metric)); ax_curve.fill_between(x, summary["low"], summary["high"], color=metric_color(metric), alpha=0.12)
        for _, row in summary.iterrows(): curve_rows.append({"panel": "B", "metric": metric, "depth": row["cell_budget_label"], "value": row["value"], "ci_low": row["low"], "ci_high": row["high"], "aggregation": "mean of source×contrast intervals; no universal threshold"})
    ax_curve.axhline(0, color="#46515C", lw=0.8); ax_curve.set_xticks(range(6), DEPTH_ORDER); ax_curve.set_xlabel("measurement depth"); ax_curve.set_ylabel(r"$D_{\mathrm{meas-ID}}$ (group mean; 90% CI endpoints)"); ax_curve.set_title("Compact depth curves; thresholds remain cell- and metric-specific", loc="left"); add_metric_legend(ax_curve, METRICS, loc="upper left", ncol=3); clean_axes(ax_curve, grid=True)
    fig.suptitle("Figure 3 | Measurement depth defines the identifiability boundary", fontsize=12, fontweight="bold"); fig.text(0.5, 0.01, "Heatmap values are R(n) = D_meas-ID(n) / D_meas-ID(full); raw values above one are retained. Gray cells are unavailable.", ha="center", fontsize=7.2, color="#46515C")
    return save_figure(fig, out_dir, "reliability_transportability_fig3_measurement_boundary"), pd.DataFrame(rows + curve_rows)


def figure4(out_dir: Path, manifests: Path) -> tuple[list[str], pd.DataFrame]:
    link_path = manifests / "reliability_transport_decision_link.csv"; link = pd.read_csv(link_path) if link_path.is_file() else pd.DataFrame(); summary = _depth_labels(pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv", dtype={"cell_budget_label": "string"})); summary = summary.loc[summary["metric"].isin(METRICS) & summary["cell_budget_label"].eq("full")].copy(); curve = summary.groupby(["metric", "decision_budget_fraction"], as_index=False).agg(retention=("retention_point", "mean"), retention_ci_low=("retention_ci_low", "mean"), retention_ci_high=("retention_ci_high", "mean"), excess_regret=("excess_regret_point", "mean"), excess_regret_ci_low=("excess_regret_ci_low", "mean"), excess_regret_ci_high=("excess_regret_ci_high", "mean"))
    fig = plt.figure(figsize=(13, 4.8)); grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.35, 1.2), wspace=0.25); ax_a, ax_b, ax_c = [fig.add_subplot(grid[0, i]) for i in range(3)]
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1); ax_a.axis("off"); add_panel_label(ax_a, "A"); ax_a.text(0.04, 0.91, "Selection under transport", fontsize=10, fontweight="bold", va="top"); ax_a.text(0.5, 0.7, "source\nranking", ha="center", va="center", fontsize=10, fontweight="bold", bbox={"boxstyle": "square,pad=0.5", "facecolor": "#EAF2F8", "edgecolor": "#7A8996"}); ax_a.text(0.5, 0.28, "target\nutility", ha="center", va="center", fontsize=10, fontweight="bold", bbox={"boxstyle": "square,pad=0.5", "facecolor": "#FBE7C6", "edgecolor": "#7A8996"}); _draw_arrow(ax_a, (0.5, 0.58), (0.5, 0.42)); ax_a.text(0.08, 0.5, "budget", rotation=90, va="center", ha="center", color="#46515C"); ax_a.text(0.5, 0.08, "misordered top-k choices create regret", ha="center", fontsize=8, fontweight="bold"); clean_axes(ax_a)
    ax_b.set_title("Canonical decision curves", loc="left", pad=14); add_panel_label(ax_b, "B")
    for metric in METRICS:
        sub = curve.loc[curve["metric"].eq(metric)].sort_values("decision_budget_fraction"); x = sub["decision_budget_fraction"].to_numpy(dtype=float); ax_b.plot(x, sub["retention"], marker="o", color=metric_color(metric), lw=1.8, label=metric_label(metric)); ax_b.fill_between(x, sub["retention_ci_low"], sub["retention_ci_high"], color=metric_color(metric), alpha=0.10)
    ax_b.set_xticks([0.05, 0.10, 0.20, 0.50], [".05", ".10", ".20", ".50"]); ax_b.set_xlabel("decision budget"); ax_b.set_ylabel("top-k retention"); ax_b.set_ylim(0, 1.05); ax_b.grid(axis="y", alpha=0.25); ax_b2 = ax_b.twinx()
    for metric in METRICS:
        sub = curve.loc[curve["metric"].eq(metric)].sort_values("decision_budget_fraction"); x = sub["decision_budget_fraction"].to_numpy(dtype=float); ax_b2.plot(x, sub["excess_regret"], ls="--", marker="x", color=metric_color(metric), alpha=0.75, lw=1.1); ax_b2.fill_between(x, sub["excess_regret_ci_low"], sub["excess_regret_ci_high"], color=metric_color(metric), alpha=0.05)
    ax_b2.set_ylabel("excess regret", color="#46515C"); ax_b.legend(frameon=False, fontsize=6.5, loc="lower right"); clean_axes(ax_b); clean_axes(ax_b2)
    ax_c.set_title("Identifiability ↔ regret", loc="left", pad=14); add_panel_label(ax_c, "C")
    if link.empty: ax_c.text(0.5, 0.5, "link unavailable", ha="center", va="center", color="#7A8996"); rho = np.nan
    else:
        for metric in METRICS:
            sub = link.loc[link["metric"].eq(metric)]; ax_c.errorbar(sub["d_meas_id"], sub["normalized_regret"], xerr=[sub["d_meas_id"] - sub["d_meas_id_ci_low"], sub["d_meas_id_ci_high"] - sub["d_meas_id"]], yerr=[sub["normalized_regret"] - sub["normalized_regret_ci_low"], sub["normalized_regret_ci_high"] - sub["normalized_regret"]], fmt="o", ms=4.5, color=metric_color(metric), ecolor=metric_color(metric), alpha=0.75, capsize=2, label=metric_label(metric))
        rho = float(spearmanr(link["d_meas_id"], link["normalized_regret"]).statistic)
    ax_c.set_xlabel(r"$D_{\mathrm{meas-ID}}$ (full depth)"); ax_c.set_ylabel("normalized regret (.10 budget)"); ax_c.text(0.03, 0.97, f"Spearman ρ = {rho:.2f}" if np.isfinite(rho) else "Spearman ρ unavailable", transform=ax_c.transAxes, va="top", fontsize=8, fontweight="bold"); ax_c.text(0.03, 0.88, "n = 18 · descriptive, not causal", transform=ax_c.transAxes, va="top", fontsize=7.5, color="#46515C"); ax_c.legend(frameon=False, fontsize=6.5, loc="best"); clean_axes(ax_c, grid=True)
    fig.suptitle("Figure 4 | Identifiable reordering changes experimental decisions", fontsize=12, fontweight="bold"); fig.text(0.5, 0.01, "Panel B ribbons are means of group-level intervals from the canonical 2,000-draw bootstrap summary. Panel C is full depth and 10% budget; the association is descriptive.", ha="center", fontsize=7.2, color="#46515C")
    source = curve.assign(panel="B", uncertainty_unit="mean of canonical group-level 90% intervals");
    if not link.empty: source = pd.concat([source, link.assign(panel="C")], ignore_index=True, sort=False)
    source = pd.concat([pd.DataFrame([{"panel": "A", "description": "source ranking → target utility under a fixed decision budget", "provenance": "conceptual schematic"}]), source], ignore_index=True, sort=False)
    return save_figure(fig, out_dir, "reliability_transportability_fig4_decision_consequence"), source


def _state_code(value: Any) -> int:
    text = str(value).lower()
    if "unresolved" in text or "tie" in text or text in {"nan", "none"}: return 0
    if "p<q" in text or "p_lt_q" in text or "p<" in text: return -1
    if "p>q" in text or "p_gt_q" in text or "p>" in text: return 1
    return 0


def _pair_state_counts(path: Path) -> pd.DataFrame:
    """Summarize the large pair-state table without materializing all rows."""
    usecols = ["source_environment_id", "target_environment_id", "metric", "source_state_discovery", "target_state_validation"]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000):
        chunk = chunk.loc[
            chunk["metric"].isin(METRICS)
            & chunk["source_environment_id"].isin(CONTEXT_ORDER)
            & chunk["target_environment_id"].isin(CONTEXT_ORDER)
            & chunk["source_environment_id"].ne(chunk["target_environment_id"])
        ].copy()
        if chunk.empty:
            continue
        source_code = chunk["source_state_discovery"].map(_state_code)
        target_code = chunk["target_state_validation"].map(_state_code)
        chunk["unresolved_count"] = ((source_code == 0) | (target_code == 0)).astype(int)
        chunk["inversion_count"] = ((source_code != 0) & (target_code != 0) & (source_code != target_code)).astype(int)
        pieces.append(chunk.groupby(["source_environment_id", "target_environment_id", "metric"], as_index=False).agg(inversion_count=("inversion_count", "sum"), unresolved_count=("unresolved_count", "sum"), pair_count=("inversion_count", "size")))
    if not pieces:
        return pd.DataFrame(columns=["source_environment_id", "target_environment_id", "metric", "inversion_count", "unresolved_count", "pair_count"])
    return pd.concat(pieces, ignore_index=True).groupby(["source_environment_id", "target_environment_id", "metric"], as_index=False).sum()


def figure5(out_dir: Path, manifests: Path) -> tuple[list[str], pd.DataFrame]:
    failure_path = manifests / "reliability_transport_failure_anatomy.csv"; failure = pd.read_csv(failure_path) if failure_path.is_file() else pd.DataFrame(); states_path = manifests / "reliability_transport_metric_pair_states.csv"; states = _pair_state_counts(states_path) if states_path.is_file() else pd.DataFrame(columns=["source_environment_id", "target_environment_id", "metric", "inversion_count", "unresolved_count", "pair_count"]); pred_path = manifests / "reliability_transport_predictability.csv"; pred = pd.read_csv(pred_path) if pred_path.is_file() else pd.DataFrame()
    fig = plt.figure(figsize=(13, 7.5)); grid = fig.add_gridspec(3, 3, height_ratios=(1.2, 1.3, 1.2), hspace=0.55, wspace=0.32); source_rows: list[dict[str, Any]] = []
    for col, metric in enumerate(METRICS):
        ax = fig.add_subplot(grid[0, col]); add_panel_label(ax, chr(ord("A") + col)); ax.set_title(metric_label(metric), loc="left", pad=10); matrix = np.zeros((3, 3)); unresolved = np.zeros((3, 3)); sub = states.loc[states["metric"].eq(metric)].copy()
        for _, row in sub.iterrows():
            source, target = str(row["source_environment_id"]), str(row["target_environment_id"])
            if source not in CONTEXT_ORDER or target not in CONTEXT_ORDER or source == target: continue
            si, ti = CONTEXT_ORDER.index(source), CONTEXT_ORDER.index(target)
            matrix[si, ti] += float(row["inversion_count"])
            unresolved[si, ti] += float(row["unresolved_count"])
        denom = matrix + unresolved; values = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom > 0); ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for si in range(3):
            for ti in range(3):
                ax.text(ti, si, "—" if si == ti else (f"{values[si, ti]:.2f}" if denom[si, ti] else "NA"), ha="center", va="center", fontsize=8); source_rows.append({"panel": "A", "metric": metric, "source": context_label(CONTEXT_ORDER[si]), "target": context_label(CONTEXT_ORDER[ti]), "inversion_fraction": values[si, ti] if si != ti else np.nan, "unresolved_count": unresolved[si, ti], "provenance": "reliability_transport_metric_pair_states.csv"})
        ax.set_xticks(range(3), [context_label(v) for v in CONTEXT_ORDER], rotation=30, ha="right"); ax.set_yticks(range(3), [context_label(v) for v in CONTEXT_ORDER]); ax.set_xlabel("target validation"); ax.set_ylabel("source discovery"); clean_axes(ax)
    if failure.empty: failure = pd.DataFrame(columns=["metric", "response_displacement", "reordering_burden", "model_disagreement", "status"])
    for col, metric in enumerate(METRICS):
        ax = fig.add_subplot(grid[1, col]); add_panel_label(ax, chr(ord("D") + col)); ax.set_title(f"{metric_label(metric)} · explanatory", loc="left", fontsize=8.5, pad=10); sub = failure.loc[failure["metric"].eq(metric) & failure["status"].eq("executed")].copy()
        if len(sub) > 1200: sub = sub.sample(1200, random_state=20260910)
        if not sub.empty:
            size = pd.to_numeric(sub["model_disagreement"], errors="coerce").fillna(0).clip(lower=0); size = 12 + 110 * size / max(float(size.quantile(0.95)), 1e-9); ax.scatter(sub["response_displacement"], sub["reordering_burden"], s=size, alpha=0.22, color=metric_color(metric), edgecolors="none")
        ax.set_xlabel("response displacement"); ax.set_ylabel("reordering burden"); ax.text(0.03, 0.96, "post-hoc explanatory", transform=ax.transAxes, va="top", fontsize=7.3, color="#46515C"); clean_axes(ax, grid=True)
        for _, row in sub.head(300).iterrows(): source_rows.append({"panel": "B", "metric": metric, "source_environment_id": row.get("source_environment_id"), "target_environment_id": row.get("target_environment_id"), "perturbation_label": row.get("perturbation_label"), "response_displacement": row.get("response_displacement"), "reordering_burden": row.get("reordering_burden"), "model_disagreement": row.get("model_disagreement"), "role": "post-hoc explanatory", "provenance": "reliability_transport_failure_anatomy.csv"})
    ax = fig.add_subplot(grid[2, :]); add_panel_label(ax, "G"); ax.set_title("Prospective prediction from source-only features", loc="left", pad=10); pred = pred.loc[pred["status"].eq("executed")].copy() if not pred.empty else pd.DataFrame()
    if not pred.empty:
        pred["spearman"] = pd.to_numeric(pred["spearman"], errors="coerce"); groups = pred.groupby(["model", "metric"], as_index=False).agg(value=("spearman", "mean"), low=("spearman", lambda x: x.quantile(0.25)), high=("spearman", lambda x: x.quantile(0.75))).sort_values(["metric", "model"], kind="stable")
        for yi, (_, row) in enumerate(groups.iterrows()):
            color = metric_color(row["metric"]); ax.errorbar(row["value"], yi, xerr=[[row["value"] - row["low"]], [row["high"] - row["value"]]], fmt="o", color=color, ecolor=color, capsize=2); ax.text(-0.02, yi, f"{row['model']} / {metric_label(row['metric'])}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7); source_rows.append({"panel": "C", "model": row["model"], "metric": row["metric"], "held_out_spearman_mean": row["value"], "iqr_low": row["low"], "iqr_high": row["high"], "source_only": True, "target_informed_feature_used": False, "provenance": "reliability_transport_predictability.csv"})
        ax.set_yticks(range(len(groups))); ax.set_yticklabels([]); ax.set_ylim(-1, len(groups))
    ax.axvline(0, color="#46515C", lw=0.8); ax.set_xlabel("held-out Spearman (mean ± IQR)"); ax.text(0.99, 0.96, "target outcomes used only for evaluation", transform=ax.transAxes, ha="right", va="top", fontsize=7.3, color="#46515C"); clean_axes(ax, grid=True)
    fig.suptitle("Figure 5 | What drives transport failure, and can we know it beforehand?", fontsize=12, fontweight="bold"); fig.text(0.5, 0.01, "Matrices are tie-aware source→target inversion fractions with unresolved pairs retained separately. Middle panels are descriptive; target-informed quantities are excluded from prospective prediction.", ha="center", fontsize=7.2, color="#46515C")
    return save_figure(fig, out_dir, "reliability_transportability_fig5_failure_anatomy"), pd.DataFrame(source_rows)


def run(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve(); manifests = root / "artifacts/manifests"; out_dir = root / "results/figures/reliability_transportability"; out_dir.mkdir(parents=True, exist_ok=True); apply_style()
    functions = [("fig1", "fig1_source.csv", lambda: figure1(out_dir)), ("fig2", "fig2_transport_atlas.csv", lambda: figure2(out_dir, manifests)), ("fig3", "fig3_measurement_boundary.csv", lambda: figure3(out_dir, manifests)), ("fig4", "fig4_decision_consequence.csv", lambda: figure4(out_dir, manifests)), ("fig5", "fig5_failure_anatomy.csv", lambda: figure5(out_dir, manifests))]
    outputs: dict[str, Any] = {}
    for key, source_name, function in functions:
        paths, source = function(); outputs[key] = {"files": paths, "source_table": _write_source(root, source_name, source), "rows": int(len(source))}
    report = {"schema_version": 2, "status": "executed", "figure_count": 5, "figure_order": ["observed_to_identifiable", "transport_atlas", "measurement_boundary", "decision_consequence", "failure_anatomy"], "formats": ["png", "pdf", "svg", "tiff"], "source_tables_directory": "results/figures/reliability_transportability/source_tables", "outputs": outputs, "data_policy": "Frozen Frangieh/Nadig data and existing matched-fixed/bootstrap summaries; unavailable cells stay gray/NA; target-informed explanatory quantities never enter prospective prediction.", "style_policy": "Centralized ptl_figure_style.py; colorblind-safe palette; no rainbow, 3-D, or long-ID axes."}
    (manifests / "reliability_transport_figures.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2, ensure_ascii=False)); return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=ROOT); run(parser.parse_args().root); return 0


if __name__ == "__main__":
    raise SystemExit(main())
