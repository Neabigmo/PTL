"""Figure 2: structured transport, strict inversions, controls and replication."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.figures.common import (
    ATLAS_DATASET, CONTEXT_ORDER, CONTRAST_ORDER, METRICS, add_panel_label,
    clean_axes, context_label, metric_color, metric_label, num, save_figure,
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


def _short(label: str) -> str:
    return label.replace("Co-culture", "Co")


def _add_metric_key(fig: plt.Figure) -> None:
    handles = [Line2D([], [], marker="o", color=metric_color(m), linestyle="None", ms=4.2, label=metric_label(m)) for m in METRICS]
    # Keep the shared key below the long figure title; a figure-level legend
    # is not included in the per-axis collision audit.
    fig.legend(handles=handles, frameon=False, loc="upper right", bbox_to_anchor=(.985, .905), ncol=3, fontsize=5.0, handletextpad=.25, columnspacing=.65)


def figure2(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    atlas_all = pd.read_csv(manifests / "reliability_transport_atlas.csv")
    atlas = atlas_all.loc[atlas_all["dataset"].eq(ATLAS_DATASET) & atlas_all["metric"].isin(METRICS)].copy()
    frangieh = atlas.loc[atlas["source_environment_id"].astype(str).str.startswith("frangieh_")].copy()
    if len(frangieh) != 27:
        raise ValueError(f"Figure 2 requires the 27-row Frangieh atlas, got {len(frangieh)}")
    stable = pd.read_csv(manifests / "stable_inversion_summary.csv")
    control = pd.read_csv(manifests / "reviewer_pseudocontext_negative_control.csv")
    nadig = atlas_all.loc[atlas_all["dataset"].eq("NadigOConner2024") & atlas_all["metric"].isin(METRICS)].copy()
    # Keep the displayed replication universe tied to the canonical full-size
    # per-direction artifact.  The aggregate atlas intentionally does not
    # carry this field, so it must not be silently replaced by a hard-coded
    # annotation in the figure.
    nadig_full = pd.read_csv(manifests / "formal_v2_claim_lock_replication_nadig_fullsize.csv")
    nadig_counts = nadig_full.loc[
        nadig_full["metric"].isin(METRICS),
        ["source_context_id", "metric", "n_perturbations"],
    ].copy()
    nadig_counts["n_perturbations"] = pd.to_numeric(nadig_counts["n_perturbations"], errors="coerce")
    if len(nadig_counts) != 6 or nadig_counts.duplicated(["source_context_id", "metric"]).any():
        raise ValueError("Figure 2 requires six unique Nadig source-context x metric count rows")
    if nadig_counts["n_perturbations"].isna().any() or nadig_counts["n_perturbations"].nunique() != 1:
        raise ValueError("Figure 2 Nadig replication counts must be finite and consistent across directions/metrics")
    nadig = nadig.merge(
        nadig_counts,
        left_on=["source_environment_id", "metric"],
        right_on=["source_context_id", "metric"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["source_context_id"])
    rows: list[dict] = []

    width = figure_size("fig2")[0]
    fig = plt.figure(figsize=(width, width * .52), constrained_layout=False)
    grid = fig.add_gridspec(1, 4, width_ratios=(.44, .22, .15, .19), wspace=.36, left=.045, right=.985, top=.80, bottom=.18)
    _add_metric_key(fig)

    # A: primary atlas. Labels and estimates use separate columns so the six
    # directed transfers remain readable at final manuscript size.
    # Keep the label column visually separated from the estimate column at
    # manuscript scale; the previous 1.28-pt gutter was too close even
    # though the text-collision audit passed.
    agrid = grid[0].subgridspec(1, 2, width_ratios=(.43, .57), wspace=.08)
    label_ax = fig.add_subplot(agrid[0]); add_panel_label(label_ax, "A")
    label_ax.set_title("Frangieh transport", loc="left", pad=5, fontweight="bold", fontsize=7.4)
    label_ax.set_xlim(0, 1); label_ax.set_ylim(-.25, 6.25); label_ax.axis("off")
    label_ax.set_xticks([]); label_ax.set_yticks([])
    label_ax.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    data_ax = fig.add_subplot(agrid[1])
    transfers = _directional_rows()
    ys = np.arange(len(transfers))[::-1] + .5
    data_ax.set_xlim(-.015, .205); data_ax.set_ylim(-.25, 6.25); data_ax.set_yticks([])
    data_ax.axvline(0, color="#8B98A4", lw=.55, ls=(0, (4, 3)), zorder=0)
    group_centers = {CONTEXT_ORDER[0]: 5.0, CONTEXT_ORDER[1]: 3.0, CONTEXT_ORDER[2]: 1.0}
    for source, center in group_centers.items():
        label_ax.text(.02, center, context_label(source), ha="left", va="center", fontsize=5.8, fontweight="bold")
        label_ax.plot([.38, .38], [center - .70, center + .70], color="#AEB7BF", lw=.5)
    label_ax.text(.15, 6.16, "Source", ha="center", va="bottom", fontsize=5.4, fontweight="bold")
    label_ax.text(.67, 6.16, "Transfer", ha="center", va="bottom", fontsize=5.4, fontweight="bold")
    for y in (1.99, 3.99):
        label_ax.plot([0, 1], [y, y], color="#D4DADF", lw=.5)
        data_ax.plot([-.015, .205], [y, y], color="#D4DADF", lw=.5)
    for yi, (source, contrast, transfer_label) in zip(ys, transfers):
        label_ax.text(.43, yi, _short(transfer_label), ha="left", va="center", fontsize=5.8)
        for metric in METRICS:
            item = frangieh.loc[
                frangieh["source_environment_id"].eq(source)
                & frangieh["left_target_environment_id"].eq(contrast[0])
                & frangieh["right_target_environment_id"].eq(contrast[1])
                & frangieh["metric"].eq(metric)
            ]
            if item.empty:
                continue
            item = item.iloc[0]
            value, lo, hi = [num(item[c]) for c in ("measurement_identifiable", "measurement_identifiable_ci_low", "measurement_identifiable_ci_high")]
            if not np.isfinite([value, lo, hi]).all():
                continue
            offset = {METRICS[0]: .15, METRICS[1]: 0.0, METRICS[2]: -.15}[metric]
            data_ax.errorbar(value, yi + offset, xerr=[[max(value - lo, 0)], [max(hi - value, 0)]], fmt="o", ms=3.0, color=metric_color(metric), ecolor=metric_color(metric), capsize=1.4, lw=.6, zorder=3)
            rows.append({"panel": "A", "metric": metric, "source_environment_id": source, "contrast": f"{context_label(contrast[0])} ↔ {context_label(contrast[1])}", "direction": transfer_label, "measurement_identifiable": value, "ci_low": lo, "ci_high": hi, "joint_identifiable": item["joint_identifiable"], "provenance": "reliability_transport_atlas.csv"})
    data_ax.set_xlabel(r"$D_{\rm adj}$", fontsize=6.4, labelpad=2); data_ax.set_xticks([0, .1, .2]); data_ax.tick_params(axis="x", labelsize=5.8, pad=1); clean_axes(data_ax, grid=True)

    # B: stable strict inversions across the same six directions.
    ax = fig.add_subplot(grid[1]); add_panel_label(ax, "B"); ax.set_title("Stable inversions", loc="left", pad=5, fontweight="bold", fontsize=7.2)
    yvals = []
    ylabels = []
    for ti, (source, contrast, label) in enumerate(transfers):
        center = len(transfers) - ti - .55
        ylabels.append((center, _short(label)))
        for metric in METRICS:
            item = stable.loc[stable["source_environment_id"].eq(source) & stable["left_target_environment_id"].eq(contrast[0]) & stable["right_target_environment_id"].eq(contrast[1]) & stable["metric"].eq(metric)]
            if item.empty:
                continue
            value = num(item.iloc[0]["stable_order_inversion_fraction"])
            if np.isfinite(value):
                y = center + {METRICS[0]: .15, METRICS[1]: 0, METRICS[2]: -.15}[metric]
                ax.plot([0, value], [y, y], color=metric_color(metric), lw=1.3, solid_capstyle="round", alpha=.72)
                ax.scatter(value, y, s=17, color=metric_color(metric), edgecolor="white", lw=.35, zorder=3)
                rows.append({"panel": "B", "metric": metric, "source_environment_id": source, "direction": label, "stable_inversion_fraction": value, "minimum_strict_support": int(num(item.iloc[0]["minimum_strict_support"])), "provenance": "stable_inversion_summary.csv"})
    ax.axvline(0, color="#B8BEC8", lw=.55, ls=(0, (3, 2))); ax.set_xlim(0, .32); ax.set_ylim(.05, len(transfers) + .15); ax.set_yticks([v for v, _ in ylabels], [l for _, l in ylabels]); ax.tick_params(axis="y", labelsize=5.8, pad=1); ax.set_xlabel("strict pair reversals", fontsize=6.0, labelpad=2); ax.tick_params(axis="x", labelsize=5.8, pad=1); clean_axes(ax, grid=True)

    # C: same-context negative control; no fake interval is drawn.
    # Keep this compact control as a single-line title.  The narrow control
    # column is intentional; wrapping the title makes it collide with the
    # shared metric key at manuscript scale.
    ax = fig.add_subplot(grid[2]); add_panel_label(ax, "C"); ax.set_title("Floor control", loc="left", pad=5, fontweight="bold", fontsize=6.7)
    ctx_offsets = {CONTEXT_ORDER[0]: -.14, CONTEXT_ORDER[1]: 0, CONTEXT_ORDER[2]: .14}
    for mi, metric in enumerate(METRICS):
        base = 2 - mi
        for source, off in ctx_offsets.items():
            item = control.loc[control["source_environment_id"].eq(source) & control["metric"].eq(metric)]
            if item.empty:
                continue
            value = num(item.iloc[0]["measurement_identifiable"])
            ax.scatter(value, base + off, s=15, color=metric_color(metric), edgecolor="white", linewidth=.35, zorder=3)
            rows.append({"panel": "C", "metric": metric, "source_environment_id": source, "same_context_ab": value, "provenance": "reviewer_pseudocontext_negative_control.csv"})
    ax.axvline(0, color="#B8BEC8", lw=.55, ls=(0, (3, 2))); ax.set_xlim(-.0005, .0065); ax.set_xticks([0, .003, .006]); ax.set_xticklabels(["0", ".003", ".006"]); ax.set_ylim(-.5, 2.5); ax.set_yticks([2, 1, 0], ["Delta", "Systema", "Abs. rank"]); ax.tick_params(axis="both", labelsize=5.8, pad=1); ax.set_xlabel(r"$D_{\rm adj}$", fontsize=5.8, labelpad=2); clean_axes(ax, grid=True)

    # D: independent Nadig replication, with direction as marker shape.
    ax = fig.add_subplot(grid[3]); add_panel_label(ax, "D"); ax.set_title("Nadig replication", loc="left", pad=5, fontweight="bold", fontsize=7.0)
    style = {"nadig_hepg2": ("HepG2 → Jurkat", "o", .10), "nadig_jurkat": ("Jurkat → HepG2", "^", -.10)}
    for mi, metric in enumerate(METRICS):
        for _, item in nadig.loc[nadig["metric"].eq(metric)].iterrows():
            direction, marker, offset = style[str(item["source_environment_id"])]
            value, lo, hi = [num(item[c]) for c in ("measurement_identifiable", "measurement_identifiable_ci_low", "measurement_identifiable_ci_high")]
            y = 2 - mi + offset
            if np.isfinite([value, lo, hi]).all():
                ax.errorbar(value, y, xerr=[[max(value - lo, 0)], [max(hi - value, 0)]], fmt=marker, ms=3.2, color=metric_color(metric), ecolor=metric_color(metric), capsize=1.3, lw=.6, zorder=3)
            rows.append({"panel": "D", "metric": metric, "direction": direction, "n_perturbations": num(item.get("n_perturbations", np.nan)), "measurement_identifiable": value, "ci_low": lo, "ci_high": hi, "provenance": "formal_v2_claim_lock_replication_nadig_fullsize.csv"})
    ax.axvline(0, color="#B8BEC8", lw=.55, ls=(0, (3, 2))); ax.set_xlim(-.06, .24); ax.set_xticks([0, .1, .2]); ax.set_ylim(-.5, 2.5); ax.set_yticks([2, 1, 0], ["Delta", "Systema", "Abs. rank"]); ax.tick_params(axis="both", labelsize=5.8, pad=1); ax.set_xlabel(r"$D_{\rm adj}$", fontsize=5.8, labelpad=2); clean_axes(ax, grid=True)
    nadig_values = pd.to_numeric(nadig["n_perturbations"], errors="coerce")
    nadig_measurements = nadig[["measurement_identifiable", "measurement_identifiable_ci_low", "measurement_identifiable_ci_high"]].apply(pd.to_numeric, errors="coerce")
    if len(nadig) == 6 and nadig_values.notna().all() and nadig_values.nunique() == 1 and nadig_measurements.notna().all().all():
        nadig_n = int(nadig_values.iloc[0])
        ax.text(.98, .98, rf"$n={nadig_n:,}$/direction", transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color="#6B7280")
    # The direction is encoded by marker shape and stated in the caption.  An
    # in-axis legend is deliberately omitted: both lower corners contain
    # real Nadig marks/intervals at manuscript scale, so even a compact key
    # would risk obscuring the data.

    fig.suptitle("Fig. 2   Context-dependent reliability transport and independent replication", fontsize=10.8, fontweight="bold", x=.02, y=.985, ha="left")
    return save_figure(fig, out_dir, "reliability_transportability_fig2_transport_atlas"), pd.DataFrame(rows)


__all__ = ["figure2"]
