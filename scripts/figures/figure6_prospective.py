"""Figure 6: source-only prospective boundary and information firewall."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figures.common import (
    FEATURES, FORBIDDEN_TARGET_FIELDS, METRICS, add_panel_label, clean_axes,
    full_predictability, metric_color, metric_label, set_title, save_figure,
)
from scripts.figures.glyphs import firewall_box


def figure6(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    pred = full_predictability(manifests); rows: list[dict] = []
    fig = plt.figure(figsize=(14, 8.1), constrained_layout=False); grid = fig.add_gridspec(2, 12, height_ratios=(1.50, .92), hspace=.55, wspace=.55)

    ax = fig.add_subplot(grid[0, 0:5]); add_panel_label(ax, "A"); set_title(ax, "Information firewall", "the prospective claim is source-only, before any target outcome exists")
    firewall_box(ax, list(FEATURES), list(FORBIDDEN_TARGET_FIELDS)); rows.append({"panel": "A", "allowed_features": ";".join(FEATURES), "forbidden_target_informed_quantities": ";".join(FORBIDDEN_TARGET_FIELDS), "provenance": "reliability_transport_predictability.csv + failure anatomy manifest"})

    ax = fig.add_subplot(grid[0, 5:12]); add_panel_label(ax, "B"); set_title(ax, "Source-only performance landscape", "two legal models × three metrics; point = task, axes carry Spearman and AUROC")
    for metric in METRICS:
        for model, marker in (("source_u_only", "o"), ("linear_source_features", "D")):
            sub = pred.loc[pred["metric"].eq(metric) & pred["model"].eq(model)].copy(); x = sub["spearman"]; y = sub["auroc_high_identifiable"]
            ax.scatter(x, y, s=19, alpha=.30, color=metric_color(metric), marker=marker, edgecolor="white", linewidth=.25)
            if not sub.empty:
                ax.errorbar(x.median(), y.median(), xerr=[[x.median() - x.quantile(.25)], [x.quantile(.75) - x.median()]], yerr=[[y.median() - y.quantile(.25)], [y.quantile(.75) - y.median()]], fmt=marker, ms=5, color=metric_color(metric), ecolor=metric_color(metric), capsize=2, lw=.8)
            rows.append({"panel": "E", "metric": metric, "model": model, "median_spearman": x.median(), "q25_spearman": x.quantile(.25), "q75_spearman": x.quantile(.75), "median_auroc": y.median(), "q25_auroc": y.quantile(.25), "q75_auroc": y.quantile(.75), "n_tasks": len(sub), "reference": .5, "calibration_status": "not_reported_train_fold_only_no_posthoc_target_calibration", "provenance": "reliability_transport_predictability.csv"})
    ax.axvline(0, color="#303840", lw=.75); ax.axhline(.5, color="#303840", lw=.75, ls="--"); ax.set_xlabel("held-out Spearman (ordering) →"); ax.set_ylabel("AUROC (high identifiability; chance=.5)"); ax.text(.03, .96, "● source U-only     ◆ linear source features\nmetric color: delta cosine / Systema / absolute rank", transform=ax.transAxes, va="top", fontsize=6.1, color="#46515C"); clean_axes(ax, grid=True)

    ax = fig.add_subplot(grid[1, 0:8]); add_panel_label(ax, "C"); set_title(ax, "Task-level prospective strip", "negative values remain visible; no heatmap aggregation")
    offsets = {"source_u_only": -.13, "linear_source_features": .13}
    order = [(metric, model) for metric in METRICS for model in ("source_u_only", "linear_source_features")]
    for yi, (metric, model) in enumerate(order):
        vals = pred.loc[pred["metric"].eq(metric) & pred["model"].eq(model), "spearman"].dropna().to_numpy(); jitter = np.linspace(-.08, .08, len(vals)) if len(vals) else np.array([])
        ax.scatter(vals, np.full(len(vals), yi) + jitter, s=12, alpha=.42, color=metric_color(metric), marker="o" if model == "source_u_only" else "D", edgecolor="white", linewidth=.2)
        if len(vals): ax.plot([np.median(vals), np.median(vals)], [yi - .18, yi + .18], color="#303840", lw=1.2); rows.append({"panel": "C", "metric": metric, "model": model, "median_spearman": np.median(vals), "q25": np.quantile(vals, .25), "q75": np.quantile(vals, .75), "n_tasks": len(vals), "provenance": "reliability_transport_predictability.csv"})
    ax.axvline(0, color="#303840", lw=.8); ax.set_yticks(range(len(order)), [f"{metric_label(metric)} · {'U-only' if model == 'source_u_only' else 'linear'}" for metric, model in order], fontsize=5.8); ax.set_xlabel("held-out Spearman (zero = no ordering signal)"); clean_axes(ax, grid=True)

    ax = fig.add_subplot(grid[1, 8:12]); add_panel_label(ax, "D"); set_title(ax, "Nadig replication", "limitation inset, not a standalone numeric result")
    ax.axis("off"); ax.add_patch(plt.Rectangle((.04, .20), .92, .62, facecolor="#F4F5F6", edgecolor="#AAB4BC", lw=.7)); ax.text(.50, .65, "UNAVAILABLE", ha="center", fontsize=10, fontweight="bold", color="#46515C"); ax.text(.50, .48, "aggregate-only dataset; no shared per-label\ntarget outcome for held-out transport", ha="center", va="center", fontsize=7.0); ax.text(.50, .27, "No numeric result · no oracle · no imputation", ha="center", fontsize=6.2, color="#9E2A2B", fontweight="bold")
    rows.append({"panel": "F", "status": "unavailable", "reason": "Nadig aggregate-only no shared per-label target", "numeric_result": False, "provenance": "Nadig Claim Lock"})
    fig.suptitle("Figure 6 | Transport failure is easier to measure than to predict prospectively", fontsize=12, fontweight="bold"); fig.text(.5, .012, "Source-only performance is shown at task level with negative results retained. The firewall and blocked Nadig inset state exactly what cannot be claimed before experiment.", ha="center", fontsize=7.0, color="#46515C")
    return save_figure(fig, out_dir, "reliability_transportability_fig6_prospective_limit"), pd.DataFrame(rows)


__all__ = ["figure6"]
