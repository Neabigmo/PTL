"""Legacy renderer retained for provenance; not part of the current paper build.

The current paper-facing prospective figure is built by
``supp_source_only_forecasting.py``.  This module is kept only so historical
renders remain reproducible and must not be used by the canonical driver.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scripts.figures.common import (
    METRICS, PRIMARY_METRIC, PRIMARY_SOURCE, PRIMARY_TARGET,
    add_panel_label, clean_axes, failure_cases, metric_color, num,
    rank_table, save_figure,
)
from scripts.ptl_figure_style import figure_size
from scripts.ptl_figure_style import metric_label


def _feature_count(value: object) -> int:
    text = str(value)
    if not text or text.lower() == "nan":
        return 0
    return len([part for part in text.split(";") if part.strip()])


def _rank_example(ax: plt.Axes, ranks: pd.DataFrame, case: pd.Series, title: str, color: str, *, show_y: bool) -> None:
    ax.scatter(ranks["source_rank"], ranks["ifng_rank"], s=11, color="#B8BEC8", alpha=.45, edgecolor="none")
    ax.plot([1, len(ranks)], [1, len(ranks)], color="#B8BEC8", lw=.65, ls=(0, (3, 2)))
    x, y = num(case["source_rank"]), num(case["ifng_rank"])
    ax.scatter([x], [y], s=42, color=color, edgecolor="white", linewidth=.55, zorder=3)
    ax.annotate(str(case["perturbation_label"]), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=6.0, color=color, fontweight="bold", arrowprops={"arrowstyle": "-", "color": color, "lw": .55})
    ax.set_title(title, fontsize=6.1, fontweight="bold", pad=3)
    ax.set_xlim(.5, len(ranks) + .5); ax.set_ylim(len(ranks) + .5, .5); ax.set_xlabel("source rank", fontsize=5.9); ax.set_ylabel("target rank" if show_y else "", fontsize=5.4); ax.tick_params(axis="y", labelsize=4.5, pad=1.0, labelleft=show_y); clean_axes(ax, grid=True)


def figure5(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    ranks = rank_table(manifests); cases = failure_cases(manifests, ranks); rows: list[dict] = []
    pred = pd.read_csv(manifests / "reliability_transport_predictability.csv")
    pred = pred.loc[pred["status"].eq("executed") & pred["dataset"].eq("FrangiehIzar2021_RNA") & pred["dataset_split"].eq("within_dataset_leave_one_perturbation_label_out")].copy()
    pred["spearman"] = pd.to_numeric(pred["spearman"], errors="coerce"); pred["n_features"] = pred["source_only_features"].map(_feature_count)
    coef = pd.read_csv(manifests / "reliability_transport_heterogeneity_coefficients.csv")
    coef = coef.loc[coef["cell_budget_label"].astype(str).str.lower().eq("full") & coef["model"].eq("ridge_source_features")].copy(); coef["coefficient"] = pd.to_numeric(coef["coefficient"], errors="coerce")
    ablation = pd.read_csv(manifests / "formal_v2_information_ablation_metrics.csv")
    ablation = ablation.loc[ablation["aggregation_level"].eq("environment_macro") & ablation["scenario"].eq("in_domain")].copy(); ablation["auroc"] = pd.to_numeric(ablation["auroc"], errors="coerce")

    fig = plt.figure(figsize=figure_size("fig5"), constrained_layout=False)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.05, .95), hspace=.52)
    top = grid[0].subgridspec(1, 2, width_ratios=(.57, .43), wspace=.36)

    ax = fig.add_subplot(top[0]); add_panel_label(ax, "A"); ax.set_title("Predictive performance (held-out targets)", loc="left", pad=5, fontweight="bold"); ax.axis("off")
    for ci, metric in enumerate(METRICS):
        small = ax.inset_axes([.02 + ci * .325, .12, .285, .78])
        sub_metric = pred.loc[pred["metric"].eq(metric)].copy(); rng = np.random.default_rng(20260911 + ci); x = sub_metric["n_features"].to_numpy(dtype=float) * np.exp(rng.normal(0, .035, len(sub_metric))); colors = np.where(sub_metric["model"].eq("source_u_only"), metric_color(metric), "#9AA4AC"); markers = np.where(sub_metric["model"].eq("source_u_only"), "o", "s")
        for marker in ("o", "s"):
            keep = markers == marker; small.scatter(x[keep], sub_metric.loc[keep, "spearman"], s=13, marker=marker, color=metric_color(metric) if marker == "o" else "#9AA4AC", alpha=.52, edgecolor="white", linewidth=.25)
        med = sub_metric.groupby("n_features", as_index=False)["spearman"].median().sort_values("n_features"); small.plot(med["n_features"], med["spearman"], color=metric_color(metric), lw=1.1, marker="o", ms=3.0, zorder=3)
        short_title = ("Delta cosine", "Systema", "Abs.-effect rank")[ci]
        small.axhline(0, color="#B8BEC8", lw=.55, ls=(0, (3, 2))); small.set_xscale("log"); small.set_xlim(.7, 8); small.set_ylim(-1.0, .35); small.set_title(short_title, fontsize=5.9, color=metric_color(metric), fontweight="bold", pad=2); small.set_xlabel("source-only features", fontsize=5.5); small.set_ylabel("Spearman ρ" if ci == 0 else "", fontsize=5.3); small.tick_params(axis="y", labelsize=4.5, pad=1.0, labelleft=(ci == 0)); clean_axes(small, grid=True)
        rows.extend({"panel": "A", **item.to_dict(), "n_features": int(item["n_features"]), "provenance": "reliability_transport_predictability.csv"} for _, item in sub_metric.iterrows())

    ax = fig.add_subplot(top[1]); add_panel_label(ax, "B"); ax.set_title("Feature groups", loc="left", pad=5, fontweight="bold")
    feature_order = ["uq_cosine_disagreement", "uq_mean_gene_variance", "uq_effect_norm_variance", "prediction_norm", "prediction_sparsity", "prediction_concentration"]; feature_labels = ["Uncertainty", "Geometry", "Magnitude", "Type", "Annotation", "Sequence"]; y0 = np.arange(len(feature_order))[::-1]
    for offset, metric in zip((.18, 0, -.18), METRICS):
        for yi, feature in zip(y0 + offset, feature_order):
            val = coef.loc[coef["metric"].eq(metric) & coef["feature"].eq(feature), "coefficient"].dropna()
            if val.empty: continue
            q25, med, q75 = val.quantile(.25), val.median(), val.quantile(.75); ax.plot([q25, q75], [yi, yi], color=metric_color(metric), lw=1.0, solid_capstyle="round"); ax.scatter(med, yi, s=22, color=metric_color(metric), edgecolor="white", linewidth=.35, zorder=3)
            rows.append({"panel": "B", "metric": metric, "feature": feature, "coefficient_median": med, "coefficient_q25": q25, "coefficient_q75": q75, "n_rows": int(len(val)), "provenance": "reliability_transport_heterogeneity_coefficients.csv"})
    ax.axvline(0, color="#B8BEC8", lw=.6, ls=(0, (3, 2))); ax.set_yticks(y0, feature_labels, fontsize=5.2); ax.tick_params(axis="y", pad=4); ax.set_xlabel("source-feature coefficient", fontsize=6.2); ax.set_ylim(-.55, 6.3); clean_axes(ax, grid=True)
    # Reserve a narrow in-panel header band above the first feature row for
    # the metric key, so it cannot collide with either the data or x-axis text.
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="None", color=metric_color(metric), ms=3.8, label=metric_label(metric)) for metric in METRICS], frameon=False, fontsize=3.9, ncol=3, loc="center", bbox_to_anchor=(.60, .93), bbox_transform=ax.transAxes, handletextpad=.24, columnspacing=.48, borderaxespad=0.0)

    lower = grid[1].subgridspec(1, 2, width_ratios=(.48, .52), wspace=.28)
    ax = fig.add_subplot(lower[0]); add_panel_label(ax, "C"); ax.set_title("Example perturbations", loc="left", pad=5, fontweight="bold"); ax.axis("off")
    colors = ("#527AA3", "#D9825B", "#3A9D8F")
    for i, case_index in enumerate((0, min(2, len(cases) - 1))):
        case = cases.iloc[case_index]
        small = ax.inset_axes([.03 + i * .49, .14, .43, .72]); _rank_example(small, ranks, case, "Low reordering" if i == 0 else "High reordering", colors[i], show_y=(i == 0)); rows.append({"panel": "C", **case.to_dict(), "provenance": "formal_v2_reliability_reordering_perturbations.csv + reliability_transport_failure_anatomy.csv"})

    ax = fig.add_subplot(lower[1]); add_panel_label(ax, "D"); ax.set_title("Incremental value of source feature groups", loc="left", pad=5, fontweight="bold")
    order = ["U", "U+P", "U+P+S", "U+P+S+N", "U+P+S+N+C"]; names = ["Uncertainty", "Geometry", "Magnitude", "Type", "All"]
    grouped = ablation.groupby("information_set")["auroc"].agg(["mean", lambda v: v.quantile(.25), lambda v: v.quantile(.75)]).reindex(order); baseline = float(grouped.iloc[0]["mean"]); delta = grouped["mean"] - baseline; lo = grouped["<lambda_0>"] - baseline; hi = grouped["<lambda_1>"] - baseline; x = np.arange(len(order)); ax.bar(x, delta, color="#4C78A8", alpha=.68, width=.62, edgecolor="white", linewidth=.35); ax.errorbar(x, delta, yerr=[delta - lo, hi - delta], fmt="none", ecolor="#263238", lw=.7, capsize=2, zorder=3); ax.axhline(0, color="#B8BEC8", lw=.6); ax.set_xticks(x, names, rotation=28, ha="right", fontsize=5.4); ax.set_ylabel("ΔAUROC vs U", fontsize=6.2); ax.text(.03, .96, "source-only ablation", transform=ax.transAxes, va="top", fontsize=5.8, color="#6B7280"); clean_axes(ax, grid=True)
    for info, name, mean, dlt, q25, q75 in zip(order, names, grouped["mean"], delta, grouped["<lambda_0>"], grouped["<lambda_1>"]): rows.append({"panel": "D", "information_set": info, "label": name, "auroc_mean": mean, "auroc_q25": q25, "auroc_q75": q75, "delta_auroc_vs_U": dlt, "provenance": "formal_v2_information_ablation_metrics.csv"})
    fig.suptitle("Supplementary Fig. S2   Source-only signals weakly predict reordering", fontsize=12.5, fontweight="bold", x=.02, ha="left")
    return save_figure(fig, out_dir, "reliability_transportability_fig5_failure_anatomy"), pd.DataFrame(rows)


__all__ = ["figure5"]
