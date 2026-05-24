from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figlib.export import save_figure
from figlib.panels import plot_context_distance, plot_split_audit_heatmap, plot_support_boxplot, read_table
from figlib.palettes import BLACK, BLUE, GRAY, GREEN, LIGHT_BLUE, ORANGE, RED, RULE, TEAL, WHITE
from figlib.style import apply_style, despine_data, framed_panel, ordered_splits, pretty_split

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
OUTPUT = ROOT / "results" / "figures_methods"


def make_s1(output_dir: Path) -> None:
    audit = read_table(TABLES, "split_audit.csv")
    apply_style()
    fig = plt.figure(figsize=(13.8, 4.9), facecolor=WHITE)
    gs = fig.add_gridspec(1, 3, wspace=0.28)
    plot_support_boxplot(fig.add_subplot(gs[0, 0]), audit)
    plot_context_distance(fig.add_subplot(gs[0, 1]), audit)
    plot_split_audit_heatmap(fig.add_subplot(gs[0, 2]), audit)
    save_figure(fig, output_dir, "supp_fig_s1_split_audit")


def make_s2(output_dir: Path) -> None:
    threshold = read_table(TABLES, "threshold_sensitivity_summary.csv")
    selective = read_table(TABLES, "selective_prediction_summary.csv")
    oof = read_table(TABLES, "ptl_oof_predictions.csv")
    apply_style()
    fig = plt.figure(figsize=(13.8, 5.0), facecolor=WHITE)
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, label, title in zip(axes, ["A", "B", "C"], ["Threshold sensitivity", "Risk-coverage diagnostics", "Calibration by score bin"]):
        framed_panel(ax, label, title)

    if "method" not in threshold.columns:
        threshold = threshold.copy()
        threshold["method"] = threshold["ablation"].where(threshold["ablation"].ne("full_PTL"), "full_PTL_random_forest")
    for method, color, label in [
        ("naive_confidence", GRAY, "Naive confidence"),
        ("full_PTL_random_forest", BLUE, "Full PTL RF"),
        ("calibrated_logistic_deployment", TEAL, "Calibrated logistic"),
    ]:
        sub = threshold[threshold["method"].eq(method)]
        if sub.empty:
            continue
        axes[0].plot(sub["threshold_multiplier"], sub["mean_false_transportability_rate"], marker="o", color=color, label=label)
    axes[0].set_xlabel("Anchor multiplier")
    axes[0].set_ylabel("False-transportability rate")
    axes[0].legend(frameon=False)
    despine_data(axes[0])

    cov = np.array([0.2, 0.4, 0.6, 0.8])
    for ab, est, label, color in [
        ("naive_confidence", "naive_confidence", "Naive confidence", GRAY),
        ("full_PTL", "random_forest", "Full PTL RF", BLUE),
        ("no_context_distance", "random_forest", "No context distance", LIGHT_BLUE),
    ]:
        row = selective[selective["ablation"].eq(ab) & selective["estimator"].eq(est)]
        if row.empty:
            continue
        r = row.iloc[0]
        risk = [r[f"false_transportability_rate_at_0p{x}"] for x in [2, 4, 6, 8]]
        axes[1].plot(cov, risk, marker="o", color=color, label=label)
    axes[1].set_xlabel("Coverage")
    axes[1].set_ylabel("False-transportability rate")
    axes[1].legend(frameon=False)
    despine_data(axes[1])

    cur = oof[oof["ablation"].eq("full_PTL") & oof["estimator"].eq("random_forest")].copy()
    cur["bin"] = pd.qcut(cur["score"], q=8, duplicates="drop")
    cal = cur.groupby("bin", observed=True).agg(score=("score", "mean"), truth=("y_true", "mean"), n=("y_true", "size")).reset_index()
    axes[2].plot([0, 1], [0, 1], color=RULE, ls="--", lw=0.7)
    axes[2].scatter(cal["score"], cal["truth"], s=np.sqrt(cal["n"]) * 2.5, color=BLUE, edgecolor=BLACK, linewidth=0.4)
    axes[2].set_xlabel("Mean PTL score")
    axes[2].set_ylabel("Observed transportability")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    despine_data(axes[2])
    save_figure(fig, output_dir, "supp_fig_s2_sensitivity_risk_calibration")


def make_s3(output_dir: Path) -> None:
    gears = read_table(TABLES, "gears_validation_summary.csv")
    apply_style()
    fig = plt.figure(figsize=(12.6, 4.9), facecolor=WHITE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.05], wspace=0.52)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    framed_panel(ax_a, "A", "GEARS completion status")
    gears = gears.copy()
    gears["status_group"] = np.where(gears["status"].astype(str).str.lower().isin(["success", "completed", "complete", "done"]), "completed", "missing")
    status = gears.groupby(["dataset_scope", "split_family", "status_group"]).size().reset_index(name="n")
    piv = status.pivot_table(index=["dataset_scope", "split_family"], columns="status_group", values="n", fill_value=0)
    if "completed" not in piv:
        piv["completed"] = 0
    if "missing" not in piv:
        piv["missing"] = 0
    y = np.arange(len(piv))
    ax_a.barh(y, piv.get("completed", 0), color=TEAL, edgecolor=BLACK, linewidth=0.4, label="Completed")
    ax_a.barh(y, piv.get("missing", 0), left=piv.get("completed", 0), color=LIGHT_BLUE, edgecolor=BLACK, linewidth=0.4, label="Missing/audited")
    def short_dataset(value: str) -> str:
        value = str(value).replace("_filtered", "")
        value = value.replace("ReplogleWeissman2022_K562_essential", "Replogle K562")
        value = value.replace("ReplogleWeissman2022_rpe1", "Replogle RPE1")
        value = value.replace("NormanWeissman2019", "Norman")
        return value

    ax_a.set_yticks(y, [f"{short_dataset(d)}\n{pretty_split(s, short=True)}" for d, s in piv.index])
    ax_a.set_xlabel("Planned rows")
    legend = ax_a.legend(frameon=True, loc="upper right")
    legend.get_frame().set_facecolor(WHITE)
    legend.get_frame().set_alpha(0.88)
    legend.get_frame().set_edgecolor(WHITE)
    despine_data(ax_a, axis="x")

    framed_panel(ax_b, "B", "Completed output-contract metrics")
    done = gears[gears["status_group"].eq("completed")].copy()
    if not done.empty:
        y2 = np.arange(len(done))
        labels = [f"{short_dataset(r.dataset_scope)}\n{pretty_split(r.split_family, short=True)} seed{int(r.seed)}" for _, r in done.iterrows()]
        ax_b.barh(y2, done["mean_cosine_non_control"], color=ORANGE, edgecolor=BLACK, linewidth=0.4)
        ax_b.set_yticks(y2, labels)
        for yi, (_, r) in zip(y2, done.iterrows()):
            ax_b.text(r["mean_cosine_non_control"] + 0.005, yi, f"genes={int(r['gene_count'])}", va="center", fontsize=7)
    ax_b.set_xlabel("Mean non-control cosine")
    despine_data(ax_b, axis="x")
    save_figure(fig, output_dir, "supp_fig_s3_gears_contract")


def make(output_dir: Path = OUTPUT) -> None:
    make_s1(output_dir)
    make_s2(output_dir)
    make_s3(output_dir)


if __name__ == "__main__":
    make()
