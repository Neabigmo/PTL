from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figlib.export import save_figure, save_panel
from figlib.panels import plot_false_transportability_bars, plot_pr_curve, plot_reliability_baseline, plot_risk_coverage, plot_roc_curve, read_table
from figlib.style import apply_style
from figlib.palettes import WHITE

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
OUTPUT = ROOT / "results" / "figures_methods"


def _rel_table(metrics):
    rel = metrics.copy()
    if "baseline" in rel.columns:
        rel["method"] = rel["baseline"]
        return rel
    rel["method"] = rel["ablation"].where(rel["estimator"].ne("random_forest"), rel["ablation"])
    rel.loc[rel["ablation"].eq("full_PTL") & rel["estimator"].eq("random_forest"), "method"] = "full_PTL_random_forest"
    rel.loc[rel["ablation"].eq("naive_confidence"), "method"] = "naive_confidence"
    rel.loc[rel["ablation"].eq("calibrated_logistic_deployment"), "method"] = "calibrated_logistic_deployment"
    return rel


def make(output_dir: Path = OUTPUT) -> None:
    apply_style()
    metrics = read_table(TABLES, "reliability_baseline_summary.csv")
    selective = read_table(TABLES, "selective_prediction_summary.csv")
    oof = read_table(TABLES, "ptl_oof_predictions.csv")
    rel = _rel_table(metrics)
    fig = plt.figure(figsize=(12.8, 9.2), facecolor=WHITE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.00, 1.08, 1.0], width_ratios=[1.08, 1.0], hspace=0.68, wspace=0.32)
    axes = {
        "A": fig.add_subplot(gs[0, :]),
        "B": fig.add_subplot(gs[1, 0]),
        "C": fig.add_subplot(gs[1, 1]),
        "D": fig.add_subplot(gs[2, 0]),
        "E": fig.add_subplot(gs[2, 1]),
    }
    plot_false_transportability_bars(axes["A"], rel)
    plot_roc_curve(axes["B"], oof)
    plot_pr_curve(axes["C"], oof)
    plot_risk_coverage(axes["D"], selective)
    plot_reliability_baseline(axes["E"], rel)
    save_figure(fig, output_dir, "fig4_methods_ptl_reliability")

    panel_specs = [
        ("A_false_transportability", lambda ax: plot_false_transportability_bars(ax, rel), (7.6, 3.4)),
        ("B_roc", lambda ax: plot_roc_curve(ax, oof), (4.8, 3.6)),
        ("C_pr", lambda ax: plot_pr_curve(ax, oof), (4.8, 3.6)),
        ("D_risk_coverage", lambda ax: plot_risk_coverage(ax, selective), (4.8, 3.6)),
        ("E_baselines", lambda ax: plot_reliability_baseline(ax, rel), (5.0, 3.8)),
    ]
    for name, fn, size in panel_specs:
        pfig, pax = plt.subplots(figsize=size, facecolor=WHITE)
        fn(pax)
        save_panel(pfig, output_dir, f"fig4_{name}")


if __name__ == "__main__":
    make()
