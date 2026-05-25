from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd

from figlib.export import save_figure
from figlib.panels import (
    plot_context_enrichment,
    plot_dataset_bar,
    plot_failure_heatmap,
    plot_failure_taxonomy,
    plot_split_audit_heatmap,
    plot_split_design_matrix,
    plot_severity_composition,
    read_table,
)
from figlib.palettes import BLACK, BLUE, GREEN, LIGHT_BLUE, LIGHT_GRAY, LIGHT_ORANGE, LIGHT_RED, LIGHT_TEAL, ORANGE, PURPLE, RED, RULE, TEAL, WHITE
from figlib.schematic import arrow, icon_heatmap, rounded_box
from figlib.style import apply_style, framed_panel
from make_fig3_rank_instability import make as make_fig3
from make_fig4_ptl_filtering import make as make_fig4
from make_supplementary import make as make_supplementary_base

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
OUTPUT = ROOT / "results" / "figures_cbac"


def _copy_stem(output_dir: Path, old: str, new: str, exts: tuple[str, ...] = ("png", "pdf", "svg", "tiff")) -> None:
    for ext in exts:
        src = output_dir / f"{old}.{ext}"
        dst = output_dir / f"{new}.{ext}"
        if src.exists() and str(src).lower() != str(dst).lower():
            shutil.copy2(src, dst)


def _module(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, subtitle: str, color: str) -> None:
    rounded_box(ax, (x, y), (w, h), edge=color, face=WHITE, radius=0.012, lw=1.0)
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top", fontsize=9.5, color=color, fontweight="bold")
    ax.text(x + w / 2, y + 0.050, subtitle, ha="center", va="bottom", fontsize=7.4, color=BLACK, linespacing=1.15)


def make_graphical_abstract(output_dir: Path = OUTPUT) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(13.28, 5.31), facecolor=WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.93, "Reliability modeling for single-cell perturbation-response prediction", fontsize=15, fontweight="bold", ha="left")
    xs = [0.04, 0.235, 0.43, 0.625, 0.82]
    titles = ["Public Perturb-seq", "Stress-transfer tests", "Completed predictions", "PTL reliability model", "Biological audit"]
    subtitles = [
        "functional genomics\nscreens and controls",
        "unseen perturbation,\nlow support, dataset shift",
        "delta signatures from\ntransparent or published models",
        "deployment features\nscore keep/abstain risk",
        "prioritize retained\npredictions; flag failures",
    ]
    colors = [BLUE, ORANGE, PURPLE, TEAL, RED]
    for i, (x, title, sub, color) in enumerate(zip(xs, titles, subtitles, colors)):
        _module(ax, x, 0.35, 0.14, 0.36, title, sub, color)
        if i < len(xs) - 1:
            arrow(ax, (x + 0.145, 0.53), (xs[i + 1] - 0.010, 0.53), lw=1.2)
    icon_heatmap(ax, 0.455, 0.47, 0.070, 0.070)
    ax.scatter([0.86, 0.89, 0.92, 0.86, 0.89, 0.92], [0.54, 0.57, 0.53, 0.46, 0.49, 0.45], s=34, color=[TEAL, TEAL, RED, TEAL, ORANGE, RED], edgecolor=BLACK, linewidth=0.35)
    ax.text(0.50, 0.18, "Core result: random-split winners do not reliably transport; PTL reduces false transportability before biological interpretation.", ha="center", fontsize=9.2, color=BLACK)
    save_figure(fig, output_dir, "Graphical_Abstract")


def make_fig1_cbac(output_dir: Path = OUTPUT) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(13.2, 6.4), facecolor=WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.015, 0.965, "Figure 1. Computational reliability study design for Perturb-seq prediction", fontsize=14, fontweight="bold", ha="left")
    flow = [
        ("A", "Functional genomics screens", "Perturb-seq signatures\nmatched controls\npublic screen metadata", BLUE, LIGHT_BLUE),
        ("B", "Transportability stress", "random anchor\nheld-out perturbation\nlow support and dataset shift", ORANGE, LIGHT_ORANGE),
        ("C", "Reliability modeling", "confidence, support,\nnovelty and split context\npredict keep/abstain", TEAL, LIGHT_TEAL),
        ("D", "Biological interpretation risk", "retained predictions\nfailure atlas\ncase-level diagnostics", RED, LIGHT_RED),
    ]
    for i, (letter, title, body, edge, face) in enumerate(flow):
        x = 0.035 + i * 0.235
        rounded_box(ax, (x, 0.34), (0.19, 0.38), edge=edge, face=face, radius=0.012, lw=1.0)
        ax.text(x + 0.020, 0.675, letter, fontsize=13, fontweight="bold", color=edge, ha="left")
        ax.text(x + 0.095, 0.655, title, fontsize=9.5, fontweight="bold", color=edge, ha="center")
        ax.text(x + 0.095, 0.445, body, fontsize=8.0, ha="center", va="center", linespacing=1.25)
        if i < len(flow) - 1:
            arrow(ax, (x + 0.195, 0.53), (x + 0.230, 0.53), lw=1.1)
    rounded_box(ax, (0.20, 0.13), (0.60, 0.10), edge=RULE, face=WHITE, radius=0.006, lw=0.8)
    ax.text(0.50, 0.18, r"Relative transportability label:  $\cos(\hat{\Delta}, \Delta) \geq \tau \times$ split-specific anchor fidelity", ha="center", va="center", fontsize=9.0)
    save_figure(fig, output_dir, "fig1_cbac_overview")


def make_fig2_cbac(output_dir: Path = OUTPUT) -> None:
    apply_style()
    manifest = read_table(TABLES, "baseline_run_matrix.csv")
    audit = read_table(TABLES, "split_audit.csv")
    fig = plt.figure(figsize=(16.2, 6.2), facecolor=WHITE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.72, 1.08], wspace=0.30)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    plot_dataset_bar(axes[0], manifest)
    plot_split_design_matrix(axes[1])
    plot_split_audit_heatmap(axes[2], audit, panel_label="C")
    axes[1].set_title("Split families encode transfer questions", loc="left", x=0.070, pad=8, fontsize=10.5, fontweight="bold")
    axes[2].set_title("Compact leakage and overlap audit", loc="left", x=0.070, pad=8, fontsize=10.5, fontweight="bold")
    save_figure(fig, output_dir, "fig2_cbac_benchmark_design")


def plot_case_examples(ax: plt.Axes, cases: pd.DataFrame, panel_label: str = "E") -> None:
    framed_panel(ax, panel_label, "Case examples: biological interpretation risk")
    ax.set_axis_off()
    palette = {
        "retained_high_transportability": (TEAL, LIGHT_TEAL, "Retained"),
        "confidence_failure_rejected": (ORANGE, LIGHT_ORANGE, "High-confidence failure"),
        "transfer_boundary_failure": (RED, LIGHT_RED, "Dataset-boundary failure"),
    }
    for i, (_, row) in enumerate(cases.iterrows()):
        x = 0.035 + i * 0.315
        edge, face, label = palette.get(str(row["case_type"]), (BLUE, LIGHT_BLUE, str(row["case_type"]).replace("_", " ")))
        rounded_box(ax, (x, 0.12), (0.285, 0.68), edge=edge, face=face, radius=0.012, lw=0.95)
        ax.text(x + 0.018, 0.73, label, ha="left", va="center", fontsize=9.0, color=edge, fontweight="bold")
        ax.text(x + 0.018, 0.63, f"Perturbation: {row['perturbation_label']}", ha="left", va="center", fontsize=8.2, color=BLACK)
        stress = str(row["split_family"]).replace("_split", "").replace("_", " ")
        ax.text(x + 0.018, 0.54, f"Stress: {stress}", ha="left", va="center", fontsize=8.2, color=BLACK)
        ax.text(x + 0.018, 0.43, f"cosine = {float(row['cosine_similarity']):.2f}", ha="left", va="center", fontsize=9.1, color=BLACK, fontweight="bold")
        ax.text(x + 0.145, 0.43, f"top-gene direction = {float(row['top8_direction_consistency']):.2f}", ha="left", va="center", fontsize=8.2, color=BLACK)
        genes = str(row["shared_top_genes"]).replace(";", ", ")
        if len(genes) > 56:
            genes = genes[:53] + "..."
        ax.text(x + 0.018, 0.27, f"Shared top genes:\n{genes}", ha="left", va="center", fontsize=7.7, color=BLACK, linespacing=1.20)


def make_fig5_cbac(output_dir: Path = OUTPUT) -> None:
    apply_style()
    atlas = read_table(TABLES, "failure_mode_atlas.csv")
    cases = read_table(TABLES, "cbac_case_examples.csv")
    fig = plt.figure(figsize=(13.8, 9.3), facecolor=WHITE)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.95, 1.02, 1.20], width_ratios=[0.95, 1.32, 1.0], hspace=0.46, wspace=0.33)
    axes = {
        "A": fig.add_subplot(gs[0, 0]),
        "B": fig.add_subplot(gs[0, 1:]),
        "C": fig.add_subplot(gs[1, :2]),
        "D": fig.add_subplot(gs[1, 2]),
        "E": fig.add_subplot(gs[2, :]),
    }
    plot_failure_taxonomy(axes["A"])
    plot_failure_heatmap(axes["B"], atlas)
    plot_severity_composition(axes["C"], atlas)
    plot_context_enrichment(axes["D"], atlas)
    plot_case_examples(axes["E"], cases)
    save_figure(fig, output_dir, "fig5_cbac_failure_cases")

    sfig, sax = plt.subplots(figsize=(11.0, 3.6), facecolor=WHITE)
    plot_case_examples(sax, cases, panel_label="S4")
    save_figure(sfig, output_dir, "supp_fig_s4_case_examples")


def make(output_dir: Path = OUTPUT) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    make_graphical_abstract(output_dir)
    make_fig1_cbac(output_dir)
    make_fig2_cbac(output_dir)
    make_fig3(output_dir)
    make_fig4(output_dir)
    make_fig5_cbac(output_dir)
    make_supplementary_base(output_dir)
    _copy_stem(output_dir, "fig3_methods_rank_instability", "fig3_cbac_ranking_instability")
    _copy_stem(output_dir, "fig4_methods_ptl_reliability", "fig4_cbac_reliability_filtering")
    _copy_stem(output_dir, "supp_fig_s1_split_audit", "Supplementary_Figure_S1")
    _copy_stem(output_dir, "supp_fig_s2_sensitivity_risk_calibration", "Supplementary_Figure_S2")
    _copy_stem(output_dir, "supp_fig_s3_gears_contract", "Supplementary_Figure_S3")
    _copy_stem(output_dir, "supp_fig_s4_case_examples", "Supplementary_Figure_S4")
    _copy_stem(output_dir, "Graphical_Abstract", "Graphical_Abstract")
    _copy_stem(output_dir, "fig1_cbac_overview", "Figure_1")
    _copy_stem(output_dir, "fig2_cbac_benchmark_design", "Figure_2")
    _copy_stem(output_dir, "fig3_cbac_ranking_instability", "Figure_3")
    _copy_stem(output_dir, "fig4_cbac_reliability_filtering", "Figure_4")
    _copy_stem(output_dir, "fig5_cbac_failure_cases", "Figure_5")
    print(f"Wrote CBAC figures to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CBAC-targeted PTL figures.")
    parser.add_argument("--output-dir", default=str(OUTPUT))
    args = parser.parse_args()
    make(Path(args.output_dir))


if __name__ == "__main__":
    main()
