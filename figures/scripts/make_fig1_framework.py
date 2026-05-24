from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

from figlib.export import save_figure, save_panel
from figlib.palettes import BLACK, BLUE, GRAY, GREEN, LIGHT_BLUE, LIGHT_GRAY, LIGHT_ORANGE, LIGHT_PURPLE, LIGHT_TEAL, ORANGE, PURPLE, RED, RULE, TEAL, WHITE
from figlib.schematic import arrow, icon_expression, icon_heatmap, icon_metadata, icon_report, mini_decisions, reliability_tree, rounded_box
from figlib.style import apply_style

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "figures_methods"


def draw_framework(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.00, 1.01, "Figure 1. PTL framework: from model prediction to transportability decision", fontsize=15, fontweight="bold", ha="left")

    modules = [
        ("A", "Input contract", BLUE, LIGHT_BLUE, 0.015, 0.82, 0.19),
        ("B", "Stress split +\nlabeler", ORANGE, LIGHT_ORANGE, 0.245, 0.82, 0.31),
        ("C", "PTL reliability layer", GREEN, LIGHT_TEAL, 0.585, 0.82, 0.23),
        ("D", "Atlas outputs", PURPLE, LIGHT_PURPLE, 0.845, 0.82, 0.14),
    ]
    for letter, title, edge, face, x, y, w in modules:
        rounded_box(ax, (x, y), (w, 0.10), edge=edge, face=face, radius=0.009, lw=1.0)
        ax.text(x + 0.055, y + 0.054, letter, ha="center", va="center", color=edge, fontsize=11, fontweight="bold")
        ax.text(x + w / 2 + 0.03, y + 0.054, title, ha="center", va="center", color=edge, fontsize=10, fontweight="bold", linespacing=0.95)

    rounded_box(ax, (0.015, 0.22), (0.19, 0.55), edge=RULE, face=WHITE)
    input_items = [
        (0.070, 0.625, "Perturbation data\n(expression)", icon_expression),
        (0.070, 0.495, "Metadata and context\n(dataset, cell type, ...)", icon_metadata),
        (0.070, 0.365, "Model prediction outputs\nPred.       Target", None),
        (0.070, 0.245, "Support and\nsplit manifest", None),
    ]
    for cx, cy, label, icon in input_items:
        if icon is not None:
            icon(ax, cx - 0.035, cy + 0.01, 1.0)
        elif "Model" in label:
            icon_heatmap(ax, cx - 0.065, cy - 0.025, 0.052, 0.060)
        else:
            icon_report(ax, cx - 0.065, cy - 0.025, 0.042, 0.060)
        ax.text(cx + 0.005, cy, label, ha="left", va="center", fontsize=7.4)

    rounded_box(ax, (0.245, 0.22), (0.31, 0.55), edge=RULE, face=WHITE)
    ax.text(0.295, 0.665, "Delta signature", ha="center", va="center", fontsize=8.0, fontweight="bold")
    ax.text(0.430, 0.665, "Context-stress\nmanifest", ha="center", va="center", fontsize=8.0, fontweight="bold")
    ax.scatter([0.30, 0.315, 0.325, 0.285, 0.337], [0.565, 0.585, 0.550, 0.535, 0.520], s=13, color=BLUE, edgecolor=WHITE, linewidth=0.25)
    ax.scatter([0.30, 0.315, 0.325, 0.285, 0.337], [0.405, 0.425, 0.390, 0.375, 0.360], s=13, color=LIGHT_GRAY, edgecolor=RULE, linewidth=0.25)
    arrow(ax, (0.310, 0.510), (0.310, 0.445), lw=0.7)
    arrow(ax, (0.310, 0.355), (0.310, 0.295), lw=0.7)
    ax.text(0.310, 0.590, "Perturbed cells\n(context c)", ha="center", fontsize=6.8)
    ax.text(0.310, 0.430, "Matched controls\n(context c)", ha="center", fontsize=6.8)
    icon_heatmap(ax, 0.265, 0.240, 0.090, 0.030)
    ax.text(0.310, 0.285, r"$\Delta_{p,c}=\bar{x}_{p,c}-\bar{x}_{0,c}$", ha="center", fontsize=8.2)
    split_rows = [
        ("Random anchor", BLUE),
        ("Held-out perturbation", ORANGE),
        ("Held-out combination", PURPLE),
        ("Low-support", GREEN),
        ("Dataset heldout", TEAL),
        ("External holdout", RED),
        ("Label rule + audit", GREEN),
    ]
    for i, (label, color) in enumerate(split_rows):
        yy = 0.585 - i * 0.055
        if i < 3:
            ax.scatter([0.402, 0.418], [yy, yy + 0.014], s=12, facecolor=WHITE, edgecolor=color, linewidth=0.8)
            ax.scatter([0.412], [yy - 0.012], s=12, marker="x", color=color, linewidth=0.8)
        elif i == 3:
            ax.bar([0.402, 0.414, 0.426], [0.018, 0.040, 0.026], width=0.007, bottom=yy - 0.020, color=color, alpha=0.55)
        elif i == 4:
            ax.add_patch(patches.Circle((0.413, yy), 0.014, facecolor=WHITE, edgecolor=color, lw=0.8))
            ax.add_patch(patches.Rectangle((0.399, yy - 0.015), 0.028, 0.030, facecolor=color, alpha=0.15, edgecolor=color, lw=0.5))
        elif i == 5:
            ax.add_patch(patches.Circle((0.413, yy), 0.015, facecolor=WHITE, edgecolor=color, lw=0.8))
            ax.plot([0.398, 0.428], [yy, yy], color=color, lw=0.7)
            ax.plot([0.413, 0.413], [yy - 0.015, yy + 0.015], color=color, lw=0.7)
        else:
            ax.add_patch(patches.RegularPolygon((0.413, yy), 5, radius=0.017, facecolor=WHITE, edgecolor=color, lw=0.8))
        ax.text(0.445, yy, label, ha="left", va="center", fontsize=6.9)

    rounded_box(ax, (0.585, 0.22), (0.23, 0.55), edge=RULE, face=WHITE)
    ax.text(0.700, 0.675, "Deployment features", ha="center", fontsize=8.0, fontweight="bold")
    features = [("Fidelity / confidence", BLUE), ("Support", ORANGE), ("Perturbation novelty", PURPLE), ("Split family (stress)", GREEN), ("Dataset / context", TEAL), ("Context distance\n(optional)", GRAY)]
    for i, (label, color) in enumerate(features):
        yy = 0.622 - i * 0.050
        rounded_box(ax, (0.618, yy - 0.017), (0.160, 0.030), edge=color, face=color + "15", radius=0.006, lw=0.55)
        ax.text(0.700, yy, label, ha="center", va="center", fontsize=6.8)
    ax.plot([0.612, 0.790], [0.333, 0.333], color=RULE, lw=0.5)
    ax.text(0.700, 0.300, "Reliability model\n(keep / abstain)", ha="center", fontsize=7.2, fontweight="bold")
    reliability_tree(ax, 0.700, 0.265, 0.48)

    rounded_box(ax, (0.845, 0.22), (0.14, 0.55), edge=RULE, face=WHITE)
    ax.text(0.915, 0.650, "Per-signature\ndecision", ha="center", fontsize=7.8, fontweight="bold")
    ax.scatter([0.885, 0.885], [0.592, 0.555], s=32, color=[TEAL, RED], edgecolor=BLACK, linewidth=0.3)
    ax.text(0.904, 0.592, "Keep", fontsize=6.9, va="center")
    ax.text(0.904, 0.555, "Abstain", fontsize=6.9, va="center")
    mini_decisions(ax, 0.872, 0.510)
    ax.text(0.915, 0.372, "Reports", ha="center", fontsize=7.8, fontweight="bold")
    icon_report(ax, 0.870, 0.290, 0.045, 0.070)
    ax.text(0.915, 0.250, "Failure-mode\natlas", ha="center", fontsize=7.8, fontweight="bold")
    icon_heatmap(ax, 0.863, 0.190, 0.090, 0.040)

    for start, end in [((0.205, 0.465), (0.242, 0.465)), ((0.555, 0.465), (0.583, 0.465)), ((0.815, 0.465), (0.842, 0.465))]:
        arrow(ax, start, end, lw=0.9)
    rounded_box(ax, (0.145, 0.055), (0.710, 0.060), edge=PURPLE, face=WHITE, radius=0.008, lw=0.8)
    ax.text(0.500, 0.085, r"Relative transportability rule:  $\cos(\hat{\Delta}, \Delta) \geq \tau \times \mathrm{median(anchor\ fidelity)}$", ha="center", va="center", fontsize=8.5, color=PURPLE)


def make(output_dir: Path = OUTPUT) -> None:
    apply_style()
    fig = plt.figure(figsize=(15.5, 8.1), facecolor=WHITE)
    ax = fig.add_axes([0.02, 0.045, 0.96, 0.92])
    draw_framework(ax)
    save_figure(fig, output_dir, "fig1_methods_workflow")

    panel_fig = plt.figure(figsize=(15.5, 8.1), facecolor=WHITE)
    panel_ax = panel_fig.add_axes([0.02, 0.02, 0.96, 0.94])
    draw_framework(panel_ax)
    save_panel(panel_fig, output_dir, "fig1_framework_full")


if __name__ == "__main__":
    make()
