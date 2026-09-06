from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figlib.export import save_figure, save_panel
from figlib.panels import plot_context_distance, plot_dataset_bar, plot_split_audit_heatmap, plot_split_design_matrix, plot_support_boxplot, read_table
from figlib.style import apply_style
from figlib.palettes import WHITE

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
OUTPUT = ROOT / "results" / "figures_methods"


def make(output_dir: Path = OUTPUT) -> None:
    apply_style()
    manifest = read_table(TABLES, "baseline_run_matrix.csv")
    audit = read_table(TABLES, "split_audit.csv")
    fig = plt.figure(figsize=(15.7, 5.6), facecolor=WHITE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.62, 1.0], wspace=0.28)
    axes = {
        "A": fig.add_subplot(gs[0, 0]),
        "B": fig.add_subplot(gs[0, 1]),
        "C": fig.add_subplot(gs[0, 2]),
    }
    plot_dataset_bar(axes["A"], manifest)
    plot_split_design_matrix(axes["B"])
    plot_split_audit_heatmap(axes["C"], audit, panel_label="C")
    save_figure(fig, output_dir, "fig2_methods_benchmark_audit")

    panel_calls = {
        "A_dataset_bar": lambda ax: plot_dataset_bar(ax, manifest),
        "B_split_design": plot_split_design_matrix,
        "C_support": lambda ax: plot_support_boxplot(ax, audit),
        "D_context_distance": lambda ax: plot_context_distance(ax, audit),
        "E_split_audit": lambda ax: plot_split_audit_heatmap(ax, audit),
    }
    for name, fn in panel_calls.items():
        pfig, pax = plt.subplots(figsize=(5.2, 3.6), facecolor=WHITE)
        fn(pax)
        save_panel(pfig, output_dir, f"fig2_{name}")


if __name__ == "__main__":
    make()
