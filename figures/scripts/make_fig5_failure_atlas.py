from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figlib.export import save_figure, save_panel
from figlib.panels import (
    plot_biological_diagnostics,
    plot_context_enrichment,
    plot_failure_heatmap,
    plot_failure_taxonomy,
    plot_severity_composition,
    read_table,
)
from figlib.style import apply_style
from figlib.palettes import WHITE

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
OUTPUT = ROOT / "results" / "figures_methods"


def make(output_dir: Path = OUTPUT) -> None:
    apply_style()
    atlas = read_table(TABLES, "failure_mode_atlas.csv")
    bio = read_table(TABLES, "ptl_retention_biology_summary.csv")
    fig = plt.figure(figsize=(12.8, 8.75), facecolor=WHITE)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.02, 1.08], width_ratios=[0.95, 1.25, 1.0], hspace=0.42, wspace=0.31)
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
    plot_biological_diagnostics(axes["E"], bio)
    save_figure(fig, output_dir, "fig5_methods_failure_biology")

    panel_specs = [
        ("A_taxonomy", lambda ax: plot_failure_taxonomy(ax), (3.6, 3.4)),
        ("B_heatmap", lambda ax: plot_failure_heatmap(ax, atlas), (6.4, 3.4)),
        ("C_composition", lambda ax: plot_severity_composition(ax, atlas), (6.2, 3.6)),
        ("D_context", lambda ax: plot_context_enrichment(ax, atlas), (4.0, 3.5)),
        ("E_biology", lambda ax: plot_biological_diagnostics(ax, bio), (7.0, 3.4)),
    ]
    for name, fn, size in panel_specs:
        pfig, pax = plt.subplots(figsize=size, facecolor=WHITE)
        fn(pax)
        save_panel(pfig, output_dir, f"fig5_{name}")


if __name__ == "__main__":
    make()
