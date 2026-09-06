from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figlib.export import save_figure, save_panel
from figlib.panels import plot_model_heatmap_with_ranks, plot_rank_trajectories, plot_transfer_zoom, read_table
from figlib.style import apply_style
from figlib.palettes import WHITE

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
OUTPUT = ROOT / "results" / "figures_methods"


def make(output_dir: Path = OUTPUT) -> None:
    apply_style()
    transfer = read_table(TABLES, "transfer_decay_summary.csv")
    fig = plt.figure(figsize=(12.2, 7.8), facecolor=WHITE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.08], hspace=0.32, wspace=0.34)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    plot_model_heatmap_with_ranks(ax_a, transfer)
    plot_rank_trajectories(ax_b, transfer)
    plot_transfer_zoom(ax_c, transfer)
    save_figure(fig, output_dir, "fig3_methods_rank_instability")

    for name, fn, size in [
        ("A_heatmap", plot_model_heatmap_with_ranks, (6.8, 5.2)),
        ("B_rank_trajectories", plot_rank_trajectories, (5.0, 3.8)),
        ("C_transfer_zoom", plot_transfer_zoom, (5.0, 3.8)),
    ]:
        pfig, pax = plt.subplots(figsize=size, facecolor=WHITE)
        fn(pax, transfer)
        save_panel(pfig, output_dir, f"fig3_{name}")


if __name__ == "__main__":
    make()
