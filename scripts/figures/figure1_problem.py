"""Figure 1: reliability ordering under a source-frozen context shift."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figures.common import (
    CONTEXT_ORDER, METRICS, add_panel_label, context_label, metric_color,
    metric_label, num, primary_depth, full_rank_table, rank_flow, save_figure,
)
from scripts.ptl_figure_style import figure_size
from scripts.figures.ribbons import filled_rank_braid, shortlist_summary_glyph


def _setup(ax: plt.Axes, rows: list[dict]) -> None:
    """Use the supplied editable experimental-setup slide as Fig.1A.

    The slide is exported once from the user-provided PPTX and then placed as
    a high-resolution figure asset.  This keeps the requested editable design
    intact while the remaining panels stay native, data-backed Python plots.
    """
    asset = Path(__file__).resolve().parents[2] / "results/figures/reliability_transportability/source_assets/fig1a_experimental_setup.png"
    if not asset.exists():
        raise FileNotFoundError(
            "Missing Fig.1A source asset. Export the provided experimental setup PPTX to "
            f"{asset} before generating the figures."
        )
    image = plt.imread(asset)
    ax.axis("off")
    ax.set_xticks([]); ax.set_yticks([])
    # Preserve the high-resolution slide pixels at the final paper scale;
    # Lanczos softens the small schematic text after PDF embedding.
    ax.imshow(image, interpolation="none", resample=False, aspect="auto")
    rows.append({"panel": "A", "description": "source-frozen predictor evaluated under Ctrl, Co-culture, and IFNγ", "provenance": "frozen Frangieh protocol"})


def _decomposition(ax: plt.Axes, depth: pd.DataFrame, rows: list[dict]) -> None:
    ax.set_xlim(0, .42); ax.set_ylim(-.55, 2.55)
    ax.set_yticks([2, 1, 0], [r"$D_{\rm cross}$", "within floor", r"$D_{\rm adj}$"])
    ax.tick_params(axis="y", length=0, pad=-1, labelsize=6.0)
    ax.set_xlabel("magnitude", fontsize=6.5, labelpad=3)
    for yi, label in enumerate(("cross_disagreement", "measurement_floor", "identifiable_divergence")):
        y = 2 - yi
        for metric in METRICS:
            item = depth.loc[depth["metric"].eq(metric) & depth["cell_budget_label"].eq("full")]
            if item.empty:
                continue
            item = item.iloc[0]
            value = num(item[label])
            lo, hi = num(item.get(f"{label}_ci_low", np.nan)), num(item.get(f"{label}_ci_high", np.nan))
            if not np.isfinite(value):
                continue
            offset = {METRICS[0]: -.055, METRICS[1]: 0, METRICS[2]: .055}[metric]
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([max(0, lo), hi], [y + offset, y + offset], color=metric_color(metric), lw=1.0, solid_capstyle="round")
            ax.scatter(value, y + offset, s=20, color=metric_color(metric), edgecolor="white", linewidth=.35, zorder=3)
            rows.append({"panel": "C", "metric": metric, "component": label, "value": value, "ci_low": lo, "ci_high": hi, "depth": "full", "provenance": "formal_v2_claim_lock_measurement_fullsize_summary.csv"})
    ax.axvline(0, color="#B8BEC8", lw=.65, zorder=0)
    ax.set_xlim(left=0); ax.grid(axis="x", color="#E1E5E8", lw=.45, alpha=.7); ax.set_axisbelow(True)


def _endpoint_labels(ax: plt.Axes, ranks: pd.DataFrame, labels: list[str]) -> None:
    """Place selected endpoint labels with final-renderer-space clearance."""
    selected = ranks.loc[ranks["perturbation_label"].astype(str).isin(set(labels))].copy()
    selected["label"] = selected["perturbation_label"].astype(str)
    selected = selected.sort_values(["ifng_rank", "label"], kind="stable")
    if selected.empty:
        return
    # Rank-space gaps are not typography-space gaps.  Relax in display points
    # after the axes have their final data limits, then map back to rank units.
    actual = selected["ifng_rank"].astype(float).to_numpy()
    x_label = 2.13
    display = np.asarray([ax.transData.transform((x_label, value))[1] for value in actual], dtype=float)
    order = np.argsort(display)
    sorted_display = display[order]
    min_clearance = 7.0 * ax.figure.dpi / 72.0
    placed_display = sorted_display.copy()
    for index in range(1, len(placed_display)):
        placed_display[index] = max(placed_display[index], placed_display[index - 1] + min_clearance)
    lower = ax.bbox.y0 + min_clearance / 2
    upper = ax.bbox.y1 - min_clearance / 2
    if placed_display[-1] > upper:
        placed_display -= placed_display[-1] - upper
    if placed_display[0] < lower:
        placed_display += lower - placed_display[0]
    placed = np.empty_like(placed_display)
    placed[order] = placed_display
    y_values = [float(ax.transData.inverted().transform((0, value))[1]) for value in placed]
    for (_, item), y in zip(selected.iterrows(), y_values):
        actual = float(item["ifng_rank"])
        ax.plot([2.0, 2.10], [actual, y], color="#9AA4AC", lw=.45, zorder=5)
        ax.text(x_label, y, str(item["perturbation_label"]), fontsize=5.05, va="center", color="#263238", zorder=6, clip_on=True)


def figure1(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    ranks = full_rank_table(manifests)
    flow, labels = rank_flow(manifests)
    # Keep deterministic quantile exemplars for labels, but draw every
    # background ribbon from the same canonical 243-label surface.
    display_ranks = ranks
    fullsize = pd.read_csv(manifests / "formal_v2_claim_lock_measurement_fullsize_summary.csv")
    depth = fullsize.loc[
        fullsize["source_environment_id"].eq(CONTEXT_ORDER[0])
        & fullsize["left_target_environment_id"].eq(CONTEXT_ORDER[0])
        & fullsize["right_target_environment_id"].eq(CONTEXT_ORDER[2])
    ].copy()
    depth = depth.rename(columns={
        "ordering_cross_disagreement": "cross_disagreement",
        "ordering_cross_disagreement_ci_low": "cross_disagreement_ci_low",
        "ordering_cross_disagreement_ci_high": "cross_disagreement_ci_high",
        "ordering_measurement_floor": "measurement_floor",
        "ordering_measurement_floor_ci_low": "measurement_floor_ci_low",
        "ordering_measurement_floor_ci_high": "measurement_floor_ci_high",
        "ordering_delta_meas_id": "identifiable_divergence",
        "ordering_delta_meas_id_ci_low": "identifiable_divergence_ci_low",
        "ordering_delta_meas_id_ci_high": "identifiable_divergence_ci_high",
    })
    depth["cell_budget_label"] = "full"
    if len(depth) != len(METRICS):
        raise ValueError(f"Fig.1 full-size primary decomposition requires {len(METRICS)} rows, got {len(depth)}")
    rows: list[dict] = []
    fig = plt.figure(figsize=figure_size("fig1"), constrained_layout=False)
    # The supplied A slide is deliberately given a full-width, near-native
    # aspect.  B/C/D share one baseline and are intentionally compact, nearly
    # square quantitative panels matching the user's reference layout.
    grid = fig.add_gridspec(2, 1, height_ratios=(2.20, 1.30), hspace=.28)
    body = grid[1].subgridspec(1, 3, width_ratios=(.39, .28, .33), wspace=.32)

    ax = fig.add_subplot(grid[0, :]); _setup(ax, rows)
    highlight_idx = np.rint(np.linspace(0, len(flow) - 1, 6)).astype(int)
    highlight_labels = [str(flow.iloc[i]["perturbation_label"]) for i in highlight_idx]
    ax = fig.add_subplot(body[0, 0]); add_panel_label(ax, "B", x=-.07, y=1.17); ax.set_title(r"Rank transport · $\Delta$ cosine", loc="left", pad=5, fontweight="bold")
    filled_rank_braid(ax, display_ranks, highlight_labels, [context_label(v) for v in CONTEXT_ORDER], max_background=None, width=.38, selected_width=1.05)
    _endpoint_labels(ax, display_ranks, highlight_labels)
    # Reserve a compact internal label column, as in the reference layout;
    # labels therefore cannot spill into panel C.
    ax.set_xlim(-.22, 2.80)
    ax.set_ylabel("rank (1 = best)", fontsize=7); ax.set_xlabel("")
    for _, item in display_ranks.loc[display_ranks["perturbation_label"].astype(str).isin(set(highlight_labels))].iterrows():
        rows.append({"panel": "B", **item.to_dict(), "selected_exemplar": True, "provenance": "frangieh_source_frozen_predictions.npz; full 243-label source-frozen risk surface"})
    ax = fig.add_subplot(body[0, 1]); add_panel_label(ax, "C"); ax.set_title("Ordering components", loc="left", pad=5, fontweight="bold"); _decomposition(ax, depth, rows)
    ax = fig.add_subplot(body[0, 2]); add_panel_label(ax, "D", x=-.07); ax.set_title("Shortlist replacement preview", loc="left", pad=5, fontweight="bold"); shortlist_summary_glyph(ax, ranks, rows, budget=.10, panel="D", provenance="frangieh_source_frozen_predictions.npz; canonical delta-cosine risk across all 243 shared labels")
    fig.suptitle("Fig. 1   Reliability ordering under biological context shift", fontsize=12, fontweight="bold", x=.02, ha="left")
    return save_figure(fig, out_dir, "reliability_transportability_fig1_graphical_abstract"), pd.DataFrame(rows)


__all__ = ["figure1"]
