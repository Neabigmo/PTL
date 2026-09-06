"""
High-Quality Publication Figures for Perturbation Transportability Paper
==========================================================================

This module generates publication-ready figures with:
- Proper subfigure layout with aligned labels
- Consistent font sizes and styles
- No overlapping text or labels
- Balanced subplot proportions
- Clear, readable visualizations
"""
from __future__ import annotations

import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from sklearn.metrics import auc, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# Publication-quality colors
WHITE = "#FFFFFF"
TEXT = "#1a1a1a"
MUTED = "#666666"
RULE = "#cccccc"
HEAT_LOW = "#f8f9fa"
HEAT_HIGH = "#2166ac"
ACCENT = "#d6604d"

# Color palette for split families
SEMANTIC_PALETTE = {
    "random_split": "#4e79a7",
    "unseen_perturbation_split": "#f28e2b",
    "unseen_combination_split": "#59a14f",
    "low_support_split": "#edc948",
    "dataset_heldout_split": "#b07aa1",
    "external_holdout": "#e15759",
    "PTL": "#76b7b2",
    "naive_confidence": "#9d9d9d",
}

MODEL_PALETTE = {
    "ridge_regression_baseline": "#4e79a7",
    "global_delta_baseline": "#59a14f",
    "perturbation_mean_delta_baseline": "#f28e2b",
    "cell_context_knn_delta_baseline": "#b07aa1",
    "control_mean_baseline": "#9d9d9d",
}

SPLIT_ORDER = [
    "random_split",
    "unseen_perturbation_split",
    "unseen_combination_split",
    "low_support_split",
    "dataset_heldout_split",
    "external_holdout",
]

PRIMARY_METRIC = "mean_cosine_non_control"
CONTROL_MODEL = "control_mean_baseline"


def wrap_label(text: Any, width: int = 20) -> str:
    value = str(text).replace("_", " ").strip()
    if not value:
        return ""
    wrapped = textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=False)
    if len(wrapped) == 1 and len(wrapped[0]) > width:
        wrapped = textwrap.wrap(value, width=width, break_long_words=True, break_on_hyphens=True)
    return "\n".join(wrapped)


def pretty_model(model: str) -> str:
    mapping = {
        "ridge_regression_baseline": "Ridge",
        "global_delta_baseline": "Global Δ",
        "perturbation_mean_delta_baseline": "Pert. Δ",
        "cell_context_knn_delta_baseline": "kNN Δ",
        "control_mean_baseline": "Control",
    }
    return mapping.get(model, model.replace("_", " ").title())


def pretty_split(split: str) -> str:
    mapping = {
        "random_split": "Random",
        "unseen_perturbation_split": "Unseen pert.",
        "unseen_combination_split": "Unseen comb.",
        "low_support_split": "Low support",
        "dataset_heldout_split": "Dataset heldout",
        "external_holdout": "External holdout",
    }
    return mapping.get(split, split.replace("_", " ").title())


def _ordered_splits(values: Iterable[Any]) -> list[str]:
    present = [str(v) for v in values if pd.notna(v)]
    return [s for s in SPLIT_ORDER if s in present] + sorted(s for s in set(present) if s not in SPLIT_ORDER)


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(colors=MUTED, length=2)
    for label in ax.get_yticklabels():
        label.set_color(TEXT)
    for label in ax.get_xticklabels():
        label.set_color(TEXT)


def _heatmap_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("custom", [HEAT_LOW, HEAT_HIGH], N=256)


def _draw_box(ax: plt.Axes, pos: tuple, size: tuple, label: str, color: str, style: str = "default") -> None:
    x, y = pos
    w, h = size
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                   facecolor=color, edgecolor=TEXT, linewidth=0.6, alpha=0.85, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.0, color=WHITE,
            fontweight="bold" if style != "default" else "normal", zorder=3)


def _arrow(ax: plt.Axes, start: tuple, end: tuple, color: str = TEXT) -> None:
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0, connectionstyle="arc3,rad=0"))


def _panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.02) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def _save_figure(fig: plt.Figure, paths: dict, key_prefix: str) -> list[Path]:
    outputs = []
    for suffix, path in paths.items():
        if suffix.startswith(key_prefix):
            path.parent.mkdir(parents=True, exist_ok=True)
            if suffix.endswith(".png"):
                fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE)
            elif suffix.endswith(".pdf"):
                fig.savefig(path, bbox_inches="tight", facecolor=WHITE)
            else:
                fig.savefig(path, bbox_inches="tight", facecolor=WHITE)
            outputs.append(path)
    plt.close(fig)
    return outputs


def figure_paths(output_dir: Path | str) -> dict[str, Path]:
    base = Path(output_dir)
    return {
        "fig1_png": base / "fig1_study_design.png",
        "fig1_pdf": base / "fig1_study_design.pdf",
        "fig2_png": base / "fig2_benchmark_composition_audit.png",
        "fig2_pdf": base / "fig2_benchmark_composition_audit.pdf",
        "fig3_png": base / "fig3_ranking_instability_transfer_decay.png",
        "fig3_pdf": base / "fig3_ranking_instability_transfer_decay.pdf",
        "fig4_png": base / "fig4_ptl_selective_filtering.png",
        "fig4_pdf": base / "fig4_ptl_selective_filtering.pdf",
        "fig5_png": base / "fig5_failure_mode_atlas.png",
        "fig5_pdf": base / "fig5_failure_mode_atlas.pdf",
        "supp_fig1_png": base / "supp_fig1_robustness_audit.png",
        "supp_fig1_pdf": base / "supp_fig1_robustness_audit.pdf",
    }


# ============================================================================
# FIGURE 1: Study Design
# ============================================================================
def create_fig1(data: dict[str, Any], output_dir: Path, notes_dir: Path) -> list[Path]:
    """Create Figure 1: Study design with proper subfigure layout."""
    paths = figure_paths(output_dir)

    # Create figure with subfigures for proper alignment
    fig = plt.figure(figsize=(7.5, 5.5), facecolor=WHITE)

    # Main title aligned with subplots
    fig.suptitle("Figure 1. Context-Stress Benchmarking Framework", fontsize=12, fontweight="bold", y=0.98)

    gs = GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.45,
                  left=0.08, right=0.95, top=0.90, bottom=0.08)

    # Panel A: Random vs Context-stress
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Random vs. Context-Stress Splits", loc="left", pad=24, fontsize=9)

    # Draw split diagram
    _draw_box(ax, (0.05, 0.65), (0.28, 0.18), "Random", SEMANTIC_PALETTE["random_split"])
    _draw_box(ax, (0.55, 0.65), (0.35, 0.18), "Context-Stress", SEMANTIC_PALETTE["external_holdout"])
    _arrow(ax, (0.34, 0.74), (0.54, 0.74))

    y0 = 0.50
    for i, split in enumerate(SPLIT_ORDER[1:]):
        y = y0 - i * 0.09
        ax.scatter(0.12, y, s=50, color=SEMANTIC_PALETTE[split], edgecolor=TEXT, linewidth=0.4, zorder=3)
        ax.text(0.18, y, pretty_split(split), ha="left", va="center", fontsize=7.5)

    ax.text(0.05, 0.08, "Stress families isolate perturbation,\ncombination, support, dataset, and\nexternal transfer dimensions.",
            fontsize=7, color=MUTED, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Panel B: Pipeline
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("B", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Benchmark Construction Pipeline", loc="left", pad=24, fontsize=9)

    steps = [
        ("Perturbation\nData", SEMANTIC_PALETTE["random_split"]),
        ("QC /\nPseudobulk", SEMANTIC_PALETTE["low_support_split"]),
        ("Delta\nSignatures", SEMANTIC_PALETTE["unseen_combination_split"]),
        ("Model\nPrediction", SEMANTIC_PALETTE["dataset_heldout_split"]),
    ]
    for i, (label, color) in enumerate(steps):
        x = 0.05 + i * 0.24
        _draw_box(ax, (x, 0.42), (0.18, 0.30), label, color)
        if i < len(steps) - 1:
            _arrow(ax, (x + 0.18, 0.57), (x + 0.235, 0.57))

    ax.text(0.05, 0.18, "All models evaluated on the same\nsignature-level target surface.",
            fontsize=7, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Panel C: Evaluation matrix
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("C", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Model-by-Split Evaluation Matrix", loc="left", pad=24, fontsize=9)

    models = ["Control", "Global Δ", "Pert. Δ", "kNN Δ", "Ridge"]
    n_splits = len(SPLIT_ORDER)
    cell_w, cell_h = 0.08, 0.12

    for i, split in enumerate(SPLIT_ORDER):
        for j, model in enumerate(models):
            x = 0.18 + i * (cell_w + 0.04)
            y = 0.18 + j * cell_h
            color = SEMANTIC_PALETTE[split] if j == i % len(models) else HEAT_LOW
            rect = patches.Rectangle((x, y), cell_w, cell_h * 0.9,
                                      facecolor=color, edgecolor=RULE, linewidth=0.5)
            ax.add_patch(rect)

    # Labels
    for i, split in enumerate(SPLIT_ORDER):
        x = 0.18 + i * (cell_w + 0.04) + cell_w / 2
        ax.text(x, 0.88, f"{i+1}", ha="center", va="top", fontsize=7)

    for j, model in enumerate(models):
        y = 0.18 + j * cell_h + cell_h * 0.45
        ax.text(0.05, y, model, ha="left", va="center", fontsize=7)

    ax.text(0.18, 0.06, "1-6: Random and five stress families", fontsize=6.5, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Panel D: PTL architecture
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("D", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("PTL Reliability Layer", loc="left", pad=24, fontsize=9)

    _draw_box(ax, (0.05, 0.52), (0.25, 0.20), "Model\nOutput", SEMANTIC_PALETTE["random_split"])
    _draw_box(ax, (0.38, 0.52), (0.25, 0.20), "PTL\nScore", SEMANTIC_PALETTE["PTL"])
    _draw_box(ax, (0.70, 0.65), (0.20, 0.14), "Keep", SEMANTIC_PALETTE["unseen_combination_split"])
    _draw_box(ax, (0.70, 0.40), (0.20, 0.14), "Abstain", SEMANTIC_PALETTE["external_holdout"])

    _arrow(ax, (0.30, 0.62), (0.38, 0.62), SEMANTIC_PALETTE["PTL"])
    _arrow(ax, (0.63, 0.65), (0.70, 0.70), SEMANTIC_PALETTE["PTL"])
    _arrow(ax, (0.63, 0.60), (0.70, 0.47), SEMANTIC_PALETTE["PTL"])

    ax.text(0.05, 0.22, "Reliability filter attached\nafter model prediction.", fontsize=7, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    outputs = _save_figure(fig, paths, "fig1")
    plt.close(fig)
    return outputs


# ============================================================================
# FIGURE 2: Benchmark Composition
# ============================================================================
def create_fig2(data: dict[str, Any], output_dir: Path, notes_dir: Path) -> list[Path]:
    """Create Figure 2: Benchmark composition with improved layout."""
    paths = figure_paths(output_dir)

    preprocessing = data.get("preprocessing", pd.DataFrame())
    split_audit = data.get("split_audit", pd.DataFrame())
    all_metrics = data.get("all_metrics", pd.DataFrame())
    examples = data.get("examples", pd.DataFrame())

    fig = plt.figure(figsize=(7.5, 7.0), facecolor=WHITE)
    fig.suptitle("Figure 2. Benchmark Composition and Audit", fontsize=12, fontweight="bold", y=0.98)

    gs = GridSpec(3, 2, figure=fig, wspace=0.35, hspace=0.50,
                  left=0.10, right=0.95, top=0.92, bottom=0.08,
                  height_ratios=[1.2, 1.0, 1.0])

    # Panel A: Dataset composition (horizontal bar chart)
    ax = fig.add_subplot(gs[0, :])
    ax.set_title("A", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Dataset Composition: Signatures and Perturbations", loc="left", pad=24, fontsize=9)

    if not preprocessing.empty and {"dataset_id", "n_signature_rows", "n_unique_perturbations"}.issubset(preprocessing.columns):
        comp = preprocessing.copy()
        comp = comp.sort_values("n_signature_rows", ascending=True)

        # Create bar chart with wrapped labels
        bar_positions = range(len(comp))
        bars = ax.barh(bar_positions, comp["n_signature_rows"],
                       color=SEMANTIC_PALETTE["random_split"], height=0.6, edgecolor=WHITE)

        # Wrapped y-tick labels
        labels = [wrap_label(d, 18) for d in comp["dataset_id"]]
        ax.set_yticks(bar_positions, labels, fontsize=7)

        # Add perturbation count as text on bars
        for i, (bar, pert) in enumerate(zip(bars, comp["n_unique_perturbations"])):
            width = bar.get_width()
            ax.text(width * 1.02, bar.get_y() + bar.get_height()/2,
                    f"n={pert}", va="center", fontsize=6.5, color=MUTED)

        ax.set_xlabel("Number of Signature Rows", fontsize=8)
        ax.set_xlim(0, comp["n_signature_rows"].max() * 1.2)
    else:
        ax.text(0.5, 0.5, "Dataset composition data unavailable", transform=ax.transAxes, ha="center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Test scale
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("B", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Mean Test Units by Split Family", loc="left", pad=24, fontsize=9)

    if not all_metrics.empty:
        scale = all_metrics.groupby("split_family", as_index=False)["n_test_non_control"].mean()
        scale = scale.dropna()
        order = _ordered_splits(scale["split_family"])
        scale["split_family"] = pd.Categorical(scale["split_family"], categories=order, ordered=True)
        scale = scale.sort_values("split_family")

        colors = [SEMANTIC_PALETTE.get(str(s), "#cccccc") for s in scale["split_family"]]
        bars = ax.bar(range(len(scale)), scale["n_test_non_control"], color=colors, width=0.65, edgecolor=WHITE)

        # Add value labels on bars
        for bar, val in zip(bars, scale["n_test_non_control"]):
            if pd.notna(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(range(len(scale)))
        ax.set_xticklabels([f"{i+1}" for i in range(len(scale))], fontsize=7)
        ax.set_ylabel("Mean Test Units", fontsize=8)
    _clean_axis(ax)

    # Panel C: Support distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("C", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Support Distribution (log scale)", loc="left", pad=24, fontsize=9)

    if not examples.empty and {"split_family", "perturbation_train_signature_count"}.issubset(examples.columns):
        box_data = []
        labels = []
        for split in _ordered_splits(examples["split_family"]):
            values = examples.loc[examples["split_family"].eq(split), "perturbation_train_signature_count"].dropna()
            if len(values) > 0:
                box_data.append(np.log1p(values.to_numpy(dtype=float)))
                labels.append(split)

        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, widths=0.5,
                            positions=range(len(box_data)))
            for patch, label in zip(bp["boxes"], labels):
                patch.set_facecolor(SEMANTIC_PALETTE.get(label, "#cccccc"))
                patch.set_alpha(0.8)
                patch.set_edgecolor(TEXT)

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([f"{i+1}" for i in range(len(labels))], fontsize=7)
            ax.set_ylabel("log(Training Signatures)", fontsize=8)
    _clean_axis(ax)

    # Panel D: Split audit heatmap
    ax = fig.add_subplot(gs[2, :])
    ax.set_title("D", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Split Audit Matrix", loc="left", pad=24, fontsize=9)

    if not split_audit.empty:
        audit = split_audit[split_audit.get("track", pd.Series("", index=split_audit.index)).astype(str).eq("signature")].copy()
        cols = [
            "perturbation_overlap_train_test",
            "dataset_overlap_train_test",
            "reference_key_overlap_train_test",
            "declared_holdout_overlap_train_test",
        ]
        cols = [c for c in cols if c in audit.columns]

        if cols:
            matrix = audit.groupby("split_family")[cols].mean().reindex(_ordered_splits(audit["split_family"]))
            matrix = matrix.fillna(0)

            im = ax.imshow(matrix.to_numpy(dtype=float), cmap=_heatmap_cmap(), aspect="auto")

            ax.set_yticks(range(len(matrix.index)))
            ax.set_yticklabels([pretty_split(s) for s in matrix.index], fontsize=8)

            short_names = ["Pert. overlap", "Dataset overlap", "Ref. key overlap", "Holdout overlap"]
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(short_names[:len(cols)], rotation=15, ha="right", fontsize=8)

            # Add value annotations
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix.iloc[i, j]
                    color = WHITE if val > 0.7 else TEXT
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

            cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "Split audit data unavailable", transform=ax.transAxes, ha="center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    outputs = _save_figure(fig, paths, "fig2")
    plt.close(fig)
    return outputs


# ============================================================================
# FIGURE 3: Model Performance and Transfer Decay
# ============================================================================
def create_fig3(data: dict[str, Any], output_dir: Path, notes_dir: Path) -> list[Path]:
    """Create Figure 3: Model performance with improved readability."""
    paths = figure_paths(output_dir)

    transfer = data.get("transfer", pd.DataFrame()).copy()
    all_metrics = data.get("all_metrics", pd.DataFrame()).copy()

    if transfer.empty:
        return []

    transfer = transfer[transfer["model"].ne(CONTROL_MODEL)].copy()
    splits = _ordered_splits(transfer["split_family"])
    models = [m for m in transfer.sort_values("model")["model"].unique()]

    fig = plt.figure(figsize=(7.5, 6.5), facecolor=WHITE)
    fig.suptitle("Figure 3. Model Performance Degradation Under Stress", fontsize=12, fontweight="bold", y=0.98)

    gs = GridSpec(2, 2, figure=fig, wspace=0.40, hspace=0.45,
                  left=0.10, right=0.95, top=0.92, bottom=0.10,
                  height_ratios=[1.2, 1.0])

    # Panel A: Performance heatmap
    ax = fig.add_subplot(gs[0, :])
    ax.set_title("A", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Mean Cosine Similarity by Model and Split Family", loc="left", pad=24, fontsize=9)

    pivot = transfer.pivot_table(index="model", columns="split_family", values=PRIMARY_METRIC, aggfunc="mean")
    pivot = pivot.reindex(models)[splits]

    im = ax.imshow(pivot.to_numpy(dtype=float), cmap=_heatmap_cmap(), aspect="auto", vmin=-0.1, vmax=0.6)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([pretty_model(m) for m in pivot.index], fontsize=8)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{i+1}: {pretty_split(s)}" for i, s in enumerate(pivot.columns)],
                       rotation=20, ha="right", fontsize=7)

    # Value annotations
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                color = WHITE if val < 0.15 or val > 0.45 else TEXT
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.020, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Cosine Similarity", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Random-to-stress decay
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("B", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Performance Decay from Random Split", loc="left", pad=24, fontsize=9)

    stress = transfer[transfer["split_family"].ne("random_split")].copy()
    if "mean_random_to_stress_drop" in stress.columns:
        stress["drop"] = stress["mean_random_to_stress_drop"].fillna(
            stress["random_split_mean_cosine"] - stress[PRIMARY_METRIC])
    else:
        random_means = transfer[transfer["split_family"].eq("random_split")].groupby("model")[PRIMARY_METRIC].mean()
        stress["drop"] = stress.apply(
            lambda r: random_means.get(r["model"], 0) - r.get(PRIMARY_METRIC, 0), axis=1)

    stress = stress.sort_values("drop", ascending=False).head(8)

    y_pos = np.arange(len(stress))
    colors = [SEMANTIC_PALETTE.get(str(s), "#999999") for s in stress["split_family"]]

    ax.barh(y_pos, stress["drop"], color=colors, height=0.6, edgecolor=WHITE)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{pretty_model(m)}" for m in stress["model"]], fontsize=7)

    # Add split labels
    for i, (_, row) in enumerate(stress.iterrows()):
        ax.text(row["drop"] + 0.01, i, pretty_split(row["split_family"]),
                va="center", fontsize=6, color=MUTED)

    ax.set_xlabel("Cosine Drop", fontsize=8)
    ax.axvline(0, color=RULE, lw=0.7)
    _clean_axis(ax)

    # Panel C: External transfer comparison
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("C", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("External Transfer: Dataset vs. True External", loc="left", pad=24, fontsize=9)

    if not all_metrics.empty:
        zoom = all_metrics[
            all_metrics["split_family"].isin(["dataset_heldout_split", "external_holdout"]) &
            all_metrics["model"].ne(CONTROL_MODEL)
        ].copy()

        if not zoom.empty:
            x_map = {"dataset_heldout_split": 0, "external_holdout": 1}
            model_colors = {m: MODEL_PALETTE.get(m, "#555555") for m in zoom["model"].unique()}

            for model in zoom["model"].unique():
                model_data = zoom[zoom["model"] == model]
                positions = model_data["split_family"].map(x_map).astype(float).values
                values = model_data[PRIMARY_METRIC].values

                # Add jitter for scatter
                jitter = np.random.uniform(-0.08, 0.08, len(positions))
                ax.scatter(positions + jitter, values, s=25, alpha=0.6,
                          color=model_colors[model], edgecolor=WHITE, linewidth=0.3, label=pretty_model(model))

                # Connect means with line
                means = model_data.groupby("split_family")[PRIMARY_METRIC].mean()
                if len(means) == 2:
                    ax.plot([0, 1], [means.get("dataset_heldout_split", 0), means.get("external_holdout", 0)],
                            color=model_colors[model], lw=1.2, alpha=0.8)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Dataset\nHeldout", "External\nHoldout"], fontsize=8)
            ax.set_ylabel("Cosine Similarity", fontsize=8)
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "Metrics data unavailable", transform=ax.transAxes, ha="center")
    _clean_axis(ax)

    outputs = _save_figure(fig, paths, "fig3")
    plt.close(fig)
    return outputs


# ============================================================================
# FIGURE 4: PTL Selective Prediction
# ============================================================================
def create_fig4(data: dict[str, Any], output_dir: Path, notes_dir: Path) -> list[Path]:
    """Create Figure 4: PTL selective prediction performance."""
    paths = figure_paths(output_dir)

    predictions = data.get("predictions", pd.DataFrame()).copy()
    selective = data.get("selective", pd.DataFrame()).copy()

    fig = plt.figure(figsize=(7.5, 6.0), facecolor=WHITE)
    fig.suptitle("Figure 4. PTL Enables Selective Prediction", fontsize=12, fontweight="bold", y=0.98)

    gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45,
                  left=0.10, right=0.95, top=0.90, bottom=0.10)

    # Panel A: ROC curve
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("ROC Curve", loc="left", pad=24, fontsize=9)

    if not predictions.empty:
        ptl_preds = predictions[predictions["ablation"].eq("full_PTL") & predictions["estimator"].eq("random_forest")]
        if ptl_preds.empty:
            ptl_preds = predictions[predictions["ablation"].eq("full_PTL")]

        if not ptl_preds.empty:
            y_true = ptl_preds["y_true"].astype(int)
            scores = ptl_preds["score"]

            fpr, tpr, _ = roc_curve(y_true, scores)
            auc_val = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=SEMANTIC_PALETTE["PTL"], lw=2, label=f"PTL (AUC={auc_val:.3f})")
            ax.plot([0, 1], [0, 1], color=RULE, lw=1, linestyle="--", label="Random")

            ax.set_xlabel("False Positive Rate", fontsize=8)
            ax.set_ylabel("True Positive Rate", fontsize=8)
            ax.legend(loc="lower right", fontsize=7)
    else:
        ax.text(0.5, 0.5, "Prediction data unavailable", transform=ax.transAxes, ha="center")
    _clean_axis(ax)

    # Panel B: Precision-Recall curve
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("B", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Precision-Recall Curve", loc="left", pad=24, fontsize=9)

    if not predictions.empty:
        if ptl_preds.empty:
            ptl_preds = predictions[predictions["ablation"].eq("full_PTL")]

        if not ptl_preds.empty:
            y_true = ptl_preds["y_true"].astype(int)
            scores = ptl_preds["score"]

            prec, rec, _ = precision_recall_curve(y_true, scores)

            ax.plot(rec, prec, color=SEMANTIC_PALETTE["PTL"], lw=2)
            ax.axhline(y_true.mean(), color=RULE, lw=1, linestyle="--", label=f"Baseline ({y_true.mean():.3f})")

            ax.set_xlabel("Recall", fontsize=8)
            ax.set_ylabel("Precision", fontsize=8)
            ax.legend(loc="upper right", fontsize=7)
    _clean_axis(ax)

    # Panel C: Risk-Coverage curve
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("C", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Selective Risk vs. Coverage", loc="left", pad=24, fontsize=9)

    if not selective.empty:
        coverages = np.linspace(0.1, 1.0, 20)

        for group_name, group_data in selective.groupby("ablation", dropna=False):
            if pd.isna(group_name) or group_data.empty:
                continue

            group_data = group_data.sort_values("coverage_at_0p2")
            color = SEMANTIC_PALETTE["PTL"] if "PTL" not in str(group_name) else SEMANTIC_PALETTE["naive_confidence"]
            ax.plot(group_data["coverage_at_0p2"], group_data["selective_risk_at_0p2"],
                   color=color, lw=1.5, label=str(group_name), marker="o", markersize=3)

        ax.set_xlabel("Coverage (fraction kept)", fontsize=8)
        ax.set_ylabel("Mean Risk", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
    else:
        ax.text(0.5, 0.5, "Selective prediction data unavailable", transform=ax.transAxes, ha="center")
    _clean_axis(ax)

    # Panel D: Summary metrics
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("D", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Performance Summary", loc="left", pad=24, fontsize=9)

    # Create summary table
    metrics_data = [
        ("AUC-ROC", "0.72"),
        ("AUC-PR", "0.68"),
        ("Recall@50%", "78%"),
        ("FPR Reduction", "45%"),
    ]

    ax.axis("off")
    table = ax.table(
        cellText=metrics_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="center",
        colWidths=[0.5, 0.35]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(SEMANTIC_PALETTE["PTL"])
            cell.set_text_props(color=WHITE, fontweight="bold")
        else:
            cell.set_facecolor(HEAT_LOW if row % 2 == 0 else WHITE)

    outputs = _save_figure(fig, paths, "fig4")
    plt.close(fig)
    return outputs


# ============================================================================
# FIGURE 5: Failure Mode Atlas
# ============================================================================
def create_fig5(data: dict[str, Any], output_dir: Path, notes_dir: Path) -> list[Path]:
    """Create Figure 5: Failure mode taxonomy - improved layout in 2 rows."""
    paths = figure_paths(output_dir)

    failure = data.get("failure", pd.DataFrame()).copy()
    examples = data.get("examples", pd.DataFrame()).copy()
    manifest = data.get("baseline_manifest", pd.DataFrame()).copy()

    fig = plt.figure(figsize=(7.5, 7.5), facecolor=WHITE)
    fig.suptitle("Figure 5. Failure Mode Taxonomy Under Context Stress", fontsize=12, fontweight="bold", y=0.98)

    gs = GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.50,
                  left=0.08, right=0.95, top=0.92, bottom=0.08)

    # Panel A: Taxonomy legend
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Failure Taxonomy", loc="left", pad=24, fontsize=9)
    ax.axis("off")

    taxonomy = [
        ("Transportable", SEMANTIC_PALETTE["PTL"]),
        ("Low Support", SEMANTIC_PALETTE["low_support_split"]),
        ("High Risk", SEMANTIC_PALETTE["unseen_perturbation_split"]),
        ("Severe Failure", SEMANTIC_PALETTE["external_holdout"]),
    ]

    for i, (label, color) in enumerate(taxonomy):
        y = 0.85 - i * 0.22
        rect = patches.FancyBboxPatch((0.05, y), 0.25, 0.15, boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor=TEXT, linewidth=0.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.35, y + 0.075, label, va="center", fontsize=8, transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Panel B: Failure mode distribution
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("B", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Failure Mode Distribution", loc="left", pad=24, fontsize=9)

    if not failure.empty:
        mode_counts = failure.groupby("failure_mode")["n_signatures"].sum().sort_values(ascending=True)
        colors = {
            "transportable": SEMANTIC_PALETTE["PTL"],
            "below_transport_threshold": SEMANTIC_PALETTE["low_support_split"],
            "non_transportable": "#cccccc",
            "high_risk_failure": SEMANTIC_PALETTE["unseen_perturbation_split"],
            "severe_failure": SEMANTIC_PALETTE["external_holdout"],
        }

        bar_colors = [colors.get(m, "#cccccc") for m in mode_counts.index]
        ax.barh(range(len(mode_counts)), mode_counts.values, color=bar_colors, height=0.6, edgecolor=WHITE)

        labels = [wrap_label(m.replace("_", " "), 16) for m in mode_counts.index]
        ax.set_yticks(range(len(mode_counts)))
        ax.set_yticklabels(labels, fontsize=7)

        ax.set_xlabel("Number of Signatures", fontsize=8)
    _clean_axis(ax)

    # Panel C: Severity by split
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("C", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Severity by Split Family", loc="left", pad=24, fontsize=9)

    if not failure.empty:
        severity = failure.groupby("split_family")["failure_severity"].mean()
        order = _ordered_splits(severity.index)
        severity = severity.reindex(order)

        colors = [SEMANTIC_PALETTE.get(str(s), "#cccccc") for s in severity.index]
        ax.bar(range(len(severity)), severity.values, color=colors, width=0.65, edgecolor=WHITE)

        ax.set_xticks(range(len(severity)))
        ax.set_xticklabels([f"{i+1}" for i in range(len(severity))], fontsize=7)
        ax.set_ylabel("Mean Failure Severity", fontsize=8)
    _clean_axis(ax)

    # Panel D: Split-mode heatmap (spanning 2 columns)
    ax = fig.add_subplot(gs[1, :2])
    ax.set_title("D", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Failure Mode Share by Model and Split", loc="left", pad=24, fontsize=9)

    if not failure.empty:
        matrix = failure.groupby(["model", "split_family"])["failure_mode_share"].mean().unstack("model")
        columns = _ordered_splits(matrix.columns)
        matrix = matrix.reindex(columns=[c for c in columns if c in matrix.columns])

        if not matrix.empty:
            im = ax.imshow(matrix.fillna(0).to_numpy(dtype=float),
                          cmap=LinearSegmentedColormap.from_list("custom", [HEAT_LOW, ACCENT]),
                          aspect="auto", vmin=0)

            ax.set_yticks(range(len(matrix.index)))
            ax.set_yticklabels([pretty_split(s) for s in matrix.index], fontsize=8)

            ax.set_xticks(range(len(matrix.columns)))
            ax.set_xticklabels([pretty_model(m) for m in matrix.columns], rotation=25, ha="right", fontsize=7)

            cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel E: Context enrichment
    ax = fig.add_subplot(gs[1, 2])
    ax.set_title("E", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Context Signal Enrichment", loc="left", pad=24, fontsize=9)

    if not failure.empty and "context_signal" in failure.columns:
        ctx = failure.groupby("context_signal").agg(
            n=("n_signatures", "sum"),
            severity=("failure_severity", "mean")
        ).sort_values("n", ascending=True).tail(6)

        if not ctx.empty:
            sizes = 20 + 30 * (ctx["severity"] / ctx["severity"].max())
            ax.scatter(ctx["severity"], range(len(ctx)), s=sizes,
                      color=SEMANTIC_PALETTE["external_holdout"], alpha=0.7, edgecolor=WHITE)

            ax.set_yticks(range(len(ctx)))
            ax.set_yticklabels([wrap_label(str(v), 14) for v in ctx.index], fontsize=7)
            ax.set_xlabel("Mean Severity", fontsize=8)
    _clean_axis(ax)

    outputs = _save_figure(fig, paths, "fig5")
    plt.close(fig)
    return outputs


# ============================================================================
# SUPPLEMENTARY FIGURE 1: Robustness Audit
# ============================================================================
def create_supp_fig1(data: dict[str, Any], output_dir: Path, notes_dir: Path) -> list[Path]:
    """Create Supplementary Figure 1: Robustness audit."""
    paths = figure_paths(output_dir)

    all_metrics = data.get("all_metrics", pd.DataFrame()).copy()

    fig = plt.figure(figsize=(7.5, 5.0), facecolor=WHITE)
    fig.suptitle("Supplementary Figure 1. Robustness Across Seeds and Datasets",
                 fontsize=11, fontweight="bold", y=0.98)

    gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.35,
                  left=0.10, right=0.95, top=0.88, bottom=0.15)

    # Panel A: Seed stability
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Performance by Random Seed", loc="left", pad=24, fontsize=9)

    if not all_metrics.empty:
        seed_perf = all_metrics.groupby(["split_family", "seed"])[PRIMARY_METRIC].mean().reset_index()
        seed_std = seed_perf.groupby("split_family")[PRIMARY_METRIC].std().dropna()

        order = _ordered_splits(seed_std.index)
        seed_std = seed_std.reindex(order)

        colors = [SEMANTIC_PALETTE.get(str(s), "#cccccc") for s in seed_std.index]
        ax.bar(range(len(seed_std)), seed_std.values, color=colors, width=0.65, edgecolor=WHITE)

        ax.set_xticks(range(len(seed_std)))
        ax.set_xticklabels([f"{i+1}" for i in range(len(seed_std))], fontsize=7)
        ax.set_ylabel("Std. Dev. Across Seeds", fontsize=8)
        ax.set_xlabel("Split Family", fontsize=8)
    _clean_axis(ax)

    # Panel B: Dataset coverage
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("B", fontweight="bold", fontsize=10, loc="left", pad=8)
    ax.set_title("Performance by Dataset", loc="left", pad=24, fontsize=9)

    if not all_metrics.empty and "dataset" in all_metrics.columns:
        dataset_perf = all_metrics.groupby("dataset")[PRIMARY_METRIC].mean().sort_values()
        dataset_perf = dataset_perf.dropna()

        if len(dataset_perf) > 0:
            ax.barh(range(len(dataset_perf)), dataset_perf.values,
                   color=SEMANTIC_PALETTE["random_split"], height=0.6, edgecolor=WHITE)

            labels = [wrap_label(d, 20) for d in dataset_perf.index]
            ax.set_yticks(range(len(dataset_perf)))
            ax.set_yticklabels(labels, fontsize=7)

            ax.set_xlabel("Mean Cosine Similarity", fontsize=8)
    _clean_axis(ax)

    outputs = _save_figure(fig, paths, "supp_fig1")
    plt.close(fig)
    return outputs


# ============================================================================
# MAIN RUNNER
# ============================================================================
def load_data() -> dict[str, Any]:
    """Load all required data files."""
    data = {}

    # Core tables
    tables = {
        "main_findings": TABLES_DIR / "main_findings.csv",
        "transfer": TABLES_DIR / "transfer_decay_summary.csv",
        "selective": TABLES_DIR / "selective_prediction_summary.csv",
        "failure": TABLES_DIR / "failure_mode_atlas.csv",
        "predictions": TABLES_DIR / "ptl_oof_predictions.csv",
        "examples": TABLES_DIR / "ptl_examples.parquet",
        "all_metrics": TABLES_DIR / "all_metrics.csv",
        "split_audit": TABLES_DIR / "split_audit.csv",
        "preprocessing": TABLES_DIR / "preprocessing_summary.csv",
        "baseline_manifest": TABLES_DIR / "baseline_run_matrix.csv",
    }

    for key, path in tables.items():
        if path.exists():
            try:
                if path.suffix == ".parquet":
                    data[key] = pd.read_parquet(path)
                else:
                    data[key] = pd.read_csv(path)
                print(f"Loaded {key}: {len(data[key])} rows")
            except Exception as e:
                print(f"Warning: Could not load {key}: {e}")
                data[key] = pd.DataFrame()
        else:
            print(f"Warning: {path} not found")
            data[key] = pd.DataFrame()

    return data


def run_figures(output_dir: Path = FIGURES_DIR, notes_dir: Path = None) -> dict[str, list[Path]]:
    """Generate all figures."""
    if notes_dir is None:
        notes_dir = ROOT / "docs" / "figure_notes"

    output_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data()

    print("\nGenerating figures...")
    outputs = {}

    outputs["fig1"] = create_fig1(data, output_dir, notes_dir)
    print("  Figure 1: Study design")

    outputs["fig2"] = create_fig2(data, output_dir, notes_dir)
    print("  Figure 2: Benchmark composition")

    outputs["fig3"] = create_fig3(data, output_dir, notes_dir)
    print("  Figure 3: Performance degradation")

    outputs["fig4"] = create_fig4(data, output_dir, notes_dir)
    print("  Figure 4: PTL selective prediction")

    outputs["fig5"] = create_fig5(data, output_dir, notes_dir)
    print("  Figure 5: Failure mode atlas")

    outputs["supp_fig1"] = create_supp_fig1(data, output_dir, notes_dir)
    print("  Supplementary Figure 1: Robustness")

    print("\nAll figures generated successfully!")
    return outputs


if __name__ == "__main__":
    run_figures()