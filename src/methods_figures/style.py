from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import patches

WHITE = "#FFFFFF"
BLACK = "#111111"
TEXT = "#111111"
MUTED = "#555555"
RULE = "#8C8C8C"
GRID = "#D8D8D8"
BLUE = "#1F5AA6"
LIGHT_BLUE = "#DCE9F7"
ORANGE = "#E64B16"
PURPLE = "#6A3D9A"
GREEN = "#1B7837"
YELLOW = "#F2C94C"
TEAL = "#007C82"
RED = "#C81D25"
GRAY = "#777777"
LIGHT_GRAY = "#F4F4F4"

SPLIT_ORDER = [
    "random_split",
    "unseen_perturbation_split",
    "unseen_combination_split",
    "low_support_split",
    "dataset_heldout_split",
    "external_holdout",
]

SPLIT_LABELS = {
    "random_split": "Random\nanchor",
    "unseen_perturbation_split": "Held-out\nperturbation",
    "unseen_combination_split": "Held-out\ncombination",
    "low_support_split": "Low\nsupport",
    "dataset_heldout_split": "Dataset\nheldout",
    "external_holdout": "External\nholdout",
}

MODEL_LABELS = {
    "ridge_regression_baseline": "Ridge",
    "global_delta_baseline": "Global delta",
    "perturbation_mean_delta_baseline": "Perturbation mean",
    "cell_context_knn_delta_baseline": "Context kNN",
    "control_mean_baseline": "Control mean",
}

MODEL_ORDER = [
    "cell_context_knn_delta_baseline",
    "global_delta_baseline",
    "perturbation_mean_delta_baseline",
    "ridge_regression_baseline",
    "control_mean_baseline",
]

MODEL_COLORS = {
    "ridge_regression_baseline": BLUE,
    "global_delta_baseline": PURPLE,
    "perturbation_mean_delta_baseline": ORANGE,
    "cell_context_knn_delta_baseline": GREEN,
    "control_mean_baseline": GRAY,
}

SPLIT_COLORS = {
    "random_split": BLUE,
    "unseen_perturbation_split": ORANGE,
    "unseen_combination_split": PURPLE,
    "low_support_split": GREEN,
    "dataset_heldout_split": TEAL,
    "external_holdout": RED,
}


def apply_methods_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.6,
            "axes.edgecolor": RULE,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.25,
            "patch.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.02, 0.96, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")


def framed_panel(ax: plt.Axes, label: str, title: str | None = None) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color(RULE)
    ax.tick_params(length=3, width=0.7)
    panel_label(ax, label)
    if title:
        ax.set_title(title, loc="left", pad=8, x=0.075, fontsize=12, fontweight="bold")


def despine_data(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.6, alpha=0.75)


def message_strip(fig: plt.Figure, text: str, y: float = 0.015, height: float = 0.045) -> None:
    rect = patches.FancyBboxPatch(
        (0.02, y),
        0.96,
        height,
        boxstyle="round,pad=0.004,rounding_size=0.004",
        transform=fig.transFigure,
        facecolor=WHITE,
        edgecolor=RULE,
        linewidth=0.8,
    )
    fig.add_artist(rect)
    fig.text(0.5, y + height / 2, text, ha="center", va="center", fontsize=11, style="italic")


def save_all(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight", "facecolor": WHITE}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def pretty_split(value: str) -> str:
    return SPLIT_LABELS.get(str(value), str(value).replace("_", " ").title())


def pretty_model(value: str) -> str:
    return MODEL_LABELS.get(str(value), str(value).replace("_baseline", "").replace("_", " ").title())


def ordered_splits(values: Iterable[str]) -> list[str]:
    present = set(map(str, values))
    return [split for split in SPLIT_ORDER if split in present] + sorted(present - set(SPLIT_ORDER))


def draw_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], title: str, subtitle: str, color: str) -> None:
    rect = patches.FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.012,rounding_size=0.014",
        facecolor=color + "18" if color.startswith("#") else WHITE,
        edgecolor=color,
        linewidth=1.0,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] * 0.63, title, ha="center", va="center", fontsize=9, fontweight="bold", color=color)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] * 0.30, subtitle, ha="center", va="center", fontsize=7.4, color=TEXT)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=0.9, shrinkA=2, shrinkB=2))
