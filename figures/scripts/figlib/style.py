from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib import patches

from .palettes import (
    BLACK,
    GRID,
    MODEL_LABELS,
    RULE,
    SPLIT_LABELS,
    SPLIT_ORDER,
    SPLIT_SHORT,
    TEXT,
    WHITE,
)

BASE_FONT = 8.2
TITLE_FONT = 9.2
PANEL_LABEL_FONT = 11.0
LINEWIDTH = 0.8
AXIS_LINEWIDTH = 0.65


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": BASE_FONT,
            "axes.labelsize": BASE_FONT,
            "axes.titlesize": TITLE_FONT,
            "xtick.labelsize": BASE_FONT - 0.6,
            "ytick.labelsize": BASE_FONT - 0.4,
            "legend.fontsize": BASE_FONT - 0.5,
            "axes.edgecolor": RULE,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.linewidth": AXIS_LINEWIDTH,
            "xtick.major.width": AXIS_LINEWIDTH,
            "ytick.major.width": AXIS_LINEWIDTH,
            "lines.linewidth": 1.15,
            "patch.linewidth": LINEWIDTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = 0.0, y: float = 1.055) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONT,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=BLACK,
        clip_on=False,
    )


def panel_title(ax: plt.Axes, label: str, title: str) -> None:
    panel_label(ax, label)
    ax.set_title(title, loc="left", x=0.075, pad=7, fontsize=TITLE_FONT, fontweight="bold")


def framed_panel(ax: plt.Axes, label: str, title: str | None = None) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_LINEWIDTH)
        spine.set_color(RULE)
    ax.tick_params(length=2.5, width=AXIS_LINEWIDTH, pad=2)
    if title is not None:
        panel_title(ax, label, title)
    else:
        panel_label(ax, label)


def despine_data(ax: plt.Axes, axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=axis, color=GRID, linestyle="-", linewidth=0.45, alpha=0.65)


def message_strip(fig: plt.Figure, text: str, y: float = 0.014, height: float = 0.044) -> None:
    rect = patches.FancyBboxPatch(
        (0.018, y),
        0.964,
        height,
        boxstyle="round,pad=0.004,rounding_size=0.004",
        transform=fig.transFigure,
        facecolor=WHITE,
        edgecolor=RULE,
        linewidth=AXIS_LINEWIDTH,
    )
    fig.add_artist(rect)
    fig.text(0.5, y + height / 2, text, ha="center", va="center", fontsize=BASE_FONT + 0.7, style="italic")


def pretty_split(value: str, short: bool = False) -> str:
    mapping = SPLIT_SHORT if short else SPLIT_LABELS
    return mapping.get(str(value), str(value).replace("_", " ").title())


def pretty_model(value: str) -> str:
    return MODEL_LABELS.get(str(value), str(value).replace("_baseline", "").replace("_", " ").title())


def ordered_splits(values: Iterable[str]) -> list[str]:
    present = set(map(str, values))
    return [s for s in SPLIT_ORDER if s in present] + sorted(present - set(SPLIT_ORDER))


def lettered_panel_figure(width: float, height: float) -> plt.Figure:
    apply_style()
    return plt.figure(figsize=(width, height), facecolor=WHITE)
