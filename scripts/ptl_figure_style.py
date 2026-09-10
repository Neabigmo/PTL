"""Shared visual grammar for the reliability-transportability figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


METRIC_LABELS = {
    "delta_cosine": "Delta cosine",
    "systema_centroid_accuracy": "Systema centroid",
    "absolute_effect_rank_agreement": "Absolute-effect rank",
}
METRIC_COLORS = {
    "delta_cosine": "#0072B2",
    "systema_centroid_accuracy": "#D55E00",
    "absolute_effect_rank_agreement": "#009E73",
}
CONTEXT_LABELS = {
    "frangieh_melanoma_control": "Ctrl",
    "frangieh_melanoma_coculture": "Co-culture",
    "frangieh_melanoma_ifng": "IFNγ",
    "nadig_hepg2": "HepG2",
    "nadig_jurkat": "Jurkat",
}
CONTEXT_ORDER = (
    "frangieh_melanoma_control",
    "frangieh_melanoma_coculture",
    "frangieh_melanoma_ifng",
)
CONTRAST_ORDER = (
    ("frangieh_melanoma_control", "frangieh_melanoma_coculture"),
    ("frangieh_melanoma_control", "frangieh_melanoma_ifng"),
    ("frangieh_melanoma_coculture", "frangieh_melanoma_ifng"),
)
DEPTH_ORDER = ("10", "20", "40", "80", "160", "full")
OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def apply_style() -> None:
    """Apply one compact, colorblind-safe style at the final figure size."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(str(metric), str(metric))


def metric_color(metric: str) -> str:
    return METRIC_COLORS.get(str(metric), "#333333")


def context_label(context: str) -> str:
    return CONTEXT_LABELS.get(str(context), str(context))


def contrast_label(left: str, right: str) -> str:
    return f"{context_label(left)} ↔ {context_label(right)}"


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="right")


def clean_axes(ax: plt.Axes, *, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#D9DDE2", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=600)
    outputs.append(png_path.name)
    for suffix in ("pdf", "svg"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path.name)
    # Render TIFF from the already rasterized PNG.  Re-rendering a large
    # constrained-layout figure directly through Matplotlib's TIFF backend can
    # multiply peak memory without improving scientific content.
    tiff_path = out_dir / f"{stem}.tiff"
    with Image.open(png_path) as image:
        image.convert("RGB").save(tiff_path, dpi=(600, 600), compression="tiff_lzw")
    outputs.append(tiff_path.name)
    plt.close(fig)
    return outputs


def add_metric_legend(ax: plt.Axes, metrics: Iterable[str], **kwargs) -> None:
    handles = [mpl.lines.Line2D([], [], color=metric_color(metric), marker="o", lw=1.7, label=metric_label(metric)) for metric in metrics]
    ax.legend(handles=handles, frameon=False, **kwargs)
