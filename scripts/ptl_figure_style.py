"""Shared visual grammar for the reliability-transportability figures."""

from __future__ import annotations

import json
import re
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
    # One fixed, deliberately muted metric grammar used across Fig. 1--6.
    "delta_cosine": "#4C78A8",
    "systema_centroid_accuracy": "#D9825B",
    "absolute_effect_rank_agreement": "#3A9D8F",
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
OKABE_ITO = ("#4C78A8", "#D9825B", "#3A9D8F", "#9A7AA0", "#B29A63", "#7D9AAA")


def apply_style() -> None:
    """Apply one compact, colorblind-safe style at the final figure size."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8.2,
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
            "text.color": "#263238",
            "axes.labelcolor": "#263238",
            "xtick.color": "#4B5560",
            "ytick.color": "#4B5560",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def paper_textwidth_in() -> float:
    """Read the active LaTeX text width, with a conservative fallback."""
    roots = [Path(__file__).resolve().parents[1] / "third_party/iclr2027/iclr2027_conference.sty", Path(__file__).resolve().parents[1] / "paper/iclr2027/main.tex"]
    for path in roots:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"\\textwidth\s+([0-9.]+)\s+true\s+in", text)
        if match:
            return float(match.group(1))
    return 5.5


def figure_size(name: str) -> tuple[float, float]:
    """Return final manuscript-size dimensions for the six main figures."""
    # Keep the final canvases compact while reserving enough height for
    # near-square quantitative panels rather than narrow vertical strips.
    ratios = {"fig1": .72, "fig2": .47, "fig3": .50, "fig4": .68, "fig5": .70, "fig6": 1.13}
    width = paper_textwidth_in()
    return width, width * ratios.get(name, .64)


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(str(metric), str(metric))


def metric_color(metric: str) -> str:
    return METRIC_COLORS.get(str(metric), "#333333")


def context_label(context: str) -> str:
    return CONTEXT_LABELS.get(str(context), str(context))


def contrast_label(left: str, right: str) -> str:
    return f"{context_label(left)} ↔ {context_label(right)}"


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.025, y: float = 1.05, ha: str = "right") -> None:
    # Keep the panel letter in the declared canvas gutter. The old -0.13
    # placement relied on a tight export crop and was clipped on fixed-canvas
    # PDF/PNG outputs, especially for full-width and narrow lower panels.
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha=ha)


def clean_axes(ax: plt.Axes, *, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#D9DEE5", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)


def audit_text_bboxes(fig: plt.Figure) -> dict:
    """Audit rendered text, legends, and panel gutters at final figure size."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    collisions: list[dict[str, object]] = []
    text_count = 0
    all_items: list[tuple[int, str, object]] = []
    for axis_index, ax in enumerate(fig.axes):
        items: list[tuple[str, object]] = []
        for text in ax.texts:
            if text.get_visible() and text.get_text().strip():
                text_count += 1
                items.append((f"text:{text.get_text()[:42]}", text.get_window_extent(renderer)))
        title = ax.title
        if title.get_visible() and title.get_text().strip():
            items.append((f"title:{title.get_text()[:42]}", title.get_window_extent(renderer)))
        for label_name, label in (("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label)):
            if label.get_visible() and label.get_text().strip():
                items.append((f"{label_name}:{label.get_text()[:42]}", label.get_window_extent(renderer)))
        for tick_name, tick_labels in (("xtick", ax.get_xticklabels()), ("ytick", ax.get_yticklabels())):
            for tick_index, tick in enumerate(tick_labels):
                if tick.get_visible() and tick.get_text().strip():
                    items.append((f"{tick_name}[{tick_index}]:{tick.get_text()[:24]}", tick.get_window_extent(renderer)))
        legend = ax.get_legend()
        if legend is not None and legend.get_visible():
            items.append(("legend", legend.get_window_extent(renderer)))
        all_items.extend((axis_index, name, box) for name, box in items)
        for i, (name_a, box_a) in enumerate(items):
            for name_b, box_b in items[i + 1:]:
                if not box_a.overlaps(box_b):
                    continue
                left = max(box_a.x0, box_b.x0); right = min(box_a.x1, box_b.x1)
                bottom = max(box_a.y0, box_b.y0); top = min(box_a.y1, box_b.y1)
                intersection = max(0.0, right - left) * max(0.0, top - bottom)
                smaller = max(min(box_a.width * box_a.height, box_b.width * box_b.height), 1.0)
                if intersection / smaller >= .05:
                    collisions.append({"axis": axis_index, "a": name_a, "b": name_b, "overlap_fraction": round(intersection / smaller, 4)})
    # Check text that spills across a neighboring axes. This catches labels
    # that are technically non-overlapping inside their own axes but occupy a
    # shared gutter at the final manuscript scale.
    for i, (axis_a, name_a, box_a) in enumerate(all_items):
        for axis_b, name_b, box_b in all_items[i + 1:]:
            if axis_a == axis_b or not box_a.overlaps(box_b):
                continue
            left = max(box_a.x0, box_b.x0); right = min(box_a.x1, box_b.x1)
            bottom = max(box_a.y0, box_b.y0); top = min(box_a.y1, box_b.y1)
            intersection = max(0.0, right - left) * max(0.0, top - bottom)
            smaller = max(min(box_a.width * box_a.height, box_b.width * box_b.height), 1.0)
            if intersection / smaller >= .05:
                collisions.append({"axes": [axis_a, axis_b], "a": name_a, "b": name_b, "overlap_fraction": round(intersection / smaller, 4)})

    # Report the smallest geometric axes gap in points; this is separate from
    # text collisions so a tight but valid plot is diagnosable rather than
    # silently passing as if no layout constraint existed.
    gaps_pt: list[float] = []
    for i, first in enumerate(fig.axes):
        for second in fig.axes[i + 1:]:
            a, b = first.bbox, second.bbox
            x_overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
            y_overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
            if x_overlap > 0:
                gap_px = max(0.0, min(abs(a.y0 - b.y1), abs(b.y0 - a.y1)))
            elif y_overlap > 0:
                gap_px = max(0.0, min(abs(a.x0 - b.x1), abs(b.x0 - a.x1)))
            else:
                continue
            gaps_pt.append(gap_px * 72.0 / fig.dpi)
    min_gutter = min(gaps_pt) if gaps_pt else None
    return {
        "axes": len(fig.axes),
        "custom_text_items": text_count,
        "collisions": collisions,
        "collision_count": len(collisions),
        "min_axes_gutter_pt": None if min_gutter is None else round(min_gutter, 2),
        "gutter_warning": bool(min_gutter is not None and min_gutter < 4.0),
    }


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    audit = audit_text_bboxes(fig)
    (out_dir / f"{stem}.layout_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if audit["collision_count"]:
        print(f"layout_audit {stem}: {audit['collision_count']} collision(s)")
    else:
        print(f"layout_audit {stem}: PASS")
    png_path = out_dir / f"{stem}.png"
    # Export the declared canvas, not a content-tight crop. A tight crop
    # changes the PDF MediaBox whenever a label extends beyond the axes,
    # silently changing the manuscript-scale figure size.
    fig.savefig(png_path, bbox_inches=None, pad_inches=0, facecolor="white", dpi=600)
    outputs.append(png_path.name)
    for suffix in ("pdf", "svg"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches=None, pad_inches=0, facecolor="white")
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
