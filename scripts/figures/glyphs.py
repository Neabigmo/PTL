"""Small semantic glyphs for states, rank pairs, atlas cells, and firewalls."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge, FancyArrowPatch
import numpy as np

from scripts.figures.common import STATE_COLORS, num


def rank_glyph(ax: plt.Axes, source_rank: float, target_rank: float, source_label: str, target_label: str, burden: float, rank_delta: float, title: str) -> None:
    ax.plot([0, 1], [source_rank, target_rank], color="#263238", lw=1.4, zorder=2)
    ax.scatter([0, 1], [source_rank, target_rank], s=58, c=["#E7EAEC", "#6B7280"], edgecolors="white", linewidth=.6, zorder=3)
    ax.text(0, source_rank, f"  {source_label}\n  rank {source_rank:.0f}", va="center", fontsize=6.0)
    ax.text(1, target_rank, f"{target_label}  \nrank {target_rank:.0f}", va="center", ha="right", fontsize=6.0)
    ax.annotate("", xy=(.83, target_rank), xytext=(.17, source_rank), arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "#9AA4AC", "connectionstyle": "arc3,rad=.2"})
    ax.set_xlim(-.35, 1.35); ax.set_ylim(max(source_rank, target_rank) + 5, max(0, min(source_rank, target_rank) - 5)); ax.set_xticks([0, 1], ["Ctrl", "IFNγ"]); ax.set_ylabel("rank (1 = best)"); ax.set_title(title, loc="left", pad=8, fontweight="bold")
    ax.text(.03, .03, f"Δrank={rank_delta:+.0f}   burden={burden:.3f}", transform=ax.transAxes, fontsize=6.2, color="#6B7280", fontweight="bold")


def state_key(value: Any) -> str:
    return str(value) if str(value) in STATE_COLORS else "unstable"


def state_dot(ax: plt.Axes, x: float, y: float, value: Any, size: float = 80) -> None:
    key = state_key(value)
    ax.scatter(x, y, s=size, color=STATE_COLORS[key], edgecolor="#303840", linewidth=.45, zorder=4)


def atlas_cell(ax: plt.Axes, x: float, y: float, value: float, lower: float, joint_lower: float, vmax: float) -> None:
    # Legacy helper retained for source-table-compatible callers.  The atlas
    # builder now uses the shape-only glyph below, not a heatmap rectangle.
    color = "#56616B" if value >= 0 else "#B8C0C7"
    marker = "^" if value >= 0 else "v"
    ax.scatter(x, y, s=25 + 90 * min(abs(value) / max(vmax, 1e-9), 1), marker=marker, facecolor=color, edgecolor="#303840", linewidth=.45)
    if num(lower) > 0:
        ax.scatter(x, y, s=10, marker="o", facecolor="white", edgecolor="#303840", linewidth=.6)
    if num(joint_lower) > 0:
        ax.scatter(x, y, s=60, marker="o", facecolors="none", edgecolors="#303840", linewidth=.7)


def nested_decomposition(ax: plt.Axes, values: list[float], labels: list[str], metric: str, metric_color: str) -> None:
    """Nested uncertainty partition: observed → floor → identifiable remainder."""
    values = [max(0.0, float(v)) if np.isfinite(float(v)) else 0.0 for v in values]
    total = max(sum(values), 1e-9)
    radii = (1.0, .70, .40)
    fills = ("#E8EBED", "#B8BEC8", metric_color)
    for radius, value, fill in zip(radii, values, fills):
        ax.add_patch(Circle((0, 0), radius, facecolor=fill, edgecolor="#303840", linewidth=.65, alpha=.95))
        ax.add_patch(Wedge((0, 0), radius, 90, 90 + 360 * value / total, facecolor="white", edgecolor="none", alpha=.55))
    ax.text(0, 0, "ID", ha="center", va="center", fontsize=6.0, fontweight="bold", color="#303840")
    ax.text(0, -1.20, metric, ha="center", va="top", fontsize=5.7, color=metric_color, fontweight="bold")
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.35, 1.15); ax.set_aspect("equal"); ax.axis("off")


def source_fingerprint(ax: plt.Axes, values: list[float], labels: list[str]) -> None:
    """Compact source-side feature fingerprint; no unsupported gene values."""
    vals = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    norm = np.zeros_like(vals) if hi <= lo else (vals - lo) / (hi - lo)
    for i, (value, label) in enumerate(zip(norm, labels)):
        x, y = i % 3, 1 - i // 3
        ax.add_patch(Rectangle((x - .37, y - .30), .74, .60, facecolor="#E2E6E8", edgecolor="#9AA4AC", lw=.45))
        ax.add_patch(Circle((x, y), .10 + .10 * float(value), facecolor="#6B7280", edgecolor="white", lw=.35, alpha=.9))
        ax.text(x, y - .42, label, ha="center", va="top", fontsize=4.6, color="#6B7280")
    ax.set_xlim(-.65, 2.65); ax.set_ylim(-.85, 1.55); ax.axis("off")


def firewall_glyphs(ax: plt.Axes, allowed: list[str], forbidden: list[str]) -> None:
    """Information firewall with compact object glyphs instead of prose lists."""
    ax.axis("off")
    ax.plot([.5, .5], [.12, .78], color="#C96A5A", lw=3.0, solid_capstyle="round", zorder=1)
    ax.add_patch(Circle((.5, .50), .055, facecolor="white", edgecolor="#C96A5A", lw=1.4, zorder=3))
    for side, names, x, face, edge, title in ((-1, allowed, .20, "#EEF3F6", "#6B7280", "source observables"), (1, forbidden, .80, "#F7E7E7", "#C96A5A", "target-only")):
        ax.text(x, .79, title, ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=edge)
        positions = ((x - .07, .64), (x + .07, .64), (x - .07, .42), (x + .07, .42), (x, .20))
        markers = ("o", "s", "P", "D", "^")
        for i, name in enumerate(names[:5]):
            xx, yy = positions[i]
            ax.scatter(xx, yy, s=175, marker=markers[i], facecolor=face, edgecolor=edge, linewidth=.8, zorder=2)
            ax.text(xx, yy, name, ha="center", va="center", fontsize=5.0, color="#263238", fontweight="bold", zorder=4)
    ax.text(.5, .03, "legal before target outcome", ha="center", va="bottom", fontsize=5.5, color="#6B7280")


def firewall_box(ax: plt.Axes, features: list[str], forbidden: list[str]) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((.02, .08), .43, .82, boxstyle="round,pad=.02", facecolor="#EAF2F8", edgecolor="#7A8996"))
    ax.add_patch(FancyBboxPatch((.56, .08), .42, .82, boxstyle="round,pad=.02", facecolor="#FBE7E7", edgecolor="#C44E52"))
    ax.text(.235, .82, "SOURCE-OBSERVABLE", ha="center", fontsize=8, fontweight="bold", color="#004B75")
    ax.text(.77, .82, "TARGET-INFORMED", ha="center", fontsize=8, fontweight="bold", color="#9E2A2B")
    ax.text(.07, .70, "Allowed before experiment", fontsize=7, fontweight="bold")
    ax.text(.07, .64, "\n".join("✓ " + item for item in features), va="top", fontsize=6.0)
    ax.text(.60, .70, "Forbidden at prediction", fontsize=7, fontweight="bold", color="#9E2A2B")
    ax.text(.60, .64, "\n".join("× " + item for item in forbidden), va="top", fontsize=6.0, color="#9E2A2B")
    ax.annotate("", xy=(.54, .48), xytext=(.47, .48), arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": "#C44E52"})
    ax.text(.50, .015, "firewall: no target data, labels, or outcomes at prediction time", ha="center", fontsize=6.2, color="#46515C")
