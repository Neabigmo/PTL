from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from .palettes import BLACK, BLUE, GRAY, GREEN, HEATMAP_FAIL, HEATMAP_POS, LIGHT_GRAY, ORANGE, RED, RULE, TEAL, WHITE


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    edge: str,
    face: str | None = None,
    radius: float = 0.012,
    lw: float = 0.9,
) -> patches.FancyBboxPatch:
    patch = patches.FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=face or WHITE,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = BLACK, lw: float = 1.0) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2))


def icon_expression(ax: plt.Axes, cx: float, cy: float, scale: float = 1.0) -> None:
    rng = np.random.default_rng(11)
    pts = rng.normal(size=(34, 2))
    pts = pts / np.abs(pts).max() * 0.032 * scale
    ax.scatter(cx + pts[:, 0], cy + pts[:, 1], s=8 * scale, color=BLUE, edgecolor=WHITE, linewidth=0.25, zorder=3)


def icon_metadata(ax: plt.Axes, cx: float, cy: float, scale: float = 1.0) -> None:
    for dx, dy in [(-0.018, 0), (0.0, 0.012), (0.018, 0)]:
        circ = patches.Circle((cx + dx * scale, cy + dy * scale), 0.010 * scale, facecolor=GRAY, edgecolor=BLACK, lw=0.35)
        ax.add_patch(circ)
        ax.add_patch(patches.Rectangle((cx + dx * scale - 0.014 * scale, cy - 0.024 * scale), 0.028 * scale, 0.018 * scale, facecolor=LIGHT_GRAY, edgecolor=BLACK, lw=0.35))


def icon_heatmap(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    vals = np.array([[0.2, 0.4, 0.7], [0.8, 0.5, 0.1], [0.3, 0.9, 0.6]])
    for i in range(3):
        for j in range(3):
            color = [LIGHT_GRAY, BLUE, HEATMAP_POS, HEATMAP_FAIL][int(vals[i, j] * 3.5)]
            ax.add_patch(patches.Rectangle((x + j * w / 3, y + (2 - i) * h / 3), w / 3, h / 3, facecolor=color, edgecolor=RULE, lw=0.25))


def icon_report(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=WHITE, edgecolor=BLACK, lw=0.5))
    ax.plot([x + 0.012, x + w - 0.012], [y + h * 0.72, y + h * 0.72], color=GRAY, lw=0.6)
    ax.plot([x + 0.012, x + w - 0.012], [y + h * 0.52, y + h * 0.52], color=GRAY, lw=0.6)
    ax.bar([x + w * 0.22, x + w * 0.45, x + w * 0.68], [h * 0.22, h * 0.35, h * 0.14], width=w * 0.12, bottom=y + h * 0.08, color=[BLUE, TEAL, ORANGE])


def mini_decisions(ax: plt.Axes, x: float, y: float, n_col: int = 5, n_row: int = 3) -> None:
    colors = [TEAL, TEAL, RED, TEAL, ORANGE]
    for r in range(n_row):
        for c in range(n_col):
            col = colors[(r + c) % len(colors)]
            ax.scatter(x + c * 0.020, y - r * 0.023, s=28, facecolor=col, edgecolor=BLACK, linewidth=0.35)

def reliability_tree(ax: plt.Axes, x: float, y: float, scale: float = 1.0) -> None:
    nodes = [(x, y), (x - 0.04 * scale, y - 0.05 * scale), (x + 0.04 * scale, y - 0.05 * scale), (x - 0.07 * scale, y - 0.10 * scale), (x - 0.01 * scale, y - 0.10 * scale), (x + 0.035 * scale, y - 0.10 * scale), (x + 0.075 * scale, y - 0.10 * scale)]
    for a, b in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=BLACK, lw=0.6)
    for i, (nx, ny) in enumerate(nodes):
        ax.scatter(nx, ny, s=70 * scale, facecolor=TEAL if i % 3 != 0 else WHITE, edgecolor=GREEN if i != 5 else RED, linewidth=0.7, zorder=3)
