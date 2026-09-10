"""Reusable ribbon and braid primitives used by the figure builders."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np


def _bezier(ax: plt.Axes, points: Sequence[tuple[float, float]], color: str, lw: float = 1.2, alpha: float = .7, zorder: int = 2) -> None:
    ax.plot([p[0] for p in points], [p[1] for p in points], color=color, lw=lw, alpha=alpha, zorder=zorder, solid_capstyle="round")


def ribbon(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, width: float, color: str, alpha: float = .35, zorder: int = 1) -> None:
    """Draw a filled cubic ribbon between two rank/state positions."""
    dx = (x1 - x0) * .46
    verts = [
        (x0, y0 - width / 2), (x0 + dx, y0 - width / 2), (x1 - dx, y1 - width / 2), (x1, y1 - width / 2),
        (x1, y1 + width / 2), (x1 - dx, y1 + width / 2), (x0 + dx, y0 + width / 2), (x0, y0 + width / 2),
        (x0, y0 - width / 2),
    ]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder))


def rank_braid(ax: plt.Axes, flow, labels: Sequence[str], context_labels: Sequence[str], palette: Sequence[str] | None = None) -> None:
    """Continuous source→context→target rank braid with selected real labels."""
    palette = list(palette or ["#0072B2", "#7A8996", "#009E73", "#C44E52"])
    xs = np.arange(3, dtype=float)
    ymax = max(float(flow[["source_rank", "coculture_rank", "ifng_rank"]].max().max()), 10)
    for index, (_, row) in enumerate(flow.iterrows()):
        label = str(row["perturbation_label"])
        hi = label in {str(value) for value in labels}
        color = palette[index % len(palette)] if hi else "#AAB4BC"
        lw = 2.2 if hi else 1.0
        alpha = .92 if hi else .42
        ys = [float(row["source_rank"]), float(row["coculture_rank"]), float(row["ifng_rank"])]
        _bezier(ax, list(zip(xs, ys)), color, lw=lw, alpha=alpha, zorder=3 if hi else 2)
        ax.scatter(xs, ys, s=17 if hi else 8, color=color, edgecolor="white", linewidth=.25, zorder=4 if hi else 3)
        if hi:
            ax.text(2.03, ys[-1], label, fontsize=6.0, va="center", color=color, fontweight="bold")
    ax.set_xlim(-.18, 2.65); ax.set_ylim(ymax + .7, .4); ax.set_xticks(xs, context_labels); ax.set_ylabel("reliability rank (1 = best)"); ax.grid(axis="y", color="#D9DDE2", lw=.45, alpha=.7); ax.set_axisbelow(True)


def alluvial(ax: plt.Axes, source: Sequence[tuple[str, float]], target: Sequence[tuple[str, float]], colors: dict[str, str], title: str = "") -> None:
    """Compact categorical alluvial; source/target values are category weights."""
    left_y = {}; right_y = {}
    cursor = 0.0
    for state, count in source:
        left_y[state] = cursor + count / 2; cursor += count
    cursor = 0.0
    for state, count in target:
        right_y[state] = cursor + count / 2; cursor += count
    total = max(cursor, 1)
    for state, count in source:
        if state not in right_y:
            continue
        target_count = dict(target).get(state, 0)
        ribbon(ax, 0, left_y[state] / total, 1, right_y[state] / total, min(.20, .60 * min(count, target_count) / total), colors.get(state, "#AAB4BC"), alpha=.38)
    for x, mapping in ((0, left_y), (1, right_y)):
        for state, y in mapping.items():
            ax.scatter(x, y / total, s=60, color=colors.get(state, "#AAB4BC"), edgecolor="#303840", linewidth=.5, zorder=5)
            ax.text(x + (-.07 if x == 0 else .07), y / total, state, ha="right" if x == 0 else "left", va="center", fontsize=6.5)
    ax.set_xlim(-.28, 1.28); ax.set_ylim(-.08, 1.08); ax.set_xticks([0, 1], ["source state", "target state"]); ax.set_yticks([]); ax.set_title(title, loc="left", pad=8, fontweight="bold"); ax.grid(False)
