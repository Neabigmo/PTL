"""Small semantic glyphs for states, rank pairs, atlas cells, and firewalls."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

from scripts.figures.common import STATE_COLORS, num


def rank_glyph(ax: plt.Axes, source_rank: float, target_rank: float, source_label: str, target_label: str, burden: float, rank_delta: float, title: str) -> None:
    ax.plot([0, 1], [source_rank, target_rank], color="#303840", lw=1.4, zorder=2)
    ax.scatter([0, 1], [source_rank, target_rank], s=58, c=["#0072B2", "#C44E52"], edgecolors="white", linewidth=.6, zorder=3)
    ax.text(0, source_rank, f"  {source_label}\n  rank {source_rank:.0f}", va="center", fontsize=6.0)
    ax.text(1, target_rank, f"{target_label}  \nrank {target_rank:.0f}", va="center", ha="right", fontsize=6.0)
    ax.annotate("", xy=(.83, target_rank), xytext=(.17, source_rank), arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "#7A8996", "connectionstyle": "arc3,rad=.2"})
    ax.set_xlim(-.35, 1.35); ax.set_ylim(max(source_rank, target_rank) + 5, max(0, min(source_rank, target_rank) - 5)); ax.set_xticks([0, 1], ["Ctrl", "IFNγ"]); ax.set_ylabel("rank (1 = best)"); ax.set_title(title, loc="left", pad=8, fontweight="bold")
    ax.text(.03, .03, f"Δrank={rank_delta:+.0f}   burden={burden:.3f}", transform=ax.transAxes, fontsize=6.2, color="#46515C", fontweight="bold")


def state_key(value: Any) -> str:
    return str(value) if str(value) in STATE_COLORS else "unstable"


def state_dot(ax: plt.Axes, x: float, y: float, value: Any, size: float = 80) -> None:
    key = state_key(value)
    ax.scatter(x, y, s=size, color=STATE_COLORS[key], edgecolor="#303840", linewidth=.45, zorder=4)


def atlas_cell(ax: plt.Axes, x: float, y: float, value: float, lower: float, joint_lower: float, vmax: float) -> None:
    color = plt.get_cmap("RdBu_r")((value + vmax) / (2 * vmax) if vmax else .5)
    ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor=color, edgecolor="white", lw=.8))
    ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=5.4, color="white" if abs(value) > vmax * .56 else "#111111")
    if num(lower) > 0:
        ax.scatter(x + .27, y - .27, s=23, marker=".", color="#111111")
    if num(joint_lower) > 0:
        ax.scatter(x + .28, y + .28, s=24, marker="o", facecolors="none", edgecolors="#111111", linewidth=.75)


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
