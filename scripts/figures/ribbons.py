"""Reusable ribbon and braid primitives used by the figure builders."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import FancyBboxPatch, PathPatch
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
    palette = list(palette or ["#6683A1", "#9AA4AC", "#789180", "#B76D5E"])
    xs = np.arange(3, dtype=float)
    ymax = max(float(flow[["source_rank", "coculture_rank", "ifng_rank"]].max().max()), 10)
    for index, (_, row) in enumerate(flow.iterrows()):
        label = str(row["perturbation_label"])
        hi = label in {str(value) for value in labels}
        color = palette[index % len(palette)] if hi else "#B7BCC4"
        lw = 2.2 if hi else 1.0
        alpha = .92 if hi else .42
        ys = [float(row["source_rank"]), float(row["coculture_rank"]), float(row["ifng_rank"])]
        _bezier(ax, list(zip(xs, ys)), color, lw=lw, alpha=alpha, zorder=3 if hi else 2)
        ax.scatter(xs, ys, s=17 if hi else 8, color=color, edgecolor="white", linewidth=.25, zorder=4 if hi else 3)
        if hi:
            ax.text(2.03, ys[-1], label, fontsize=6.0, va="center", color=color, fontweight="bold")
    ax.set_xlim(-.18, 2.65); ax.set_ylim(ymax + .7, .4); ax.set_xticks(xs, context_labels); ax.set_ylabel("reliability rank (1 = best)"); ax.grid(axis="y", color="#D9DEE5", lw=.45, alpha=.7); ax.set_axisbelow(True)


def filled_rank_braid(
    ax: plt.Axes,
    flow,
    labels: Sequence[str],
    context_labels: Sequence[str],
    *,
    selected_color: str = "#263238",
    background_color: str = "#B7BCC4",
    width: float = .34,
    selected_width: float = .78,
    max_background: int | None = None,
) -> None:
    """Draw a compact bump-alluvial rank braid with filled bands.

    The old implementation used a collection of thin lines.  This version
    keeps every real perturbation trajectory, but gives each trajectory a
    narrow filled ribbon so crossings and rank zones remain legible after
    two-column reduction.  Only registered exemplars receive a dark outline.
    """
    xs = np.arange(3, dtype=float)
    selected = {str(value) for value in labels}
    ordered = flow.sort_values("source_rank", kind="stable")
    if max_background is not None and len(ordered) > max_background:
        keep = set(ordered.iloc[:max_background]["perturbation_label"].astype(str)) | selected
        ordered = ordered.loc[ordered["perturbation_label"].astype(str).isin(keep)]
    ymax = max(float(flow[["source_rank", "coculture_rank", "ifng_rank"]].max().max()), 10)
    # Broad rank zones are structural context rather than an extra encoding.
    for lo, hi, color in ((1, max(2, ymax * .28), "#F7F8F9"), (max(2, ymax * .28), max(3, ymax * .65), "#EEF1F3"), (max(3, ymax * .65), ymax + 1, "#F8F9FA")):
        ax.axhspan(lo, hi, color=color, zorder=0)
    for _, row in ordered.iterrows():
        label = str(row["perturbation_label"])
        hi = label in selected
        ys = [float(row["source_rank"]), float(row["coculture_rank"]), float(row["ifng_rank"])]
        col = selected_color if hi else background_color
        ribbon(ax, xs[0], ys[0], xs[1], ys[1], selected_width if hi else width, col, alpha=.64 if hi else .18, zorder=3 if hi else 1)
        ribbon(ax, xs[1], ys[1], xs[2], ys[2], selected_width if hi else width, col, alpha=.64 if hi else .18, zorder=3 if hi else 1)
        ax.plot(xs, ys, color=col, lw=1.05 if hi else .40, alpha=.97 if hi else .48, zorder=4 if hi else 2)
        if hi:
            ax.scatter(xs, ys, s=17, facecolor="white", edgecolor=selected_color, linewidth=.7, zorder=5)
    ax.set_xlim(-.22, 2.32)
    ax.set_ylim(ymax + 1.0, .4)
    ax.set_xticks(xs, context_labels)
    ax.set_ylabel("rank (1 = best)")
    ax.grid(axis="y", color="#D9DEE5", lw=.42, alpha=.65)
    ax.set_axisbelow(True)


def shortlist_alluvial(
    ax: plt.Axes,
    ranks,
    rows: list[dict],
    budget: float = .10,
    *,
    label_fontsize: float = 5.2,
    summary_fontsize: float = 6.0,
    tick_labels: tuple[str, str] = ("source top-k", "target top-k"),
    show_summary: bool = True,
    panel: str = "F",
    provenance: str = "formal_v2_reliability_reordering_perturbations.csv",
    label_limit: int | None = None,
    include_all_labels: bool = False,
    summary_xy: tuple[float, float] = (.5, .82),
    summary_label: str | None = None,
) -> None:
    """Parallel-sets shortlist replacement with neutral category grammar.

    ``include_all_labels`` keeps the rendered shortlist compact while making
    the source table a complete accounting of the underlying label universe.
    Non-selected labels are recorded as ``not_selected`` rather than being
    silently dropped from the provenance table.
    """
    n = len(ranks)
    k = max(1, int(np.ceil(budget * n)))
    source = set(ranks.nsmallest(k, "source_risk")["perturbation_label"].astype(str))
    target = set(ranks.nsmallest(k, "ifng_risk")["perturbation_label"].astype(str))
    labels = sorted(source | target, key=lambda v: (int(ranks.loc[ranks["perturbation_label"].eq(v), "source_rank"].iloc[0]), v))
    left_order = sorted(source | target, key=lambda v: (v not in source, int(ranks.loc[ranks["perturbation_label"].eq(v), "source_rank"].iloc[0]), v))
    right_order = sorted(source | target, key=lambda v: (v not in target, int(ranks.loc[ranks["perturbation_label"].eq(v), "ifng_rank"].iloc[0]), v))
    categories = {
        label: ("retained" if label in source and label in target else ("dropped" if label in source else "new"))
        for label in labels
    }
    left_y = {v: i + .5 for i, v in enumerate(left_order)}
    right_y = {v: i + .5 for i, v in enumerate(right_order)}
    total = max(len(labels), 1)
    if label_limit is None:
        visible_labels = set(labels)
    else:
        # The full source-frozen surface can contain many top-k members. Keep
        # only a few exemplars per outcome category, and select exemplars by
        # available whitespace on both sides so endpoint labels do not stack.
        visible_labels = set()
        selected_positions: list[tuple[float, float]] = []
        for category in ("retained", "dropped", "new"):
            category_labels = [label for label in labels if categories[label] == category]
            for _ in range(max(0, int(label_limit))):
                candidates = [label for label in category_labels if label not in visible_labels]
                if not candidates:
                    break
                def spacing(label: str) -> float:
                    point = (left_y[label] / total, right_y[label] / total)
                    if not selected_positions:
                        return float("inf")
                    return min(min(abs(point[0] - other[0]), abs(point[1] - other[1])) for other in selected_positions)
                candidate = max(candidates, key=spacing)
                # A visible label is useful only if it has enough vertical
                # clearance on both source and target sides.
                if selected_positions and spacing(candidate) < .075:
                    break
                visible_labels.add(candidate)
                selected_positions.append((left_y[candidate] / total, right_y[candidate] / total))
    for label in labels:
        row = ranks.loc[ranks["perturbation_label"].astype(str).eq(label)].iloc[0]
        retained = label in source and label in target
        category = "retained" if retained else ("dropped" if label in source else "new")
        style = {"retained": ("#527AA3", "-"), "dropped": ("#C96A5A", "-"), "new": ("#6F9B87", "-")}[category]
        ribbon_width = min(.075, .70 / total)
        ribbon(ax, 0, left_y[label] / total, 1, right_y[label] / total, ribbon_width, style[0], alpha=.52, zorder=1)
        ax.plot([0, 1], [left_y[label] / total, right_y[label] / total], color=style[0], lw=1.0, ls=style[1], alpha=.86, zorder=2)
        ax.scatter([0, 1], [left_y[label] / total, right_y[label] / total], s=20, facecolor="white", edgecolor="#263238", linewidth=.55, zorder=3)
        if label in visible_labels:
            ax.text(-.045, left_y[label] / total, label, ha="right", va="center", fontsize=label_fontsize)
            ax.text(1.045, right_y[label] / total, label, ha="left", va="center", fontsize=label_fontsize)
        rows.append({"panel": panel, "perturbation_label": label, "source_rank": row["source_rank"], "target_rank": row["ifng_rank"], "source_selected": label in source, "target_selected": label in target, "budget_fraction": budget, "shortlist_category": category, "provenance": provenance})
    if include_all_labels:
        plotted_labels = set(labels)
        for label in ranks["perturbation_label"].astype(str):
            if label in plotted_labels:
                continue
            row = ranks.loc[ranks["perturbation_label"].astype(str).eq(label)].iloc[0]
            rows.append({"panel": panel, "perturbation_label": label, "source_rank": row["source_rank"], "target_rank": row["ifng_rank"], "source_selected": False, "target_selected": False, "budget_fraction": budget, "shortlist_category": "not_selected", "provenance": provenance})
    ax.set_xlim(-.38, 1.38); ax.set_ylim(-.08, 1.08); ax.set_xticks([0, 1], tick_labels); ax.set_yticks([])
    if show_summary:
        label = summary_label or f"retained {len(source & target)}/{k} · dropped {len(source - target)} · new {len(target - source)}"
        ax.text(summary_xy[0], summary_xy[1], label, transform=ax.transAxes, ha="center", va="top", fontsize=summary_fontsize, fontweight="bold")


def shortlist_summary_glyph(
    ax: plt.Axes,
    ranks,
    rows: list[dict],
    budget: float = .10,
    *,
    panel: str = "D",
    provenance: str = "formal_v2_reliability_reordering_perturbations.csv",
) -> None:
    """Draw the compact graphical-abstract preview of a real shortlist.

    This intentionally does not repeat the detailed Fig.4 alluvial.  The
    source and target stacks retain the exact canonical counts, while three
    broad ribbons encode only the decision summary (retained, dropped, and
    newly entered).  Every canonical label is still written to ``rows`` so
    the source table remains auditable.
    """
    n = len(ranks)
    k = max(1, int(np.ceil(budget * n)))
    source = set(ranks.nsmallest(k, "source_risk")["perturbation_label"].astype(str))
    target = set(ranks.nsmallest(k, "ifng_risk")["perturbation_label"].astype(str))
    categories = {
        str(label): ("retained" if str(label) in source and str(label) in target
                     else ("dropped" if str(label) in source else
                           ("new" if str(label) in target else "not_selected")))
        for label in ranks["perturbation_label"].astype(str)
    }
    counts = {
        "retained": len(source & target),
        "dropped": len(source - target),
        "new": len(target - source),
    }
    colors = {"retained": "#527AA3", "dropped": "#C96A5A", "new": "#6F9B87"}

    # Record the complete canonical accounting without drawing 243 tiny
    # objects into the compact preview.
    for label in ranks["perturbation_label"].astype(str):
        row = ranks.loc[ranks["perturbation_label"].astype(str).eq(label)].iloc[0]
        rows.append({
            "panel": panel,
            "perturbation_label": label,
            "source_rank": row["source_rank"],
            "target_rank": row["ifng_rank"],
            "source_selected": label in source,
            "target_selected": label in target,
            "budget_fraction": budget,
            "shortlist_category": categories[label],
            "provenance": provenance,
        })

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    # The two stacks encode the same k positions on either side.  Their
    # unequal segments are proportional to the true retained/replaced counts.
    x_left, x_right, stack_w = .10, .78, .12
    stack_y, stack_h = .28, .46
    retained_h = stack_h * counts["retained"] / k
    replaced_h = stack_h - retained_h
    for x, side in ((x_left, "source"), (x_right, "target")):
        ax.add_patch(FancyBboxPatch(
            (x, stack_y), stack_w, stack_h,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor="#F5F7F8", edgecolor="#AAB4BC", linewidth=.65, zorder=1,
        ))
        if side == "source":
            ax.add_patch(FancyBboxPatch((x, stack_y), stack_w, retained_h,
                                        boxstyle="square,pad=0", facecolor=colors["retained"],
                                        edgecolor="none", alpha=.78, zorder=2))
            ax.add_patch(FancyBboxPatch((x, stack_y + retained_h), stack_w, replaced_h,
                                        boxstyle="square,pad=0", facecolor=colors["dropped"],
                                        edgecolor="none", alpha=.68, zorder=2))
        else:
            ax.add_patch(FancyBboxPatch((x, stack_y), stack_w, retained_h,
                                        boxstyle="square,pad=0", facecolor=colors["retained"],
                                        edgecolor="none", alpha=.78, zorder=2))
            ax.add_patch(FancyBboxPatch((x, stack_y + retained_h), stack_w, replaced_h,
                                        boxstyle="square,pad=0", facecolor=colors["new"],
                                        edgecolor="none", alpha=.68, zorder=2))
    # Three illustrative category ribbons replace the detailed alluvial; the
    # widths and endpoints are the exact count summary, not invented values.
    ribbon(ax, x_left + stack_w, stack_y + retained_h / 2,
           x_right, stack_y + retained_h / 2, max(.018, retained_h * .72),
           colors["retained"], alpha=.58, zorder=0)
    ribbon(ax, x_left + stack_w, stack_y + retained_h + replaced_h * .52,
           .51, .84, max(.035, replaced_h * .72), colors["dropped"], alpha=.42, zorder=0)
    ribbon(ax, .49, .16, x_right, stack_y + retained_h + replaced_h * .48,
           max(.035, replaced_h * .72), colors["new"], alpha=.42, zorder=0)
    ax.text(x_left + stack_w / 2, .79, f"source top 10%\n(k = {k})", ha="center", va="bottom", fontsize=5.7)
    ax.text(x_right + stack_w / 2, .79, f"target top 10%\n(k = {k})", ha="center", va="bottom", fontsize=5.7)
    ax.text(.50, .08, f"{counts['retained']} retained  ·  {counts['dropped']} dropped  ·  {counts['new']} new", ha="center", va="bottom", fontsize=5.7, color="#46515C")


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
