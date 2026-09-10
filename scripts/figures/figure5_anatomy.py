"""Figure 5: heterogeneous anatomy of individual transport failures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figures.common import (
    METRICS, PRIMARY_METRIC, PRIMARY_SOURCE, PRIMARY_TARGET, STATE_COLORS,
    STATE_ORDER, add_panel_label, clean_axes, failure_cases, metric_color,
    metric_label, num, primary_risks, rank_table, set_title, spearman_pair,
    state_counts, save_figure,
)
from scripts.figures.glyphs import rank_glyph
from scripts.figures.ribbons import ribbon


def _state_alluvial(ax: plt.Axes, states: pd.DataFrame, rows: list[dict]) -> None:
    agg = states.groupby(["source_state", "target_state"], as_index=False)["count"].sum()
    source = agg.groupby("source_state")["count"].sum().reindex(STATE_ORDER, fill_value=0)
    target = agg.groupby("target_state")["count"].sum().reindex(STATE_ORDER, fill_value=0)
    total = float(max(source.sum(), 1)); left = {}; right = {}; c = 0.
    for state, value in source.items(): left[state] = c + float(value) / 2; c += float(value)
    c = 0.
    for state, value in target.items(): right[state] = c + float(value) / 2; c += float(value)
    for _, item in agg.iterrows():
        src, tgt, count = item["source_state"], item["target_state"], float(item["count"])
        if count <= 0: continue
        ribbon(ax, 0, left[src] / total, 1, right[tgt] / total, min(.12, .58 * count / total), STATE_COLORS.get(src, "#C7CDD2"), alpha=.28, zorder=1)
        rows.append(item.to_dict())
    for x, mapping, title in ((0, left, "source state"), (1, right, "target state")):
        for state, pos in mapping.items():
            ax.scatter(x, pos / total, s=50, color=STATE_COLORS[state], edgecolor="#303840", linewidth=.5, zorder=4)
            ax.text(x + (-.055 if x == 0 else .055), pos / total, state, ha="right" if x == 0 else "left", va="center", fontsize=6.6)
        ax.text(x, 1.065, title, ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xlim(-.25, 1.25); ax.set_ylim(-.04, 1.10); ax.set_xticks([]); ax.set_yticks([]); ax.text(.50, -.02, "blue/red = stable direction · gray = tie/unstable after strict-support floor 8", transform=ax.transAxes, ha="center", va="top", fontsize=6.0, color="#46515C")


def figure5(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    ranks = rank_table(manifests); cases = failure_cases(manifests, ranks); risks = primary_risks(manifests); failure = pd.read_csv(manifests / "reliability_transport_failure_anatomy.csv"); complete = failure.loc[failure["status"].eq("executed")].copy(); rows: list[dict] = []
    fig = plt.figure(figsize=(14, 8.1), constrained_layout=False)
    grid = fig.add_gridspec(2, 12, height_ratios=(1.50, 1.0), hspace=.55, wspace=.50)
    for ci, (_, case) in enumerate(cases.iterrows()):
        ax = fig.add_subplot(grid[0, ci * 4:(ci + 1) * 4]); add_panel_label(ax, chr(ord("A") + ci)); q = int(round(case["quantile"] * 100))
        set_title(ax, f"{case['perturbation_label']} · q{q} burden", "real label; rank displacement, risk distribution, and model members")
        rank_glyph(ax, case["source_rank"], case["ifng_rank"], "Ctrl", "IFNγ", case["reordering_burden"], case["rank_delta"], "")
        risk_sub = risks.loc[risks["perturbation_label"].eq(case["perturbation_label"]) & risks["cell_budget_label"].eq("full")]
        if not risk_sub.empty:
            source_values = pd.to_numeric(risk_sub["left_risk"], errors="coerce").dropna(); target_values = pd.to_numeric(risk_sub["right_risk"], errors="coerce").dropna()
            inset = ax.inset_axes([.40, .08, .56, .27]); inset.violinplot([source_values, target_values], positions=[0, 1], vert=False, widths=.65, showmeans=False, showmedians=True, showextrema=False)
            for body, color in zip(inset.collections, ["#0072B2", "#C44E52"]): body.set_facecolor(color); body.set_alpha(.32); body.set_edgecolor(color)
            inset.set_yticks([0, 1], ["S", "T"], fontsize=5); inset.set_xlabel("risk distribution", fontsize=5); inset.tick_params(axis="x", labelsize=5); clean_axes(inset)
            member_values = [float(v) for v in source_values.iloc[:3]] + [float(v) for v in target_values.iloc[:3]]
            ax.text(.42, .38, "3 frozen member/seed points shown", transform=ax.transAxes, fontsize=5.5, color="#46515C")
        ax.text(.03, .04, f"response={num(case['response_displacement']):.3f} · effect={num(case['effect_magnitude']):.3f}\npost-hoc descriptors; no mechanistic annotation", transform=ax.transAxes, fontsize=5.8, color="#46515C")
        rows.append({"panel": chr(ord("A") + ci), **case.to_dict(), "risk_distribution": "30 seed × matched measurement replicate at full depth", "model_members_shown": 3, "provenance": "reliability_transport_failure_anatomy.csv + reliability_transport_measurement_depth_matched_fixed_risks.csv"})

    ax = fig.add_subplot(grid[1, 0:5]); add_panel_label(ax, "D"); set_title(ax, "State transitions across contexts", "stable, inversion, tie, and unresolved branches remain distinct")
    states = state_counts(manifests / "reliability_transport_metric_pair_states.csv"); _state_alluvial(ax, states, rows)
    for _, item in states.iterrows():
        row = item.to_dict(); row["panel"] = "D"; row["provenance"] = "reliability_transport_metric_pair_states.csv"; rows.append(row)

    ax = fig.add_subplot(grid[1, 5:12]); add_panel_label(ax, "E"); set_title(ax, "Response displacement is not a metric-universal explanation", "complete rows only · raw points + density structure · explanatory, post-hoc")
    complete["reordering_burden"] = pd.to_numeric(complete["reordering_burden"], errors="coerce"); complete["response_displacement"] = pd.to_numeric(complete["response_displacement"], errors="coerce")
    complete = complete.dropna(subset=["reordering_burden", "response_displacement"])
    ax.hexbin(complete["reordering_burden"], complete["response_displacement"], gridsize=24, mincnt=1, cmap="Blues", bins="log", alpha=.42, linewidths=.15)
    rhos = {}
    for metric in METRICS:
        sub = complete.loc[complete["metric"].eq(metric)]
        rhos[metric] = spearman_pair(sub)
        ax.scatter(sub["reordering_burden"], sub["response_displacement"], s=8, color=metric_color(metric), alpha=.22, label=metric_label(metric))
        rows.append({"panel": "E summary", "metric": metric, "n_complete": len(sub), "spearman": rhos[metric], "provenance": "reliability_transport_failure_anatomy.csv", "no_imputation": True})
    overall = spearman_pair(complete); med = complete.groupby("metric", as_index=False)[["reordering_burden", "response_displacement"]].median()
    for _, item in med.iterrows():
        ax.scatter(item["reordering_burden"], item["response_displacement"], s=76, marker="D", color=metric_color(item["metric"]), edgecolor="white", linewidth=.8, zorder=5)
    ax.axvline(0, color="#303840", lw=.7); ax.text(.03, .97, f"overall ρ={overall:.2f} · complete 219/2784\nΔ cosine {rhos.get('delta_cosine', np.nan):.2f} · Systema {rhos.get('systema_centroid_accuracy', np.nan):.2f} · rank {rhos.get('absolute_effect_rank_agreement', np.nan):.2f}", transform=ax.transAxes, va="top", fontsize=6.8, fontweight="bold")
    coverage = pd.read_csv(manifests / "reliability_transport_failure_anatomy_coverage.csv"); coverage["fraction"] = coverage["complete"] / coverage["total"]
    ax.text(.98, .04, "coverage: 219/2784; no imputation\nfull coverage table in Supplement", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="#46515C")
    ax.set_xlabel("reordering burden"); ax.set_ylabel("response displacement"); ax.legend(frameon=False, fontsize=5.7, loc="upper right"); clean_axes(ax, grid=True)
    rows.extend({"panel": "F", **item.to_dict(), "no_imputation": True, "provenance": "reliability_transport_failure_anatomy_coverage.csv"} for _, item in coverage.iterrows())

    fig.suptitle("Figure 5 | Individual transport failures have heterogeneous, partly unobserved anatomy", fontsize=12, fontweight="bold")
    fig.text(.5, .012, "Case identities are deterministic q10/q50/q90 burden exemplars. State classification enforces minimum strict support 8; explanatory response fields never enter prospective prediction.", ha="center", fontsize=7.0, color="#46515C")
    return save_figure(fig, out_dir, "reliability_transportability_fig5_failure_anatomy"), pd.DataFrame(rows)


__all__ = ["figure5"]
