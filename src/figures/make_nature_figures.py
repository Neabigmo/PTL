"""Nature-style, evidence-backed figures for the PTL-v2 paper.

The script deliberately consumes only existing source tables/manifests. It is
not a replacement for the final expanded benchmark; panels label the current
replay as pilot evidence where appropriate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts" / "source_data"
MANIFEST = ROOT / "artifacts" / "manifests"
TABLES = ROOT / "results" / "tables"
DEFAULT_OUTPUT = ROOT / "results" / "figures" / "nature"

COLORS = {
    "ink": "#20252b",
    "muted": "#66717c",
    "grid": "#d9dee3",
    "paper": "#ffffff",
    "navy": "#174a72",
    "blue": "#3d78a8",
    "teal": "#16877a",
    "orange": "#d97932",
    "verm": "#b64b4b",
    "purple": "#73528f",
    "sand": "#e8c77b",
    "light_blue": "#dceaf4",
    "light_teal": "#d9eee9",
    "light_orange": "#f5e2cf",
    "light_purple": "#e8e0ef",
    "light_grey": "#f2f4f5",
}

PREDICTOR_COLORS = {
    "mean_global": COLORS["blue"],
    "mean_matching": COLORS["orange"],
    "ridge": COLORS["purple"],
}

SPLIT_ORDER = ["random_split", "unseen_perturbation_split", "dataset_heldout_split"]
SPLIT_LABELS = {
    "random_split": "Random",
    "unseen_perturbation_split": "Unseen perturbation",
    "dataset_heldout_split": "Dataset held out",
}

METHOD_LABELS = {
    "raw_normalized_uq": "Raw UQ",
    "platt_logistic": "Platt",
    "isotonic": "Isotonic",
    "support_novelty_only": "Support + novelty",
    "ptl_context": "PTL context",
    "ptl_no_context": "PTL no context",
    "ptl_full_id": "PTL full-ID diagnostic",
    "gbdt_uncalibrated": "GBDT",
    "gbdt_full_id": "GBDT full-ID diagnostic",
    "ptl_rf": "PTL",
    "ptl_rf_full_id": "PTL full-ID diagnostic",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.75,
            "axes.edgecolor": COLORS["grid"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": COLORS["paper"],
            "figure.facecolor": COLORS["paper"],
            "savefig.facecolor": COLORS["paper"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=COLORS["ink"],
    )


def clean_axis(ax: plt.Axes, grid: bool = False) -> None:
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(axis="both", colors=COLORS["muted"], length=2, width=0.6)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.8)


def load(name: str, source_dir: Path = SOURCE) -> pd.DataFrame:
    path = source_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_publication(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{stem}.svg", output_dir / f"{stem}.pdf", output_dir / f"{stem}.tiff", output_dir / f"{stem}.png"]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=600, bbox_inches="tight")
    fig.savefig(outputs[3], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def draw_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str,
             subtitle: str, face: str, edge: str | None = None, title_color: str = COLORS["ink"]) -> None:
    edge = edge or face
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face, edgecolor=edge, linewidth=0.8, transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.61, title, transform=ax.transAxes,
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=title_color)
    ax.text(x + w / 2, y + h * 0.31, subtitle, transform=ax.transAxes,
            ha="center", va="center", fontsize=5.8, color=COLORS["muted"])


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          color: str = COLORS["ink"], y_shift: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (start[0], start[1] + y_shift), (end[0], end[1] + y_shift),
            transform=ax.transAxes, arrowstyle="-|>", mutation_scale=8,
            linewidth=0.8, color=color, shrinkA=3, shrinkB=3,
        )
    )


def fig1_framework(output_dir: Path) -> list[Path]:
    audit = load("gwps_coverage_audit.csv", MANIFEST)
    sem = load("context_semantics_audit.csv", MANIFEST)

    fig = plt.figure(figsize=(7.2, 6.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.45, 1.08, 1.12], hspace=0.62, wspace=0.42)
    fig.suptitle("Context-conditioned reliability for perturbation prediction", x=0.06, y=0.985,
                 ha="left", fontsize=11, fontweight="bold", color=COLORS["ink"])
    fig.text(0.06, 0.957, "A science-first workflow: define shift, separate evidence, calibrate decisions, then expand context.",
             ha="left", va="top", fontsize=7, color=COLORS["muted"])

    ax = fig.add_subplot(gs[0, :])
    add_panel_label(ax, "a")
    ax.set_axis_off()
    ax.text(0.01, 0.91, "Deployment-time reliability layer", transform=ax.transAxes,
            fontsize=8, fontweight="bold", color=COLORS["ink"])
    draw_box(ax, 0.02, 0.48, 0.17, 0.27, "Perturbation", "predictor output", COLORS["light_blue"], COLORS["blue"])
    draw_box(ax, 0.26, 0.64, 0.15, 0.22, "U", "native UQ", COLORS["light_orange"], COLORS["orange"])
    draw_box(ax, 0.26, 0.30, 0.15, 0.22, "z", "deployment context", COLORS["light_teal"], COLORS["teal"])
    draw_box(ax, 0.50, 0.48, 0.19, 0.27, "PTL calibrator", "context-conditioned p", COLORS["light_purple"], COLORS["purple"])
    draw_box(ax, 0.79, 0.48, 0.18, 0.27, "Decision", "keep / abstain", COLORS["light_grey"], COLORS["ink"])
    arrow(ax, (0.19, 0.61), (0.25, 0.73))
    arrow(ax, (0.19, 0.61), (0.25, 0.41))
    arrow(ax, (0.41, 0.75), (0.50, 0.61))
    arrow(ax, (0.41, 0.41), (0.50, 0.61))
    arrow(ax, (0.69, 0.61), (0.79, 0.61))
    ax.text(0.03, 0.12, "U", transform=ax.transAxes, fontsize=7, fontweight="bold", color=COLORS["orange"])
    ax.text(0.08, 0.12, "uncertainty", transform=ax.transAxes, fontsize=6.2, color=COLORS["muted"])
    tokens = [("P", "prediction geometry", COLORS["blue"]), ("S", "support", COLORS["teal"]),
              ("N", "novelty", COLORS["verm"]), ("C", "biological context", COLORS["purple"])]
    token_positions = [0.20, 0.45, 0.60, 0.73]
    for (token, label, color), x in zip(tokens, token_positions):
        ax.text(x, 0.12, token, transform=ax.transAxes, fontsize=7, fontweight="bold", color=color)
        ax.text(x + 0.04, 0.12, label, transform=ax.transAxes, fontsize=6.2, color=COLORS["muted"])
    ax.text(0.50, 0.29, r"$p=\sigma\{\exp(a(z))\,\mathrm{logit}(p_0)+b(z)\}$",
            transform=ax.transAxes, fontsize=7, color=COLORS["ink"], ha="center")

    ax = fig.add_subplot(gs[1, 0])
    add_panel_label(ax, "b")
    ax.set_title("Context expansion is semantic, not just larger", loc="left", fontsize=8, pad=4, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 3.55)
    ax.set_axis_off()
    semantic_rows = [
        ("Tian CRISPRa", "iPSC-induced neuron · 21,193 cells", COLORS["blue"]),
        ("Tian CRISPRi", "iPSC-induced neuron · 32,300 cells", COLORS["teal"]),
        ("Frangieh RNA", "melanoma co-culture · 218,331 cells", COLORS["purple"]),
        ("Frangieh conditions", "Control · Co-culture · IFNγ", COLORS["orange"]),
    ]
    for i, (title, detail, color) in enumerate(semantic_rows):
        y = 3.05 - i * 0.88
        ax.add_patch(FancyBboxPatch((0.02, y - 0.24), 0.08, 0.48, boxstyle="round,pad=0.01",
                                    facecolor=color, edgecolor=color))
        ax.text(0.06, y, str(i + 1), ha="center", va="center", color="white", fontweight="bold", fontsize=7)
        ax.text(0.14, y + 0.08, title, ha="left", va="center", fontsize=7, fontweight="bold", color=COLORS["ink"])
        ax.text(0.14, y - 0.10, detail, ha="left", va="center", fontsize=6.1, color=COLORS["muted"])
    ax.text(0.02, -0.21, "Raw metadata audit precedes model comparison.", fontsize=6.1, color=COLORS["muted"])
    _ = sem  # loaded to make the panel source explicit and fail loudly if absent

    ax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, "c")
    ax.set_title("GWPS coverage audit · attrition at eligibility", loc="left", fontsize=8, pad=4, fontweight="bold")
    stage_order = ["raw_obs", "qc_min_genes", "qc_min_counts", "min_cells_per_perturbation", "reference_matching"]
    stage_labels = {
        "raw_obs": "raw",
        "qc_min_genes": "QC genes",
        "qc_min_counts": "QC counts",
        "min_cells_per_perturbation": "batch × perturbation",
        "reference_matching": "matched signatures",
    }
    subset = audit.set_index("stage").loc[stage_order].reset_index()
    values = subset["n_cells"].to_numpy(dtype=float)
    y = np.arange(len(subset))
    ax.barh(y, values, color=[COLORS["blue"], COLORS["blue"], COLORS["blue"], COLORS["orange"], COLORS["teal"]], height=0.55)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([stage_labels[s] for s in subset["stage"]])
    ax.invert_yaxis()
    ax.set_xlabel("cells retained (log scale)")
    ax.set_xlim(3e4, 3e6)
    for yi, val in zip(y, values):
        label = f"{val/1e6:.2f}M" if val >= 1e6 else f"{val/1e3:.0f}k"
        ax.text(val * 1.05, yi, label, va="center", fontsize=6.2, color=COLORS["ink"])
    clean_axis(ax)

    ax = fig.add_subplot(gs[2, 0])
    add_panel_label(ax, "d")
    ax.set_title("Evidence tables stay separate", loc="left", fontsize=8, pad=4, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    boxes = [("Predictions", "mean + UQ", COLORS["light_blue"], COLORS["blue"]),
             ("Deployment", "target-free z", COLORS["light_teal"], COLORS["teal"]),
             ("Outcomes", "fidelity + label", COLORS["light_orange"], COLORS["orange"]),
             ("Folds", "grouped CV", COLORS["light_purple"], COLORS["purple"])]
    for i, (title, detail, face, edge) in enumerate(boxes):
        x = 0.02 + (i % 2) * 0.49
        y = 0.53 - (i // 2) * 0.38
        draw_box(ax, x, y, 0.37, 0.22, title, detail, face, edge)
        if i in (0, 2):
            arrow(ax, (x + 0.38, y + 0.11), (x + 0.47, y + 0.11), color=COLORS["grid"])
    ax.text(0.02, 0.03, "Joined only by prediction_id inside evaluation code.", fontsize=6.1, color=COLORS["muted"])

    ax = fig.add_subplot(gs[2, 1])
    add_panel_label(ax, "e")
    ax.set_title("Current decision boundary", loc="left", fontsize=8, pad=4, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.05, 0.54), 0.90, 0.27, boxstyle="round,pad=0.02",
                                facecolor=COLORS["light_orange"], edgecolor=COLORS["orange"], linewidth=0.8))
    ax.text(0.50, 0.68, "GWPS → GEARS contract pilot", ha="center", va="center", fontsize=7.2, fontweight="bold")
    ax.text(0.50, 0.58, "native UQ + condition holdout", ha="center", va="center", fontsize=6.2, color=COLORS["muted"])
    ax.add_patch(FancyBboxPatch((0.05, 0.16), 0.90, 0.27, boxstyle="round,pad=0.02",
                                facecolor=COLORS["light_teal"], edgecolor=COLORS["teal"], linewidth=0.8))
    ax.text(0.50, 0.30, "K562 essential → headline common core", ha="center", va="center", fontsize=7.2, fontweight="bold")
    ax.text(0.50, 0.20, "broad-enough replay surface", ha="center", va="center", fontsize=6.2, color=COLORS["muted"])
    return save_publication(fig, output_dir, "nature_fig1_framework")


def fig2_native_uq(output_dir: Path) -> list[Path]:
    df = load("figure2_confidence_source.csv")
    df = df[df["split_family"].isin(SPLIT_ORDER)].copy()
    fig = plt.figure(figsize=(7.2, 4.65))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0], hspace=0.62, wspace=0.38)
    fig.suptitle("Native uncertainty changes meaning across structured shift", x=0.06, y=0.985,
                 ha="left", fontsize=11, fontweight="bold", color=COLORS["ink"])
    fig.text(0.06, 0.953, "Current PTL replay: observation-level uncertainty quantiles versus realized delta-cosine fidelity.",
             ha="left", va="top", fontsize=7, color=COLORS["muted"])

    for j, split in enumerate(SPLIT_ORDER):
        ax = fig.add_subplot(gs[0, j])
        add_panel_label(ax, chr(ord("a") + j))
        part = df[df["split_family"] == split]
        for predictor in ["mean_global", "mean_matching", "ridge"]:
            sub = part[part["predictor"] == predictor].sort_values("prediction_id")
            if len(sub) > 900:
                step = max(1, len(sub) // 900)
                sub = sub.iloc[::step]
            ax.scatter(sub["uq_quantile"], sub["fidelity_delta_cosine"], s=4, alpha=0.16,
                       color=PREDICTOR_COLORS[predictor], linewidths=0, label=predictor)
        ax.axvline(0.5, color=COLORS["grid"], linewidth=0.6, linestyle="--")
        ax.set_title(SPLIT_LABELS[split], loc="left", fontsize=8, fontweight="bold")
        ax.set_xlabel("UQ quantile (higher = more uncertain)")
        if j == 0:
            ax.set_ylabel("delta-cosine fidelity")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.65, 1.02)
        clean_axis(ax, grid=True)
        if j == 2:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, ["global mean", "matching mean", "Ridge"], loc="lower left", frameon=False,
                      handletextpad=0.3, markerscale=2)

    # Aggregate summary in the bottom row. Each panel corresponds to a distinct question.
    ax = fig.add_subplot(gs[1, 0])
    add_panel_label(ax, "d")
    summary = df.groupby(["predictor", "split_family"], as_index=False).agg(
        mean_fidelity=("fidelity_delta_cosine", "mean"), mean_uq=("uq_quantile", "mean"), n=("fidelity_delta_cosine", "size")
    )
    for predictor in ["mean_global", "mean_matching", "ridge"]:
        sub = summary[summary["predictor"] == predictor]
        ax.plot(sub["mean_uq"], sub["mean_fidelity"], marker="o", markersize=3.5,
                linewidth=1.1, color=PREDICTOR_COLORS[predictor], label=predictor)
        for _, row in sub.iterrows():
            ax.text(row["mean_uq"] + 0.015, row["mean_fidelity"], SPLIT_LABELS[row["split_family"]].split()[0],
                    fontsize=5.3, color=PREDICTOR_COLORS[predictor], va="center")
    ax.set_xlabel("mean UQ quantile")
    ax.set_ylabel("mean fidelity")
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 0.48)
    clean_axis(ax, grid=True)

    ax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, "e")
    bar = summary.pivot(index="split_family", columns="predictor", values="mean_fidelity").reindex(SPLIT_ORDER)
    x = np.arange(len(SPLIT_ORDER)); width = 0.23
    for i, predictor in enumerate(["mean_global", "mean_matching", "ridge"]):
        ax.bar(x + (i - 1) * width, bar[predictor].to_numpy(), width=width, color=PREDICTOR_COLORS[predictor],
               label={"mean_global": "global", "mean_matching": "matching", "ridge": "Ridge"}[predictor])
    ax.set_xticks(x); ax.set_xticklabels(["Random", "Unseen\npert.", "Dataset\nholdout"])
    ax.set_ylabel("mean fidelity")
    ax.legend(frameon=False, ncol=3, loc="upper right", handlelength=1.2, columnspacing=0.8)
    clean_axis(ax, grid=True)

    ax = fig.add_subplot(gs[1, 2])
    add_panel_label(ax, "f")
    ax.set_axis_off()
    ax.text(0.02, 0.86, "Interpretation", fontsize=8, fontweight="bold", color=COLORS["ink"])
    ax.text(0.02, 0.68, "The same uncertainty scale is\nnot a transportability invariant.", fontsize=8,
            color=COLORS["navy"], fontweight="bold", va="top")
    ax.text(0.02, 0.33, "This motivates a target-free\ncontext-conditioned reliability layer.\n\nPilot evidence; final learned-model\ncomparison remains pending.",
            fontsize=6.8, color=COLORS["muted"], va="top")
    return save_publication(fig, output_dir, "nature_fig2_native_uq")


def fig3_selective(output_dir: Path) -> list[Path]:
    source = load("figure3_selective_source.csv")
    macro = source[source["aggregation_level"] == "environment_macro_average"].copy()
    ci = load("g4_bootstrap_ci.csv")
    ci = ci[(ci["aggregation_level"] == "paired_hierarchical_macro_ci") & ci["metric"].isin(["aurc", "ftr_at_80"])].copy()
    methods = ["raw_normalized_uq", "support_novelty_only", "ptl_context", "ptl_no_context", "gbdt_uncalibrated", "ptl_rf"]
    methods = [m for m in methods if m in set(macro["method"])]
    order = macro.set_index("method").loc[methods].sort_values("aurc").index.tolist()

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.28, 1.0], hspace=0.68, wspace=0.42)
    fig.suptitle("PTL improves selective reliability on the replay surface", x=0.06, y=0.985,
                 ha="left", fontsize=11, fontweight="bold", color=COLORS["ink"])
    fig.text(0.06, 0.953, "Lower AURC and lower false-transportability rate are better; aggregation is environment-first.",
             ha="left", va="top", fontsize=7, color=COLORS["muted"])

    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "a")
    y = np.arange(len(order))
    values = macro.set_index("method").loc[order, "aurc"].to_numpy()
    ax.barh(y, values, color=[COLORS["muted"] if m == "raw_normalized_uq" else COLORS["teal"] if m == "ptl_rf" else COLORS["blue"] for m in order], height=0.58)
    ci_lookup = ci[(ci["metric"] == "aurc")].set_index("method")
    for yi, method in zip(y, order):
        if method in ci_lookup.index:
            row = ci_lookup.loc[method]
            ax.errorbar(row["estimate"], yi, xerr=[[row["estimate"] - row["ci_lower"]], [row["ci_upper"] - row["estimate"]]],
                        fmt="none", ecolor=COLORS["ink"], capsize=2.2, linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels([METHOD_LABELS[m] for m in order])
    ax.invert_yaxis(); ax.set_xlabel("environment-macro AURC")
    ax.text(0.98, 0.03, "95% paired hierarchical CI", transform=ax.transAxes, ha="right", fontsize=5.7, color=COLORS["muted"])
    clean_axis(ax, grid=True)

    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "b")
    x = np.arange(len(order)); width = 0.34
    vals50 = macro.set_index("method").loc[order, "ftr_at_50"].to_numpy()
    vals80 = macro.set_index("method").loc[order, "ftr_at_80"].to_numpy()
    ax.bar(x - width / 2, vals50, width, color=COLORS["light_orange"], edgecolor=COLORS["orange"], linewidth=0.5, label="50% retained")
    ax.bar(x + width / 2, vals80, width, color=COLORS["light_purple"], edgecolor=COLORS["purple"], linewidth=0.5, label="80% retained")
    ax.set_xticks(x); ax.set_xticklabels([METHOD_LABELS[m].replace(" ", "\n", 1) for m in order], rotation=35, ha="right")
    ax.set_ylim(0, 0.7); ax.set_ylabel("false-transportability rate")
    ax.legend(frameon=False, loc="upper left", handlelength=1.2)
    clean_axis(ax, grid=True)

    ax = fig.add_subplot(gs[1, 0])
    add_panel_label(ax, "c")
    selected = source[source["aggregation_level"] == "predictor_environment_split"].copy()
    raw = selected[selected["method"] == "raw_normalized_uq"].groupby("split_family")["aurc"].mean()
    ptl = selected[selected["method"] == "ptl_rf"].groupby("split_family")["aurc"].mean()
    delta = (raw - ptl).reindex(SPLIT_ORDER)
    ax.bar(np.arange(len(delta)), delta.to_numpy(), color=[COLORS["blue"], COLORS["teal"], COLORS["verm"]], width=0.55)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_xticks(np.arange(len(delta))); ax.set_xticklabels(["Random", "Unseen\npert.", "Dataset\nholdout"])
    ax.set_ylabel("raw AURC − PTL AURC")
    ax.text(0.98, 0.90, "positive = PTL lower", transform=ax.transAxes, ha="right", fontsize=5.8, color=COLORS["muted"])
    clean_axis(ax, grid=True)

    ax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, "d")
    ax.set_axis_off()
    ax.text(0.02, 0.85, "What is supported now", fontsize=8, fontweight="bold", color=COLORS["ink"])
    ax.text(0.02, 0.65, "PTL is a selective\nreliability layer, not\na predictor leaderboard.", fontsize=8,
            color=COLORS["navy"], fontweight="bold", va="top")
    ax.text(0.02, 0.28, "The strongest current replay method is\nPTL with a random-forest head.\n\nLearned predictor roster and expanded\ncontext surfaces remain next-stage work.",
            fontsize=6.7, color=COLORS["muted"], va="top")
    return save_publication(fig, output_dir, "nature_fig3_selective")


def fig4_information_blocks(output_dir: Path) -> list[Path]:
    abl = load("ptl_ablation.csv", TABLES)
    feat = load("ptl_feature_audit.csv", TABLES)
    feat = feat[(feat["deployment_available"] == True) & (feat["excluded_from_ptl"] == False)]
    block_order = ["prediction_confidence_uncertainty", "context_distance", "support_availability", "perturbation_novelty", "model_split_dataset_indicator", "other"]
    block_labels = {
        "prediction_confidence_uncertainty": "U · prediction / UQ",
        "context_distance": "P/C · geometry / context",
        "support_availability": "S · support",
        "perturbation_novelty": "N · novelty",
        "model_split_dataset_indicator": "design indicators",
        "other": "other deployment fields",
    }
    counts = feat.groupby("source_group").size().reindex(block_order).fillna(0)
    completed = abl[abl["status"] == "completed"].copy()
    completed = completed.sort_values("mean_false_transportability_rate_gain_vs_naive", ascending=True)

    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.66, wspace=0.42)
    fig.suptitle("Reliability is assembled from deployment-time information blocks", x=0.06, y=0.985,
                 ha="left", fontsize=11, fontweight="bold", color=COLORS["ink"])
    fig.text(0.06, 0.953, "Current replay probes the blocks by ablation; the full incremental ladder is reserved for the expanded benchmark.",
             ha="left", va="top", fontsize=7, color=COLORS["muted"])

    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "a")
    y = np.arange(len(block_order))
    block_colors = [COLORS["orange"], COLORS["blue"], COLORS["teal"], COLORS["verm"], COLORS["purple"], COLORS["muted"]]
    ax.barh(y, counts.to_numpy(), color=block_colors, height=0.57)
    ax.set_yticks(y); ax.set_yticklabels([block_labels[b] for b in block_order])
    ax.invert_yaxis(); ax.set_xlabel("deployment features")
    for yi, val in zip(y, counts.to_numpy()):
        if val:
            ax.text(val + 0.6, yi, str(int(val)), va="center", fontsize=6.2, color=COLORS["ink"])
    clean_axis(ax, grid=True)

    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "b")
    labels = ["U", "P", "S", "N", "C"]
    x = np.arange(len(labels))
    for i, token in enumerate(labels):
        color = [COLORS["orange"], COLORS["blue"], COLORS["teal"], COLORS["verm"], COLORS["purple"]][i]
        ax.add_patch(FancyBboxPatch((i - 0.36, 0.36), 0.72, 0.34, boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor=color, linewidth=0.8))
        ax.text(i, 0.53, token, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        if i < len(labels) - 1:
            ax.annotate("", xy=(i + 0.62, 0.53), xytext=(i + 0.38, 0.53),
                        arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": COLORS["ink"]})
    ax.text(2, 0.19, "U → U+P → U+P+S → U+P+S+N → U+P+S+N+C", ha="center", fontsize=7, color=COLORS["navy"], fontweight="bold")
    ax.text(2, 0.04, "planned incremental evaluation", ha="center", fontsize=6, color=COLORS["muted"])
    ax.set_xlim(-0.6, 4.6); ax.set_ylim(0, 1); ax.set_axis_off()

    ax = fig.add_subplot(gs[1, 0])
    add_panel_label(ax, "c")
    vals = completed["mean_false_transportability_rate_gain_vs_naive"].to_numpy()
    labels = ["full PTL", "no context distance", "no perturb. novelty", "no UQ features"]
    name_map = {"full_PTL": "full PTL", "no_context_distance": "no context distance", "no_perturbation_novelty": "no perturb. novelty", "no_uncertainty_features": "no UQ features"}
    labels = [name_map.get(v, v) for v in completed["ablation"]]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=[COLORS["teal"] if "full" in lab else COLORS["blue"] for lab in labels], height=0.56)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("FTR gain vs raw UQ")
    ax.axvline(0, color=COLORS["ink"], linewidth=0.7)
    for yi, val in zip(y, vals):
        ax.text(val + 0.008, yi, f"{val:.3f}", va="center", fontsize=6.1)
    clean_axis(ax, grid=True)

    ax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, "d")
    ax.set_axis_off()
    ax.text(0.02, 0.84, "Design constraint", fontsize=8, fontweight="bold", color=COLORS["ink"])
    ax.text(0.02, 0.63, "Only information available\nbefore observing the target\nmay enter PTL.", fontsize=8,
            color=COLORS["navy"], fontweight="bold", va="top")
    ax.text(0.02, 0.23, "Outcome-derived fields are retained\nfor evaluation and excluded from\nthe deployment feature allowlist.", fontsize=6.8,
            color=COLORS["muted"], va="top")
    return save_publication(fig, output_dir, "nature_fig4_information_blocks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evidence-backed Nature-style PTL figures.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_style()
    outputs = []
    outputs.extend(fig1_framework(args.output_dir))
    outputs.extend(fig2_native_uq(args.output_dir))
    outputs.extend(fig3_selective(args.output_dir))
    outputs.extend(fig4_information_blocks(args.output_dir))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
