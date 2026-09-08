"""Create the controlled reliability-reordering diagnostic figure.

The figure is deliberately tied to the materialized Frangieh first-stage
artifacts.  It shows condition-level rank crossings, descriptive response
program displacement, and the monotonic recalibration counterfactual without
turning any of those evaluation outcomes into deployment features.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_grouping_loss import artifact_paths  # noqa: E402


NAVY = "#12263A"
BLUE = "#1F6F8B"
TEAL = "#2A9D8F"
CORAL = "#D95D52"
GOLD = "#E9B44C"
PURPLE = "#7661A8"
SLATE = "#6C7A89"
GRID = "#DCE5E8"

PAIR_LABELS = {
    ("frangieh_melanoma_control", "frangieh_melanoma_coculture"): "control → co-culture",
    ("frangieh_melanoma_control", "frangieh_melanoma_ifng"): "control → IFNγ",
    ("frangieh_melanoma_coculture", "frangieh_melanoma_ifng"): "co-culture → IFNγ",
}
PAIR_COLORS = {
    "control → co-culture": BLUE,
    "control → IFNγ": CORAL,
    "co-culture → IFNγ": PURPLE,
}
PREDICTOR_LABELS = {
    "mean_matching": "matching mean",
    "strong_linear": "Ahlmann–Eltze",
    "slim_string": "SLIM",
}
PREDICTOR_COLORS = {"mean_matching": BLUE, "strong_linear": PURPLE, "slim_string": GOLD}


def style() -> None:
    mpl.rcParams.update({
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=10, fontweight="bold", color=NAVY, va="bottom")
    ax.set_title(title, loc="left", pad=7, color=NAVY, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0)
    outputs: list[str] = []
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600}), ("tiff", {"dpi": 600})):
        output = path.with_suffix(f".{suffix}")
        fig.savefig(output, bbox_inches="tight", **kwargs)
        outputs.append(output.relative_to(ROOT).as_posix())
    plt.close(fig)
    return outputs


def _primary_pair_predictions(left: str, right: str) -> pd.DataFrame:
    predictions = pd.read_csv(artifact_paths(ROOT, 20260907)[0])
    selected = predictions.loc[
        predictions["scenario"].eq("in_domain")
        & predictions["split_seed"].eq(20260907)
        & predictions["predictor"].eq("strong_linear")
        & predictions["environment_key"].isin([left, right]),
        ["environment_key", "perturbation_label", "continuous_risk"],
    ].copy()
    left_frame = selected.loc[selected["environment_key"].eq(left)].rename(columns={"continuous_risk": "risk_left"})
    right_frame = selected.loc[selected["environment_key"].eq(right)].rename(columns={"continuous_risk": "risk_right"})
    merged = left_frame.merge(right_frame, on="perturbation_label", how="inner", validate="one_to_one")
    merged["rank_left"] = rankdata(merged["risk_left"].to_numpy(dtype=float), method="average")
    merged["rank_right"] = rankdata(merged["risk_right"].to_numpy(dtype=float), method="average")
    return merged.sort_values("perturbation_label", kind="stable").reset_index(drop=True)


def make_figure() -> tuple[list[str], dict[str, object]]:
    reordering = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_reliability_reordering.csv")
    perturbations = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_reliability_reordering_perturbations.csv")
    recalibration = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_recalibration_counterfactual.csv")
    reordering = reordering.loc[reordering["method"].eq("ptl_rf") & reordering["predictor"].eq("strong_linear")].copy()
    perturbations = perturbations.loc[perturbations["method"].eq("ptl_rf") & perturbations["predictor"].eq("strong_linear")].copy()

    fig = plt.figure(figsize=(7.25, 6.35))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.0, 1.05], hspace=0.55, wspace=0.40)

    ax = fig.add_subplot(grid[0, 0])
    panel(ax, "a", "Biology changes which perturbations are risky")
    pairs = list(PAIR_LABELS.values())
    x = np.arange(len(pairs))
    for index, pair_label in enumerate(pairs):
        pair_rows = reordering.loc[
            reordering.apply(lambda row: PAIR_LABELS.get((row["left_environment_id"], row["right_environment_id"])) == pair_label, axis=1)
        ]
        values = pair_rows["risk_inversion_rate"].to_numpy(dtype=float)
        ax.scatter(np.full(len(values), index), values, s=18, color=PAIR_COLORS[pair_label], alpha=0.65, edgecolor="white", linewidth=0.35, zorder=3)
        ax.plot([index - 0.16, index + 0.16], [np.mean(values), np.mean(values)], color=NAVY, lw=2.2, solid_capstyle="round", zorder=4)
        ax.text(index, np.mean(values) + 0.035, f"{np.mean(values):.2f}", ha="center", va="bottom", fontsize=6.5, color=NAVY)
    ax.set_xticks(x, pairs, rotation=22, ha="right", fontsize=5.7)
    ax.set_ylabel("risk-order inversion rate")
    ax.set_ylim(0, 0.55)
    ax.text(0.02, 0.96, "PTL · 5 split seeds · 10–11 shared labels", transform=ax.transAxes, va="top", fontsize=5.8, color=SLATE)

    ax = fig.add_subplot(grid[0, 1])
    panel(ax, "b", "Matched perturbations cross risk ranks")
    left, right = "frangieh_melanoma_control", "frangieh_melanoma_ifng"
    matched = _primary_pair_predictions(left, right)
    for _, row in matched.iterrows():
        displacement = row["rank_right"] - row["rank_left"]
        color = CORAL if displacement > 0 else BLUE if displacement < 0 else "#B7C5CB"
        alpha = 0.50 if displacement != 0 else 0.22
        ax.plot([0, 1], [row["rank_left"], row["rank_right"]], color=color, alpha=alpha, lw=0.95 if displacement != 0 else 0.65)
    ax.set_xticks([0, 1], ["control", "IFNγ"])
    ax.set_ylabel("within-condition risk rank")
    ax.invert_yaxis()
    ax.set_ylim(len(matched) + 1, 0)
    ax.text(0.02, 0.96, f"Ahlmann–Eltze · n={len(matched)} · held-out risk", transform=ax.transAxes, va="top", fontsize=5.8, color=SLATE)
    ax.legend(handles=[
        mpl.lines.Line2D([], [], color=CORAL, lw=1.5, label="higher risk rank"),
        mpl.lines.Line2D([], [], color=BLUE, lw=1.5, label="lower risk rank"),
    ], fontsize=5.5, loc="lower right")

    ax = fig.add_subplot(grid[1, 0])
    panel(ax, "c", "Response programs shift")
    for pair_label, pair_frame in perturbations.groupby(
        perturbations.apply(lambda row: PAIR_LABELS.get((row["left_environment_id"], row["right_environment_id"])), axis=1),
        sort=False,
    ):
        pair_frame = pair_frame.dropna(subset=["response_program_shift", "absolute_risk_change"])
        ax.scatter(pair_frame["response_program_shift"], pair_frame["absolute_risk_change"], s=13, alpha=0.45, color=PAIR_COLORS[pair_label], label=pair_label, edgecolor="white", linewidth=0.25)
        rho = spearmanr(pair_frame["response_program_shift"], pair_frame["absolute_risk_change"]).statistic
        ax.text(0.03, 0.96 - 0.10 * list(PAIR_LABELS.values()).index(pair_label), f"{pair_label}: ρ={rho:+.2f}", transform=ax.transAxes, fontsize=5.6, color=PAIR_COLORS[pair_label], va="top")
    ax.set_xlabel("response-program shift (1 − cosine)")
    ax.set_ylabel("absolute risk change")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.text(0.03, 0.02, "descriptive response vectors; no causal claim", transform=ax.transAxes, fontsize=5.7, color=SLATE)

    ax = fig.add_subplot(grid[1, 1])
    panel(ax, "d", "Calibration cannot fix rank")
    predictors = ["mean_matching", "strong_linear", "slim_string"]
    positions = np.arange(len(predictors))
    for position, predictor in zip(positions, predictors):
        values = recalibration.loc[recalibration["predictor"].eq(predictor), "platt_delta_brier"].dropna().to_numpy(dtype=float)
        jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) else np.array([])
        ax.scatter(np.full(len(values), position) + jitter, values, s=16, alpha=0.65, color=PREDICTOR_COLORS[predictor], edgecolor="white", linewidth=0.3)
        ax.plot([position - 0.17, position + 0.17], [np.mean(values), np.mean(values)], color=NAVY, lw=2.0, solid_capstyle="round")
    ax.axhline(0, color=NAVY, lw=0.75)
    ax.set_xticks(positions, [PREDICTOR_LABELS[p] for p in predictors], rotation=18, ha="right", fontsize=5.8)
    ax.set_ylabel("Platt Δ Brier vs base score")
    ax.text(0.02, 0.96, "45 groups · ΔAURC = 0 · order disagreements = 0", transform=ax.transAxes, va="top", fontsize=5.8, color=SLATE)
    ax.text(0.02, 0.04, "negative = better probability calibration", transform=ax.transAxes, fontsize=5.7, color=SLATE)

    fig.suptitle("Biological context rewrites the reliability ordering", x=0.03, y=1.015, ha="left", fontsize=12, fontweight="bold", color=NAVY)
    outputs = save_figure(fig, ROOT / "results/figures/iclr_formal/formal_fig1_reliability_reordering")
    manifest = {
        "schema_version": 1,
        "figure": "formal_fig1_reliability_reordering",
        "claim": "matched biological conditions can reorder perturbation reliability; monotonic calibration cannot repair strict ranking",
        "status": "first_stage_frangieh_diagnostic",
        "source_data": [
            "artifacts/manifests/formal_v2_reliability_reordering.csv",
            "artifacts/manifests/formal_v2_reliability_reordering_perturbations.csv",
            "artifacts/manifests/formal_v2_recalibration_counterfactual.csv",
        ],
        "outputs": outputs,
    }
    (ROOT / "artifacts/manifests/formal_v2_reliability_reordering_figure.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return outputs, manifest


def main() -> int:
    style()
    outputs, manifest = make_figure()
    print(json.dumps({"outputs": outputs, "manifest": manifest}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
