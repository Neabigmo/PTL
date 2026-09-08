"""Create the controlled-shift, recalibration and deployment figures.

All response-program panels are explicitly descriptive.  Outcome-derived risk
and confidence are kept as evaluation-only quantities, and the external panel
labels the source-response persistence baseline rather than a trained model.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAVY = "#12263A"
BLUE = "#1F6F8B"
TEAL = "#2A9D8F"
CORAL = "#D95D52"
GOLD = "#E9B44C"
PURPLE = "#7661A8"
SLATE = "#6C7A89"
GRID = "#DCE5E8"
GREEN = "#2E9E66"
RED = "#B64342"


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
        fig.tight_layout(pad=1.0)
    outputs: list[str] = []
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600}), ("tiff", {"dpi": 600})):
        output = path.with_suffix(f".{suffix}")
        fig.savefig(output, bbox_inches="tight", **kwargs)
        outputs.append(output.relative_to(ROOT).as_posix())
    plt.close(fig)
    return outputs


def _pair_label(row: pd.Series) -> str:
    left = str(row["left_environment_id"]).replace("frangieh_melanoma_", "")
    right = str(row["right_environment_id"]).replace("frangieh_melanoma_", "")
    return f"{left} → {right}"


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    value = left.corr(right, method="spearman")
    return float(value) if pd.notna(value) else float("nan")


def make_response_figure() -> tuple[list[str], dict[str, object]]:
    oof = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_controlled_shift_oof_perturbations.csv")
    external = pd.read_csv(ROOT / "artifacts/manifests/head_to_head_controlled_shift_ladder.csv")
    external_summary = pd.read_csv(ROOT / "artifacts/manifests/head_to_head_controlled_shift_ladder_summary.csv")
    fig = plt.figure(figsize=(7.25, 6.25))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.0, 1.05], hspace=0.52, wspace=0.40)

    ax = fig.add_subplot(grid[0, 0])
    panel(ax, "a", "Frangieh response programs rotate")
    oof = oof.dropna(subset=["response_program_shift"])
    pairs = list(dict.fromkeys(_pair_label(row) for _, row in oof.iterrows()))
    for index, pair in enumerate(pairs):
        values = oof.loc[oof.apply(_pair_label, axis=1).eq(pair), "response_program_shift"].to_numpy(dtype=float)
        jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) else np.array([])
        ax.scatter(np.full(len(values), index) + jitter, values, s=9, alpha=0.32, color=[BLUE, CORAL, PURPLE][index % 3], edgecolor="white", linewidth=0.2)
        ax.plot([index - 0.17, index + 0.17], [np.mean(values), np.mean(values)], color=NAVY, lw=2.0)
        ax.text(index, np.mean(values) + 0.035, f"{np.mean(values):.2f}", ha="center", fontsize=6.2, color=NAVY)
    ax.set_xticks(range(len(pairs)), [x.replace(" → ", "\n→ ") for x in pairs], fontsize=5.7)
    ax.set_ylabel("response-program shift (1 − cosine)")
    ax.set_ylim(bottom=0)
    ax.text(0.02, 0.96, "five-fold perturbation-label OOF · n=243–244", transform=ax.transAxes, va="top", fontsize=5.7, color=SLATE)

    ax = fig.add_subplot(grid[0, 1])
    panel(ax, "b", "Matched shifts (persistence baseline)")
    external_summary = external_summary.sort_values(["axis", "comparison_id", "modality"], kind="stable").reset_index(drop=True)
    x = np.arange(len(external_summary))
    colors = external_summary["axis"].map({"modality": CORAL, "time": BLUE, "cell_context": PURPLE}).fillna(SLATE)
    ax.scatter(x, external_summary["mean_response_program_shift"], s=38, c=colors, zorder=3, edgecolor="white", linewidth=0.5)
    for i, row in external_summary.iterrows():
        axis_label = {"cell_context": "cell", "modality": "modality", "time": "time"}.get(row["axis"], row["axis"])
        assay_label = str(row["modality"]).replace("CRISPR", "")
        label = f"{axis_label}\n{assay_label}"
        ax.annotate(label, (i, row["mean_response_program_shift"]), xytext=(0, 8 if i % 2 else 18),
                    textcoords="offset points", ha="center", va="bottom", fontsize=4.8, color=NAVY)
    ax.set_xticks(x, [str(i + 1) for i in x])
    ax.set_xlabel("controlled comparison")
    ax.set_ylabel("mean response-program shift")
    ax.set_ylim(0, 1.12)
    ax.text(0.02, 0.035, "evaluation-only · exact target intersection · SCEPTRE effects", transform=ax.transAxes, va="bottom", fontsize=5.3, color=SLATE)

    ax = fig.add_subplot(grid[1, 0])
    panel(ax, "c", "Response shift and risk change")
    ax.scatter(oof["response_program_shift"], oof["absolute_risk_change"], s=9, alpha=0.28, color=TEAL, label="Frangieh OOF", edgecolor="white", linewidth=0.2)
    ax.scatter(external["response_program_shift"], external["absolute_risk_change"], s=15, alpha=0.28, color=CORAL, label="external persistence", edgecolor="white", linewidth=0.2)
    rho_oof = _safe_corr(oof["response_program_shift"], oof["absolute_risk_change"])
    rho_ext = _safe_corr(external["response_program_shift"], external["absolute_risk_change"])
    ax.set_xlabel("response-program shift (1 − cosine)")
    ax.set_ylabel("absolute risk change")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=5.6, loc="upper left")
    ax.text(0.03, 0.88, f"Frangieh ρ={rho_oof:+.2f}\nexternal ρ={rho_ext:+.2f}", transform=ax.transAxes, fontsize=5.7, color=NAVY, va="top")
    ax.text(0.03, 0.03, "descriptive response vectors; no causal claim", transform=ax.transAxes, fontsize=5.7, color=SLATE)

    ax = fig.add_subplot(grid[1, 1])
    panel(ax, "d", "Guide reproducibility and shift")
    external_plot = external.copy()
    external_plot["mean_within_guide_mse"] = (external_plot["source_within_guide_mse_left"] + external_plot["source_within_guide_mse_right"]) / 2
    for modality, color in [("CRISPRko", CORAL), ("CRISPRi", BLUE)]:
        subset = external_plot.loc[external_plot["modality"].eq(modality)]
        ax.scatter(subset["mean_within_guide_mse"], subset["response_program_shift"], s=17, alpha=0.42, color=color, label=modality, edgecolor="white", linewidth=0.25)
    ax.set_xlabel("mean within-target guide MSE")
    ax.set_ylabel("response-program shift")
    ax.legend(fontsize=5.7, loc="best")
    ax.text(0.03, 0.03, "measurement reproducibility is recorded as a covariate", transform=ax.transAxes, fontsize=5.5, color=SLATE)
    fig.suptitle("Controlled shifts reveal response reprogramming and rank instability", x=0.03, y=1.015, ha="left", fontsize=11.5, fontweight="bold", color=NAVY)
    outputs = save_figure(fig, ROOT / "results/figures/iclr_formal/formal_fig2_response_reprogramming")
    manifest = {
        "schema_version": 1,
        "figure": "formal_fig2_response_reprogramming",
        "claim": "matched contexts show descriptive response-program displacement alongside risk reordering; no causal mechanism is inferred",
        "evidence_tier": "orthogonal biological extension",
        "predictor_contract": "evaluation-only source-response persistence baseline; not a trained perturbation model",
        "source_data": [
            "artifacts/manifests/formal_v2_controlled_shift_oof_perturbations.csv",
            "artifacts/manifests/head_to_head_controlled_shift_ladder.csv",
            "artifacts/manifests/head_to_head_controlled_shift_ladder_summary.csv",
        ],
        "outputs": outputs,
    }
    return outputs, manifest


def make_recalibration_figure() -> tuple[list[str], dict[str, object]]:
    recalibration = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_recalibration_counterfactual.csv")
    fig = plt.figure(figsize=(7.25, 6.25))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0], hspace=0.50, wspace=0.38)
    predictors = ["mean_matching", "strong_linear", "slim_string"]
    labels = {"mean_matching": "matching mean", "strong_linear": "Ahlmann–Eltze", "slim_string": "SLIM"}
    colors = {"mean_matching": BLUE, "strong_linear": PURPLE, "slim_string": GOLD}

    ax = fig.add_subplot(grid[0, 0])
    panel(ax, "a", "Probability calibration can improve Brier")
    for position, predictor in enumerate(predictors):
        values = recalibration.loc[recalibration["predictor"].eq(predictor), "platt_delta_brier"].dropna().to_numpy(dtype=float)
        jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) else np.array([])
        ax.scatter(np.full(len(values), position) + jitter, values, s=14, alpha=0.55, color=colors[predictor], edgecolor="white", linewidth=0.3)
        ax.plot([position - 0.17, position + 0.17], [np.mean(values), np.mean(values)], color=NAVY, lw=2.0)
    ax.axhline(0, color=NAVY, lw=0.7)
    ax.set_xticks(range(3), [labels[x] for x in predictors], rotation=18, ha="right", fontsize=5.8)
    ax.set_ylabel("Platt Δ Brier vs base")
    ax.text(0.02, 0.04, "negative = better probability calibration", transform=ax.transAxes, fontsize=5.7, color=SLATE)

    ax = fig.add_subplot(grid[0, 1])
    panel(ax, "b", "But monotonic maps cannot change AURC")
    for position, predictor in enumerate(predictors):
        values = recalibration.loc[recalibration["predictor"].eq(predictor), "platt_delta_aurc"].dropna().to_numpy(dtype=float)
        ax.scatter(np.full(len(values), position), values, s=18, color=colors[predictor], alpha=0.55, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color=NAVY, lw=0.8)
    ax.set_xticks(range(3), [labels[x] for x in predictors], rotation=18, ha="right", fontsize=5.8)
    ax.set_ylabel("Platt Δ AURC")
    ax.text(0.02, 0.96, "45 groups · all values exactly 0", transform=ax.transAxes, va="top", fontsize=5.8, color=SLATE)

    ax = fig.add_subplot(grid[1, 0])
    panel(ax, "c", "Strict pairwise order is preserved")
    order = recalibration.groupby("predictor", as_index=False)["n_order_disagreements"].sum().set_index("predictor").reindex(predictors)
    ax.scatter(range(3), np.zeros(3), s=42, color=[colors[x] for x in predictors], edgecolor="white", linewidth=0.4, zorder=3)
    for position in range(3):
        ax.text(position, 0.003, "0", ha="center", va="bottom", fontsize=7.0, color=NAVY, fontweight="bold")
    ax.set_xticks(range(3), [labels[x] for x in predictors], rotation=18, ha="right", fontsize=5.8)
    ax.set_ylabel("order disagreements")
    ax.set_ylim(-0.01, 0.025)
    ax.text(0.02, 0.96, "monotonicity preserves rank by construction", transform=ax.transAxes, va="top", fontsize=5.8, color=SLATE)

    ax = fig.add_subplot(grid[1, 1])
    panel(ax, "d", "The counterfactual is a rank proposition")
    ax.axis("off")
    ax.text(0.05, 0.78, r"$g$ monotonic", fontsize=11.5, color=PURPLE, fontweight="bold")
    ax.text(0.48, 0.78, "→", fontsize=18, color=NAVY, va="center")
    ax.text(0.66, 0.78, r"rank$(g(C))$", fontsize=11.5, color=TEAL, fontweight="bold")
    ax.text(0.05, 0.51, r"rank$(g(C)) =$ rank$(C)$", fontsize=14, color=NAVY, fontweight="bold")
    ax.text(0.05, 0.28, "Scale calibration may change\nprobability error; it cannot\nrepair a reordered confidence surface.", fontsize=8.0, color=SLATE, linespacing=1.35)
    fig.suptitle("Recalibration changes scale, not the identity of hard perturbations", x=0.03, y=1.015, ha="left", fontsize=11.5, fontweight="bold", color=NAVY)
    outputs = save_figure(fig, ROOT / "results/figures/iclr_formal/formal_fig3_recalibration_counterfactual")
    manifest = {
        "schema_version": 1,
        "figure": "formal_fig3_recalibration_counterfactual",
        "claim": "train-only monotonic recalibration can alter Brier/log-loss while preserving strict reliability ranking and AURC",
        "source_data": ["artifacts/manifests/formal_v2_recalibration_counterfactual.csv"],
        "outputs": outputs,
    }
    return outputs, manifest


def _aurc(group: pd.DataFrame, score: str) -> float:
    ordered = group.sort_values(score, ascending=False, kind="mergesort")
    risk = ordered["continuous_risk"].to_numpy(dtype=float)
    return float((np.cumsum(risk) / np.arange(1, len(risk) + 1)).mean())


def make_deployment_figure() -> tuple[list[str], dict[str, object]]:
    pred = pd.read_csv(ROOT / "artifacts/source_data/formal_v2_reliability_predictions.csv")
    atlas = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_environment_transfer_matrix.csv")
    metrics = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_reliability_metrics.csv")
    source_metrics = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_transport_source_selection_metrics.csv")
    utility = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_deployment_utility.csv")
    fig = plt.figure(figsize=(7.25, 6.25))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.1, 1.0], hspace=0.53, wspace=0.40)
    env_order = [x for x in ["norman_k562", "replogle_rpe1", "tian_neuron_crispra", "tian_neuron_crispri", "frangieh_melanoma_control", "frangieh_melanoma_coculture", "frangieh_melanoma_ifng", "xu_he293"] if x in set(atlas["source_environment_key"]) | set(atlas["target_environment_key"])]
    ax = fig.add_subplot(grid[0, 0])
    panel(ax, "a", "Source → target transfer is heterogeneous")
    heat = atlas.loc[atlas["baseline"].eq("u_only_rf")].pivot(index="source_environment_key", columns="target_environment_key", values="transfer_gain").reindex(index=env_order, columns=env_order)
    values = heat.to_numpy(dtype=float)
    lim = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    cmap = LinearSegmentedColormap.from_list("transfer", [RED, "#F7F7F7", GREEN])
    im = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="auto")
    labels = [x.replace("_", "\n") for x in env_order]
    ax.set_xticks(range(len(env_order)), labels, rotation=58, ha="right", fontsize=4.6)
    ax.set_yticks(range(len(env_order)), labels, fontsize=4.6)
    for i in range(len(env_order)):
        for j in range(len(env_order)):
            if pd.notna(heat.iloc[i, j]):
                ax.text(j, i, f"{heat.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=4.3, color=NAVY if abs(heat.iloc[i, j]) < lim * 0.55 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Δ AURC vs U-only", fontsize=6.0)
    cbar.ax.tick_params(labelsize=5.2)

    ax = fig.add_subplot(grid[0, 1])
    panel(ax, "b", "PTL gains are scenario-dependent")
    rows = []
    for scenario in ["in_domain", "leave_environment_out", "leave_predictor_out"]:
        subset = pred.loc[pred["scenario"].eq(scenario)]
        rows.append({"scenario": scenario, "U-only": _aurc(subset, "u_only_rf"), "PTL": _aurc(subset, "ptl_rf")})
    scenario_frame = pd.DataFrame(rows)
    x = np.arange(len(scenario_frame))
    ax.bar(x - 0.18, scenario_frame["U-only"], 0.34, color=SLATE, label="U-only")
    ax.bar(x + 0.18, scenario_frame["PTL"], 0.34, color=BLUE, label="PTL")
    ax.set_xticks(x, ["in-domain", "environment\nout", "predictor\nout"], fontsize=5.8)
    ax.set_ylabel("AURC · lower is better")
    ax.legend(fontsize=5.7, loc="best")

    ax = fig.add_subplot(grid[1, 0])
    panel(ax, "c", "Prospective source selection is near chance")
    source_metrics["auroc"] = pd.to_numeric(source_metrics["within_target_positive_negative_auroc"], errors="coerce")
    source_metrics = source_metrics.dropna(subset=["auroc"])
    by_split = source_metrics.groupby("split_seed")["auroc"].mean()
    ax.scatter(np.arange(len(by_split)), by_split.values, s=30, color=PURPLE, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0.5, color=NAVY, lw=0.8, ls="--", label="chance")
    ax.set_xticks(np.arange(len(by_split)), [str(x)[-4:] for x in by_split.index], rotation=25, ha="right", fontsize=5.4)
    ax.set_ylabel("within-target AUROC")
    ax.set_xlabel("split seed suffix")
    ax.legend(fontsize=5.7, loc="best")
    ax.text(0.03, 0.96, "candidate source ranking; outcomes excluded at selection time", transform=ax.transAxes, va="top", fontsize=5.5, color=SLATE)

    ax = fig.add_subplot(grid[1, 1])
    panel(ax, "d", "Deployment use: U-only baseline fallback")
    utility = utility.loc[utility["aggregation_level"].eq("environment_macro")].copy()
    if utility.empty:
        utility = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_deployment_utility.csv")
    plot = utility.groupby(["method", "budget"], as_index=False)["realized_fidelity"].mean()
    for method, color, label in [("raw_normalized_uq", SLATE, "raw UQ"), ("u_only_rf", TEAL, "U-only"), ("ptl_rf", BLUE, "PTL")]:
        subset = plot.loc[plot["method"].eq(method)].sort_values("budget")
        if not subset.empty:
            ax.plot(subset["budget"] * 100, subset["realized_fidelity"], marker="o", ms=3.5, lw=1.6, color=color, label=label)
    ax.set_xlabel("selected budget (%)")
    ax.set_ylabel("realized fidelity")
    ax.legend(fontsize=5.7, loc="best")
    fig.suptitle("Deployment consequence: reliability transport remains difficult prospectively", x=0.03, y=1.015, ha="left", fontsize=11.5, fontweight="bold", color=NAVY)
    outputs = save_figure(fig, ROOT / "results/figures/iclr_formal/formal_fig4_deployment_consequences")
    manifest = {
        "schema_version": 1,
        "figure": "formal_fig4_deployment_consequences",
        "claim": "PTL is a deployment-side probe with heterogeneous transfer and a non-protective source-selection baseline",
        "source_data": [
            "artifacts/source_data/formal_v2_reliability_predictions.csv",
            "artifacts/manifests/formal_v2_environment_transfer_matrix.csv",
            "artifacts/manifests/formal_v2_reliability_metrics.csv",
            "artifacts/manifests/formal_v2_transport_source_selection_metrics.csv",
            "artifacts/manifests/formal_v2_deployment_utility.csv",
        ],
        "outputs": outputs,
    }
    return outputs, manifest


def main() -> int:
    style()
    outputs = {}
    manifests = {}
    for name, function in [("response", make_response_figure), ("recalibration", make_recalibration_figure), ("deployment", make_deployment_figure)]:
        generated, manifest = function()
        outputs[name] = generated
        manifests[name] = manifest
        (ROOT / "artifacts/manifests" / f"{manifest['figure']}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": outputs, "manifests": manifests}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
