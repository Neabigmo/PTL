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


def make_measurement_figure() -> tuple[list[str], dict[str, object]]:
    """Visualize the raw-cell lock and its real minimum-cell sensitivity."""

    primary = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_summary.csv")
    sensitivity = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_claim_lock_measurement_summary.csv")
    metrics = ["delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement"]
    metric_labels = {metrics[0]: "delta cosine", metrics[1]: "Systema centroid", metrics[2]: "absolute-effect rank"}
    metric_colors = {metrics[0]: CORAL, metrics[1]: BLUE, metrics[2]: PURPLE}

    fig = plt.figure(figsize=(7.25, 6.25))
    grid = fig.add_gridspec(2, 2, width_ratios=[0.95, 1.05], height_ratios=[0.95, 1.05], hspace=0.56, wspace=0.42)

    ax = fig.add_subplot(grid[0, 0])
    panel(ax, "a", "The lock separates signal from cell noise")
    ax.axis("off")
    boxes = [
        (0.03, 0.57, 0.25, 0.23, "frozen\npredictor", BLUE),
        (0.37, 0.57, 0.25, 0.23, "raw-cell\nhalves", TEAL),
        (0.71, 0.57, 0.25, 0.23, "cross vs\nnoise floors", CORAL),
    ]
    for x, y, width, height, label, color in boxes:
        ax.add_patch(mpl.patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor=color, edgecolor="white", linewidth=1.0, alpha=0.92, transform=ax.transAxes))
        ax.text(x + width / 2, y + height / 2, label, transform=ax.transAxes, ha="center", va="center", color="white", fontsize=6.4, fontweight="bold")
    for x in (0.285, 0.625):
        ax.add_patch(mpl.patches.FancyArrowPatch((x, 0.685), (x + 0.07, 0.685), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=12, linewidth=1.5, color=NAVY))
    ax.text(0.5, 0.30, "30 split seeds · same 243-label contract\nfull-size with-replacement pseudo-replicates\ncontrols resampled independently", transform=ax.transAxes, ha="center", va="center", fontsize=7.0, color=NAVY, linespacing=1.45)
    ax.text(0.5, 0.08, "positive Δjoint would require cross-context displacement\nto exceed both measured components", transform=ax.transAxes, ha="center", va="center", fontsize=6.0, color=SLATE)

    ax = fig.add_subplot(grid[0, 1])
    panel(ax, "b", "Split-half sensitivity uses the same labels")
    pair_order = list(dict.fromkeys(zip(primary["left_target_environment_id"], primary["right_target_environment_id"])))
    pair_labels = [f"{left.replace('frangieh_melanoma_', '')}\n→ {right.replace('frangieh_melanoma_', '')}" for left, right in pair_order]
    def count_for(frame: pd.DataFrame, pair: tuple[str, str]) -> float:
        rows = frame.loc[frame["left_target_environment_id"].eq(pair[0]) & frame["right_target_environment_id"].eq(pair[1])]
        return float(rows["n_perturbations_primary_min"].mean())
    x = np.arange(len(pair_order))
    primary_counts = [count_for(primary, pair) for pair in pair_order]
    sensitivity_counts = [count_for(sensitivity, pair) for pair in pair_order]
    ax.bar(x - 0.18, primary_counts, 0.34, color=SLATE, label="full-size primary")
    ax.bar(x + 0.18, sensitivity_counts, 0.34, color=GOLD, label="split-half sensitivity")
    for values, offset in ((primary_counts, -0.18), (sensitivity_counts, 0.18)):
        for pos, value in zip(x, values):
            ax.text(pos + offset, value + 2, f"{value:.0f}", ha="center", fontsize=5.8, color=NAVY)
    ax.set_xticks(x, pair_labels, fontsize=5.8)
    ax.set_ylabel("matched perturbations")
    ax.set_ylim(0, max(primary_counts + sensitivity_counts) * 1.18)
    ax.legend(fontsize=5.7, loc="lower left")
    ax.text(0.02, 0.96, "same raw cells, folds, metrics and bootstrap policy", transform=ax.transAxes, va="top", fontsize=5.5, color=SLATE)

    ax = fig.add_subplot(grid[1, 0])
    panel(ax, "c", "Metric-specific excess can differ")
    for metric_index, metric in enumerate(metrics):
        values = primary.loc[primary["metric"].eq(metric), "delta_joint"].to_numpy(dtype=float)
        jitter = np.linspace(-0.16, 0.16, len(values)) if len(values) else np.array([])
        ax.scatter(np.full(len(values), metric_index) + jitter, values, s=12, alpha=0.34, color=metric_colors[metric], edgecolor="white", linewidth=0.25)
        ax.plot([metric_index - 0.19, metric_index + 0.19], [np.mean(values), np.mean(values)], color=NAVY, lw=2.0, solid_capstyle="round")
        ax.text(metric_index, np.mean(values) + 0.014, f"macro {np.mean(values):+.3f}", ha="center", fontsize=5.7, color=NAVY)
    ax.axhline(0, color=NAVY, lw=0.8)
    ax.set_xticks(range(len(metrics)), [metric_labels[m] for m in metrics], rotation=18, ha="right", fontsize=5.7)
    ax.set_ylabel("Δjoint")
    ax.set_ylim(-0.10, 0.10)
    ax.text(0.02, 0.04, "each point = source context × target pair", transform=ax.transAxes, fontsize=5.5, color=SLATE)

    ax = fig.add_subplot(grid[1, 1])
    panel(ax, "d", "Primary and split-half floors are separated")
    grouped = []
    for metric in metrics:
        for label, frame, color, offset in [("full-size primary", primary, SLATE, -0.13), ("split-half sensitivity", sensitivity, GOLD, 0.13)]:
            rows = frame.loc[frame["metric"].eq(metric)]
            grouped.append((metric, label, float(rows["delta_joint"].mean()), float(rows["delta_joint_ci_low"].mean()), float(rows["delta_joint_ci_high"].mean()), color, offset))
    for metric_index, metric in enumerate(metrics):
        for current_metric, label, value, low, high, color, offset in grouped:
            if current_metric != metric:
                continue
            ax.errorbar(metric_index + offset, value, yerr=[[value - low], [high - value]], fmt="o", ms=5.0, color=color, ecolor=color, elinewidth=1.0, capsize=2.0, markeredgecolor="white", markeredgewidth=0.4, label=label if metric_index == 0 else None)
    ax.axhline(0, color=NAVY, lw=0.8)
    ax.set_xticks(range(len(metrics)), [metric_labels[m] for m in metrics], rotation=18, ha="right", fontsize=5.7)
    ax.set_ylabel("macro Δjoint with 95% CI means")
    ax.set_ylim(-0.10, 0.10)
    ax.legend(fontsize=5.7, loc="lower left")
    ax.text(0.02, 0.96, "full-size primary; original split-half audit as sensitivity", transform=ax.transAxes, va="top", fontsize=5.5, color=SLATE)

    fig.suptitle("Raw-cell measurement lock: corrected excess is estimated above the full-size floor", x=0.03, y=1.015, ha="left", fontsize=11.5, fontweight="bold", color=NAVY)
    outputs = save_figure(fig, ROOT / "results/figures/iclr_formal/formal_fig2_measurement_reliability")
    manifest = {
        "schema_version": 1,
        "figure": "formal_fig2_measurement_reliability",
        "claim": "matched raw-cell measurement and joint floors are explicitly separated; full-size nonparametric resampling is primary and the original split-half audit is sensitivity evidence",
        "source_data": [
            "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_summary.csv",
            "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_ordering.csv",
            "artifacts/manifests/formal_v2_claim_lock_measurement_summary.csv",
            "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize.json",
        ],
        "outputs": outputs,
    }
    return outputs, manifest


def make_replication_boundary_figure() -> tuple[list[str], dict[str, object]]:
    """Show the outcome-blind Nadig replication and its measurement boundary."""

    registry = pd.read_csv(ROOT / "artifacts/manifests/reordering_replication_candidate_registry.csv")
    report = json.loads((ROOT / "artifacts/manifests/reordering_replication_candidate_registry.json").read_text(encoding="utf-8"))
    fullsize_summary_path = ROOT / "artifacts/manifests/formal_v2_claim_lock_replication_nadig_fullsize.csv"
    split_half_summary_path = ROOT / "artifacts/manifests/formal_v2_claim_lock_replication_nadig.csv"
    summary_path = fullsize_summary_path if fullsize_summary_path.is_file() else split_half_summary_path
    summary = pd.read_csv(summary_path)
    execution_report_path = summary_path.with_suffix(".json")
    execution_report = json.loads(execution_report_path.read_text(encoding="utf-8")) if execution_report_path.is_file() else {}
    summary["analysis_label"] = summary["analysis_label"].astype(str)
    summary["metric"] = summary["metric"].astype(str)
    metric_order = ["delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement"]
    metric_labels = {
        "delta_cosine": "delta cosine",
        "systema_centroid_accuracy": "Systema",
        "absolute_effect_rank_agreement": "effect-rank",
    }
    metric_colors = {"delta_cosine": BLUE, "systema_centroid_accuracy": TEAL, "absolute_effect_rank_agreement": PURPLE}
    fig = plt.figure(figsize=(7.25, 6.10))
    grid = fig.add_gridspec(2, 2, width_ratios=[0.95, 1.05], height_ratios=[1.0, 1.05], hspace=0.53, wspace=0.42)

    ax = fig.add_subplot(grid[0, 0])
    panel(ax, "a", "Outcome-blind selection precedes risk")
    ax.axis("off")
    stages = [(0.06, "metadata\nscan", f"{report['candidate_count']} pairs", BLUE), (0.37, "eligibility\ngate", f"{report['eligible_candidate_count']} pass", TEAL), (0.68, "Claim Lock\nrisk", "Nadig · once", CORAL)]
    for index, (x, title, detail, color) in enumerate(stages):
        ax.add_patch(mpl.patches.FancyBboxPatch((x, 0.50), 0.24, 0.25, boxstyle="round,pad=0.02,rounding_size=0.025", facecolor=color, edgecolor="white", linewidth=1.0, transform=ax.transAxes))
        ax.text(x + 0.12, 0.64, title, transform=ax.transAxes, ha="center", va="center", color="white", fontsize=7.0, fontweight="bold")
        ax.text(x + 0.12, 0.53, detail, transform=ax.transAxes, ha="center", va="center", color="white", fontsize=6.0)
        if index < len(stages) - 1:
            ax.add_patch(mpl.patches.FancyArrowPatch((x + 0.25, 0.625), (x + 0.30, 0.625), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=11, linewidth=1.3, color=NAVY))
    ax.text(0.5, 0.22, "shared labels · controls · raw cells · modality · guide signatures\nplus explicit batch/context provenance", transform=ax.transAxes, ha="center", va="center", fontsize=6.7, color=NAVY, linespacing=1.45)
    ax.text(0.5, 0.07, "No target predictions, risks or inversions entered candidate selection.", transform=ax.transAxes, ha="center", va="center", fontsize=5.9, color=SLATE)

    ax = fig.add_subplot(grid[0, 1])
    panel(ax, "b", "Primary: full-size cross-context and floors" if fullsize_summary_path.is_file() else "Primary: cross-context ≈ joint")
    primary = summary.loc[summary["analysis_label"].eq("primary_min20")]
    means = primary.groupby("metric", sort=False)[["cross_d", "measurement_floor_d", "joint_floor_d"]].mean().reindex(metric_order)
    x = np.arange(len(metric_order))
    width = 0.22
    for offset, column, label, color in [(-width, "cross_d", "cross-context", CORAL), (0.0, "measurement_floor_d", "measurement", GOLD), (width, "joint_floor_d", "joint", TEAL)]:
        ax.bar(x + offset, means[column].to_numpy(dtype=float), width=width, label=label, color=color, alpha=0.90)
    ax.set_xticks(x, [metric_labels[m] for m in metric_order], fontsize=5.6)
    ax.set_ylabel("rank displacement / floor")
    ax.set_ylim(0, 0.40)
    ax.legend(fontsize=5.1, loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.8)
    shared_count = int(execution_report.get("common_perturbation_count", 2392))
    evaluated_count = int(primary["n_perturbations"].max())
    ax.text(0.02, 0.86, f"{shared_count:,} shared perturbations · {evaluated_count:,} labels evaluated", transform=ax.transAxes, va="top", fontsize=5.2, color=NAVY)
    ax.text(0.02, 0.02, "three metrics; macro mean across HepG2 and Jurkat source contexts", transform=ax.transAxes, va="bottom", fontsize=4.9, color=SLATE)

    ax = fig.add_subplot(grid[1, 0])
    if fullsize_summary_path.is_file():
        panel(ax, "c", "Full-size vs split-half sensitivity")
        split_half = pd.read_csv(split_half_summary_path)
        comparison = pd.concat([
            primary.assign(comparison_label="full-size primary"),
            split_half.loc[split_half["analysis_label"].eq("primary_min20")].assign(comparison_label="split-half sensitivity"),
        ], ignore_index=True)
        grouped = comparison.groupby(["comparison_label", "metric"], sort=False)["delta_joint"].mean().reset_index()
        lower = comparison.groupby(["comparison_label", "metric"], sort=False)["delta_joint_ci_low"].mean().reset_index(name="low")
        upper = comparison.groupby(["comparison_label", "metric"], sort=False)["delta_joint_ci_high"].mean().reset_index(name="high")
        grouped = grouped.merge(lower, on=["comparison_label", "metric"]).merge(upper, on=["comparison_label", "metric"])
        threshold_order = ["full-size primary", "split-half sensitivity"]
        positions = np.arange(len(threshold_order))
        x_labels = ["full-size\nprimary", "split-half\nsensitivity"]
    else:
        panel(ax, "c", "Nadig split-half sensitivity by cell threshold")
        sensitivity = summary.loc[summary["analysis_label"].isin(["primary_min20", "sensitivity_min40"])]
        grouped = sensitivity.groupby(["analysis_label", "metric"], sort=False)["delta_joint"].mean().reset_index()
        lower = sensitivity.groupby(["analysis_label", "metric"], sort=False)["delta_joint_ci_low"].mean().reset_index(name="low")
        upper = sensitivity.groupby(["analysis_label", "metric"], sort=False)["delta_joint_ci_high"].mean().reset_index(name="high")
        grouped = grouped.merge(lower, on=["analysis_label", "metric"]).merge(upper, on=["analysis_label", "metric"])
        threshold_order = ["primary_min20", "sensitivity_min40"]
        positions = np.arange(len(threshold_order))
        x_labels = ["min20\n(n=1,255)", "min40\n(n=345)"]
    offsets = np.linspace(-0.22, 0.22, len(metric_order))
    for offset, metric in zip(offsets, metric_order):
        label_column = "comparison_label" if fullsize_summary_path.is_file() else "analysis_label"
        rows = grouped.loc[grouped["metric"].eq(metric)].set_index(label_column).reindex(threshold_order)
        y = rows["delta_joint"].to_numpy(dtype=float)
        yerr = np.vstack([y - rows["low"].to_numpy(dtype=float), rows["high"].to_numpy(dtype=float) - y])
        ax.errorbar(positions + offset, y, yerr=yerr, fmt="o", ms=4.5, capsize=2.5, lw=1.0, color=metric_colors[metric], label=metric_labels[metric], zorder=3)
    ax.axhline(0, color=NAVY, lw=0.8)
    ax.set_xticks(positions, x_labels, fontsize=5.7)
    ax.set_ylabel("Δ joint floor")
    interval_values = pd.concat([grouped["low"], grouped["high"]], ignore_index=True).to_numpy(dtype=float)
    interval_limit = max(0.065, 1.20 * float(np.max(np.abs(interval_values))))
    ax.set_ylim(-interval_limit, interval_limit)
    ax.legend(fontsize=5.2, loc="upper right", ncol=1)
    ax.text(0.02, 0.04, "row-level bootstrap intervals are shown; no threshold is hidden", transform=ax.transAxes, fontsize=5.2, color=SLATE)

    ax = fig.add_subplot(grid[1, 1])
    panel(ax, "d", "Modern-model gates")
    feasibility_path = ROOT / "artifacts/manifests/modern_predictor_feasibility.json"
    if feasibility_path.is_file():
        feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
        model_rows = feasibility.get("rows", [])
        gate_names = ["official\nsource", "checkpoint", "source-only\ntraining", "frozen\nvector", "target\nblind", "Claim\nLock"]
        matrix = np.asarray([
            [
                int(bool(row.get("official_source_checkout"))),
                int(bool(row.get("official_checkpoint_verified"))),
                int(bool(row.get("source_only_training_verified"))),
                int(bool(row.get("heldout_prediction_vector_verified"))),
                int(bool(row.get("target_outcome_blind_verified"))),
                int(bool(row.get("claim_lock_eligible"))),
            ]
            for row in model_rows
        ], dtype=float)
        ax.imshow(matrix, cmap=mpl.colors.ListedColormap(["#F3F5F6", TEAL]), vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(gate_names)), gate_names, fontsize=4.8)
        ax.set_yticks(range(len(model_rows)), [str(row["predictor_id"]).replace("official_", "") for row in model_rows], fontsize=5.8)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, "✓" if matrix[i, j] else "—", ha="center", va="center", fontsize=7.0, color="white" if matrix[i, j] else SLATE)
        ax.set_title("No modern audit is silently promoted to Claim Lock", loc="left", pad=7, color=NAVY, fontweight="bold", fontsize=8.5)
        ax.text(0.02, -0.22, "TxPert: official K562 cached inference succeeds; scGPT: ABI-blocked;\nGEARS: external contract only.", transform=ax.transAxes, fontsize=5.5, color=SLATE, va="top")
    else:
        ax.text(0.5, 0.5, "modern-model feasibility manifest pending", ha="center", va="center", transform=ax.transAxes, color=SLATE)

    fig.suptitle("Independent replication confirms the measurement-limited boundary", x=0.03, y=1.015, ha="left", fontsize=11.5, fontweight="bold", color=NAVY)
    outputs = save_figure(fig, ROOT / "results/figures/iclr_formal/formal_fig3_replication_boundary")
    manifest = {
        "schema_version": 1,
        "figure": "formal_fig3_replication_boundary",
        "claim": "the outcome-blind registry selected the Nadig HepG2/Jurkat pair with supported independent-context provenance; one fixed Claim Lock was executed and remains bounded by raw-cell measurement and joint floors",
        "source_data": [
            "artifacts/manifests/reordering_replication_candidate_registry.csv",
            "artifacts/manifests/reordering_replication_candidate_registry.json",
            summary_path.relative_to(ROOT).as_posix(),
            summary_path.with_name(summary_path.stem + "_floors.csv").relative_to(ROOT).as_posix(),
            "artifacts/manifests/formal_v2_claim_lock_replication_nadig_floors.csv",
        ],
        "outputs": outputs,
    }
    return outputs, manifest


def make_response_figure() -> tuple[list[str], dict[str, object]]:
    oof = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_controlled_shift_oof_perturbations.csv")
    external = pd.read_csv(ROOT / "artifacts/manifests/head_to_head_controlled_shift_ladder.csv")
    external_summary = pd.read_csv(ROOT / "artifacts/manifests/head_to_head_controlled_shift_ladder_summary.csv")
    inference = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_controlled_shift_oof_inference.csv")
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
    panel(ax, "c", "Response shift is not a stable rank proxy")
    ax.scatter(oof["response_program_shift"], oof["normalized_risk_rank_displacement"], s=9, alpha=0.28, color=TEAL, label="Frangieh OOF", edgecolor="white", linewidth=0.2)
    rho_oof = _safe_corr(oof["response_program_shift"], oof["normalized_risk_rank_displacement"])
    ax.set_xlabel("response-program shift (1 − cosine)")
    ax.set_ylabel("normalized risk-rank displacement")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=5.6, loc="upper left")
    association_range = (inference["response_shift_rank_displacement_spearman"].min(), inference["response_shift_rank_displacement_spearman"].max())
    ax.text(0.03, 0.88, f"pooled ρ={rho_oof:+.2f}\npairwise ρ={association_range[0]:+.2f} to {association_range[1]:+.2f}", transform=ax.transAxes, fontsize=5.7, color=NAVY, va="top")
    ax.text(0.03, 0.03, "label-bootstrap CIs cross 0 in 2/3 pairs; descriptive only", transform=ax.transAxes, fontsize=5.5, color=SLATE)

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
    fig.suptitle("Controlled shifts reveal response reprogramming, not a stable rank proxy", x=0.03, y=1.015, ha="left", fontsize=11.5, fontweight="bold", color=NAVY)
    outputs = save_figure(fig, ROOT / "results/figures/iclr_formal/formal_fig2_response_reprogramming")
    manifest = {
        "schema_version": 1,
        "figure": "formal_fig2_response_reprogramming",
        "claim": "matched contexts show descriptive response-program displacement, while coarse response shift is not a stable proxy for reliability rank displacement; no causal mechanism is inferred",
        "evidence_tier": "orthogonal biological extension",
        "predictor_contract": "evaluation-only source-response persistence baseline; not a trained perturbation model",
        "source_data": [
            "artifacts/manifests/formal_v2_controlled_shift_oof_perturbations.csv",
            "artifacts/manifests/formal_v2_controlled_shift_oof_inference.csv",
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
    for name, function in [("measurement", make_measurement_figure), ("replication_boundary", make_replication_boundary_figure), ("deployment", make_deployment_figure)]:
        generated, manifest = function()
        outputs[name] = generated
        manifests[name] = manifest
        (ROOT / "artifacts/manifests" / f"{manifest['figure']}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": outputs, "manifests": manifests}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
