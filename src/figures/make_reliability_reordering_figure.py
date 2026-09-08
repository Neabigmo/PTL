"""Create the controlled reliability-reordering diagnostic figure.

The figure is tied to the full-surface Frangieh OOF audit and its
perturbation-level Scientific Lock.  Figure 3 owns recalibration; this figure
therefore uses its final panel to show confidence tracking against a
label-preserving permutation null.
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
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import safe_rowwise_cosine  # noqa: E402
from scripts.run_formal_v2_predictors import read_ground_truth  # noqa: E402

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


def _primary_pair_predictions(left: str, right: str) -> pd.DataFrame:
    payload = np.load(ROOT / "artifacts/source_data/frangieh_source_frozen_predictions.npz", allow_pickle=False)
    labels = payload["perturbation_label"].astype(str).tolist()
    panel = payload["evaluation_gene_symbols"].astype(str).tolist()
    environments = payload["source_environment"].astype(str).tolist()
    source_index = environments.index("frangieh_melanoma_control")
    prediction = payload["prediction"][source_index].mean(axis=1)
    truth = {}
    for environment in (left, right):
        manifest = pd.read_csv(ROOT / "artifacts/manifests/environment_registry.csv")
        row = manifest.loc[manifest["environment_key"].eq(environment)].iloc[0]
        source = str(pd.read_csv(ROOT / "artifacts/manifests/biological_instance_registry.csv", dtype=str)
                     .loc[lambda frame: frame["environment_key"].eq(environment), "ground_truth_path"].iloc[0])
        frame = read_ground_truth(ROOT, source, genes=panel)
        frame = frame.loc[frame["environment_key"].eq(environment)].set_index("perturbation_label")
        truth[environment] = frame.loc[labels, panel].to_numpy(dtype=float)
    left_risk = 1.0 - safe_rowwise_cosine(truth[left], prediction)
    right_risk = 1.0 - safe_rowwise_cosine(truth[right], prediction)
    merged = pd.DataFrame({"perturbation_label": labels, "risk_left": left_risk, "risk_right": right_risk})
    merged["rank_left"] = rankdata(merged["risk_left"].to_numpy(dtype=float), method="average")
    merged["rank_right"] = rankdata(merged["risk_right"].to_numpy(dtype=float), method="average")
    return merged.sort_values("perturbation_label", kind="stable").reset_index(drop=True)


def make_figure() -> tuple[list[str], dict[str, object]]:
    inference = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_controlled_shift_oof_inference.csv")
    noise_floor = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_controlled_shift_noise_floor.csv")
    source_frozen = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_claim_lock_source_frozen_reordering.csv")
    measurement = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_claim_lock_measurement_summary.csv")

    fig = plt.figure(figsize=(7.25, 6.35))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.0, 1.05], hspace=0.55, wspace=0.40)

    ax = fig.add_subplot(grid[0, 0])
    panel(ax, "a", "Frozen predictors still reorder risk")
    pairs = list(PAIR_LABELS.values())
    x = np.arange(len(pairs))
    source_markers = {"frangieh_melanoma_control": "o", "frangieh_melanoma_coculture": "s", "frangieh_melanoma_ifng": "^"}
    source_offsets = {"frangieh_melanoma_control": -0.12, "frangieh_melanoma_coculture": 0.0, "frangieh_melanoma_ifng": 0.12}
    for index, pair_label in enumerate(pairs):
        left, right = next(pair for pair, label in PAIR_LABELS.items() if label == pair_label)
        pair_rows = source_frozen.loc[
            source_frozen["left_target_environment_id"].eq(left)
            & source_frozen["right_target_environment_id"].eq(right)
            & source_frozen["metric"].eq("delta_cosine")
        ]
        for _, row in pair_rows.iterrows():
            value = float(row["normalized_rank_displacement_mean"])
            low = value - float(row["normalized_rank_displacement_ci_low"])
            high = float(row["normalized_rank_displacement_ci_high"]) - value
            source = str(row["source_environment_id"])
            ax.errorbar(index + source_offsets[source], value, yerr=[[low], [high]], fmt=source_markers[source], ms=4.5,
                        color=PAIR_COLORS[pair_label], ecolor=PAIR_COLORS[pair_label], elinewidth=0.85, capsize=1.8,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=4)
    ax.set_xticks(x, pairs, rotation=22, ha="right", fontsize=5.7)
    ax.set_ylabel("normalized rank displacement")
    ax.set_ylim(0, 0.35)
    ax.text(0.02, 0.06, "source-frozen · delta cosine · 95% label bootstrap CI", transform=ax.transAxes, va="bottom", fontsize=5.8, color=SLATE)
    ax.legend(handles=[mpl.lines.Line2D([], [], marker=marker, color=NAVY, lw=0, markersize=4, label=label.replace("frangieh_melanoma_", ""))
                         for label, marker in source_markers.items()], fontsize=5.2, loc="upper right")

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
    ax.text(0.02, 0.96, f"same control-source predictor · n={len(matched)} · held-out target truth", transform=ax.transAxes, va="top", fontsize=5.8, color=SLATE)
    ax.legend(handles=[
        mpl.lines.Line2D([], [], color=CORAL, lw=1.5, label="higher risk rank"),
        mpl.lines.Line2D([], [], color=BLUE, lw=1.5, label="lower risk rank"),
    ], fontsize=5.5, loc="lower right")

    ax = fig.add_subplot(grid[1, 0])
    panel(ax, "c", "Joint-noise floor blurs excess")
    metric_labels = {"delta_cosine": "delta\ncosine", "systema_centroid_accuracy": "Systema\ncentroid", "absolute_effect_rank_agreement": "absolute\neffect rank"}
    metric_order = list(metric_labels)
    for metric_index, metric in enumerate(metric_order):
        rows = measurement.loc[measurement["metric"].eq(metric)]
        for _, row in rows.iterrows():
            value = float(row["delta_joint"])
            low = value - float(row["delta_joint_ci_low"])
            high = float(row["delta_joint_ci_high"]) - value
            ax.errorbar(metric_index + np.random.default_rng(int(row["bootstrap_seed"])) .uniform(-0.18, 0.18), value,
                        yerr=[[low], [high]], fmt="o", ms=2.8, color=SLATE, ecolor=SLATE, elinewidth=0.65,
                        capsize=1.1, alpha=0.55, zorder=3)
    ax.axhline(0, color=NAVY, lw=1.0)
    ax.set_xticks(np.arange(len(metric_order)), list(metric_labels.values()), fontsize=5.7)
    ax.set_ylabel("cross $D$ − joint-floor $D$")
    ax.set_ylim(-0.12, 0.06)
    ax.text(0.02, 0.96, "30 raw-cell split seeds · 2,000 paired label bootstrap draws", transform=ax.transAxes, va="top", fontsize=5.5, color=SLATE)
    ax.text(0.02, 0.04, "positive values would support excess cross-context signal; intervals mostly cross 0", transform=ax.transAxes, fontsize=5.2, color=SLATE)

    ax = fig.add_subplot(grid[1, 1])
    panel(ax, "d", "Confidence tracks less than the null")
    positions = np.arange(len(pairs))
    for position, pair_label in zip(positions, pairs):
        row = inference.loc[
            inference.apply(lambda item: PAIR_LABELS.get((item["left_environment_id"], item["right_environment_id"])) == pair_label, axis=1)
        ].iloc[0]
        observed = float(row["confidence_tracking_rate"])
        observed_low = float(row["confidence_tracking_rate_ci_low"])
        observed_high = float(row["confidence_tracking_rate_ci_high"])
        null = float(row["permutation_null_mean"])
        null_low = float(row["permutation_null_ci_low"])
        null_high = float(row["permutation_null_ci_high"])
        ax.errorbar(position - 0.075, observed, yerr=[[observed - observed_low], [observed_high - observed]],
                    fmt="o", ms=5.0, color=PAIR_COLORS[pair_label], ecolor=PAIR_COLORS[pair_label],
                    elinewidth=1.1, capsize=2.0, markeredgecolor="white", markeredgewidth=0.45, zorder=4)
        ax.errorbar(position + 0.075, null, yerr=[[null - null_low], [null_high - null]],
                    fmt="s", ms=4.0, color=SLATE, ecolor=SLATE, elinewidth=1.0, capsize=2.0,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    ax.set_xticks(positions, pairs, rotation=22, ha="right", fontsize=5.7)
    ax.set_ylabel("confidence tracking rate")
    ax.set_ylim(0, 0.5)
    ax.legend(handles=[
        mpl.lines.Line2D([], [], color=BLUE, marker="o", lw=0, markersize=4.5, label="observed"),
        mpl.lines.Line2D([], [], color=SLATE, marker="s", lw=0, markersize=4.0, label="permutation null"),
    ], fontsize=5.4, loc="upper right")
    ax.text(0.02, 0.96, "95% label bootstrap / permutation intervals", transform=ax.transAxes, va="top", fontsize=5.5, color=SLATE)
    ax.text(0.02, 0.04, "null shuffles confidence within each environment", transform=ax.transAxes, fontsize=5.5, color=SLATE)

    fig.suptitle("Biological context can reorder reliability, but the excess signal is not noise-locked", x=0.03, y=1.015, ha="left", fontsize=11.2, fontweight="bold", color=NAVY)
    outputs = save_figure(fig, ROOT / "results/figures/iclr_formal/formal_fig1_reliability_reordering")
    manifest = {
        "schema_version": 1,
        "figure": "formal_fig1_reliability_reordering",
        "claim": "source-frozen predictors show cross-context rank displacement, while matched raw-cell joint-noise deltas do not support a stable excess beyond measurement plus model noise",
        "status": "claim_lock_frangieh_source_frozen_and_noise_audit",
        "source_data": [
            "artifacts/source_data/frangieh_source_frozen_predictions.npz",
            "artifacts/manifests/formal_v2_claim_lock_source_frozen_reordering.csv",
            "artifacts/manifests/formal_v2_claim_lock_measurement_summary.csv",
            "artifacts/manifests/formal_v2_controlled_shift_oof_predictions.csv",
            "artifacts/manifests/formal_v2_controlled_shift_oof_inference.csv",
            "artifacts/manifests/formal_v2_controlled_shift_noise_floor.csv",
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
