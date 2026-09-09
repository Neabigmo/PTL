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
    noise_floor = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_controlled_shift_noise_floor.csv")
    source_frozen = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_claim_lock_source_frozen_reordering.csv")
    measurement = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_summary.csv")

    fig = plt.figure(figsize=(7.25, 6.35))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.0, 1.05], hspace=0.55, wspace=0.40)
    pairs = list(PAIR_LABELS.values())

    ax = fig.add_subplot(grid[0, 0])
    panel(ax, "a", "Frozen predictors displace risk ranks")
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
    matched = _primary_pair_predictions("frangieh_melanoma_control", "frangieh_melanoma_ifng")
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
    panel(ax, "c", "Pairwise order shift meets noise floors")
    metric_labels = {"delta_cosine": "delta\ncosine", "systema_centroid_accuracy": "Systema\ncentroid", "absolute_effect_rank_agreement": "absolute\neffect rank"}
    metric_order = list(metric_labels)
    component_specs = [("ordering_cross_disagreement", "cross", CORAL, -0.24), ("ordering_measurement_floor", "measurement", BLUE, -0.08), ("ordering_joint_floor", "joint", PURPLE, 0.08)]
    for metric_index, metric in enumerate(metric_order):
        rows = measurement.loc[measurement["metric"].eq(metric)].reset_index(drop=True)
        for key, label, color, offset in component_specs:
            values = rows[key].to_numpy(dtype=float)
            ax.scatter(np.full(len(values), metric_index + offset), values, s=11, alpha=0.25, color=color, edgecolor="white", linewidth=0.2)
            ax.plot([metric_index + offset - 0.07, metric_index + offset + 0.07], [np.mean(values), np.mean(values)], color=color, lw=2.0, solid_capstyle="round")
    ax.axhline(0, color=NAVY, lw=0.8)
    ax.set_xticks(np.arange(len(metric_order)), list(metric_labels.values()), fontsize=5.7)
    ax.set_ylabel("pairwise order disagreement")
    ax.set_ylim(0, 1.0)
    ax.legend(handles=[mpl.lines.Line2D([], [], marker="o", color=color, lw=2, markersize=3.5, label=label) for _, label, color, _ in component_specs], fontsize=5.0, loc="upper left", ncol=2)
    ax.text(0.02, 0.04, "primary estimand: tie-aware pairwise order; U-statistic within floors", transform=ax.transAxes, fontsize=5.1, color=SLATE)

    ax = fig.add_subplot(grid[1, 1])
    panel(ax, "d", "Identifiable excess is metric-dependent")
    pair_keys = list(PAIR_LABELS)
    pair_short = [PAIR_LABELS[key].replace("frangieh_melanoma_", "") for key in pair_keys]
    metric_colors = {"delta_cosine": CORAL, "systema_centroid_accuracy": BLUE, "absolute_effect_rank_agreement": PURPLE}
    metric_names = {"delta_cosine": "delta cosine", "systema_centroid_accuracy": "Systema centroid", "absolute_effect_rank_agreement": "absolute-effect rank"}
    offsets = {"delta_cosine": 0.22, "systema_centroid_accuracy": 0.0, "absolute_effect_rank_agreement": -0.22}
    for pair_index, (left, right) in enumerate(pair_keys):
        for metric in metric_order:
            rows = measurement.loc[
                measurement["left_target_environment_id"].eq(left)
                & measurement["right_target_environment_id"].eq(right)
                & measurement["metric"].eq(metric)
            ]
            for _, row in rows.iterrows():
                value = float(row["ordering_delta_joint_id"])
                ax.errorbar(pair_index + offsets[metric], value,
                            yerr=[[value - float(row["ordering_delta_joint_id_ci_low"])], [float(row["ordering_delta_joint_id_ci_high"]) - value]],
                            fmt="o", ms=3.4, color=metric_colors[metric], ecolor=metric_colors[metric], elinewidth=0.65,
                            capsize=1.2, alpha=0.65, markeredgecolor="white", markeredgewidth=0.25)
    ax.axhline(0, color=NAVY, lw=0.9)
    ax.set_xticks(np.arange(len(pair_keys)), pair_short, rotation=22, ha="right", fontsize=5.5)
    ax.set_ylabel(r"ordering excess = cross − U-statistic joint floor")
    ax.set_ylim(-0.5, 0.5)
    ax.legend(handles=[mpl.lines.Line2D([], [], marker="o", color=color, lw=0, markersize=4, label=metric_names[metric]) for metric, color in metric_colors.items()], fontsize=5.1, loc="upper left")
    ax.text(0.02, 0.04, "30 full-size raw-cell seeds · 2,000 paired label-bootstrap draws", transform=ax.transAxes, fontsize=5.2, color=SLATE)

    fig.suptitle("Measurement reliability limits claims of context-dependent reordering", x=0.03, y=1.015, ha="left", fontsize=11.2, fontweight="bold", color=NAVY)
    outputs = save_figure(fig, ROOT / "results/figures/iclr_formal/formal_fig1_reliability_reordering")
    manifest = {
        "schema_version": 1,
        "figure": "formal_fig1_reliability_reordering",
        "claim": "source-frozen predictors show cross-context rank displacement, while the primary pairwise-order excess is separated from secondary rank displacement and remains metric-dependent",
        "status": "claim_lock_frangieh_source_frozen_and_noise_audit",
        "source_data": [
            "artifacts/source_data/frangieh_source_frozen_predictions.npz",
            "artifacts/manifests/formal_v2_claim_lock_source_frozen_reordering.csv",
            "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_summary.csv",
            "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_ordering.csv",
            "artifacts/manifests/formal_v2_controlled_shift_noise_floor.csv",
            "artifacts/manifests/reordering_replication_candidate_registry.json",
            "artifacts/manifests/formal_v2_claim_lock_guide_id_semantics.json",
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
