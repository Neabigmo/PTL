"""Render the matched source-frozen predictor-family transport comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SURFACE = ROOT / "artifacts/manifests/predictor_families_v2/canonical_four_family_transport.csv"
COMPARISON = ROOT / "artifacts/manifests/predictor_families_v2/canonical_four_family_comparison.csv"
OUT = ROOT / "results/figures/reliability_transportability/predictor_family_transport"

FAMILIES = ["bilinear_ridge", "rbf_krr", "source_only_mlp", "source_only_latent_mlp", "official_gears"]
FAMILY_LABELS = ["Bilinear", "RBF KRR", "Direct MLP", "Latent MLP", "GEARS"]
METRICS = ["delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement"]
COLORS = {"delta_cosine": "#4C78A8", "systema_centroid_accuracy": "#B87557", "absolute_effect_rank_agreement": "#3A8E82"}
MARKERS = {"delta_cosine": "o", "systema_centroid_accuracy": "s", "absolute_effect_rank_agreement": "^"}


def main() -> None:
    surface = pd.read_csv(SURFACE)
    comparison = pd.read_csv(COMPARISON)
    tasks = (surface[["transfer_id", "metric"]].drop_duplicates()
             .assign(metric=lambda x: pd.Categorical(x["metric"], METRICS, ordered=True))
             .sort_values(["metric", "transfer_id"]))
    task_keys = list(tasks.itertuples(index=False, name=None))
    lookup = surface.set_index(["family", "transfer_id", "metric"])["D_adj"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": .7})
    fig = plt.figure(figsize=(7.15, 3.35))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.65, 1), left=.08, right=.98, bottom=.19, top=.82, wspace=.33)
    ax = fig.add_subplot(gs[0, 0])
    values = np.array([[float(lookup.loc[(fam, transfer, metric)]) for transfer, metric in task_keys] for fam in FAMILIES])
    scale = max(float(np.nanpercentile(np.abs(values), 95)), 1e-6)
    for row, fam in enumerate(FAMILIES):
        for col, (transfer, metric) in enumerate(task_keys):
            value = values[row, col]
            ax.scatter(col, row, s=18 + 150 * min(abs(value) / scale, 1), marker=MARKERS[metric],
                       color=COLORS[metric], alpha=.28 + .68 * min(abs(value) / scale, 1),
                       edgecolor="white", linewidth=.35)
    for boundary in (5.5, 11.5):
        ax.axvline(boundary, color="#D9DEE5", lw=.8)
    ax.set_yticks(range(len(FAMILIES)), FAMILY_LABELS)
    short = [t.replace("frangieh_melanoma_", "").replace("coculture", "Co").replace("control", "Ctrl").replace("ifng", "IFNγ").replace("->", "→") for t, _ in task_keys]
    ax.set_xticks(range(len(task_keys)), short, rotation=65, ha="right", fontsize=5.3)
    ax.invert_yaxis(); ax.set_xlim(-.7, len(task_keys)-.3)
    ax.set_title("A   Matched transport surface", loc="left", fontweight="bold", fontsize=8.4, pad=9)
    ax.set_xlabel("Directed transfer within metric blocks")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    y = np.arange(len(FAMILIES))
    offsets = np.array([-.18, 0, .18])
    for j, metric in enumerate(METRICS):
        sub = comparison.loc[comparison["metric"].eq(metric)].set_index("family")
        x = [float(sub.loc[fam, "spearman_D_adj_vs_reference"]) for fam in FAMILIES]
        ax.scatter(x, y + offsets[j], s=29, marker=MARKERS[metric], color=COLORS[metric],
                   edgecolor="white", linewidth=.4, label=metric.replace("_", " "))
    ax.axvline(0, color="#B8BEC8", lw=.8, ls=(0, (3, 2)))
    ax.set_xlim(-1.02, 1.04); ax.set_yticks(y, FAMILY_LABELS); ax.invert_yaxis()
    ax.set_xlabel("Spearman $\\rho$ vs. bilinear profile")
    ax.set_title("B   Profile concordance", loc="left", fontweight="bold", fontsize=8.4, pad=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=5.7, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=1,
              handletextpad=.4, labelspacing=.25)

    fig.suptitle("Strict source-frozen predictor families preserve a shared but non-identical transport structure",
                 x=.08, ha="left", fontsize=10.2, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
