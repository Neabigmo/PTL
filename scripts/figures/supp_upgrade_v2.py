"""Supplementary Figure: corrected finite-measurement calibration and predictor families."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ptl_figure_style import apply_style, clean_axes, metric_color, metric_label, save_figure


OUT = ROOT / "results/figures/reliability_transportability"
METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")
FAMILIES = ("bilinear_ridge", "rbf_kernel_ridge_source_only", "source_only_mlp", "source_only_latent_mlp")
FAMILY_LABELS = {"bilinear_ridge": "Bilinear", "rbf_kernel_ridge_source_only": "RBF", "source_only_mlp": "MLP", "source_only_latent_mlp": "Latent MLP"}


def _panel_label(ax: plt.Axes, label: str, *, x: float = -0.22, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def build() -> list[str]:
    apply_style()
    simulation = pd.read_csv(ROOT / "artifacts/manifests/simulation_v21/finite_measurement_primary_v21.csv")
    profiles = pd.read_csv(ROOT / "artifacts/manifests/neural_family_transport_profiles.csv")
    profile_rows = profiles.loc[profiles["predictor_family"].isin(FAMILIES)].copy()
    fidelity_rows = []
    # JSON keeps nested source-side adequacy; read it with json for stable key access.
    import json
    payload = json.loads((ROOT / "artifacts/manifests/neural_family_transport_profiles.json").read_text(encoding="utf-8"))
    for family, values in payload["source_adequacy"].items():
        for source in ("frangieh_melanoma_control", "frangieh_melanoma_coculture", "frangieh_melanoma_ifng"):
            fidelity_rows.append({"family": family, "source": source, "fidelity": values[f"source_fidelity::{source}"]})
    fidelity = pd.DataFrame(fidelity_rows)

    fig = plt.figure(figsize=(5.5, 3.8), facecolor="white")
    fig.suptitle("Finite-measurement calibration and source-only family sensitivity", x=.02, y=.985, ha="left", fontsize=10.5, fontweight="bold")
    grid = fig.add_gridspec(2, 2, left=.10, right=.93, top=.88, bottom=.15, wspace=.34, hspace=.48)

    ax = fig.add_subplot(grid[0, 0]); _panel_label(ax, "A", x=-0.34)
    for law, color, label in (("gaussian_heteroscedastic", "#4C78A8", "Gaussian"), ("student_t_heavy_tailed", "#8B6F61", "Student-t")):
        sub = simulation.loc[simulation["law"].eq(law)].groupby("depth", as_index=False)["d_adj_mae_to_truth"].mean()
        ax.plot(sub["depth"], sub["d_adj_mae_to_truth"], marker="o", ms=3.5, lw=1.5, color=color, label=label)
    ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 40, 80, 160], ["5", "10", "20", "40", "80", "160"])
    ax.set_xlabel("Cells per condition"); ax.set_ylabel(r"$D_{\rm adj}$ MAE to finite truth")
    ax.legend(frameon=False, fontsize=6.4, loc="upper right")
    clean_axes(ax, grid=True)

    ax = fig.add_subplot(grid[0, 1]); _panel_label(ax, "B")
    pivot = simulation.loc[
        simulation["law"].eq("gaussian_heteroscedastic")
        & simulation["resolution"].eq(0.1)
        & simulation["noise_scale"].eq(1.0)
    ].pivot_table(index="inversion_fraction", columns="depth", values="d_adj_bias_to_truth", aggfunc="mean")
    bias_values = pivot.to_numpy(dtype=float)
    color_limit = max(0.001, float(np.nanpercentile(np.abs(bias_values), 95)) * 1.25)
    im = ax.imshow(
        bias_values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(pivot.columns)), [str(int(v)) for v in pivot.columns]); ax.set_yticks(np.arange(len(pivot.index)), [f"{v:g}" for v in pivot.index])
    ax.set_xlabel("Cells per condition"); ax.set_ylabel("Inversion setting")
    ax.set_title("Gaussian bias (resolution 0.1; noise 1.0)", loc="left", fontsize=8, pad=3, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=.045, pad=.035); cbar.set_label("estimate − finite truth", fontsize=6.5); cbar.ax.tick_params(labelsize=6)

    ax = fig.add_subplot(grid[1, 0]); _panel_label(ax, "C")
    family_positions = np.arange(len(FAMILIES))
    sources = list(fidelity["source"].unique())
    offsets = np.linspace(-.18, .18, len(sources))
    for offset, source in zip(offsets, sources):
        sub = fidelity.loc[fidelity["source"].eq(source)].set_index("family").reindex(FAMILIES)
        ax.scatter(family_positions + offset, sub["fidelity"], s=24, color="#6B7280", edgecolor="white", linewidth=.4, zorder=3)
    ax.axhline(fidelity["fidelity"].mean(), color="#B8BEC8", lw=.8, ls=(0, (3, 2)))
    ax.set_xticks(family_positions, ["Bilin.", "RBF", "MLP", "Latent"], rotation=25, ha="right")
    ax.set_ylabel("Source OOF fidelity")
    ax.set_ylim(0.25, .43)
    clean_axes(ax, grid=True)
    ax.text(.02, .96, "one dot per source context", transform=ax.transAxes, fontsize=6.3, color="#6B7280", va="top")

    ax = fig.add_subplot(grid[1, 1]); _panel_label(ax, "D", x=-.38)
    rng = np.random.default_rng(20260915)
    for family_index, family in enumerate(FAMILIES):
        sub = profile_rows.loc[profile_rows["predictor_family"].eq(family)]
        for metric in METRICS:
            values = sub.loc[sub["metric"].eq(metric), "d_adj_deterministic_profile"].to_numpy(float)
            jitter = rng.normal(0, .055, len(values))
            ax.scatter(np.full(len(values), family_index) + jitter, values, s=4.2, alpha=.25, color=metric_color(metric), edgecolor="none", rasterized=True)
            ax.scatter(family_index + (METRICS.index(metric) - 1) * .09, np.median(values), s=22, color=metric_color(metric), edgecolor="white", linewidth=.45, zorder=5)
    ax.set_xticks(family_positions, ["Bilin.", "RBF", "MLP", "Latent"], rotation=25, ha="right")
    ax.set_ylabel("Deterministic ordering profile")
    handles = [plt.Line2D([], [], marker="o", color=metric_color(m), lw=0, ms=4, label=metric_label(m)) for m in METRICS]
    ax.legend(handles=handles, frameon=False, fontsize=6.0, loc="upper left")
    clean_axes(ax, grid=True)

    return save_figure(fig, OUT, "reliability_transportability_supp_fig5_upgrade_v2")


if __name__ == "__main__":
    print(build())
