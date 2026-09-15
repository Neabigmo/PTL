"""Legacy renderer retained for provenance; not part of the current paper build.

The current paper-facing supplementary figures are built by the explicit
``supp_*`` modules.  This historical renderer is not called by the canonical
driver and must not be mistaken for Supplementary Fig. S1.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from scripts.figures.common import FEATURES, METRICS, add_panel_label, clean_axes, failure_cases, full_predictability, metric_color, metric_label, num, rank_table, save_figure
from scripts.ptl_figure_style import figure_size


FORWARD_TRANSFERS = (
    ("frangieh_melanoma_control", "frangieh_melanoma_coculture", "C→Co"),
    ("frangieh_melanoma_control", "frangieh_melanoma_ifng", "C→IFNγ"),
    ("frangieh_melanoma_coculture", "frangieh_melanoma_ifng", "Co→IFNγ"),
)


def _forward_summary(manifests: Path) -> pd.DataFrame:
    link = pd.read_csv(manifests / "reliability_transport_decision_link.csv")
    link = link.loc[link["cell_budget_label"].astype(str).str.lower().eq("full")].copy()
    transfer_map = pd.DataFrame(FORWARD_TRANSFERS, columns=["source_environment_id", "target_environment_id", "transfer"])
    link = link.merge(transfer_map, on=["source_environment_id", "target_environment_id"], how="inner")
    link = link.groupby(["transfer", "metric"], as_index=False).agg(d_adj=("d_meas_id", "mean"), regret=("normalized_regret", "mean"))
    het = pd.read_csv(manifests / "reliability_transport_heterogeneity.csv")
    het = het.loc[het["cell_budget_label"].astype(str).str.lower().eq("full") & het["model"].eq("ridge_source_features")].copy()
    pieces = []
    for source, target, label in FORWARD_TRANSFERS:
        sub = het.loc[het["source_environment_id"].eq(source) & het["left_target_environment_id"].eq(source) & het["right_target_environment_id"].eq(target)]
        if not sub.empty:
            pieces.append(sub.assign(transfer=label))
    if pieces:
        pred = pd.concat(pieces, ignore_index=True).groupby(["transfer", "metric"], as_index=False)["spearman"].mean().rename(columns={"spearman": "predictability"})
        return link.merge(pred, on=["transfer", "metric"], how="left")
    return link.assign(predictability=np.nan)


def _rank_card(ax: plt.Axes, case: pd.Series, color: str, title: str) -> None:
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    source_rank, target_rank = num(case["source_rank"]), num(case["ifng_rank"])
    ax.text(.02, .98, title, va="top", fontsize=4.7, fontweight="bold")
    ax.text(.04, .68, "Ctrl", fontsize=4.2, color="#6B7280"); ax.text(.96, .68, "IFNγ", fontsize=4.2, color="#6B7280", ha="right")
    ax.plot([.16, .84], [.43, .43], color="#B8BEC8", lw=.8, zorder=1)
    ax.scatter([.16, .84], [.43, .43], s=20, facecolor="white", edgecolor=color, linewidth=.75, zorder=2)
    ax.text(.16, .20, f"{source_rank:.0f}", ha="center", fontsize=4.2); ax.text(.84, .20, f"{target_rank:.0f}", ha="center", fontsize=4.2)
    ax.text(.50, .018, f"{str(case['perturbation_label'])}, burden {num(case['reordering_burden']):.3f}", ha="center", fontsize=4.15, color=color, fontweight="bold")


def figure6(out_dir: Path, manifests: Path, registry: dict) -> tuple[list[str], pd.DataFrame]:
    summary = _forward_summary(manifests); pred = full_predictability(manifests); ranks = rank_table(manifests); cases = failure_cases(manifests, ranks); rows: list[dict] = []
    coeff = pd.read_csv(manifests / "reliability_transport_heterogeneity_coefficients.csv"); coeff = coeff.loc[coeff["cell_budget_label"].astype(str).str.lower().eq("full") & coeff["model"].eq("ridge_source_features")].copy(); coeff["coefficient"] = pd.to_numeric(coeff["coefficient"], errors="coerce")

    fig = plt.figure(figsize=figure_size("fig6"), constrained_layout=False)
    # A is the visual entry point.  The remaining evidence panels occupy one
    # compact evidence row, with B deliberately transposed to a portrait map.
    # Let the native-proportion schematic carry roughly the same visual weight
    # as the B–D evidence group below it.
    fig.subplots_adjust(top=.98, bottom=.12, left=.045, right=.985)

    # Use an explicit figure-level box so the source-proportion schematic does
    # not inherit a tall grid row and acquire artificial white space.
    ax = fig.add_axes([.045, .54, .91, .34]); asset = Path(__file__).resolve().parents[2] / "results/figures/reliability_transportability/source_assets/fig6a_information_firewall.png"
    if not asset.exists():
        raise FileNotFoundError(f"Missing Fig.6A source asset: {asset}")
    # Preserve the supplied schematic's native slide proportion; the source is
    # high-resolution and is never resampled by matplotlib.
    image = plt.imread(asset)
    # The supplied slide contains presentation-style white margins. Remove
    # only those margins, preserving every colored/text element and the native
    # aspect ratio of the remaining schematic.
    rgb = image[..., :3]
    nonwhite = np.min(rgb, axis=2) < .985
    occupied_y = np.flatnonzero(nonwhite.any(axis=1))
    if occupied_y.size:
        pad = max(8, int(image.shape[0] * .008))
        y0 = max(0, int(occupied_y[0]) - pad)
        y1 = min(image.shape[0], int(occupied_y[-1]) + pad + 1)
        image = image[y0:y1, ...]
    ax.imshow(image, interpolation="none", resample=False, aspect="equal"); ax.set_anchor("N"); ax.axis("off")
    rows.append({"panel": "A", "allowed_features": ";".join(FEATURES), "source_asset": "provided editable information-overview slide", "provenance": "provided Fig.6A editable asset"})

    # Pull the evidence row up toward A and make it shallower.  This keeps the
    # schematic readable while removing the unused vertical band between rows.
    evidence = fig.add_gridspec(1, 4, width_ratios=(.27, .25, .22, .26), left=.045, right=.985, bottom=.29, top=.53, wspace=.62)

    # B: rotate the former wide 3×9 overview into a compact 9×3 portrait map.
    ax = fig.add_subplot(evidence[0]); add_panel_label(ax, "B"); ax.set_title("Forward summary", loc="left", pad=10, fontweight="bold")
    transfers = [x[2] for x in FORWARD_TRANSFERS]; columns = [(metric, transfer) for metric in METRICS for transfer in transfers]; row_specs = [("$D_{adj}$", "d_adj", "Blues"), ("$ρ_{src}$", "predictability", "Greens"), ("regret", "regret", "Reds")]
    matrix = np.array([[num(summary.loc[summary["metric"].eq(metric) & summary["transfer"].eq(transfer), field].mean()) for metric, transfer in columns] for _, field, _ in row_specs], dtype=float)
    scales = np.nanmax(np.abs(matrix), axis=1); scales[scales == 0] = 1.0
    cmap = plt.get_cmap("Blues")
    row_labels = []
    for ri, (metric, transfer) in enumerate(columns):
        y = len(columns) - 1 - ri
        short_metric = ("Δ", "Sys", "Rank")[METRICS.index(metric)]
        row_labels.append(f"{short_metric} {transfer.replace('IFNγ', 'I')}")
        for ci, (_, _, _) in enumerate(row_specs):
            value = matrix[ci, ri]
            color = cmap(.16 + .76 * (abs(value) / scales[ci] if np.isfinite(value) else 0))
            ax.add_patch(Rectangle((ci, y), 1, 1, facecolor=color, edgecolor="white", linewidth=.55))
    for ci, (label, _, _) in enumerate(row_specs):
        ax.text(ci + .5, 9.13, label, ha="center", va="bottom", fontsize=4.3, fontweight="bold")
    for ri, label in enumerate(row_labels):
        ax.text(-.10, len(columns) - .5 - ri, label, ha="right", va="center", fontsize=3.85, color="#46515C")
    scale_ax = ax.inset_axes([.96, .14, .050, .54])
    scale_ax.imshow(np.linspace(.16, .92, 128)[:, None], cmap="Blues", aspect="auto", origin="lower")
    scale_ax.set_xticks([]); scale_ax.set_yticks([0, 127], ["low", "high"], fontsize=3.2); scale_ax.tick_params(length=0, pad=1)
    for spine in scale_ax.spines.values(): spine.set_visible(False)
    ax.text(3.43, .10, "relative\nvalue", ha="center", va="bottom", fontsize=3.35, color="#6B7280")
    ax.set_xlim(-1.25, 4.00); ax.set_ylim(-.12, 9.42); ax.axis("off")
    for _, item in summary.iterrows(): rows.append({"panel": "B", **item.to_dict(), "provenance": "reliability_transport_decision_link.csv + reliability_transport_heterogeneity.csv"})

    ax = fig.add_subplot(evidence[1]); add_panel_label(ax, "C"); ax.set_title("Reliability map", loc="left", pad=10, fontweight="bold"); ax.axvspan(-.5, 0, color="#F4E7E5", alpha=.30, zorder=0); ax.axvspan(0, .18, color="#E5F0EA", alpha=.22, zorder=0); ax.axvline(0, color="#B8BEC8", lw=.55, ls=(0, (3, 2)))
    for metric in METRICS:
        sub = summary.loc[summary["metric"].eq(metric)].sort_values("transfer"); ax.plot(sub["predictability"], sub["d_adj"], color="#B8BEC8", lw=.55, alpha=.75, zorder=1); ax.scatter(sub["predictability"], sub["d_adj"], s=27, color=metric_color(metric), edgecolor="white", linewidth=.4, zorder=2, label=metric_label(metric))
    ax.set_xlim(-.5, .18); ax.set_xticks([-.5, -.25, 0, .1]); ax.set_yscale("log"); ax.set_ylim(max(.001, float(summary["d_adj"].min()) * .7), float(summary["d_adj"].max()) * 1.6); ax.set_xlabel("source $ρ$", fontsize=5.7, labelpad=1); ax.set_ylabel("$D_{adj}$", fontsize=5.7); ax.tick_params(axis="both", labelsize=4.4, pad=1.0); clean_axes(ax, grid=True); ax.legend(frameon=False, fontsize=3.55, loc="lower left", bbox_to_anchor=(.01, .015), ncol=1, handletextpad=.2, labelspacing=.24, borderaxespad=0.0)
    for _, item in summary.iterrows(): rows.append({"panel": "C", **item.to_dict(), "provenance": "reliability_transport_decision_link.csv + reliability_transport_heterogeneity.csv"})

    ax = fig.add_subplot(evidence[2]); add_panel_label(ax, "D"); ax.set_title("Feature importance", loc="left", pad=10, fontweight="bold"); feature_order = ["uq_cosine_disagreement", "uq_mean_gene_variance", "uq_effect_norm_variance", "prediction_norm", "prediction_sparsity", "prediction_concentration"]; feature_labels = ["Uncertainty", "Geometry", "Magnitude", "Type", "Annotation", "Sequence"]
    all_values = []
    for metric in METRICS:
        for feature in feature_order:
            for source, target, _ in FORWARD_TRANSFERS:
                all_values.extend(coeff.loc[coeff["source_environment_id"].eq(source) & coeff["left_target_environment_id"].eq(source) & coeff["right_target_environment_id"].eq(target) & coeff["metric"].eq(metric) & coeff["feature"].eq(feature), "coefficient"].dropna().tolist())
    norm = TwoSlopeNorm(vmin=min(all_values + [-.01]), vcenter=0, vmax=max(all_values + [.01]))
    importance = []
    for metric in METRICS:
        values = []
        for feature in feature_order:
            pieces = []
            for source, target, _ in FORWARD_TRANSFERS:
                pieces.extend(coeff.loc[coeff["source_environment_id"].eq(source) & coeff["left_target_environment_id"].eq(source) & coeff["right_target_environment_id"].eq(target) & coeff["metric"].eq(metric) & coeff["feature"].eq(feature), "coefficient"].dropna().tolist())
            values.append(float(np.nanmedian(pieces)) if pieces else np.nan)
        importance.append(values)
        for feature, value in zip(feature_order, values): rows.append({"panel": "D", "metric": metric, "feature": feature, "coefficient": value, "provenance": "reliability_transport_heterogeneity_coefficients.csv"})
    image = ax.imshow(np.column_stack(importance), cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(3), ["Δ", "Sys", "Rank"], fontsize=4.2)
    for label, metric in zip(ax.get_xticklabels(), METRICS): label.set_color(metric_color(metric)); label.set_fontweight("bold")
    ax.set_yticks(range(len(feature_labels)), feature_labels, fontsize=4.05); ax.tick_params(axis="both", length=0, pad=1.0)
    colorbar = fig.colorbar(image, ax=ax, fraction=.075, pad=.05); colorbar.ax.tick_params(labelsize=3.6, length=1, pad=1); colorbar.set_label("coefficient", fontsize=3.8, labelpad=1)

    ax = fig.add_subplot(evidence[3]); add_panel_label(ax, "E"); ax.set_title("Examples", loc="left", pad=10, fontweight="bold"); ax.axis("off")
    example_titles = ("Stable", "Context shift", "Partial"); example_colors = ("#527AA3", "#D9825B", "#3A9D8F")
    for i, (_, case) in enumerate(cases.iterrows()):
        if i >= 3: break
        small = ax.inset_axes([.02, .70 - i * .33, .96, .26]); _rank_card(small, case, example_colors[i], example_titles[i]); rows.append({"panel": "E", **case.to_dict(), "provenance": "formal_v2_reliability_reordering_perturbations.csv + reliability_transport_failure_anatomy.csv"})
    fig.suptitle("Supplementary Fig. S1   A unified view of reliability, reordering and biological consequence", fontsize=12.5, fontweight="bold", x=.02, ha="left")
    return save_figure(fig, out_dir, "reliability_transportability_fig6_prospective_limit"), pd.DataFrame(rows)


__all__ = ["figure6"]
