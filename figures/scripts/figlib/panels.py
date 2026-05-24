from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from .palettes import (
    BASELINE_LABELS,
    BLACK,
    BLUE,
    GRAY,
    GREEN,
    GRID,
    HEATMAP_AUDIT,
    HEATMAP_FAIL,
    HEATMAP_NEG,
    HEATMAP_POS,
    HEATMAP_ZERO,
    LIGHT_BLUE,
    LIGHT_GRAY,
    LIGHT_GREEN,
    LIGHT_ORANGE,
    LIGHT_RED,
    LIGHT_TEAL,
    MID_GRAY,
    MODEL_COLORS,
    MODEL_ORDER,
    ORANGE,
    PURPLE,
    RED,
    RULE,
    SPLIT_COLORS,
    TEAL,
    WHITE,
    YELLOW,
)
from .style import despine_data, framed_panel, ordered_splits, pretty_model, pretty_split


def read_table(root: Path, name: str, **kwargs) -> pd.DataFrame:
    path = root / name
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, **kwargs)


AUDIT_CMAP = LinearSegmentedColormap.from_list("audit_muted", [LIGHT_GRAY, "#C9D8E2", HEATMAP_AUDIT])
FIDELITY_CMAP = LinearSegmentedColormap.from_list("fidelity_muted", [HEATMAP_NEG, HEATMAP_ZERO, HEATMAP_POS])
FAILURE_CMAP = LinearSegmentedColormap.from_list("failure_muted", [LIGHT_GRAY, "#E4C7C1", HEATMAP_FAIL])


def _cell_text_color(value: float, vmin: float, vmax: float, dark_threshold: float = 0.70) -> str:
    if vmax <= vmin:
        return BLACK
    normalized = (value - vmin) / (vmax - vmin)
    return WHITE if normalized >= dark_threshold else BLACK


def _draw_cell_grid(ax: plt.Axes, n_rows: int, n_cols: int) -> None:
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=WHITE, linewidth=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)


def plot_dataset_bar(ax: plt.Axes, manifest: pd.DataFrame) -> None:
    framed_panel(ax, "A", "Benchmark surface (by dataset)")
    manifest = manifest[~manifest["dataset_scope"].astype(str).eq("all_datasets")].copy()
    ds = manifest.groupby("dataset_scope").agg(runs=("run_id", "count"), targets=("heldout_target", "nunique")).reset_index()
    ds = ds.sort_values("runs", ascending=True).tail(8)
    labels = [x.replace("_filtered", "").replace("Weissman", "").replace("_essential", " K562") for x in ds["dataset_scope"]]
    ax.barh(range(len(ds)), ds["runs"], color=LIGHT_BLUE, edgecolor=BLUE, linewidth=0.75)
    ax.scatter(ds["targets"] * 10, range(len(ds)), color=ORANGE, s=24, label="targets x10", zorder=3)
    ax.set_yticks(range(len(ds)), labels)
    ax.set_xlabel("Run rows")
    ax.set_xscale("log")
    ax.legend(loc="lower right", frameon=False, handlelength=1.1, borderaxespad=0.2)
    despine_data(ax)


def plot_split_design_matrix(ax: plt.Axes) -> None:
    framed_panel(ax, "B", "Split families and what they test")
    ax.set_axis_off()
    cols = [pretty_split(s) for s in ["random_split", "unseen_perturbation_split", "unseen_combination_split", "low_support_split", "dataset_heldout_split", "external_holdout"]]
    rows = ["Perturbation overlap", "Combination overlap", "Support stress", "Dataset boundary", "External screen", "Controls available", "Leakage audited"]
    vals = [
        ["High", "Low", "High", "High", "Low", "None"],
        ["High", "High", "Low", "High", "Low", "None"],
        ["No", "No", "No", "Yes", "No", "No"],
        ["No", "No", "No", "No", "Yes", "Yes"],
        ["No", "No", "No", "No", "No", "Yes"],
        ["yes", "yes", "yes", "yes", "yes", "yes"],
        ["pass", "pass", "pass", "pass", "pass", "pass"],
    ]
    table = ax.table(cellText=vals, rowLabels=rows, colLabels=cols, cellLoc="center", loc="center", bbox=[0.05, 0.02, 0.92, 0.88])
    table.auto_set_font_size(False)
    table.set_fontsize(6.9)
    table.scale(1.0, 1.16)
    split_keys = ["random_split", "unseen_perturbation_split", "unseen_combination_split", "low_support_split", "dataset_heldout_split", "external_holdout"]
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.45)
        if r == 0 and c >= 0:
            cell.set_text_props(fontweight="bold", color=SPLIT_COLORS[split_keys[c]])


def plot_support_boxplot(ax: plt.Axes, split_audit: pd.DataFrame) -> None:
    framed_panel(ax, "C", "Training support (log10 cells)")
    ready = split_audit[split_audit["track"].eq("signature") & split_audit["status"].eq("ready")].copy()
    splits = ordered_splits(ready["split_family"])
    data = [np.log10(ready.loc[ready["split_family"].eq(s), "train_units"].clip(lower=1)) for s in splits]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, split in zip(bp["boxes"], splits):
        patch.set(facecolor=SPLIT_COLORS.get(split, GRAY), alpha=0.45, edgecolor=BLACK)
    for med in bp["medians"]:
        med.set(color=BLACK, linewidth=0.9)
    ax.set_xticks(range(1, len(splits) + 1), [pretty_split(s, short=True) for s in splits])
    ax.set_ylabel("log10 train units")
    despine_data(ax)


def plot_context_distance(ax: plt.Axes, split_audit: pd.DataFrame) -> None:
    framed_panel(ax, "D", "Train-test context distance")
    ready = split_audit[split_audit["track"].eq("signature") & split_audit["status"].eq("ready")].copy()
    splits = ordered_splits(ready["split_family"])
    y = ready.groupby("split_family")["train_test_centroid_l2"].mean().reindex(splits).fillna(0)
    ax.plot(range(len(splits)), y, color=BLUE, marker="o", markersize=4)
    ax.set_xticks(range(len(splits)), [pretty_split(s, short=True) for s in splits])
    ax.set_ylabel("Composite index")
    despine_data(ax)


def plot_split_audit_heatmap(ax: plt.Axes, split_audit: pd.DataFrame, panel_label: str = "E") -> None:
    framed_panel(ax, panel_label, "Split audit (overlap/leakage)")
    ready = split_audit[split_audit["track"].eq("signature") & split_audit["status"].eq("ready")].copy()
    splits = ordered_splits(ready["split_family"])
    metrics = {
        "Pert. overlap": "perturbation_overlap_train_test",
        "Control overlap": "reference_key_overlap_train_test",
        "Dataset overlap": "dataset_overlap_train_test",
        "Declared overlap": "declared_holdout_overlap_train_test",
    }
    mat = pd.DataFrame({name: ready.groupby("split_family")[col].mean().reindex(splits).fillna(0) for name, col in metrics.items()}).T
    mat = mat / mat.max(axis=1).replace(0, 1).values[:, None]
    im = ax.imshow(mat.to_numpy(), cmap=AUDIT_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(splits)), [pretty_split(s, short=True) for s in splits])
    ax.set_yticks(range(len(mat.index)), mat.index)
    _draw_cell_grid(ax, mat.shape[0], mat.shape[1])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = float(mat.iloc[i, j])
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.6, color=_cell_text_color(value, 0, 1))
    plt.colorbar(im, ax=ax, fraction=0.042, pad=0.045, label="Normalized")


def plot_model_heatmap_with_ranks(ax: plt.Axes, transfer: pd.DataFrame) -> None:
    framed_panel(ax, "A", "Mean non-control cosine and rank")
    splits = ordered_splits(transfer["split_family"])
    models = [m for m in MODEL_ORDER if m in set(transfer["model"])]
    matrix = transfer.pivot(index="model", columns="split_family", values="mean_cosine_non_control").reindex(models).reindex(columns=splits)
    ranks = matrix.rank(axis=0, ascending=False, method="min")
    vmin, vmax = -0.12, 0.70
    im = ax.imshow(matrix.to_numpy(), cmap=FIDELITY_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(splits)), [pretty_split(s) for s in splits])
    ax.set_yticks(range(len(models)), [pretty_model(m) for m in models])
    _draw_cell_grid(ax, len(models), len(splits))
    for i, model in enumerate(models):
        for j, split in enumerate(splits):
            value = matrix.loc[model, split]
            rank = ranks.loc[model, split]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}\n(#{int(rank)})", ha="center", va="center", fontsize=6.9, color=_cell_text_color(float(value), vmin, vmax, 0.78))
    plt.colorbar(im, ax=ax, fraction=0.033, pad=0.035, label="Cosine")


def plot_rank_trajectories(ax: plt.Axes, transfer: pd.DataFrame) -> None:
    framed_panel(ax, "B", "Rank trajectories")
    splits = ordered_splits(transfer["split_family"])
    for model in [m for m in MODEL_ORDER if m in set(transfer["model"])]:
        sub = transfer[transfer["model"].eq(model)].set_index("split_family").reindex(splits)
        color = MODEL_COLORS.get(model, GRAY)
        alpha = 0.55 if model == "control_mean_baseline" else 0.95
        ax.plot(range(len(splits)), sub["family_rank"], marker="o", color=color, label=pretty_model(model), alpha=alpha)
    ax.invert_yaxis()
    ax.set_xticks(range(len(splits)), [pretty_split(s, short=True) for s in splits])
    ax.set_ylabel("Rank (lower is better)")
    ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False, handlelength=1.4)
    despine_data(ax)


def plot_transfer_zoom(ax: plt.Axes, transfer: pd.DataFrame) -> None:
    framed_panel(ax, "C", "Dataset-heldout vs external holdout")
    keep = ["dataset_heldout_split", "external_holdout"]
    for model in [m for m in MODEL_ORDER if m in set(transfer["model"])]:
        sub = transfer[transfer["model"].eq(model)].set_index("split_family").reindex(keep)
        color = MODEL_COLORS.get(model, GRAY)
        ax.plot([0, 1], sub["mean_cosine_non_control"], marker="o", color=color, label=pretty_model(model), alpha=0.55 if model == "control_mean_baseline" else 0.95)
    ax.axhline(0, color=RULE, lw=0.6, ls="--")
    ax.set_xticks([0, 1], ["Dataset\nheldout", "External\nholdout"])
    ax.set_ylabel("Mean non-control cosine")
    despine_data(ax)


def curve_rows(oof: pd.DataFrame, choices: list[tuple[str, str]]) -> dict[str, pd.DataFrame]:
    out = {}
    for ablation, estimator in choices:
        cur = oof[oof["ablation"].eq(ablation) & oof["estimator"].eq(estimator)].copy()
        if not cur.empty:
            out[f"{ablation}:{estimator}"] = cur
    return out


def plot_false_transportability_bars(ax: plt.Axes, rel: pd.DataFrame) -> None:
    framed_panel(ax, "A", "False-transportability rate among accepted predictions")
    order = ["naive_confidence", "calibrated_logistic_deployment", "full_PTL_random_forest", "no_context_distance", "support_only", "novelty_only"]
    sub = rel[rel["method"].isin(order)].copy()
    sub["order"] = sub["method"].map({k: i for i, k in enumerate(order)})
    sub = sub.sort_values("order")
    y = np.arange(len(sub))[::-1]
    colors = [MID_GRAY if m == "naive_confidence" else BLUE if m == "full_PTL_random_forest" else LIGHT_BLUE for m in sub["method"]]
    ax.barh(y, sub["mean_false_transportability_rate"], color=colors, edgecolor=BLACK, linewidth=0.55)
    for yi, val, method in zip(y, sub["mean_false_transportability_rate"], sub["method"]):
        ax.text(val + 0.018, yi, f"{val:.2f}", va="center", fontsize=8)
        if method == "full_PTL_random_forest":
            ax.text(0.47, yi, "66% relative\nreduction", color=BLUE, fontweight="bold", va="center", fontsize=8.2)
    ax.set_yticks(y, [BASELINE_LABELS.get(m, m) + (" (primary)" if m == "full_PTL_random_forest" else "") for m in sub["method"]])
    ax.set_xlim(0, 0.82)
    ax.set_xlabel("False-transportability rate")
    despine_data(ax, axis="x")


def plot_roc_curve(ax: plt.Axes, oof: pd.DataFrame) -> None:
    framed_panel(ax, "B", "ROC curves")
    choices = [("naive_confidence", "naive_confidence"), ("full_PTL", "random_forest"), ("no_context_distance", "random_forest")]
    labels = ["Naive confidence", "Full PTL RF", "No context distance"]
    colors = [GRAY, BLUE, LIGHT_BLUE]
    for (ab, est), label, color in zip(choices, labels, colors):
        cur = oof[oof["ablation"].eq(ab) & oof["estimator"].eq(est)]
        if cur.empty:
            continue
        fpr, tpr, _ = roc_curve(cur["y_true"], cur["score"])
        ax.plot(fpr, tpr, color=color, ls="--" if "Naive" in label else "-", label=f"{label} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], color=RULE, lw=0.6, ls=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    legend = ax.legend(loc="lower right", frameon=True, handlelength=1.6)
    legend.get_frame().set_facecolor(WHITE)
    legend.get_frame().set_alpha(0.86)
    legend.get_frame().set_edgecolor(WHITE)
    despine_data(ax)


def plot_pr_curve(ax: plt.Axes, oof: pd.DataFrame) -> None:
    framed_panel(ax, "C", "Precision-recall curves")
    choices = [("naive_confidence", "naive_confidence"), ("full_PTL", "random_forest"), ("no_context_distance", "random_forest")]
    labels = ["Naive confidence", "Full PTL RF", "No context distance"]
    colors = [GRAY, BLUE, LIGHT_BLUE]
    for (ab, est), label, color in zip(choices, labels, colors):
        cur = oof[oof["ablation"].eq(ab) & oof["estimator"].eq(est)]
        if cur.empty:
            continue
        precision, recall, _ = precision_recall_curve(cur["y_true"], cur["score"])
        ax.plot(recall, precision, color=color, ls="--" if "Naive" in label else "-", label=f"{label} (AP={auc(recall, precision):.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    legend = ax.legend(loc="lower center", bbox_to_anchor=(0.56, 0.02), ncol=1, frameon=True, handlelength=1.6)
    legend.get_frame().set_facecolor(WHITE)
    legend.get_frame().set_alpha(0.86)
    legend.get_frame().set_edgecolor(WHITE)
    despine_data(ax)


def plot_risk_coverage(ax: plt.Axes, selective: pd.DataFrame) -> None:
    framed_panel(ax, "D", "Risk-coverage")
    rows = [
        ("naive_confidence", "naive_confidence", "Naive confidence", GRAY),
        ("full_PTL", "random_forest", "Full PTL RF", BLUE),
        ("no_context_distance", "random_forest", "No context distance", LIGHT_BLUE),
    ]
    cov = np.array([0.2, 0.4, 0.6, 0.8])
    for ab, est, label, color in rows:
        row = selective[selective["ablation"].eq(ab) & selective["estimator"].eq(est)]
        if row.empty:
            continue
        r = row.iloc[0]
        risk = np.array([r[f"false_transportability_rate_at_0p{x}"] for x in [2, 4, 6, 8]], dtype=float)
        ax.plot(cov, risk, marker="o", color=color, label=label)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("False-transportability rate")
    ax.set_xlim(0.15, 0.85)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    despine_data(ax)


def plot_reliability_baseline(ax: plt.Axes, rel: pd.DataFrame) -> None:
    framed_panel(ax, "E", "Reliability baseline comparison")
    order = ["support_only", "split_family_only", "novelty_only", "support_plus_novelty", "calibrated_logistic_deployment", "full_PTL_random_forest"]
    sub = rel[rel["method"].isin(order)].copy()
    sub["order"] = sub["method"].map({k: i for i, k in enumerate(order)})
    sub = sub.sort_values("order", ascending=False)
    colors = [BLUE if m == "full_PTL_random_forest" else LIGHT_BLUE for m in sub["method"]]
    y = np.arange(len(sub))
    ax.barh(y, sub["mean_false_transportability_rate"], color=colors, edgecolor=BLACK, linewidth=0.5)
    ax.set_yticks(y, [BASELINE_LABELS.get(m, m) for m in sub["method"]])
    for yi, val in zip(y, sub["mean_false_transportability_rate"]):
        ax.text(val + 0.012, yi, f"{val:.2f}", va="center")
    ax.set_xlim(0, max(0.6, sub["mean_false_transportability_rate"].max() + 0.1))
    ax.set_xlabel("False-transportability rate")
    despine_data(ax, axis="x")


def plot_failure_taxonomy(ax: plt.Axes) -> None:
    framed_panel(ax, "A", "Failure taxonomy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.14, 0.72, "Transportable\n(kept and above threshold)", TEAL, LIGHT_TEAL),
        (0.14, 0.50, "Below threshold\n(but not severe)", ORANGE, LIGHT_ORANGE),
        (0.14, 0.28, "High-risk error\n(accepted but non-transportable)", RED, LIGHT_RED),
        (0.14, 0.06, "Severe failure\n(large degradation / wrong direction)", RED, "#E2BDB7"),
    ]
    for x, y, text, edge, face in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.74, 0.16, facecolor=face, edgecolor=edge, lw=0.75))
        ax.text(x + 0.37, y + 0.08, text, ha="center", va="center", fontweight="bold", fontsize=7.1)


def plot_failure_heatmap(ax: plt.Axes, atlas: pd.DataFrame) -> None:
    framed_panel(ax, "B", "Severe failure share (%)")
    sev = atlas[atlas["failure_mode"].eq("severe_failure")].copy()
    splits = ordered_splits(atlas["split_family"])
    models = [m for m in MODEL_ORDER if m in set(sev["model"])]
    mat = sev.pivot(index="model", columns="split_family", values="failure_mode_share").reindex(models).reindex(columns=splits).fillna(0) * 100
    vmin, vmax = 0, max(80, mat.max().max())
    im = ax.imshow(mat, cmap=FAILURE_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(splits)), [pretty_split(s, short=True) for s in splits])
    ax.set_yticks(range(len(models)), [pretty_model(m) for m in models])
    _draw_cell_grid(ax, mat.shape[0], mat.shape[1])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = float(mat.iloc[i, j])
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.9, color=_cell_text_color(value, vmin, vmax, 0.70))
    plt.colorbar(im, ax=ax, fraction=0.040, pad=0.035, label="%")


def plot_severity_composition(ax: plt.Axes, atlas: pd.DataFrame) -> None:
    framed_panel(ax, "C", "Severity composition within splits (%)")
    colors = {"transportable": LIGHT_GREEN, "below_transport_threshold": YELLOW, "high_risk_failure": ORANGE, "severe_failure": RED, "non_transportable": LIGHT_GRAY}
    splits = ordered_splits(atlas["split_family"])
    modes = ["transportable", "below_transport_threshold", "high_risk_failure", "severe_failure"]
    agg = atlas.groupby(["split_family", "failure_mode"])["n_signatures"].sum().reset_index()
    piv = agg.pivot(index="split_family", columns="failure_mode", values="n_signatures").reindex(splits).fillna(0)
    props = piv.div(piv.sum(axis=1).replace(0, 1), axis=0)[modes].fillna(0)
    left = np.zeros(len(props))
    y = np.arange(len(props))
    for mode in modes:
        vals = props[mode].to_numpy() * 100
        ax.barh(y, vals, left=left, color=colors[mode], edgecolor=WHITE, linewidth=0.4, label=mode.replace("_", " "))
        left += vals
    ax.set_yticks(y, [pretty_split(s, short=True) for s in splits])
    ax.set_xlabel("Within-split fraction (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, 1.10), ncol=4, frameon=False, columnspacing=0.8, handlelength=1.2)
    despine_data(ax, axis="x")


def plot_context_enrichment(ax: plt.Axes, atlas: pd.DataFrame) -> None:
    framed_panel(ax, "D", "Context enrichment")
    sev = atlas[atlas["failure_mode"].eq("severe_failure")].copy()
    metrics = [
        ("Perturbation novelty", 1 - sev["mean_perturbation_seen"].mean()),
        ("Combination novelty", 1 - sev["mean_component_seen_fraction"].mean()),
        ("Low support", np.clip(1 - np.log10(sev["mean_n_cells"].clip(lower=1)).mean() / 4, 0, 1)),
        ("Dataset boundary", 1 - sev["mean_reference_seen"].mean()),
    ]
    n = np.sqrt(sev["n_signatures"].sum())
    y = np.arange(len(metrics))[::-1]
    values = [m[1] for m in metrics]
    ax.scatter(values, y, s=max(50, n / 3), color=[RED, ORANGE, YELLOW, GREEN], edgecolor=BLACK, linewidth=0.45)
    for yi, (label, val) in zip(y, metrics):
        ax.text(min(val + 0.025, 0.86), yi, f"n={int(sev['n_signatures'].sum()):,}", va="center", fontsize=6.8)
    ax.set_yticks(y, [m[0] for m in metrics])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean severe-failure context signal")
    despine_data(ax, axis="x")


def plot_biological_diagnostics(ax: plt.Axes, bio: pd.DataFrame) -> None:
    framed_panel(ax, "E", "Biological diagnostics of retained and rejected predictions")
    group_col = "retention_group" if "retention_group" in bio.columns else "ptl_decision"
    metrics = [
        ("transportable_rate", "Transportable\nlabel rate"),
        ("perturbation_retrieval_top1_mean", "Top-1\nretrieval"),
        ("mean_pathway_cosine_mean", "Pathway cosine\n(diagnostic)"),
    ]
    groups = ["rejected_high_risk", "retained"]
    x = np.arange(len(metrics))
    width = 0.34
    for offset, group, color in [(-width / 2, "rejected_high_risk", ORANGE), (width / 2, "retained", TEAL)]:
        vals = []
        for col, _ in metrics:
            row = bio[bio[group_col].eq(group)]
            vals.append(float(row[col].iloc[0]) if not row.empty and col in row.columns else np.nan)
        ax.bar(x + offset, vals, width, color=color, alpha=0.65, edgecolor=BLACK, linewidth=0.45, label="Rejected high-risk" if group == "rejected_high_risk" else "Retained")
    ax.set_xticks(x, [m[1] for m in metrics])
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98), handlelength=1.2)
    despine_data(ax)
