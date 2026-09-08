"""Create publication-ready formal-v2 Figures 1--4.

Figure contract
---------------
Core conclusion: reliability shifts under biological domain change and can be
decomposed into difficulty, ranking and tested semantic components.
Archetype: asymmetric mixed-modality quantitative composite.
Backend: Python/matplotlib only; double-column vector exports plus 600-dpi
previews. All panels read formal-v2 artifacts and never the historical pilot.
Review risks addressed: target-side leakage, unequal UQ baselines, biological
instance pseudo-replication, and unsupported predictor rows.
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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

# Nature-figure contract: editable text in vector output.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAVY = "#12263A"
BLUE = "#1F6F8B"
TEAL = "#2A9D8F"
MINT = "#B8E0D2"
CORAL = "#D95D52"
GOLD = "#E9B44C"
PURPLE = "#7661A8"
SLATE = "#6C7A89"
LIGHT = "#F4F7F8"
GRID = "#DCE5E8"
GREEN = "#2E9E66"
RED = "#B64342"

ENV_ORDER = [
    "norman_k562", "replogle_rpe1",
    "tian_neuron_crispra", "tian_neuron_crispri", "frangieh_melanoma_control",
    "frangieh_melanoma_coculture", "frangieh_melanoma_ifng", "xu_he293",
]
ENV_LABELS = {
    "norman_k562": "Norman\nK562",
    "replogle_rpe1": "RPE1",
    "tian_neuron_crispra": "Tian\nCRISPRa",
    "tian_neuron_crispri": "Tian\nCRISPRi",
    "frangieh_melanoma_control": "Frangieh\ncontrol",
    "frangieh_melanoma_coculture": "Frangieh\nco-culture",
    "frangieh_melanoma_ifng": "Frangieh\nIFNγ",
    "xu_he293": "Xu\nHEK293",
}
PREDICTOR_LABELS = {"mean_matching": "Matching mean", "strong_linear": "Ahlmann–Eltze", "slim_string": "SLIM"}
PREDICTOR_COLORS = {"mean_matching": BLUE, "strong_linear": PURPLE, "slim_string": GOLD}
INFO_ORDER = [
    "U_scalar_raw", "U_scalar_best_validation", "U_scalar_platt", "U_scalar_isotonic",
    "U", "U+P", "U+P+S", "U+P+S+N", "U+P+S+N+C",
]
INFO_LABELS = {
    "U_scalar_raw": "raw scalar",
    "U_scalar_best_validation": "best scalar",
    "U_scalar_platt": "Platt",
    "U_scalar_isotonic": "isotonic",
    "U": "U",
    "U+P": "U+P",
    "U+P+S": "U+P+S",
    "U+P+S+N": "U+P+S+N",
    "U+P+S+N+C": "U+P+S+N+C",
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
    })


def panel(ax: plt.Axes, label: str, title: str | None = None) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=10, fontweight="bold", color=NAVY, va="bottom")
    if title:
        ax.set_title(title, loc="left", pad=7, color=NAVY, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)


def save_publication(fig: plt.Figure, base: Path) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    # Nested gridspecs plus colorbars trigger a benign matplotlib warning even
    # though the exported layout is reviewed below. Keep the stable layout
    # used by the figure contract and silence only that known warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
        fig.tight_layout(pad=1.0)
    outputs = []
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600}), ("tiff", {"dpi": 600})):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path.relative_to(ROOT).as_posix())
    plt.close(fig)
    return outputs


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = ROOT / "artifacts/source_data/formal_v2_reliability_predictions.csv"
    metrics = ROOT / "artifacts/manifests/formal_v2_reliability_metrics.csv"
    ablation = ROOT / "artifacts/manifests/formal_v2_information_ablation_metrics.csv"
    atlas = ROOT / "artifacts/manifests/formal_v2_environment_transfer_matrix.csv"
    paths = [source, metrics, ablation, atlas]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("formal figure inputs missing: " + ", ".join(missing))
    return pd.read_csv(source), pd.read_csv(metrics), pd.read_csv(ablation), pd.read_csv(atlas)


def _aurc(group: pd.DataFrame, score: str) -> float:
    ordered = group.sort_values(score, ascending=False, kind="mergesort")
    risk = ordered["continuous_risk"].to_numpy(dtype=float)
    return float((np.cumsum(risk) / np.arange(1, len(risk) + 1)).mean())


def figure2(pred: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = pred.loc[pred["scenario"].eq("in_domain")].copy()
    data["confidence"] = pd.to_numeric(data["raw_normalized_uq"], errors="coerce")
    data["risk"] = pd.to_numeric(data["continuous_risk"], errors="coerce")
    data["confidence_decile"] = data.groupby(["environment_key", "predictor"], observed=True)["confidence"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 10, labels=False) + 1 if len(values) >= 10 else 5
    )
    fig = plt.figure(figsize=(7.25, 6.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.05, 1], hspace=0.48, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a", "The same confidence quantile has different realized risk")
    chosen = [env for env in ["norman_k562", "replogle_rpe1", "frangieh_melanoma_ifng", "xu_he293"] if env in set(data["environment_key"])]
    for env, color in zip(chosen, [BLUE, TEAL, CORAL]):
        subset = data.loc[data["environment_key"].eq(env)]
        curve = subset.groupby("confidence_decile", observed=True)["risk"].mean().reindex(range(1, 11))
        ax.plot(curve.index, curve.values, color=color, lw=2.0, marker="o", ms=3.5, label=ENV_LABELS[env].replace("\n", " "))
    ax.axvspan(7.5, 10.5, color=MINT, alpha=0.35, zorder=0)
    ax.text(8.95, ax.get_ylim()[1] * 0.93, "high confidence", ha="center", fontsize=6.5, color=GREEN)
    ax.set_xlabel("within environment × predictor confidence decile")
    ax.set_ylabel("realized risk (1 − fidelity)")
    ax.set_xticks([1, 5, 10])
    ax.legend(fontsize=6, loc="upper right")

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b", "Controlled microscope: matched perturbations")
    controlled_path = ROOT / "artifacts/manifests/formal_v2_controlled_shift.csv"
    if controlled_path.is_file():
        controlled = pd.read_csv(controlled_path)
        controlled = controlled.loc[controlled["method"].eq("ptl_rf") & controlled["status"].eq("executed")].copy()
        controlled["pair"] = controlled["left_environment_id"].map(lambda value: ENV_LABELS.get(value, value).replace("\n", " ")) + " → " + controlled["right_environment_id"].map(lambda value: ENV_LABELS.get(value, value).replace("\n", " "))
        controlled["pair"] = controlled["pair"].str.replace("Frangieh ", "", regex=False)
        plot = controlled.groupby(["comparison", "pair"], as_index=False)["centered_confidence_mapping_gap"].mean()
        plot = plot.sort_values(["comparison", "pair"])
        x = np.arange(len(plot))
        colors = [CORAL if value >= 0 else TEAL for value in plot["centered_confidence_mapping_gap"]]
        ax.bar(x, plot["centered_confidence_mapping_gap"], color=colors, width=0.68)
        ax.axhline(0, color=NAVY, lw=0.8)
        ax.set_xticks(x, [value.replace(" → ", "\n→ ") for value in plot["pair"]], rotation=0, fontsize=5.0)
        ax.set_ylabel("centered matched-bin risk gap")
        ax.text(0.02, 0.96, "shared perturbation labels; PTL", transform=ax.transAxes, va="top", fontsize=5.8, color=SLATE)
    else:
        ax.text(0.5, 0.5, "controlled-shift artifact pending", ha="center", va="center", transform=ax.transAxes, color=SLATE)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "c", "Raw UQ: ranking varies by environment")
    heat = data.groupby(["environment_key", "predictor"], observed=True).apply(lambda group: _aurc(group, "confidence"), include_groups=False).unstack()
    heat = heat.reindex(index=[e for e in ENV_ORDER if e in heat.index], columns=list(PREDICTOR_LABELS))
    im = ax.imshow(heat.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto", vmin=float(np.nanmin(heat.values)), vmax=float(np.nanmax(heat.values)))
    ax.set_xticks(range(3), [PREDICTOR_LABELS[p] for p in heat.columns], rotation=22, ha="right")
    ax.set_yticks(range(len(heat)), [ENV_LABELS[e] for e in heat.index])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if heat.iloc[i, j] > heat.values.mean() else NAVY)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("AURC · lower is better", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d", "Conditional grouping loss")
    grouping_path = ROOT / "artifacts/manifests/formal_v2_grouping_loss_summary.json"
    if grouping_path.is_file():
        grouping = json.loads(grouping_path.read_text(encoding="utf-8"))
        grouping_frame = pd.DataFrame(grouping.get("summary", []))
        if not grouping_frame.empty:
            plot = grouping_frame.groupby(["predictor", "method"], as_index=False)["mean_contextual_grouping_loss"].mean()
            plot["x"] = plot["predictor"].map({"mean_matching": 0, "strong_linear": 1, "slim_string": 2})
            for method, color, label in [("raw_normalized_uq", SLATE, "raw UQ"), ("u_only_rf", TEAL, "U-only RF"), ("ptl_rf", BLUE, "PTL")]:
                subset = plot.loc[plot["method"].eq(method)].sort_values("x")
                ax.plot(subset["x"], subset["mean_contextual_grouping_loss"], marker="o", lw=1.5, color=color, label=label)
            ax.set_xticks([0, 1, 2], [PREDICTOR_LABELS.get(p, p) for p in ["mean_matching", "strong_linear", "slim_string"]], rotation=24, ha="right", fontsize=5.6)
            ax.set_ylabel("contextual grouping loss")
            ax.legend(fontsize=5.4, loc="best")
            ax.text(0.02, 0.96, "bins fixed from calibration scores", transform=ax.transAxes, va="top", fontsize=5.8, color=SLATE)
        else:
            ax.text(0.5, 0.5, "no valid grouping-loss rows", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "grouping-loss artifact pending", ha="center", va="center", transform=ax.transAxes, color=SLATE)
    fig.suptitle("Reliability Shift: matched confidence does not erase environment heterogeneity", x=0.03, y=1.015, ha="left", fontsize=11.5, fontweight="bold", color=NAVY)
    return save_publication(fig, out_dir / "formal_fig2_confidence_semantics")


def figure3(pred: pd.DataFrame, atlas: pd.DataFrame, out_dir: Path) -> list[str]:
    data = pred.loc[pred["scenario"].eq("leave_predictor_out")].copy()
    fig = plt.figure(figsize=(7.25, 6.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], height_ratios=[1.15, 1], hspace=0.52, wspace=0.4)
    u_only = atlas.loc[atlas["baseline"].eq("u_only_rf")].copy()
    heat = u_only.pivot(index="source_environment_key", columns="target_environment_key", values="transfer_gain").reindex(index=ENV_ORDER, columns=ENV_ORDER)
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a", "Reliability transfer is heterogeneous across source → target pairs")
    lim = float(np.nanmax(np.abs(heat.to_numpy(dtype=float))))
    cmap = LinearSegmentedColormap.from_list("transfer", [RED, "#F7F7F7", GREEN])
    im = ax.imshow(heat.to_numpy(dtype=float), cmap=cmap, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="auto")
    heat = heat.reindex(index=[e for e in ENV_ORDER if e in heat.index], columns=[e for e in ENV_ORDER if e in heat.columns])
    ax.set_xticks(range(len(heat.columns)), [ENV_LABELS[e] for e in heat.columns], rotation=55, ha="right", fontsize=5.0)
    ax.set_yticks(range(len(heat.index)), [ENV_LABELS[e] for e in heat.index], fontsize=5.2)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=5.2, color=NAVY if abs(val) < lim * 0.55 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Δ AURC vs U-only RF", fontsize=6.4)
    cbar.ax.tick_params(labelsize=5.5)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b", "Target-level transfer gains vary and can be negative")
    off = u_only.loc[~u_only["source_environment_id"].eq(u_only["target_environment_id"])]
    target_gain = off.groupby("target_environment_key", observed=True)["transfer_gain"].mean().reindex(ENV_ORDER).dropna().sort_values()
    colors = [CORAL if value < 0 else GREEN for value in target_gain.values]
    ax.barh(np.arange(len(target_gain)), target_gain.values, color=colors, alpha=0.9, height=0.62)
    ax.axvline(0, color=NAVY, lw=0.8)
    ax.set_yticks(np.arange(len(target_gain)), [ENV_LABELS[e].replace("\n", " ") for e in target_gain.index], fontsize=5.5)
    ax.set_xlabel("mean off-diagonal transfer gain")
    for y, value in enumerate(target_gain.values):
        ax.text(value + (0.003 if value >= 0 else -0.003), y, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=5.8, color=NAVY)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "c", "Context similarity is descriptive, not a prospective guarantee")
    off = off.copy()
    off["context_group"] = np.where(off["same_context_modality"].eq(1), "same cell + modality", "different context")
    groups = [off.loc[off["context_group"].eq(name), "transfer_gain"].to_numpy() for name in ["different context", "same cell + modality"]]
    ax.boxplot(groups, positions=[0, 1], widths=0.5, patch_artist=True, showfliers=False,
               boxprops={"facecolor": "#DCEEF0", "edgecolor": BLUE}, medianprops={"color": NAVY},
               whiskerprops={"color": SLATE}, capprops={"color": SLATE})
    jitter = np.random.default_rng(20260907).uniform(-0.08, 0.08, sum(len(g) for g in groups))
    start = 0
    for index, values in enumerate(groups):
        ax.scatter(np.full(len(values), index) + jitter[start:start + len(values)], values, s=10, alpha=0.45, color=BLUE if index else CORAL, edgecolor="white", linewidth=0.25)
        start += len(values)
    ax.axhline(0, color=NAVY, lw=0.7)
    ax.set_xticks([0, 1], ["different\ncontext", "same cell +\nmodality"])
    ax.set_ylabel("source → target Δ AURC")
    ax.text(0.02, 0.96, f"off-diagonal pairs: n={len(off)}", transform=ax.transAxes, va="top", fontsize=6, color=SLATE)

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d", "Holdout contrast: asymmetry is not established")
    subset = data.groupby("heldout_predictor", observed=True)
    rows = []
    for predictor, group in subset:
        rows.append({"predictor": predictor, "u_only": _aurc(group, "u_only_rf"), "ptl": _aurc(group, "ptl_rf")})
    compare = pd.DataFrame(rows).set_index("predictor").reindex(["mean_matching", "strong_linear", "slim_string"])
    x = np.arange(len(compare))
    width = 0.34
    ax.bar(x - width / 2, compare["u_only"], width, label="U-only RF", color=SLATE)
    ax.bar(x + width / 2, compare["ptl"], width, label="PTL", color=BLUE)
    ax.set_xticks(x, [PREDICTOR_LABELS.get(p, p) for p in compare.index], rotation=20, ha="right", fontsize=5.6)
    ax.set_ylabel("AURC · lower is better")
    ax.legend(fontsize=5.7, loc="upper right")
    fig.suptitle("Reliability transfer is heterogeneous and difficult to predict prospectively", x=0.03, y=1.015, ha="left", fontsize=12, fontweight="bold", color=NAVY)
    return save_publication(fig, out_dir / "formal_fig3_reliability_transfer_atlas")


def figure4(pred: pd.DataFrame, ablation: pd.DataFrame, out_dir: Path) -> list[str]:
    fig = plt.figure(figsize=(7.25, 6.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1, 1], hspace=0.50, wspace=0.38)
    macro = ablation.loc[(ablation["aggregation_level"].eq("environment_macro")) & (ablation["scenario"].isin(["in_domain", "leave_environment_out", "leave_predictor_out"]))].copy()
    macro["information_set"] = pd.Categorical(macro["information_set"], categories=INFO_ORDER, ordered=True)
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a", "What information matters beyond predictor uncertainty?")
    for scenario, color, label in [("in_domain", BLUE, "in-domain"), ("leave_environment_out", TEAL, "leave environment out"), ("leave_predictor_out", PURPLE, "leave predictor out")]:
        subset = macro.loc[macro["scenario"].eq(scenario)].sort_values("information_set")
        ax.plot(np.arange(len(subset)), subset["aurc"], marker="o", ms=3.0, lw=1.7, color=color, label=label)
    ax.set_xticks(np.arange(len(INFO_ORDER)), [INFO_LABELS[name] for name in INFO_ORDER], rotation=45, ha="right", fontsize=5.5)
    ax.set_ylabel("environment-macro AURC")
    ax.legend(fontsize=5.8, loc="best")
    ax.axvspan(4.5, 8.5, color="#E8F3F1", alpha=0.5, zorder=0)
    ax.text(6.5, ax.get_ylim()[1] * 0.97, "context surface", ha="center", fontsize=5.8, color=TEAL)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b", "PTL versus U-only is scenario-dependent")
    data = pred.loc[pred["scenario"].eq("in_domain")].copy()
    groups = []
    labels = []
    for subset_name, subset in [("all test rows", data), ("reliable subset", data.loc[data["reliable_label"].eq(1)])]:
        for method, label in [("u_only_rf", "U-only"), ("ptl_rf", "PTL")]:
            groups.append(_aurc(subset, method))
            labels.append(f"{subset_name}\n{label}")
    x = np.arange(len(groups))
    ax.bar(x, groups, color=[SLATE, BLUE, "#A7CFC2", TEAL], width=0.66)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=5.4)
    ax.set_ylabel("AURC · lower is better")
    ax.text(0.02, 0.96, f"reliable subset n={int(data['reliable_label'].sum())}", transform=ax.transAxes, va="top", fontsize=6, color=SLATE)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "c", "One score, several selective outcomes")
    metric = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_reliability_metrics.csv")
    metric = metric.loc[(metric["aggregation_level"].eq("environment_macro")) & (metric["scenario"].isin(["in_domain", "leave_environment_out", "leave_predictor_out"])) & (metric["method"].isin(["u_only_rf", "ptl_rf"]))]
    rows = []
    for scenario in ["in_domain", "leave_environment_out", "leave_predictor_out"]:
        for column, label in [("aurc", "AURC"), ("risk_at_80", "risk@80"), ("ftr_at_80", "FTR@80")]:
            values = metric.loc[metric["scenario"].eq(scenario)].set_index("method")[column]
            if {"u_only_rf", "ptl_rf"}.issubset(values.index):
                rows.append({"scenario": scenario, "metric": label, "gain": float(values["u_only_rf"] - values["ptl_rf"])})
    robustness = pd.DataFrame(rows)
    pivot = robustness.pivot(index="metric", columns="scenario", values="gain").reindex(["AURC", "risk@80", "FTR@80"])
    xpos = np.arange(len(pivot))
    width = 0.24
    for i, (scenario, color, label) in enumerate([("in_domain", BLUE, "in-domain"), ("leave_environment_out", TEAL, "LO environment"), ("leave_predictor_out", PURPLE, "LO predictor")]):
        ax.bar(xpos + (i - 1) * width, pivot[scenario], width, color=color, label=label)
    ax.axhline(0, color=NAVY, lw=0.8)
    ax.set_xticks(xpos, pivot.index)
    ax.set_ylabel("U-only − PTL (positive is better)")
    ax.legend(fontsize=5.5, loc="upper right")

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d", "Modern-model audit: validation-backed GEARS")
    gears_path = ROOT / "artifacts/manifests/formal_v2_gears_reliability_summary.json"
    if gears_path.is_file():
        gears = json.loads(gears_path.read_text(encoding="utf-8"))
        rows = pd.DataFrame(gears.get("shift_by_seed", []))
        if not rows.empty:
            x = np.arange(len(rows))
            width = 0.34
            ax.bar(x - width / 2, rows["contextual_grouping_loss"], width, color=BLUE, label="grouping loss")
            ax.bar(x + width / 2, rows["difficulty_shift_range"], width, color=GOLD, label="difficulty range")
            ax.set_xticks(x, [f"seed {int(value)}" for value in rows["gears_seed"]])
            ax.set_ylabel("shift magnitude")
            ax.legend(fontsize=5.5, loc="best")
            ax.text(0.02, 0.96, "Norman + RPE1; bins from validation", transform=ax.transAxes, va="top", fontsize=5.8, color=SLATE)
        else:
            ax.text(0.5, 0.5, "no GEARS shift rows", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "GEARS validation audit pending", ha="center", va="center", transform=ax.transAxes, color=SLATE)
    fig.suptitle("Reliability Shift across controlled, transport and modern-model audits", x=0.03, y=1.015, ha="left", fontsize=11.5, fontweight="bold", color=NAVY)
    return save_publication(fig, out_dir / "formal_fig4_information_and_robustness")


def figure1(pred: pd.DataFrame, metrics: pd.DataFrame, atlas: pd.DataFrame, out_dir: Path) -> list[str]:
    """Hero figure: the Reliability Shift proof chain."""

    shift_path = ROOT / "artifacts/manifests/formal_v2_reliability_shift_decomposition.csv"
    semantic_path = ROOT / "artifacts/manifests/formal_v2_reliability_shift_semantic.csv"
    if not shift_path.is_file() or not semantic_path.is_file():
        raise FileNotFoundError("Reliability Shift artifacts are required for Figure 1")
    shift = pd.read_csv(shift_path)
    semantic = pd.read_csv(semantic_path)
    primary = shift.loc[(shift["split_seed"].eq(20260907)) & (shift["method"].eq("ptl_rf"))].copy()
    primary_semantic = semantic.loc[(semantic["split_seed"].eq(20260907)) & (semantic["method"].eq("ptl_rf"))].copy()
    fig = plt.figure(figsize=(7.25, 6.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1.05, 1], hspace=0.48, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a", "Difficulty shift: mean risk differs by environment")
    difficulty = primary.groupby("environment_key", observed=True)["difficulty_mean_risk"].mean().reindex([e for e in ENV_ORDER if e in set(primary["environment_key"])])
    x = np.arange(len(difficulty))
    ax.bar(x, difficulty.values, color=BLUE, alpha=0.9)
    ax.set_xticks(x, [ENV_LABELS[e].replace("\n", " ") for e in difficulty.index], rotation=55, ha="right", fontsize=4.8)
    ax.set_ylabel("mean realized risk")

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b", "Ranking shift: confidence–risk concordance varies")
    ranking = primary.groupby("environment_key", observed=True)["ranking_spearman_confidence_vs_negative_risk"].mean().reindex(difficulty.index)
    colors = [TEAL if value >= 0 else CORAL for value in ranking.values]
    ax.axhline(0, color=NAVY, lw=0.8)
    ax.bar(np.arange(len(ranking)), ranking.values, color=colors)
    ax.set_xticks(np.arange(len(ranking)), [ENV_LABELS[e].replace("\n", " ") for e in ranking.index], rotation=55, ha="right", fontsize=4.8)
    ax.set_ylabel("Spearman(confidence, −risk)")
    ax.set_ylim(-1.0, 1.0)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "c", "Testing semantic shift after difficulty control")
    if not primary_semantic.empty:
        display = primary_semantic.sort_values("predictor")
        labels = [PREDICTOR_LABELS.get(value, value) for value in display["predictor"]]
        x = np.arange(len(display))
        width = 0.35
        ax.bar(x - width / 2, display["additive_null_mse"], width, color=SLATE, label="additive null")
        ax.bar(x + width / 2, display["environment_interaction_mse"], width, color=PURPLE, label="environment × confidence")
        ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=5.3)
        ax.set_ylabel("held-out risk MSE")
        ax.legend(fontsize=5.5, loc="best")

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d", "Interaction gain is tested, not assumed")
    if not primary_semantic.empty:
        display = primary_semantic.sort_values("predictor")
        x = np.arange(len(display))
        center = display["semantic_shift_bootstrap_mean"]
        lower = center - display["semantic_shift_ci_lower"]
        upper = display["semantic_shift_ci_upper"] - center
        ax.errorbar(x, center, yerr=[lower, upper], fmt="o", color=PURPLE, capsize=4, lw=1.2)
        ax.axhline(0, color=NAVY, lw=0.8)
        ax.set_xticks(x, [PREDICTOR_LABELS.get(value, value) for value in display["predictor"]], rotation=25, ha="right", fontsize=5.3)
        ax.set_ylabel("interaction improvement\n(additive MSE − interaction MSE)")
    fig.suptitle("Reliability Shift: difficulty and ranking are established; semantic interaction is tested", x=0.03, y=1.015, ha="left", fontsize=11.5, fontweight="bold", color=NAVY)
    return save_publication(fig, out_dir / "formal_fig1_same_confidence_different_risk")


def main() -> int:
    style()
    pred, metrics, ablation, atlas = load_data()
    out_dir = ROOT / "results/figures/iclr_formal"
    outputs = {
        "figure_1": figure1(pred, metrics, atlas, out_dir),
        "figure_2": figure2(pred, metrics, out_dir),
        "figure_3": figure3(pred, atlas, out_dir),
        "figure_4": figure4(pred, ablation, out_dir),
    }
    manifest = {
        "schema_version": 1,
        "backend": "python_matplotlib",
        "core_conclusion": "Reliability Shift changes difficulty and ranking across biology; the semantic interaction is tested with a null rather than assumed, while source-specific transfer remains heterogeneous",
        "source_data": [
            "artifacts/source_data/formal_v2_reliability_predictions.csv",
            "artifacts/manifests/formal_v2_reliability_metrics.csv",
            "artifacts/manifests/formal_v2_information_ablation_metrics.csv",
            "artifacts/manifests/formal_v2_environment_transfer_matrix.csv",
            "artifacts/manifests/formal_v2_environment_transfer_stability.csv",
            "artifacts/manifests/formal_v2_transport_predictor_summary.json",
            "artifacts/manifests/formal_v2_metric_robustness.csv",
            "artifacts/manifests/formal_v2_metric_robustness_selective.csv",
            "artifacts/manifests/formal_v2_raw_cell_split_half.csv",
            "artifacts/manifests/formal_v2_grouping_loss.csv",
            "artifacts/manifests/formal_v2_grouping_loss_summary.json",
            "artifacts/manifests/formal_v2_reliability_shift_decomposition.csv",
            "artifacts/manifests/formal_v2_reliability_shift_decomposition.json",
            "artifacts/manifests/formal_v2_reliability_shift_semantic.csv",
            "artifacts/manifests/formal_v2_controlled_shift.csv",
            "artifacts/manifests/formal_v2_controlled_shift.json",
            "artifacts/source_data/formal_v2_gears_reliability_predictions.csv",
            "artifacts/manifests/formal_v2_gears_reliability.csv",
            "artifacts/manifests/formal_v2_gears_reliability_summary.json",
            "artifacts/manifests/formal_v2_multisplit_reliability.csv",
            "artifacts/manifests/formal_v2_multisplit_reliability_summary.json",
            "artifacts/manifests/formal_v2_systema_robustness.csv",
            "artifacts/manifests/formal_v2_reproducibility_stratified_reliability.csv",
            "artifacts/manifests/formal_v2_reproducibility_reliability_shift.csv",
            "artifacts/manifests/formal_v2_predictor_metrics.csv",
            "artifacts/manifests/formal_v2_feature_transform_manifest.json",
        ],
        "statistics": {
            "split": "formal validation calibration -> untouched test evaluation",
            "aggregation": "environment then macro average",
            "uncertainty": "paired hierarchical bootstrap is reported in formal_v2_reliability_bootstrap_ci.csv",
            "unsupported_predictors": "excluded from scientific panels",
        },
        "outputs": outputs,
    }
    path = ROOT / "artifacts/manifests/formal_v2_figure_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
