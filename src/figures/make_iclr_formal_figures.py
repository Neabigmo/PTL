"""Create publication-ready formal-v2 Figures 2--4.

Figure contract
---------------
Core conclusion: confidence changes meaning across biology, while reliability
transfer follows a measurable source-to-target structure.
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
    "norman_k562", "replogle_k562_essential", "replogle_rpe1",
    "tian_neuron_crispra", "tian_neuron_crispri", "frangieh_melanoma_control",
    "frangieh_melanoma_coculture", "frangieh_melanoma_ifng",
]
ENV_LABELS = {
    "norman_k562": "Norman\nK562",
    "replogle_k562_essential": "K562\nessential",
    "replogle_rpe1": "RPE1",
    "tian_neuron_crispra": "Tian\nCRISPRa",
    "tian_neuron_crispri": "Tian\nCRISPRi",
    "frangieh_melanoma_control": "Frangieh\ncontrol",
    "frangieh_melanoma_coculture": "Frangieh\nco-culture",
    "frangieh_melanoma_ifng": "Frangieh\nIFNγ",
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
    chosen = ["norman_k562", "replogle_rpe1", "frangieh_melanoma_ifng"]
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
    panel(ax, "b", "High confidence is not a universal risk level")
    high = data.loc[data["confidence"].ge(data.groupby("predictor")["confidence"].transform("quantile", 0.8))]
    envs = [env for env in ENV_ORDER if env in set(high["environment_key"])]
    vals = [high.loc[high["environment_key"].eq(env), "risk"].to_numpy() for env in envs]
    positions = np.arange(len(envs))
    box = ax.boxplot(vals, positions=positions, widths=0.58, patch_artist=True, showfliers=False,
                     medianprops={"color": NAVY, "linewidth": 1.0},
                     boxprops={"facecolor": "#DCEEF0", "edgecolor": BLUE, "linewidth": 0.8},
                     whiskerprops={"color": SLATE}, capprops={"color": SLATE})
    ax.set_xticks(positions)
    ax.set_xticklabels([ENV_LABELS[e] for e in envs], rotation=45, ha="right", fontsize=5.4)
    ax.set_ylabel("risk among top 20% confidence")
    ax.set_ylim(bottom=0)
    ax.text(0.02, 0.96, f"n={len(high)} test rows", transform=ax.transAxes, va="top", fontsize=6, color=SLATE)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "c", "Raw-UQ discrimination varies by environment and predictor")
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
    panel(ax, "d", "Matched confidence, contrasting outcomes")
    sample = data.sample(min(1400, len(data)), random_state=20260907)
    ax.scatter(sample["confidence"], sample["risk"], s=6, alpha=0.14, color=SLATE, linewidths=0)
    high_conf = data.loc[data["confidence"].ge(0.75)]
    if len(high_conf) >= 2:
        values = high_conf.sort_values("confidence")
        best_pair = None
        best_gap = -1.0
        for i in range(len(values)):
            upper = values.iloc[i + 1:]
            close = upper.loc[(upper["confidence"] - values.iloc[i]["confidence"]).abs() <= 0.015]
            if len(close):
                candidate = close.iloc[(close["risk"] - values.iloc[i]["risk"]).abs().argmax()]
                gap = abs(float(candidate["risk"]) - float(values.iloc[i]["risk"]))
                if gap > best_gap:
                    best_pair = (values.iloc[i], candidate)
                    best_gap = gap
        if best_pair is not None:
            for row, color, label in zip(best_pair, [GREEN, CORAL], ["success", "failure"]):
                ax.scatter([row["confidence"]], [row["risk"]], s=44, color=color, edgecolor="white", linewidth=0.8, zorder=3)
                ax.annotate(label, (row["confidence"], row["risk"]), xytext=(5, 5 if label == "success" else -12), textcoords="offset points", fontsize=6, color=color, fontweight="bold")
            ax.annotate("same confidence\n≠ same risk", xy=(np.mean([r["confidence"] for r in best_pair]), np.mean([r["risk"] for r in best_pair])), xytext=(0.80, 0.30), textcoords="axes fraction", arrowprops={"arrowstyle": "-", "color": CORAL, "lw": 0.8}, fontsize=7, color=NAVY, ha="center", fontweight="bold")
    ax.set_xlabel("formal deployment confidence")
    ax.set_ylabel("realized risk")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    fig.suptitle("Confidence changes meaning across biological environments", x=0.03, y=1.015, ha="left", fontsize=12, fontweight="bold", color=NAVY)
    return save_publication(fig, out_dir / "formal_fig2_confidence_semantics")


def figure3(pred: pd.DataFrame, atlas: pd.DataFrame, out_dir: Path) -> list[str]:
    data = pred.loc[pred["scenario"].eq("leave_predictor_out")].copy()
    fig = plt.figure(figsize=(7.25, 6.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], height_ratios=[1.15, 1], hspace=0.52, wspace=0.4)
    u_only = atlas.loc[atlas["baseline"].eq("u_only_rf")].copy()
    heat = u_only.pivot(index="source_environment_key", columns="target_environment_key", values="transfer_gain").reindex(index=ENV_ORDER, columns=ENV_ORDER)
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a", "Reliability transfer follows a source → target landscape")
    lim = float(np.nanmax(np.abs(heat.to_numpy(dtype=float))))
    cmap = LinearSegmentedColormap.from_list("transfer", [RED, "#F7F7F7", GREEN])
    im = ax.imshow(heat.to_numpy(dtype=float), cmap=cmap, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="auto")
    ax.set_xticks(range(8), [ENV_LABELS[e] for e in ENV_ORDER], rotation=55, ha="right", fontsize=5.0)
    ax.set_yticks(range(8), [ENV_LABELS[e] for e in ENV_ORDER], fontsize=5.2)
    for i in range(8):
        for j in range(8):
            val = heat.iloc[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=5.2, color=NAVY if abs(val) < lim * 0.55 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Δ AURC vs U-only RF", fontsize=6.4)
    cbar.ax.tick_params(labelsize=5.5)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b", "Some targets receive reusable reliability knowledge")
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
    panel(ax, "c", "Context similarity is a measurable transfer descriptor")
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
    panel(ax, "d", "Predictor transfer is evaluated on held-out families")
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
    fig.suptitle("Confidence does not travel uniformly — reliability has structure", x=0.03, y=1.015, ha="left", fontsize=12, fontweight="bold", color=NAVY)
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
    panel(ax, "b", "The gain remains after restricting to reliable outcomes")
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
    panel(ax, "d", "Formal predictor surface used for every headline panel")
    pm = pd.read_csv(ROOT / "artifacts/manifests/formal_v2_predictor_metrics.csv")
    pm = pm.loc[pm["split_id"].eq("ptl_biological_instance_split_v1")].copy()
    pm = pm.loc[pm["predictor"].isin(PREDICTOR_LABELS)]
    fidelity = pm.groupby(["environment_key", "predictor"], observed=True)["mean_fidelity"].mean().unstack().reindex(index=ENV_ORDER, columns=list(PREDICTOR_LABELS))
    im = ax.imshow(fidelity.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(0.7, float(np.nanmax(fidelity.values))))
    ax.set_xticks(range(3), [PREDICTOR_LABELS[p] for p in fidelity.columns], rotation=25, ha="right", fontsize=5.5)
    ax.set_yticks(range(len(fidelity)), [ENV_LABELS[e] for e in fidelity.index], fontsize=5.3)
    for i in range(fidelity.shape[0]):
        for j in range(fidelity.shape[1]):
            value = fidelity.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color="white" if value > fidelity.values.mean() else NAVY)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("mean test fidelity", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.5)
    fig.suptitle("Deployment-safe information improves selective reliability", x=0.03, y=1.015, ha="left", fontsize=12, fontweight="bold", color=NAVY)
    return save_publication(fig, out_dir / "formal_fig4_information_and_robustness")


def main() -> int:
    style()
    pred, metrics, ablation, atlas = load_data()
    out_dir = ROOT / "results/figures/iclr_formal"
    outputs = {
        "figure_2": figure2(pred, metrics, out_dir),
        "figure_3": figure3(pred, atlas, out_dir),
        "figure_4": figure4(pred, ablation, out_dir),
    }
    manifest = {
        "schema_version": 1,
        "backend": "python_matplotlib",
        "core_conclusion": "confidence changes meaning across biology, while reliability transfer follows a measurable source-to-target structure",
        "source_data": [
            "artifacts/source_data/formal_v2_reliability_predictions.csv",
            "artifacts/manifests/formal_v2_reliability_metrics.csv",
            "artifacts/manifests/formal_v2_information_ablation_metrics.csv",
            "artifacts/manifests/formal_v2_environment_transfer_matrix.csv",
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
