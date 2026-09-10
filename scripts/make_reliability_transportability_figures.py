"""Generate the five reliability-transportability figures and source tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PALETTE = {
    "delta_cosine": "#0072B2",
    "systema_centroid_accuracy": "#D55E00",
    "absolute_effect_rank_agreement": "#009E73",
    "observed": "#333333",
    "identifiable": "#CC79A7",
}


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[str]:
    paths: list[str] = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 300})):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(path.name)
    plt.close(fig)
    return paths


def _theme() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })


def figure1(out_dir: Path) -> tuple[list[str], pd.DataFrame]:
    stages = [
        ("Observed\nreordering", "cross-context\norder disagreement"),
        ("Identifiable\ncomponent", "subtract U-statistic\nmeasurement floor"),
        ("Structure", "context × metric ×\npredictor features"),
        ("Depth\nboundary", "cell-budget resolution\nand detectability"),
        ("Decision\nconsequence", "retention, regret,\nbudget-specific utility"),
        ("Prospective\npredictability", "source-only features;\nheld-out target"),
    ]
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis("off")
    for index, (title, subtitle) in enumerate(stages):
        x = index * 1.65
        color = ["#D9EAF7", "#E8DDF4", "#DDF2E8", "#FBE7C6", "#F6D6D6", "#E4E4E4"][index]
        ax.text(x, 0.5, title, ha="center", va="center", fontsize=11, weight="bold", bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": "#555555", "linewidth": 0.8})
        ax.text(x, 0.12, subtitle, ha="center", va="center", fontsize=8, color="#333333")
        if index < len(stages) - 1:
            ax.annotate("", xy=(x + 1.15, 0.5), xytext=(x + 0.45, 0.5), arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 1.2})
    ax.text(4.1, 0.92, "Reliability transportability: from a visible ranking difference to an auditable decision boundary", ha="center", va="center", fontsize=13, weight="bold")
    ax.set_xlim(-0.85, 8.3)
    ax.set_ylim(-0.1, 1.1)
    table = pd.DataFrame({"stage": [item[0].replace("\n", " ") for item in stages], "operational_definition": [item[1].replace("\n", " ") for item in stages]})
    return _save(fig, out_dir, "reliability_transportability_fig1_concept"), table


def figure2(out_dir: Path, manifests: Path) -> tuple[list[str], pd.DataFrame]:
    atlas = pd.read_csv(manifests / "reliability_transport_atlas.csv")
    atlas["plot_value"] = atlas["observed_D"].combine_first(atlas.get("observed_value"))
    atlas["plot_low"] = atlas["observed_D_ci_low"].combine_first(atlas.get("observed_value_ci_low"))
    atlas["plot_high"] = atlas["observed_D_ci_high"].combine_first(atlas.get("observed_value_ci_high"))
    atlas = atlas.loc[atlas["plot_value"].notna()].copy()
    atlas["label"] = (
        atlas["dataset"].astype(str) + " | " + atlas["source_environment_id"].astype(str)
        + " | " + atlas["left_target_environment_id"].astype(str) + "→" + atlas["right_target_environment_id"].astype(str)
        + " | " + atlas["metric"].astype(str) + " | " + atlas["predictor"].astype(str) + " | T" + atlas["evidence_tier"].astype(str)
    )
    atlas = atlas.sort_values(["metric", "dataset", "source_environment_id"], kind="stable").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.19 * len(atlas) + 1.5)))
    y = np.arange(len(atlas))
    for metric, group in atlas.groupby("metric", sort=False):
        idx = group.index.to_numpy()
        ax.errorbar(group["plot_value"], idx, xerr=[group["plot_value"] - group["plot_low"], group["plot_high"] - group["plot_value"]], fmt="o", ms=4, color=PALETTE.get(metric, "#333333"), label=metric)
    ax.set_yticks(y)
    ax.set_yticklabels(atlas["label"], fontsize=6)
    ax.set_xlabel("Metric-specific observed transfer value (90% CI; T1/T2 shown in labels)")
    ax.set_title("Figure 2 | Pair-level transport atlas")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    return _save(fig, out_dir, "reliability_transportability_fig2_forest"), atlas


def _ensure_depth_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "cell_budget_order" not in frame:
        frame["cell_budget_order"] = frame["cell_budget"].rank(method="dense").astype(int)
    if "cell_budget_label" not in frame:
        frame["cell_budget_label"] = frame["cell_budget"].map(lambda value: "full" if value >= 10**8 else str(int(value)))
    return frame


def figure3(out_dir: Path, manifests: Path) -> tuple[list[str], pd.DataFrame]:
    macro_path = manifests / "reliability_transport_measurement_depth_matched_fixed_macro_bootstrap.csv"
    if macro_path.is_file():
        macro = _ensure_depth_labels(pd.read_csv(macro_path))
        aggregate = macro.groupby(["metric", "cell_budget_order", "cell_budget_label", "bootstrap_draw"], sort=True, observed=True)["identifiable_divergence"].mean().reset_index()
        aggregate = aggregate.groupby(["metric", "cell_budget_order", "cell_budget_label"], sort=True, observed=True)["identifiable_divergence"].agg(
            identifiable_divergence="mean", ci_low=lambda x: x.quantile(0.05), ci_high=lambda x: x.quantile(0.95)
        ).reset_index()
        aggregate["uncertainty_unit"] = "synchronized perturbation-label × measurement-seed macro bootstrap"
    else:
        path = manifests / "reliability_transport_measurement_depth_summary.csv"
        depth = _ensure_depth_labels(pd.read_csv(path))
        aggregate = depth.groupby(["metric", "cell_budget_order", "cell_budget_label"], sort=True, observed=True).agg(
            identifiable_divergence=("identifiable_divergence", "mean"), ci_low=("identifiable_divergence_ci_low", "mean"), ci_high=("identifiable_divergence_ci_high", "mean")
        ).reset_index()
        aggregate["uncertainty_unit"] = "legacy summary fallback"
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for metric, group in aggregate.groupby("metric", sort=False):
        group = group.sort_values("cell_budget_order")
        x = group["cell_budget_order"].to_numpy()
        ax.plot(x, group["identifiable_divergence"], marker="o", lw=1.8, color=PALETTE.get(metric, "#333333"), label=metric)
        ax.fill_between(x, group["ci_low"], group["ci_high"], color=PALETTE.get(metric, "#333333"), alpha=0.14)
    labels = aggregate.sort_values("cell_budget_order").drop_duplicates("cell_budget_order")["cell_budget_label"].tolist()
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Cells per raw-cell pseudoreplicate (full = matched full-size depth)")
    ax.set_ylabel("Measurement-identifiable ordering divergence")
    ax.set_title("Figure 3 | Measurement depth sets the resolution boundary")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    return _save(fig, out_dir, "reliability_transportability_fig3_measurement_depth"), aggregate


def figure4(out_dir: Path, manifests: Path) -> tuple[list[str], pd.DataFrame]:
    summary_path = manifests / "reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv"
    if summary_path.is_file():
        summary = _ensure_depth_labels(pd.read_csv(summary_path, dtype={"cell_budget_label": "string"}))
        aggregate = summary.groupby(["metric", "cell_budget_order", "cell_budget_label", "decision_budget_fraction"], sort=True, observed=True).agg(
            retention=("retention_point", "mean"), retention_ci_low=("retention_ci_low", "mean"), retention_ci_high=("retention_ci_high", "mean"),
            excess_regret=("excess_regret_point", "mean"), excess_regret_ci_low=("excess_regret_ci_low", "mean"), excess_regret_ci_high=("excess_regret_ci_high", "mean"),
            regret=("regret_point", "mean"), regret_ci_low=("regret_ci_low", "mean"), regret_ci_high=("regret_ci_high", "mean"),
        ).reset_index()
        aggregate["uncertainty_unit"] = "2,000-draw perturbation-label × measurement-seed bootstrap; point = frozen 30-seed mean"
    else:
        path = manifests / "reliability_transport_measurement_depth_matched_fixed_decision.csv"
        if not path.is_file():
            path = manifests / "reliability_transport_measurement_depth_decision.csv"
        decision = _ensure_depth_labels(pd.read_csv(path))
        if "excess_regret" not in decision:
            decision["excess_regret"] = np.nan
        aggregate = decision.groupby(["metric", "cell_budget_order", "cell_budget_label", "decision_budget_fraction"], sort=True, observed=True).agg(
            retention=("retention", "mean"), retention_ci_low=("retention", lambda x: x.quantile(0.05)), retention_ci_high=("retention", lambda x: x.quantile(0.95)),
            excess_regret=("excess_regret", "mean"), excess_regret_ci_low=("excess_regret", lambda x: x.quantile(0.05)), excess_regret_ci_high=("excess_regret", lambda x: x.quantile(0.95)),
            regret=("regret", "mean"), regret_ci_low=("regret", lambda x: x.quantile(0.05)), regret_ci_high=("regret", lambda x: x.quantile(0.95)),
        ).reset_index()
        aggregate["uncertainty_unit"] = "legacy 30-seed decision rows"
    full = aggregate.loc[aggregate["cell_budget_order"].eq(aggregate["cell_budget_order"].max())].copy()
    if full.empty:
        full = aggregate
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True)
    for metric, group in full.groupby("metric", sort=False):
        group = group.sort_values("decision_budget_fraction")
        axes[0].plot(group["decision_budget_fraction"], group["retention"], marker="o", label=metric, color=PALETTE.get(metric, "#333333"))
        axes[0].fill_between(group["decision_budget_fraction"], group["retention_ci_low"], group["retention_ci_high"], color=PALETTE.get(metric, "#333333"), alpha=0.12)
        if group["excess_regret"].notna().any():
            axes[1].plot(group["decision_budget_fraction"], group["excess_regret"], marker="o", label=metric, color=PALETTE.get(metric, "#333333"))
            axes[1].fill_between(group["decision_budget_fraction"], group["excess_regret_ci_low"], group["excess_regret_ci_high"], color=PALETTE.get(metric, "#333333"), alpha=0.12)
    axes[0].set_title("Top-k retention")
    axes[0].set_ylabel("Fraction retained from target oracle")
    axes[1].set_title("Excess target regret")
    axes[1].set_ylabel("Cross-context regret − measurement floor")
    for ax in axes:
        ax.set_xlabel("Decision budget fraction")
        ax.set_xticks(DEFAULT_BUDGETS := [0.05, 0.10, 0.20, 0.50])
        ax.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Figure 4 | Transportability changes the decision at a fixed budget", y=1.02, fontsize=11)
    return _save(fig, out_dir, "reliability_transportability_fig4_decision_curves"), aggregate


def figure5(out_dir: Path, manifests: Path) -> tuple[list[str], pd.DataFrame]:
    metric_path = manifests / "reliability_transport_metric_dependence.csv"
    predict_path = manifests / "reliability_transport_predictability.csv"
    metric = pd.read_csv(metric_path) if metric_path.is_file() else pd.DataFrame()
    predict = pd.read_csv(predict_path) if predict_path.is_file() else pd.DataFrame()
    if metric.empty:
        aggregate = pd.DataFrame({"metric_from": [], "metric_to": [], "stable_inversion_fraction": []})
    else:
        aggregate = metric.groupby(["metric_from", "metric_to"], as_index=False).agg(stable_inversion_fraction=("stable_inversion_fraction", "mean"), stable_same_fraction=("stable_same_fraction", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    if not aggregate.empty:
        labels = aggregate["metric_from"] + "→" + aggregate["metric_to"]
        axes[0].bar(np.arange(len(aggregate)), aggregate["stable_inversion_fraction"], color="#CC79A7")
        axes[0].set_xticks(np.arange(len(aggregate)), labels, rotation=65, ha="right", fontsize=7)
    axes[0].set_ylabel("Stable transition inversion fraction")
    axes[0].set_title("Tie-aware metric transitions")
    if not predict.empty:
        executed = predict.loc[predict["status"].eq("executed")].copy()
        if not executed.empty:
            summary = executed.groupby("model", as_index=False)["spearman"].mean()
            axes[1].bar(np.arange(len(summary)), summary["spearman"], color="#0072B2")
            axes[1].set_xticks(np.arange(len(summary)), summary["model"], rotation=45, ha="right")
    axes[1].axhline(0, color="#555555", lw=0.8)
    axes[1].set_ylabel("Held-out Spearman")
    axes[1].set_title("Source-only predictability (target held out)")
    for ax in axes:
        ax.grid(alpha=0.2)
    return _save(fig, out_dir, "reliability_transportability_fig5_perturbation_drivers"), aggregate


def run(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    out_dir = root / "results/figures/reliability_transportability"
    out_dir.mkdir(parents=True, exist_ok=True)
    _theme()
    figure_functions = [
        ("fig1", lambda: figure1(out_dir)),
        ("fig2", lambda: figure2(out_dir, manifests)),
        ("fig3", lambda: figure3(out_dir, manifests)),
        ("fig4", lambda: figure4(out_dir, manifests)),
        ("fig5", lambda: figure5(out_dir, manifests)),
    ]
    outputs: dict[str, Any] = {}
    for key, function in figure_functions:
        paths, source = function()
        source_path = out_dir / f"{key}_source.csv"
        source.to_csv(source_path, index=False)
        outputs[key] = {"files": paths, "source": source_path.relative_to(root).as_posix(), "rows": int(len(source))}
    report = {
        "schema_version": 1,
        "status": "executed",
        "figure_count": 5,
        "formats": ["png", "pdf", "svg", "tiff"],
        "color_policy": "colorblind-safe Okabe-Ito-inspired palette; decision ribbons use the canonical 2,000-draw bootstrap summary and other ribbons use their declared synchronized macro source",
        "outputs": outputs,
    }
    report_path = manifests / "reliability_transport_figures.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
