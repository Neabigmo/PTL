"""Shared data loaders and visual helpers for the canonical PTL figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.ptl_figure_style import (
    CONTEXT_ORDER, CONTRAST_ORDER, DEPTH_ORDER, add_panel_label, clean_axes,
    context_label, metric_color, metric_label, save_figure,
)

METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")
PRIMARY_SOURCE = "frangieh_melanoma_control"
PRIMARY_TARGET = "frangieh_melanoma_ifng"
PRIMARY_METRIC = "delta_cosine"
ATLAS_DATASET = "FrangiehIzar2021_RNA"
STATE_ORDER = ("-1", "tie", "+1", "unstable")
FEATURES = (
    "uq_cosine_disagreement", "uq_mean_gene_variance", "uq_effect_norm_variance",
    "prediction_norm", "prediction_sparsity", "prediction_concentration",
)
FORBIDDEN_TARGET_FIELDS = (
    "response_displacement", "effect_magnitude", "target_risk", "D_meas-ID", "target failure label",
)
DEPTH_VALUE = {"10": 10.0, "20": 20.0, "40": 40.0, "80": 80.0, "160": 160.0, "full": 1e9}
STATE_COLORS = {"-1": "#0072B2", "+1": "#C44E52", "tie": "#7A8996", "unstable": "#C7CDD2"}
FLOOR_COLORS = {"cross": "#303840", "measurement": "#B8C0C7", "joint": "#6F7D88"}


def num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def source_path(root: Path, name: str) -> Path:
    path = root / "results/figures/reliability_transportability/source_tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_source(root: Path, name: str, frame: pd.DataFrame) -> str:
    path = source_path(root, name)
    frame.to_csv(path, index=False)
    return path.relative_to(root).as_posix()


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def depth_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "cell_budget_label" not in frame.columns:
        frame["cell_budget_label"] = frame["cell_budget"].map(lambda x: "full" if num(x) >= 1e8 else str(int(num(x))))
    frame["cell_budget_label"] = frame["cell_budget_label"].astype(str).str.lower()
    return frame


def budget_label(value: Any) -> str:
    value = num(value)
    return "full" if value >= 1e8 else str(int(value))


def set_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", pad=8, fontweight="bold")
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, va="bottom", fontsize=6.8, color="#46515C")


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], **kwargs: Any) -> None:
    defaults = {"arrowstyle": "-|>", "lw": 1.0, "color": "#46515C", "mutation_scale": 10}
    defaults.update(kwargs)
    ax.annotate("", xy=end, xytext=start, arrowprops=defaults)


def rank_table(manifests: Path) -> pd.DataFrame:
    frame = pd.read_csv(manifests.parent / "source_data/formal_v2_reliability_predictions.csv")
    frame = frame.loc[frame["environment_key"].isin(CONTEXT_ORDER) & frame["predictor"].eq("mean_matching") & frame["scenario"].eq("in_domain")].copy()
    values: dict[str, dict[str, list[float]]] = {context: {} for context in CONTEXT_ORDER}
    for _, row in frame.iterrows():
        values[str(row["environment_key"])].setdefault(str(row["perturbation_label"]), []).append(num(row["continuous_risk"]))
    direct = pd.read_csv(manifests / "formal_v2_reliability_reordering_perturbations.csv")
    direct = direct.loc[direct["predictor"].eq("mean_matching") & direct["method"].eq("ptl_rf")]
    primary = direct.loc[direct["left_environment_id"].eq(PRIMARY_SOURCE) & direct["right_environment_id"].eq(PRIMARY_TARGET)]
    values[PRIMARY_SOURCE] = {str(label): [num(value)] for label, value in primary.groupby("perturbation_label")["risk_left"].mean().items()}
    values[PRIMARY_TARGET] = {str(label): [num(value)] for label, value in primary.groupby("perturbation_label")["risk_right"].mean().items()}
    labels = set.union(*(set(values[context]) for context in CONTEXT_ORDER))
    if len(set(values[PRIMARY_SOURCE]) & set(values[PRIMARY_TARGET])) < 10:
        raise ValueError("Fewer than ten labels are common to the primary canonical contrast")
    ranked: dict[str, pd.DataFrame] = {}
    for context in CONTEXT_ORDER:
        sub = pd.DataFrame({"perturbation_label": sorted(labels)})
        sub["risk"] = sub["perturbation_label"].map(lambda label: float(np.nanmean(values[context][label])) if label in values[context] else np.nan)
        sub = sub.dropna(subset=["risk"]).sort_values(["risk", "perturbation_label"], kind="stable").reset_index(drop=True)
        sub["rank"] = np.arange(1, len(sub) + 1)
        ranked[context] = sub
    out = ranked[CONTEXT_ORDER[0]][["perturbation_label", "risk", "rank"]].rename(columns={"risk": "source_risk", "rank": "source_rank"})
    for context, prefix in zip(CONTEXT_ORDER[1:], ("coculture", "ifng")):
        out = out.merge(ranked[context][["perturbation_label", "risk", "rank"]].rename(columns={"risk": f"{prefix}_risk", "rank": f"{prefix}_rank"}), on="perturbation_label", how="left")
    return out


def rank_flow(manifests: Path) -> tuple[pd.DataFrame, list[str]]:
    ranks = rank_table(manifests)
    ordered = ranks.sort_values(["source_rank", "perturbation_label"], kind="stable").reset_index(drop=True)
    selected: list[str] = []
    for quantile in np.arange(0.05, 1.0, 0.10):
        target_rank = quantile * (len(ordered) + 1)
        candidates = ordered.assign(_distance=(ordered["source_rank"] - target_rank).abs()).sort_values(["_distance", "perturbation_label"], kind="stable")
        selected.append(next(str(value) for value in candidates["perturbation_label"] if str(value) not in selected))
    return ranks.loc[ranks["perturbation_label"].isin(selected)].sort_values("source_rank", kind="stable").reset_index(drop=True), selected


def failure_cases(manifests: Path, ranks: pd.DataFrame) -> pd.DataFrame:
    failure = pd.read_csv(manifests / "reliability_transport_failure_anatomy.csv")
    failure = failure.loc[failure["status"].eq("executed") & failure["source_environment_id"].eq(PRIMARY_SOURCE) & failure["target_environment_id"].eq(PRIMARY_TARGET) & failure["metric"].eq(PRIMARY_METRIC)].copy()
    failure["reordering_burden"] = pd.to_numeric(failure["reordering_burden"], errors="coerce")
    failure = failure.sort_values(["reordering_burden", "perturbation_label"], kind="stable").reset_index(drop=True)
    cases: list[dict[str, Any]] = []
    for quantile in (0.10, 0.50, 0.90):
        position = quantile * (len(failure) - 1)
        candidate = failure.assign(_distance=np.abs(failure.index.to_numpy() - position)).sort_values(["_distance", "perturbation_label"], kind="stable").iloc[0].to_dict()
        candidate.pop("_distance", None)
        candidate["quantile"] = quantile
        cases.append(candidate)
    out = pd.DataFrame(cases).merge(ranks, on="perturbation_label", how="left", validate="one_to_one")
    out["rank_delta"] = out["ifng_rank"] - out["source_rank"]
    return out


def build_exemplar_registry(root: Path) -> dict[str, Any]:
    manifests = root / "artifacts/manifests"
    ranks, selected = rank_flow(manifests)
    cases = failure_cases(manifests, rank_table(manifests))
    full = pd.read_csv(manifests / "reliability_transport_failure_anatomy.csv")
    complete = full.loc[full["status"].eq("executed") & full["source_environment_id"].eq(PRIMARY_SOURCE) & full["target_environment_id"].eq(PRIMARY_TARGET) & full["metric"].eq(PRIMARY_METRIC)]
    registry = {
        "schema_version": 1, "status": "executed",
        "primary_display_context": {"source_environment_id": PRIMARY_SOURCE, "target_environment_id": PRIMARY_TARGET, "metric": PRIMARY_METRIC, "reason": "fixed semantic exemplar, first frozen metric, canonical Ctrl→IFNγ shift; not selected by observed magnitude"},
        "rank_flow": {"selection_rule": "source-only rank quantiles 5%, 15%, ..., 95%; nearest source rank, alphabetical tie-break", "n_labels": len(selected), "perturbation_labels": selected, "entries": json_records(ranks.loc[ranks["perturbation_label"].isin(selected)]), "provenance": "artifacts/source_data/formal_v2_reliability_predictions.csv; predictor=mean_matching; scenario=in_domain"},
        "failure_cases": {"selection_rule": "complete primary failure rows at burden quantiles q10/q50/q90; nearest burden, alphabetical tie-break", "complete_rows": int(len(complete)), "cases": json_records(cases[["quantile", "perturbation_label", "source_rank", "ifng_rank", "rank_delta", "reordering_burden", "response_displacement", "effect_magnitude", "model_disagreement", "source_uncertainty"]]), "provenance": "artifacts/manifests/reliability_transport_failure_anatomy.csv; status=executed; no imputation"},
        "anti_fabrication": {"visual_references_only": True, "unavailable_target_informed_quantities_not_used_as_prospective_features": True, "minimum_strict_support": 8, "no_new_experiment_or_retraining": True},
    }
    (manifests / "figure_exemplar_registry.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return registry


def state_label(value: Any, strict_count: Any) -> str:
    """Map pair states while enforcing the frozen strict-support floor."""
    text, strict = str(value).lower(), num(strict_count)
    if strict < 8:
        return "unstable"
    if "p<q" in text or "p_lt_q" in text or "p<" in text:
        return "-1"
    if "p>q" in text or "p_gt_q" in text or "p>" in text:
        return "+1"
    return "tie" if strict == 0 else "unstable"


def state_counts(path: Path, source: str | None = None, target: str | None = None, metric: str | None = None) -> pd.DataFrame:
    usecols = ["source_environment_id", "target_environment_id", "metric", "source_state_discovery", "target_state_validation", "source_strict_count", "target_strict_count"]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000):
        keep = chunk.loc[chunk["metric"].isin(METRICS) & chunk["source_environment_id"].isin(CONTEXT_ORDER) & chunk["target_environment_id"].isin(CONTEXT_ORDER) & chunk["source_environment_id"].ne(chunk["target_environment_id"])].copy()
        if source is not None:
            keep = keep.loc[keep["source_environment_id"].eq(source) & keep["target_environment_id"].eq(target) & keep["metric"].eq(metric)]
        if not keep.empty:
            keep["source_state"] = [state_label(v, n) for v, n in zip(keep["source_state_discovery"], keep["source_strict_count"])]
            keep["target_state"] = [state_label(v, n) for v, n in zip(keep["target_state_validation"], keep["target_strict_count"])]
            pieces.append(keep[["metric", "source_state", "target_state"]])
    if not pieces:
        return pd.DataFrame(columns=["metric", "source_state", "target_state", "count", "total", "fraction"])
    counts = pd.concat(pieces, ignore_index=True).value_counts(["metric", "source_state", "target_state"]).rename("count").reset_index()
    totals = counts.groupby("metric", as_index=False)["count"].sum().rename(columns={"count": "total"})
    counts = counts.merge(totals, on="metric", validate="many_to_one")
    counts["fraction"] = counts["count"] / counts["total"]
    return counts


def primary_depth(manifests: Path, metric: str | None = None) -> pd.DataFrame:
    depth = depth_labels(pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_summary.csv"))
    sub = depth.loc[depth["source_environment_id"].eq(PRIMARY_SOURCE) & depth["left_target_environment_id"].eq(PRIMARY_SOURCE) & depth["right_target_environment_id"].eq(PRIMARY_TARGET)]
    return sub.loc[sub["metric"].eq(metric)] if metric else sub


def primary_risks(manifests: Path) -> pd.DataFrame:
    path = manifests / "reliability_transport_measurement_depth_matched_fixed_risks.csv"
    cols = ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "perturbation_label", "measurement_replicate", "cell_budget_label", "left_risk", "right_risk"]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=cols, dtype={"cell_budget_label": "string"}, chunksize=250_000):
        keep = chunk.loc[chunk["source_environment_id"].eq(PRIMARY_SOURCE) & chunk["left_target_environment_id"].eq(PRIMARY_SOURCE) & chunk["right_target_environment_id"].eq(PRIMARY_TARGET) & chunk["metric"].eq(PRIMARY_METRIC)]
        if not keep.empty:
            pieces.append(keep)
    if not pieces:
        raise ValueError("Missing primary matched-fixed risk rows")
    return pd.concat(pieces, ignore_index=True)


def decision_summary(manifests: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv")
    link = pd.read_csv(manifests / "reliability_transport_decision_link.csv")
    return summary, link


def spearman_pair(frame: pd.DataFrame, x: str = "response_displacement", y: str = "reordering_burden") -> float:
    values = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(values) < 3 or values.nunique().min() < 2:
        return float("nan")
    return float(spearmanr(values[x], values[y]).statistic)


def full_predictability(manifests: Path) -> pd.DataFrame:
    pred = pd.read_csv(manifests / "reliability_transport_predictability.csv")
    return pred.loc[pred["status"].eq("executed") & pred["cell_budget_label"].eq("full") & pred["dataset"].eq(ATLAS_DATASET) & pred["dataset_split"].eq("within_dataset_leave_one_perturbation_label_out") & pred["model"].isin(["source_u_only", "linear_source_features"])].assign(spearman=lambda d: pd.to_numeric(d["spearman"], errors="coerce"), auroc_high_identifiable=lambda d: pd.to_numeric(d["auroc_high_identifiable"], errors="coerce"))


def add_metric_legend(ax: plt.Axes, metrics=METRICS, **kwargs: Any) -> None:
    handles = [mpl.lines.Line2D([], [], color=metric_color(metric), marker="o", lw=1.7, label=metric_label(metric)) for metric in metrics]
    ax.legend(handles=handles, frameon=False, **kwargs)


__all__ = ["METRICS", "PRIMARY_SOURCE", "PRIMARY_TARGET", "PRIMARY_METRIC", "ATLAS_DATASET", "STATE_ORDER", "STATE_COLORS", "FLOOR_COLORS", "FEATURES", "FORBIDDEN_TARGET_FIELDS", "DEPTH_ORDER", "CONTEXT_ORDER", "CONTRAST_ORDER", "DEPTH_VALUE", "num", "source_path", "write_source", "json_records", "depth_labels", "budget_label", "set_title", "arrow", "rank_table", "rank_flow", "failure_cases", "build_exemplar_registry", "state_label", "state_counts", "primary_depth", "primary_risks", "decision_summary", "spearman_pair", "full_predictability", "add_metric_legend", "add_panel_label", "clean_axes", "context_label", "metric_color", "metric_label", "save_figure"]
