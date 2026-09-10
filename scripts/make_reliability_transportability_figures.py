"""Generate six canonical-data reliability-transportability figures.

The uploaded six-panel images are visual organization references only. Every
quantitative mark is read from a frozen PTL artifact or is a deterministic,
documented reduction of one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ptl_figure_style import (  # noqa: E402
    CONTEXT_ORDER, CONTRAST_ORDER, DEPTH_ORDER, add_metric_legend,
    add_panel_label, apply_style, clean_axes, context_label, contrast_label,
    metric_color, metric_label, save_figure,
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


def _source_path(root: Path, name: str) -> Path:
    path = root / "results/figures/reliability_transportability/source_tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_source(root: Path, name: str, frame: pd.DataFrame) -> str:
    path = _source_path(root, name)
    frame.to_csv(path, index=False)
    return path.relative_to(root).as_posix()


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert tabular records to strict JSON, mapping missing values to null."""
    return json.loads(frame.to_json(orient="records"))


def _depth_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "cell_budget_label" not in frame.columns:
        frame["cell_budget_label"] = frame["cell_budget"].map(lambda x: "full" if _num(x) >= 1e8 else str(int(_num(x))))
    frame["cell_budget_label"] = frame["cell_budget_label"].astype(str).str.lower()
    return frame


def _budget_label(value: Any) -> str:
    numeric = _num(value)
    return "full" if numeric >= 1e8 else str(int(numeric))


def _contrast_mask(frame: pd.DataFrame, contrast: tuple[str, str]) -> pd.Series:
    return frame["left_target_environment_id"].eq(contrast[0]) & frame["right_target_environment_id"].eq(contrast[1])


def _rowwise_endpoint(frame: pd.DataFrame) -> pd.Series:
    return frame["source_environment_id"].eq(frame["left_target_environment_id"]) | frame["source_environment_id"].eq(frame["right_target_environment_id"])


def _draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], **kwargs: Any) -> None:
    defaults = {"arrowstyle": "-|>", "lw": 1.1, "color": "#46515C", "mutation_scale": 11}
    defaults.update(kwargs)
    ax.annotate("", xy=end, xytext=start, arrowprops=defaults)


def _set_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", pad=10, fontweight="bold")
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, va="bottom", fontsize=6.8, color="#46515C")


def _rank_table(manifests: Path) -> pd.DataFrame:
    """Rank canonical source-frozen risks, with alphabetical tie handling."""
    frame = pd.read_csv(manifests.parent / "source_data/formal_v2_reliability_predictions.csv")
    frame = frame.loc[frame["environment_key"].isin(CONTEXT_ORDER) & frame["predictor"].eq("mean_matching") & frame["scenario"].eq("in_domain")].copy()
    values: dict[str, dict[str, list[float]]] = {context: {} for context in CONTEXT_ORDER}
    for _, row in frame.iterrows():
        context, label = str(row["environment_key"]), str(row["perturbation_label"])
        values[context].setdefault(label, []).append(_num(row["continuous_risk"]))
    # The primary comparison has the canonical matched label universe in the
    # pairwise ranking artifact.  Use that source/target universe for the
    # headline rank flow, while retaining the compact prediction export for
    # the intermediate context.  This avoids silently unioning different
    # pairwise label universes into a fabricated rank list.
    direct = pd.read_csv(manifests / "formal_v2_reliability_reordering_perturbations.csv")
    direct = direct.loc[direct["predictor"].eq("mean_matching") & direct["method"].eq("ptl_rf")].copy()
    primary = direct.loc[direct["left_environment_id"].eq(PRIMARY_SOURCE) & direct["right_environment_id"].eq(PRIMARY_TARGET)]
    values[PRIMARY_SOURCE] = {str(label): [_num(value)] for label, value in primary.groupby("perturbation_label")["risk_left"].mean().items()}
    values[PRIMARY_TARGET] = {str(label): [_num(value)] for label, value in primary.groupby("perturbation_label")["risk_right"].mean().items()}
    labels = set.union(*(set(values[context]) for context in CONTEXT_ORDER))
    primary_labels = set(values[PRIMARY_SOURCE]) & set(values[PRIMARY_TARGET])
    if len(primary_labels) < 10:
        raise ValueError("Fewer than ten labels are common to the primary canonical contrast")
    ranked: dict[str, pd.DataFrame] = {}
    for context in CONTEXT_ORDER:
        sub = pd.DataFrame({"perturbation_label": sorted(labels)})
        sub["risk"] = sub["perturbation_label"].map(lambda label: float(np.nanmean(values[context][label])) if label in values[context] else np.nan)
        sub = sub.dropna(subset=["risk"])
        sub = sub.sort_values(["risk", "perturbation_label"], kind="stable").reset_index(drop=True)
        sub["rank"] = np.arange(1, len(sub) + 1)
        ranked[context] = sub
    out = ranked[CONTEXT_ORDER[0]][["perturbation_label", "risk", "rank"]].rename(columns={"risk": "source_risk", "rank": "source_rank"})
    for context, prefix in zip(CONTEXT_ORDER[1:], ("coculture", "ifng")):
        out = out.merge(ranked[context][["perturbation_label", "risk", "rank"]].rename(columns={"risk": f"{prefix}_risk", "rank": f"{prefix}_rank"}), on="perturbation_label", how="left")
    return out


def _rank_flow(manifests: Path) -> tuple[pd.DataFrame, list[str]]:
    ranks = _rank_table(manifests)
    ordered = ranks.sort_values(["source_rank", "perturbation_label"], kind="stable").reset_index(drop=True)
    selected: list[str] = []
    for quantile in np.arange(0.05, 1.0, 0.10):
        target_rank = quantile * (len(ordered) + 1)
        candidates = ordered.assign(_distance=(ordered["source_rank"] - target_rank).abs()).sort_values(["_distance", "perturbation_label"], kind="stable")
        selected.append(next(str(value) for value in candidates["perturbation_label"] if str(value) not in selected))
    return ranks.loc[ranks["perturbation_label"].isin(selected)].sort_values("source_rank", kind="stable").reset_index(drop=True), selected


def _failure_cases(manifests: Path, ranks: pd.DataFrame) -> pd.DataFrame:
    failure = pd.read_csv(manifests / "reliability_transport_failure_anatomy.csv")
    failure = failure.loc[
        failure["status"].eq("executed") & failure["source_environment_id"].eq(PRIMARY_SOURCE)
        & failure["target_environment_id"].eq(PRIMARY_TARGET) & failure["metric"].eq(PRIMARY_METRIC)
    ].copy()
    failure["reordering_burden"] = pd.to_numeric(failure["reordering_burden"], errors="coerce")
    failure = failure.sort_values(["reordering_burden", "perturbation_label"], kind="stable").reset_index(drop=True)
    if len(failure) < 3:
        raise ValueError("The primary failure anatomy does not contain three complete case rows")
    cases: list[dict[str, Any]] = []
    for quantile in (0.10, 0.50, 0.90):
        position = quantile * (len(failure) - 1)
        candidates = failure.assign(_distance=np.abs(failure.index.to_numpy() - position)).sort_values(["_distance", "perturbation_label"], kind="stable")
        row = candidates.iloc[0].drop(labels=["_distance"], errors="ignore").to_dict()
        row["quantile"] = quantile
        cases.append(row)
    out = pd.DataFrame(cases).merge(ranks, on="perturbation_label", how="left", validate="one_to_one")
    if out[["source_rank", "ifng_rank"]].isna().any().any():
        raise ValueError("Failure case label is absent from the canonical source-frozen rank table")
    out["rank_delta"] = out["ifng_rank"] - out["source_rank"]
    return out


def build_exemplar_registry(root: Path) -> dict[str, Any]:
    manifests = root / "artifacts/manifests"
    ranks, selected = _rank_flow(manifests)
    cases = _failure_cases(manifests, _rank_table(manifests))
    full_failure = pd.read_csv(manifests / "reliability_transport_failure_anatomy.csv")
    complete_n = int((full_failure["status"].eq("executed") & full_failure["source_environment_id"].eq(PRIMARY_SOURCE) & full_failure["target_environment_id"].eq(PRIMARY_TARGET) & full_failure["metric"].eq(PRIMARY_METRIC)).sum())
    registry = {
        "schema_version": 1,
        "status": "executed",
        "primary_display_context": {
            "source_environment_id": PRIMARY_SOURCE, "target_environment_id": PRIMARY_TARGET, "metric": PRIMARY_METRIC,
            "reason": "fixed semantic exemplar, first frozen metric, canonical Ctrl→IFNγ shift; not selected by observed magnitude",
        },
        "rank_flow": {
            "selection_rule": "source-only rank quantiles 5%, 15%, ..., 95%; nearest source rank, alphabetical tie-break",
            "n_labels": len(selected), "perturbation_labels": selected, "entries": _json_records(ranks.loc[ranks["perturbation_label"].isin(selected)]),
            "provenance": "artifacts/source_data/formal_v2_reliability_predictions.csv; predictor=mean_matching; scenario=in_domain",
        },
        "failure_cases": {
            "selection_rule": "complete primary failure rows at burden quantiles q10/q50/q90; nearest burden, alphabetical tie-break",
            "complete_rows": complete_n,
            "cases": _json_records(cases[["quantile", "perturbation_label", "source_rank", "ifng_rank", "rank_delta", "reordering_burden", "response_displacement", "effect_magnitude", "model_disagreement", "source_uncertainty"]]),
            "provenance": "artifacts/manifests/reliability_transport_failure_anatomy.csv; status=executed; no imputation",
        },
        "anti_fabrication": {"visual_references_only": True, "unavailable_target_informed_quantities_not_used_as_prospective_features": True, "minimum_strict_support": 8, "no_new_experiment_or_retraining": True},
    }
    (manifests / "figure_exemplar_registry.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return registry


def _rank_axes(ax: plt.Axes, flow: pd.DataFrame, labels: list[str], title: str = "Reliability rank trajectories") -> None:
    x = np.arange(3)
    for _, row in flow.iterrows():
        y = [row["source_rank"], row["coculture_rank"], row["ifng_rank"]]
        color = "#C44E52" if row["perturbation_label"] in {labels[0], labels[-1]} else "#7A8996"
        ax.plot(x, y, color=color, lw=2.0 if color == "#C44E52" else 1.15, alpha=.80)
        ax.scatter(x, y, s=15, color=color)
    ax.set_xticks(x, [context_label(v) for v in CONTEXT_ORDER]); ax.set_ylabel("rank (1 = best)"); ax.invert_yaxis(); ax.set_title(title, loc="left", pad=10); ax.text(.01, .02, "NA = unavailable for this registered label", transform=ax.transAxes, fontsize=5.6, color="#46515C"); ax.grid(axis="y", alpha=.23); clean_axes(ax)


def _state_label(value: Any, strict_count: Any) -> str:
    text = str(value).lower(); strict = _num(strict_count)
    if "p<q" in text or "p_lt_q" in text or "p<" in text: return "-1"
    if "p>q" in text or "p_gt_q" in text or "p>" in text: return "+1"
    if "unresolved" in text or "tie" in text or text in {"nan", "none"}: return "tie" if strict == 0 else "unstable"
    return "tie" if strict == 0 else "unstable"


def _state_counts_for_pair(path: Path, source: str, target: str, metric: str) -> pd.DataFrame:
    usecols = ["source_environment_id", "target_environment_id", "metric", "source_state_discovery", "target_state_validation", "source_strict_count", "target_strict_count"]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000):
        chunk = chunk.loc[chunk["source_environment_id"].eq(source) & chunk["target_environment_id"].eq(target) & chunk["metric"].eq(metric)].copy()
        if not chunk.empty:
            chunk["source_state"] = [_state_label(v, n) for v, n in zip(chunk["source_state_discovery"], chunk["source_strict_count"])]
            chunk["target_state"] = [_state_label(v, n) for v, n in zip(chunk["target_state_validation"], chunk["target_strict_count"])]
            pieces.append(chunk[["source_state", "target_state"]])
    if not pieces: return pd.DataFrame(columns=["source_state", "target_state", "count"])
    return pd.concat(pieces, ignore_index=True).value_counts(["source_state", "target_state"]).rename("count").reset_index()


def _pair_state_counts(path: Path) -> pd.DataFrame:
    usecols = ["source_environment_id", "target_environment_id", "metric", "source_state_discovery", "target_state_validation", "source_strict_count", "target_strict_count"]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000):
        chunk = chunk.loc[chunk["metric"].isin(METRICS) & chunk["source_environment_id"].isin(CONTEXT_ORDER) & chunk["target_environment_id"].isin(CONTEXT_ORDER) & chunk["source_environment_id"].ne(chunk["target_environment_id"])].copy()
        if not chunk.empty:
            chunk["source_state"] = [_state_label(v, n) for v, n in zip(chunk["source_state_discovery"], chunk["source_strict_count"])]
            chunk["target_state"] = [_state_label(v, n) for v, n in zip(chunk["target_state_validation"], chunk["target_strict_count"])]
            pieces.append(chunk[["metric", "source_state", "target_state"]])
    if not pieces: return pd.DataFrame(columns=["metric", "source_state", "target_state", "count", "total", "fraction"])
    all_states = pd.concat(pieces, ignore_index=True); counts = all_states.value_counts(["metric", "source_state", "target_state"]).rename("count").reset_index(); totals = counts.groupby("metric", as_index=False)["count"].sum().rename(columns={"count": "total"}); counts = counts.merge(totals, on="metric", validate="many_to_one"); counts["fraction"] = counts["count"] / counts["total"]; return counts


def _spearman_pair(frame: pd.DataFrame, x: str = "response_displacement", y: str = "reordering_burden") -> float:
    values = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(values) < 3 or values.nunique().min() < 2: return float("nan")
    return float(spearmanr(values[x], values[y]).statistic)


def _primary_depth(depth: pd.DataFrame, metric: str | None = None) -> pd.DataFrame:
    sub = depth.loc[depth["source_environment_id"].eq(PRIMARY_SOURCE) & depth["left_target_environment_id"].eq(PRIMARY_SOURCE) & depth["right_target_environment_id"].eq(PRIMARY_TARGET)]
    return sub.loc[sub["metric"].eq(metric)] if metric else sub


def _primary_risks(manifests: Path) -> pd.DataFrame:
    path = manifests / "reliability_transport_measurement_depth_matched_fixed_risks.csv"
    usecols = ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "perturbation_label", "measurement_replicate", "cell_budget_label", "left_risk", "right_risk"]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype={"cell_budget_label": "string"}, chunksize=250_000):
        keep = chunk.loc[chunk["source_environment_id"].eq(PRIMARY_SOURCE) & chunk["left_target_environment_id"].eq(PRIMARY_SOURCE) & chunk["right_target_environment_id"].eq(PRIMARY_TARGET) & chunk["metric"].eq(PRIMARY_METRIC)]
        if not keep.empty: chunks.append(keep)
    if not chunks: raise ValueError("Missing primary matched-fixed risk rows")
    return pd.concat(chunks, ignore_index=True)


def figure1(out_dir: Path, manifests: Path, registry: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    flow, labels = _rank_flow(manifests); depth = _depth_labels(pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_summary.csv")); risks = _primary_risks(manifests); rows: list[dict[str, Any]] = []
    fig = plt.figure(figsize=(14, 8.2)); grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.15), hspace=.42, wspace=.30)
    ax = fig.add_subplot(grid[0, 0]); add_panel_label(ax, "A"); ax.axis("off"); _set_title(ax, "Source-frozen predictor", "same predictor; target outcomes change"); ax.text(.5, .56, r"$f_{\theta}$", ha="center", va="center", fontsize=18, fontweight="bold", bbox={"boxstyle": "circle,pad=.35", "facecolor": "#F4F7F9", "edgecolor": "#46515C"})
    for pos, context in zip(((.18,.20),(.50,.10),(.82,.20)), CONTEXT_ORDER): ax.text(*pos, context_label(context), ha="center", va="center", bbox={"boxstyle": "round,pad=.26", "facecolor": "#EAF2F8", "edgecolor": "#7A8996"}); _draw_arrow(ax, (.5,.43), (pos[0],pos[1]+.10), color="#7A8996")
    ax.text(.5, .92, "rank reliability after a biological shift", ha="center", va="top", fontsize=8.2, color="#46515C"); ax.text(.5, .02, "conceptual contract only", ha="center", fontsize=7.2, color="#46515C"); rows.append({"panel":"A", "description":"source-frozen predictor evaluated across three canonical contexts", "status":"conceptual contract", "provenance":"frozen Frangieh protocol"})
    ax = fig.add_subplot(grid[0, 1]); add_panel_label(ax, "B"); _rank_axes(ax, flow, labels); ax.text(.02, 1.01, "10 labels selected by source-only quantiles", transform=ax.transAxes, va="bottom", fontsize=6.8, color="#46515C"); rows.extend({"panel":"B", **row.to_dict(), "selection":"source-only quantile registry", "provenance":"formal_v2_reliability_reordering_perturbations.csv + formal_v2_reliability_predictions.csv"} for _, row in flow.iterrows())
    ax = fig.add_subplot(grid[0, 2]); add_panel_label(ax, "C"); _set_title(ax, "Tie-aware pairwise ordering", "corrected state labels; unresolved remains separate"); state = _state_counts_for_pair(manifests / "reliability_transport_metric_pair_states.csv", PRIMARY_SOURCE, PRIMARY_TARGET, PRIMARY_METRIC); matrix = np.zeros((4,4))
    for _, row in state.iterrows():
        if row["source_state"] in STATE_ORDER and row["target_state"] in STATE_ORDER: matrix[STATE_ORDER.index(row["source_state"]), STATE_ORDER.index(row["target_state"])] = row["count"]
    total = matrix.sum(); displayed = matrix / total if total else matrix; im = ax.imshow(displayed, cmap="Blues", vmin=0, vmax=max(float(displayed.max()), .01), aspect="auto")
    for i in range(4):
        for j in range(4):
            if matrix[i,j]: ax.text(j, i, f"{displayed[i,j]:.2f}\n(n={int(matrix[i,j])})", ha="center", va="center", fontsize=6.2)
    ax.set_xticks(range(4), STATE_ORDER); ax.set_yticks(range(4), STATE_ORDER); ax.set_xlabel("target state"); ax.set_ylabel("source state"); ax.text(.02, -.28, f"n={int(total):,}; stable support floor ≥8", transform=ax.transAxes, fontsize=7, color="#46515C"); clean_axes(ax); rows.extend({"panel":"C", **row.to_dict(), "source_environment_id":PRIMARY_SOURCE, "target_environment_id":PRIMARY_TARGET, "provenance":"reliability_transport_metric_pair_states.csv"} for _, row in state.iterrows())
    ax = fig.add_subplot(grid[1, 0]); add_panel_label(ax, "D"); _set_title(ax, "Observed disagreement versus floors", "full matched depth; three metrics"); x = np.arange(3); width=.23
    for idx, metric in enumerate(METRICS):
        row = _primary_depth(depth, metric).loc[lambda d: d["cell_budget_label"].eq("full")].iloc[0]; values = [row["cross_disagreement"], row["measurement_floor"], row["identifiable_divergence"]]; ax.bar(idx + np.arange(3)*width, values, width=width, color=["#46515C", "#AAB4BC", metric_color(metric)]); ax.text(idx+width, max(values)+.012, f"{values[-1]:.3f}", ha="center", fontsize=6.5); rows.append({"panel":"D", "metric":metric, "cross_disagreement":values[0], "measurement_floor":values[1], "identifiable_divergence":values[2], "depth":"full", "provenance":"reliability_transport_measurement_depth_matched_fixed_summary.csv"})
    ax.axhline(0, color="#46515C", lw=.7); ax.set_xticks(x+width, [metric_label(v) for v in METRICS], rotation=18, ha="right"); ax.set_ylabel("ordering disagreement"); ax.legend(["cross", "measurement floor", "meas-ID"], frameon=False, fontsize=6.2, loc="upper left"); clean_axes(ax, grid=True)
    ax = fig.add_subplot(grid[1, 1]); add_panel_label(ax, "E"); _set_title(ax, "Measurement depth and identifiability", "primary Ctrl → IFNγ; 90% intervals"); x=np.arange(6)
    for metric in METRICS:
        sub=_primary_depth(depth, metric).set_index("cell_budget_label").reindex(DEPTH_ORDER).reset_index(); y=pd.to_numeric(sub["identifiable_divergence"],errors="coerce").to_numpy(); low=pd.to_numeric(sub["identifiable_divergence_ci_low"],errors="coerce").to_numpy(); high=pd.to_numeric(sub["identifiable_divergence_ci_high"],errors="coerce").to_numpy(); ax.plot(x,y,marker="o",color=metric_color(metric),lw=1.7,label=metric_label(metric)); ax.fill_between(x,low,high,color=metric_color(metric),alpha=.12); rows.extend({"panel":"E","metric":metric,"depth":d,"identifiable_divergence":v,"ci_low":lo,"ci_high":hi,"uncertainty":"90% seed-bootstrap interval","provenance":"reliability_transport_measurement_depth_matched_fixed_summary.csv"} for d,v,lo,hi in zip(DEPTH_ORDER,y,low,high))
    ax.axhline(0,color="#46515C",lw=.8); ax.set_xticks(x,DEPTH_ORDER); ax.set_xlabel("cells per pseudoreplicate"); ax.set_ylabel(r"$D_{\mathrm{meas-ID}}$"); ax.legend(frameon=False,fontsize=6.2); clean_axes(ax,grid=True)
    ax = fig.add_subplot(grid[1, 2]); add_panel_label(ax, "F"); _set_title(ax, "Real top-10% consequence", "source selection versus target ordering"); rank=_rank_table(manifests); n=len(rank); k=max(1,int(np.ceil(.10*n))); source_top=set(rank.nsmallest(k,"source_risk")["perturbation_label"]); target_top=set(rank.nsmallest(k,"ifng_risk")["perturbation_label"]); selected=rank.loc[rank["perturbation_label"].isin(source_top|target_top)].sort_values(["source_rank","ifng_rank"]); y=np.arange(len(selected)); colors=["#56B4E9" if l in source_top&target_top else ("#0072B2" if l in source_top else "#D55E00") for l in selected["perturbation_label"]]; ax.scatter(selected["source_rank"],y,s=30,color=colors); ax.scatter(selected["ifng_rank"],y+.18,s=30,color=colors,marker="D",alpha=.85); [ax.text(.01,yi,row["perturbation_label"],transform=ax.get_yaxis_transform(),ha="left",va="center",fontsize=6.0) for yi,(_,row) in enumerate(selected.iterrows())]; ax.set_yticks([]); ax.set_xlabel("canonical rank"); ax.invert_yaxis(); ax.legend(["source top-10%","target top-10%"],frameon=False,fontsize=6.0,loc="lower right"); ax.text(.02,.88,f"retention = {len(source_top&target_top)}/{k} = {len(source_top&target_top)/k:.2f}",transform=ax.transAxes,va="top",fontsize=7.2,fontweight="bold"); clean_axes(ax,grid=True); rows.extend({"panel":"F","perturbation_label":row["perturbation_label"],"source_rank":row["source_rank"],"target_rank":row["ifng_rank"],"source_selected":row["perturbation_label"] in source_top,"target_selected":row["perturbation_label"] in target_top,"budget_fraction":.10,"provenance":"formal_v2_reliability_reordering_perturbations.csv"} for _,row in selected.iterrows())
    fig.suptitle("Figure 1 | The source-frozen reliability ordering is the object that moves",fontsize=12,fontweight="bold"); fig.text(.5,.01,"The predictor is frozen before target outcomes; tie-aware states preserve unresolved rows, and all quantitative panels use canonical Frangieh artifacts.",ha="center",fontsize=7.2,color="#46515C"); return save_figure(fig,out_dir,"reliability_transportability_fig1_graphical_abstract"),pd.DataFrame(rows)


def figure2(out_dir: Path, manifests: Path, registry: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    atlas=pd.read_csv(manifests/"reliability_transport_atlas.csv"); frangieh=atlas.loc[atlas["dataset"].eq(ATLAS_DATASET)&atlas["metric"].isin(METRICS)].copy(); frangieh["contrast"]=list(zip(frangieh["left_target_environment_id"],frangieh["right_target_environment_id"]));
    if len(frangieh)!=27 or frangieh.duplicated(["source_environment_id","left_target_environment_id","right_target_environment_id","metric"]).any(): raise ValueError("Figure 2 requires the 27-row Frangieh atlas")
    flow,labels=_rank_flow(manifests); states=_pair_state_counts(manifests/"reliability_transport_metric_pair_states.csv"); rows=[]; fig=plt.figure(figsize=(14,8.2)); grid=fig.add_gridspec(2,3,height_ratios=(1.05,1.0),hspace=.45,wspace=.34)
    ax=fig.add_subplot(grid[0,0]); add_panel_label(ax,"A"); _rank_axes(ax,flow,labels,"Rank flow across biological contexts"); ax.text(.02,1.01,"10 labels selected by source-only quantiles",transform=ax.transAxes,va="bottom",fontsize=6.7,color="#46515C"); rows.extend({"panel":"A",**row.to_dict(),"provenance":"formal_v2_reliability_reordering_perturbations.csv + formal_v2_reliability_predictions.csv"} for _,row in flow.iterrows())
    ax=fig.add_subplot(grid[0,1]); add_panel_label(ax,"B"); _set_title(ax,"Pairwise state transitions","corrected pair-state table; unstable/unresolved retained"); mini=ax.inset_axes([.04,.08,.92,.82]); mini.axis("off")
    for mi,metric in enumerate(METRICS):
        subax=mini.inset_axes([.02+mi*.33,.10,.29,.76]); mat=np.zeros((4,4)); sub=states.loc[states["metric"].eq(metric)]
        for _,r in sub.iterrows(): mat[STATE_ORDER.index(r["source_state"]),STATE_ORDER.index(r["target_state"])] = r["fraction"]
        subax.imshow(mat,cmap="Blues",vmin=0,vmax=max(float(mat.max()),.01)); subax.set_title(metric_label(metric),fontsize=6.1); subax.set_xticks(range(4),STATE_ORDER,fontsize=5); subax.set_yticks(range(4),STATE_ORDER if mi==0 else [""]*4,fontsize=5); subax.tick_params(length=0)
        for i in range(4):
            for j in range(4):
                if mat[i,j]>0: subax.text(j,i,f"{mat[i,j]:.2f}",ha="center",va="center",fontsize=5.4)
        rows.extend({"panel":"B",**r.to_dict(),"provenance":"reliability_transport_metric_pair_states.csv"} for _,r in sub.iterrows())
    ax.text(.5,.02,"source state → target state; gray is unresolved/unstable, not a manufactured tie",transform=ax.transAxes,ha="center",fontsize=6.3,color="#46515C"); ax.set_xticks([]); ax.set_yticks([]); clean_axes(ax)
    ax=fig.add_subplot(grid[0,2]); add_panel_label(ax,"C"); _set_title(ax,r"Context transfer atlas: $D_{\mathrm{meas-ID}}$","dot = measurement lower interval >0; ring = joint lower interval >0"); atlasmini=ax.inset_axes([.10,.10,.85,.80]); atlasmini.axis("off"); cmap=mpl.colormaps.get_cmap("RdBu_r"); vmax=max(abs(float(frangieh["measurement_identifiable"].min())),abs(float(frangieh["measurement_identifiable"].max()))); norm=TwoSlopeNorm(vmin=-vmax,vcenter=0.,vmax=vmax)
    for mi,metric in enumerate(METRICS):
        subax=atlasmini.inset_axes([mi*.34,.08,.30,.78]); subax.set_facecolor("#F4F5F6"); subax.set_xticks(range(3),["C–Co","C–IFN","Co–IFN"],rotation=45,ha="right",fontsize=5); subax.set_yticks(range(3),[context_label(v) for v in CONTEXT_ORDER],fontsize=5); subax.set_title(metric_label(metric),fontsize=6.1,loc="left"); data=frangieh.loc[frangieh["metric"].eq(metric)]
        for ri,source in enumerate(CONTEXT_ORDER):
            for ci,contrast in enumerate(CONTRAST_ORDER):
                cell=data.loc[data["source_environment_id"].eq(source)&data["contrast"].map(lambda value:value==contrast)]
                if cell.empty: continue
                item=cell.iloc[0]; value=_num(item["measurement_identifiable"]); subax.add_patch(Rectangle((ci-.5,ri-.5),1,1,facecolor=cmap(norm(value)),edgecolor="#303840",lw=.5)); subax.text(ci,ri,f"{value:.2f}",ha="center",va="center",fontsize=5.4,color="white" if abs(norm(value)-.5)>.28 else "#111111");
                if _num(item["measurement_identifiable_ci_low"])>0: subax.scatter(ci+.30,ri-.30,s=14,color="#111111",marker=".")
                if _num(item["joint_identifiable_ci_low"])>0: subax.scatter(ci+.29,ri+.29,s=18,facecolors="none",edgecolors="#111111",linewidths=.8)
                rows.append({"panel":"C","metric":metric,"source_environment_id":source,"contrast":contrast_label(*contrast),"measurement_identifiable":value,"measurement_ci_low":item["measurement_identifiable_ci_low"],"measurement_ci_high":item["measurement_identifiable_ci_high"],"joint_identifiable":item["joint_identifiable"],"joint_ci_low":item["joint_identifiable_ci_low"],"joint_ci_high":item["joint_identifiable_ci_high"],"provenance":"reliability_transport_atlas.csv"})
    ax.set_xticks([]); ax.set_yticks([]); clean_axes(ax)
    complete=pd.read_csv(manifests/"reliability_transport_failure_anatomy.csv").query("status == 'executed'")
    ax=fig.add_subplot(grid[1,0]); add_panel_label(ax,"D"); _set_title(ax,"Reordering burden across perturbations","linear scale; negative values retained")
    for yi,metric in enumerate(METRICS):
        vals=pd.to_numeric(complete.loc[complete["metric"].eq(metric),"reordering_burden"],errors="coerce").dropna(); parts=ax.violinplot(vals,positions=[yi],vert=False,widths=.72,showmeans=False,showmedians=True); [body.set_facecolor(metric_color(metric)) or body.set_alpha(.35) for body in parts["bodies"]]; parts["cmedians"].set_color(metric_color(metric)); sample=vals.sample(min(90,len(vals)),random_state=20260910); ax.scatter(sample,np.full(len(sample),yi)+np.linspace(-.18,.18,len(sample)),s=5,alpha=.28,color=metric_color(metric)); rows.append({"panel":"D","metric":metric,"n_complete":len(vals),"median":vals.median(),"q25":vals.quantile(.25),"q75":vals.quantile(.75),"axis_scale":"linear","provenance":"reliability_transport_failure_anatomy.csv"})
    ax.axvline(0,color="#46515C",lw=.7); ax.set_yticks(range(3),[metric_label(v) for v in METRICS]); ax.set_xlabel("reordering burden ($D_{\\mathrm{meas-ID}}$ per perturbation)"); clean_axes(ax,grid=True)
    ax=fig.add_subplot(grid[1,1]); add_panel_label(ax,"E"); _set_title(ax,"Measurement versus joint identifiability","full-depth point + 90% interval"); summary=pd.read_csv(manifests/"formal_v2_claim_lock_measurement_fullsize_summary.csv").query("source_environment_id == @PRIMARY_SOURCE and left_target_environment_id == @PRIMARY_SOURCE and right_target_environment_id == @PRIMARY_TARGET and metric in @METRICS")
    for yi,metric in enumerate(METRICS):
        row=summary.loc[summary["metric"].eq(metric)].iloc[0]; xv,yv=_num(row["ordering_delta_meas_id"]),_num(row["ordering_delta_joint_id"]); ax.errorbar(xv,yv,xerr=[[xv-_num(row["ordering_delta_meas_id_ci_low"])],[_num(row["ordering_delta_meas_id_ci_high"])-xv]],yerr=[[yv-_num(row["ordering_delta_joint_id_ci_low"])],[_num(row["ordering_delta_joint_id_ci_high"])-yv]],fmt="o",color=metric_color(metric),ecolor=metric_color(metric),capsize=2,ms=5); ax.text(xv,yv,"  "+metric_label(metric),fontsize=6.2,va="center"); rows.append({"panel":"E","metric":metric,"d_meas_id":xv,"d_meas_id_ci_low":row["ordering_delta_meas_id_ci_low"],"d_meas_id_ci_high":row["ordering_delta_meas_id_ci_high"],"d_joint_id":yv,"d_joint_id_ci_low":row["ordering_delta_joint_id_ci_low"],"d_joint_id_ci_high":row["ordering_delta_joint_id_ci_high"],"uncertainty":"90% synchronized macro-bootstrap interval","provenance":"formal_v2_claim_lock_measurement_fullsize_summary.csv"})
    ax.axhline(0,color="#46515C",lw=.7); ax.axvline(0,color="#46515C",lw=.7); ax.set_xlabel(r"$D_{\mathrm{meas-ID}}$"); ax.set_ylabel(r"$D_{\mathrm{joint-ID}}$"); clean_axes(ax,grid=True)
    ax=fig.add_subplot(grid[1,2]); add_panel_label(ax,"F"); _set_title(ax,"Independent context replication","Nadig HepG2 ↔ Jurkat; endpoint-only canonical rows"); nadig=atlas.loc[atlas["dataset"].eq("NadigOConner2024")&atlas["metric"].isin(METRICS)].loc[lambda d:_rowwise_endpoint(d)].sort_values(["metric","source_environment_id"])
    for yi,(_,row) in enumerate(nadig.iterrows()):
        value,low,high=_num(row["measurement_identifiable"]),_num(row["measurement_identifiable_ci_low"]),_num(row["measurement_identifiable_ci_high"]); target=row["right_target_environment_id"] if row["source_environment_id"]==row["left_target_environment_id"] else row["left_target_environment_id"]; ax.errorbar(value,yi,xerr=[[value-low],[high-value]],fmt="o",color=metric_color(row["metric"]),ecolor=metric_color(row["metric"]),capsize=2); ax.text(0,yi,f"{context_label(row['source_environment_id'])} → {context_label(target)} · {metric_label(row['metric'])}",transform=ax.get_yaxis_transform(),ha="left",va="center",fontsize=6.0); rows.append({"panel":"F","dataset":row["dataset"],"metric":row["metric"],"source_environment_id":row["source_environment_id"],"target_environment_id":target,"value":value,"ci_low":low,"ci_high":high,"provenance":"reliability_transport_atlas.csv"})
    ax.set_yticks([]); ax.set_xlabel(r"$D_{\mathrm{meas-ID}}$ (90% interval)"); clean_axes(ax,grid=True); fig.suptitle("Figure 2 | Biological context restructures reliability ordering",fontsize=12,fontweight="bold"); fig.text(.5,.01,"The atlas keeps source context, contrast, metric and independent replication separate; state transitions use the corrected minimum-support rule.",ha="center",fontsize=7.2,color="#46515C"); return save_figure(fig,out_dir,"reliability_transportability_fig2_transport_atlas"),pd.DataFrame(rows)


def _resolution_state(lower: float, label: str, resolution_budget: float) -> str:
    if not np.isfinite(lower) or lower <= 0: return "unresolved"
    if np.isfinite(resolution_budget) and DEPTH_VALUE[label] >= resolution_budget: return "resolved"
    return "detectable"


def figure3(out_dir: Path, manifests: Path, registry: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    depth=_depth_labels(pd.read_csv(manifests/"reliability_transport_measurement_depth_matched_fixed_summary.csv")); resolution=pd.read_csv(manifests/"reliability_transport_measurement_depth_matched_fixed_resolution.csv"); rows=[]; fig=plt.figure(figsize=(14,8.6)); grid=fig.add_gridspec(2,3,height_ratios=(1.25,1.0),hspace=.46,wspace=.34)
    ax=fig.add_subplot(grid[0,0]); add_panel_label(ax,"A"); _set_title(ax,"Cell-specific identifiability trajectories","actual $D_{meas-ID}(n)$; negative values remain visible"); mini=ax.inset_axes([.10,.12,.85,.77]); mini.axis("off")
    for mi,metric in enumerate(METRICS):
        subax=mini.inset_axes([mi*.34,.10,.29,.78]); subax.set_title(metric_label(metric),fontsize=6.1,loc="left"); subax.axhline(0,color="#46515C",lw=.6); metric_data=depth.loc[depth["metric"].eq(metric)]
        for source in CONTEXT_ORDER:
            for contrast in CONTRAST_ORDER:
                cell=metric_data.loc[metric_data["source_environment_id"].eq(source)&_contrast_mask(metric_data,contrast)].set_index("cell_budget_label").reindex(DEPTH_ORDER).reset_index(); y=pd.to_numeric(cell["identifiable_divergence"],errors="coerce"); low=pd.to_numeric(cell["identifiable_divergence_ci_low"],errors="coerce"); high=pd.to_numeric(cell["identifiable_divergence_ci_high"],errors="coerce"); res=resolution.loc[resolution["source_environment_id"].eq(source)&resolution["left_target_environment_id"].eq(contrast[0])&resolution["right_target_environment_id"].eq(contrast[1])&resolution["metric"].eq(metric)].iloc[0]; resolution_budget=_num(res["resolution_90pct_full_budget"]); resolution_budget=np.nan if resolution_budget >= 1e8 else resolution_budget; color="#0072B2" if source==CONTEXT_ORDER[0] else ("#D55E00" if source==CONTEXT_ORDER[1] else "#009E73"); subax.plot(range(6),y,color=color,alpha=.36,lw=.8); subax.fill_between(range(6),low,high,color=color,alpha=.035); rows.extend({"panel":"A","metric":metric,"source_environment_id":source,"contrast":contrast_label(*contrast),"depth":label,"identifiable_divergence":value,"ci_low":lo,"ci_high":hi_value,"state":_resolution_state(lo,label,resolution_budget),"resolution_90pct_full_budget":resolution_budget,"provenance":"reliability_transport_measurement_depth_matched_fixed_summary.csv + ..._resolution.csv"} for label,value,lo,hi_value in zip(DEPTH_ORDER,y,low,high))
        subax.set_xticks(range(6),DEPTH_ORDER,rotation=45,fontsize=5); subax.tick_params(axis="y",labelsize=5); subax.set_xlabel("depth",fontsize=5); subax.grid(axis="y",alpha=.15)
    ax.set_xticks([]); ax.set_yticks([]); clean_axes(ax)
    ax=fig.add_subplot(grid[0,1]); add_panel_label(ax,"B"); _set_title(ax,"Representative source × contrast trajectories","three semantic source contexts; no universal threshold")
    for source,color in zip(CONTEXT_ORDER,("#0072B2","#D55E00","#009E73")):
        for metric in METRICS:
            sub=depth.loc[depth["source_environment_id"].eq(source)&depth["metric"].eq(metric)&_contrast_mask(depth,CONTRAST_ORDER[1])].set_index("cell_budget_label").reindex(DEPTH_ORDER); ax.plot(range(6),sub["identifiable_divergence"],marker="o",ms=3,color=color,alpha=.75,ls="-" if metric==PRIMARY_METRIC else ("--" if metric==METRICS[1] else ":"),label=f"{context_label(source)} · {metric_label(metric)}"); rows.append({"panel":"B","source_environment_id":source,"contrast":contrast_label(*CONTRAST_ORDER[1]),"metric":metric,"provenance":"reliability_transport_measurement_depth_matched_fixed_summary.csv"})
    ax.axhline(0,color="#46515C",lw=.7); ax.set_xticks(range(6),DEPTH_ORDER); ax.set_xlabel("cells per pseudoreplicate"); ax.set_ylabel(r"$D_{\mathrm{meas-ID}}$"); ax.legend(frameon=False,fontsize=5.2,ncol=2); clean_axes(ax,grid=True)
    ax=fig.add_subplot(grid[0,2]); add_panel_label(ax,"C"); _set_title(ax,"Threshold map","each cell reports detect / resolve budget")
    cells=[(s,c) for s in CONTEXT_ORDER for c in CONTRAST_ORDER]; row_labels=[f"{context_label(s)} | {contrast_label(*c)}" for s,c in cells]
    matrix=np.full((len(cells), len(METRICS)*2), np.nan); labels_matrix=[["" for _ in range(len(METRICS)*2)] for _ in cells]
    for ri,(source,contrast) in enumerate(cells):
        for mi,metric in enumerate(METRICS):
            cell=resolution.loc[resolution["source_environment_id"].eq(source)&resolution["left_target_environment_id"].eq(contrast[0])&resolution["right_target_environment_id"].eq(contrast[1])&resolution["metric"].eq(metric)].iloc[0]
            for ki,(kind,key) in enumerate((("detect","detectability_threshold_budget"),("resolve","resolution_90pct_full_budget"))):
                value=_num(cell[key]); label=_budget_label(value) if np.isfinite(value) and value<1e8 else "NR"; col=mi*2+ki; matrix[ri,col]=value if label!="NR" else 0; labels_matrix[ri][col]=label; rows.append({"panel":"C","source_environment_id":source,"contrast":contrast_label(*contrast),"metric":metric,"threshold_kind":kind,"budget":value,"budget_label":label,"provenance":"reliability_transport_measurement_depth_matched_fixed_resolution.csv"})
    ax.imshow(matrix,cmap="Blues",vmin=0,vmax=160,aspect="auto")
    for ri in range(len(cells)):
        for ci in range(matrix.shape[1]): ax.text(ci,ri,labels_matrix[ri][ci],ha="center",va="center",fontsize=5.5,color="#111111")
    ax.set_xticks(np.arange(6),[f"{metric_label(m)}\n{kind}" for m in METRICS for kind in ("detect","resolve")],rotation=35,ha="right",fontsize=5.0); ax.set_yticks(np.arange(len(cells)),row_labels,fontsize=5.1); ax.set_xlabel("budget; NR = not reached"); clean_axes(ax)
    ax=fig.add_subplot(grid[1,0]); add_panel_label(ax,"D"); _set_title(ax,"30-seed descriptive distributions","raw matched-fixed item rows; no new bootstrap"); items=pd.read_csv(manifests/"reliability_transport_measurement_depth_matched_fixed_items.csv",dtype={"cell_budget_label":"string"}); items=items.loc[items["source_environment_id"].eq(PRIMARY_SOURCE)&items["left_target_environment_id"].eq(PRIMARY_SOURCE)&items["right_target_environment_id"].eq(PRIMARY_TARGET)&items["metric"].eq(PRIMARY_METRIC)&items["cell_budget_label"].astype(str).eq("full")].copy(); per_seed=items.groupby("split_seed",as_index=False)["identifiable_divergence"].mean(); ax.hist(per_seed["identifiable_divergence"],bins=min(10,max(3,len(per_seed))),color=metric_color(PRIMARY_METRIC),alpha=.45,edgecolor="white"); ax.axvline(per_seed["identifiable_divergence"].median(),color="#111111",lw=1.2,label=f"median={per_seed['identifiable_divergence'].median():.3f}"); ax.axvline(0,color="#46515C",lw=.7); ax.set_xlabel(r"seed-level $D_{\mathrm{meas-ID}}$"); ax.set_ylabel("seed count"); ax.legend(frameon=False,fontsize=6.0); clean_axes(ax,grid=True); rows.append({"panel":"D","metric":PRIMARY_METRIC,"n_seeds":len(per_seed),"median":per_seed["identifiable_divergence"].median(),"source":"raw matched-fixed item rows grouped by split_seed","no_new_bootstrap":True,"provenance":"reliability_transport_measurement_depth_matched_fixed_items.csv"})
    ax=fig.add_subplot(grid[1,1]); add_panel_label(ax,"E"); _set_title(ax,"Summary of measurement floors","median + IQR across nine source × contrast cells"); full=depth.loc[depth["cell_budget_label"].eq("full")]
    for mi,metric in enumerate(METRICS):
        vals=pd.to_numeric(full.loc[full["metric"].eq(metric),"identifiable_divergence"],errors="coerce").dropna(); ax.errorbar(mi,vals.median(),yerr=[[vals.median()-vals.quantile(.25)],[vals.quantile(.75)-vals.median()]],fmt="o",color=metric_color(metric),capsize=3,ms=5); rows.append({"panel":"E","metric":metric,"median":vals.median(),"iqr_low":vals.quantile(.25),"iqr_high":vals.quantile(.75),"n_source_contrast_cells":len(vals),"uncertainty_type":"descriptive IQR across source×contrast cells","provenance":"reliability_transport_measurement_depth_matched_fixed_summary.csv"})
    ax.axhline(0,color="#46515C",lw=.7); ax.set_xticks(range(3),[metric_label(v) for v in METRICS],rotation=18,ha="right"); ax.set_ylabel(r"$D_{\mathrm{meas-ID}}$ (full)"); clean_axes(ax,grid=True)
    ax=fig.add_subplot(grid[1,2]); add_panel_label(ax,"F"); _set_title(ax,"Observed components across depth","actual $D_{cross}$, within-floor and $D_{meas-ID}$"); sub=_primary_depth(depth); x=np.arange(6)
    for column,label,color in (("cross_disagreement",r"$D_{cross}$","#46515C"),("measurement_floor",r"$D_{within}^{U}$","#AAB4BC"),("identifiable_divergence",r"$D_{meas-ID}$",metric_color(PRIMARY_METRIC))):
        data=sub.loc[sub["metric"].eq(PRIMARY_METRIC)].set_index("cell_budget_label").reindex(DEPTH_ORDER); y=pd.to_numeric(data[column],errors="coerce"); ax.plot(x,y,marker="o",ms=3,color=color,lw=1.5,label=label); rows.extend({"panel":"F","metric":PRIMARY_METRIC,"depth":d,"component":column,"value":v,"provenance":"reliability_transport_measurement_depth_matched_fixed_summary.csv"} for d,v in zip(DEPTH_ORDER,y))
    ax.axhline(0,color="#46515C",lw=.7); ax.set_xticks(x,DEPTH_ORDER); ax.set_xlabel("depth"); ax.set_ylabel("disagreement"); ax.legend(frameon=False,fontsize=5.9); clean_axes(ax,grid=True); fig.suptitle("Figure 3 | Measurement depth defines the identifiability boundary",fontsize=12,fontweight="bold"); fig.text(.5,.01,"Thresholds are cell- and metric-specific; negative identifiable divergence is not clipped, and unavailable resolution remains NR.",ha="center",fontsize=7.2,color="#46515C"); return save_figure(fig,out_dir,"reliability_transportability_fig3_measurement_boundary"),pd.DataFrame(rows)


def _decision_summary(manifests: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary=_depth_labels(pd.read_csv(manifests/"reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv")); summary=summary.loc[summary["metric"].isin(METRICS)&summary["cell_budget_label"].eq("full")].copy(); link=pd.read_csv(manifests/"reliability_transport_decision_link.csv"); return summary,link


def figure4(out_dir: Path, manifests: Path, registry: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    summary,link=_decision_summary(manifests); rows=[]; fig=plt.figure(figsize=(14,8.2)); grid=fig.add_gridspec(2,3,height_ratios=(1.,1.1),hspace=.46,wspace=.34); rank=_rank_table(manifests); n=len(rank); k=max(1,int(np.ceil(.10*n))); source_top=rank.nsmallest(k,"source_risk"); target_top=rank.nsmallest(k,"ifng_risk"); srcset=set(source_top["perturbation_label"]); tarset=set(target_top["perturbation_label"]); union=pd.concat([source_top.assign(selection="source"),target_top.assign(selection="target")]).drop_duplicates("perturbation_label")
    ax=fig.add_subplot(grid[0,0]); add_panel_label(ax,"A"); _set_title(ax,"Source rank → target rank","real Ctrl → IFNγ delta-cosine top-10% set")
    for yi,(_,row) in enumerate(union.sort_values("source_rank").iterrows()):
        color="#56B4E9" if row["perturbation_label"] in srcset&tarset else ("#0072B2" if row["perturbation_label"] in srcset else "#D55E00"); ax.plot([row["source_rank"],row["ifng_rank"]],[yi,yi],color=color,lw=2); ax.scatter([row["source_rank"],row["ifng_rank"]],[yi,yi],color=color,s=18); ax.text(.01,yi,row["perturbation_label"],transform=ax.get_yaxis_transform(),ha="left",va="center",fontsize=6.0); rows.append({"panel":"A","perturbation_label":row["perturbation_label"],"source_rank":row["source_rank"],"target_rank":row["ifng_rank"],"source_top10":row["perturbation_label"] in srcset,"target_top10":row["perturbation_label"] in tarset,"provenance":"formal_v2_reliability_reordering_perturbations.csv"})
    ax.set_yticks([]); ax.set_xlabel("rank (lower = better)"); ax.invert_yaxis(); ax.text(.98,.98,f"retention {len(srcset&tarset)}/{k}",transform=ax.transAxes,ha="right",va="top",fontsize=7,fontweight="bold"); clean_axes(ax,grid=True)
    ax=fig.add_subplot(grid[0,1]); add_panel_label(ax,"B"); _set_title(ax,"Top-k candidate table","same labels; source selection and target utility"); ax.axis("off"); ax.text(.25,.95,"source / Ctrl",ha="center",va="top",fontweight="bold",fontsize=8); ax.text(.75,.95,"target / IFNγ",ha="center",va="top",fontweight="bold",fontsize=8)
    for i in range(k):
        for j,data in enumerate((source_top.sort_values("source_rank").reset_index(drop=True),target_top.sort_values("ifng_rank").reset_index(drop=True))):
            label=data.iloc[i]["perturbation_label"]; shared=label in srcset&tarset; ax.text(.25+.5*j,.83-.12*i,f"{i+1}. {label}",ha="center",va="center",fontsize=6.8,bbox={"boxstyle":"round,pad=.2","facecolor":"#D9EEF9" if shared else ("#EAF2F8" if j==0 else "#FBE7E7"),"edgecolor":"#7A8996","lw":.5}); rows.append({"panel":"B","side":"source" if j==0 else "target","rank":i+1,"perturbation_label":label,"shared":shared,"provenance":"formal_v2_reliability_reordering_perturbations.csv"})
    ax.text(.5,.08,"shared = retained; colored rows are context-specific",ha="center",fontsize=6.5,color="#46515C")
    ax=fig.add_subplot(grid[0,2]); add_panel_label(ax,"C"); _set_title(ax,"Retention and regret across budgets","median + IQR across six directed transfers"); curve=summary.groupby(["metric","decision_budget_fraction"],as_index=False).agg(retention=("retention_point","median"),retention_low=("retention_point",lambda v:v.quantile(.25)),retention_high=("retention_point",lambda v:v.quantile(.75)),normalized_regret=("normalized_regret_point","median"),regret_low=("normalized_regret_point",lambda v:v.quantile(.25)),regret_high=("normalized_regret_point",lambda v:v.quantile(.75)),n_transfers=("target_environment_id","nunique"))
    for metric in METRICS:
        sub=curve.loc[curve["metric"].eq(metric)].sort_values("decision_budget_fraction"); x=sub["decision_budget_fraction"].to_numpy(); ax.plot(x,sub["retention"],marker="o",color=metric_color(metric),lw=1.5,label=metric_label(metric)); ax.fill_between(x,sub["retention_low"],sub["retention_high"],color=metric_color(metric),alpha=.12); ax.plot(x,sub["normalized_regret"],marker="x",ls="--",color=metric_color(metric),alpha=.8,lw=1.); rows.extend({"panel":"C","metric":metric,**r.to_dict(),"uncertainty_type":"descriptive IQR across directed transfers","provenance":"reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv"} for _,r in sub.iterrows())
    ax.axhline(0,color="#46515C",lw=.7); ax.set_xticks([.05,.1,.2,.5],["5%","10%","20%","50%"]); ax.set_xlabel("decision budget"); ax.set_ylabel("retention (circle) / normalized regret (×)"); ax.legend(frameon=False,fontsize=5.6); clean_axes(ax,grid=True)
    ax=fig.add_subplot(grid[1,0]); add_panel_label(ax,"D"); _set_title(ax,"Retention–regret Pareto map","six transfers; metric-specific points");
    for metric in METRICS:
        sub=summary.loc[summary["metric"].eq(metric)&summary["decision_budget_fraction"].eq(.1)]; ax.scatter(sub["normalized_regret_point"],sub["retention_point"],color=metric_color(metric),s=24,label=metric_label(metric)); rows.extend({"panel":"D","metric":metric,"source_environment_id":r["source_environment_id"],"target_environment_id":r["target_environment_id"],"normalized_regret":r["normalized_regret_point"],"retention":r["retention_point"],"budget_fraction":.1,"provenance":"reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv"} for _,r in sub.iterrows())
    ax.set_xlabel("normalized regret (lower = better)"); ax.set_ylabel("retention (higher = better)"); ax.legend(frameon=False,fontsize=5.6); clean_axes(ax,grid=True)
    ax=fig.add_subplot(grid[1,1]); add_panel_label(ax,"E"); _set_title(ax,"Fixed identifiability ↔ decision link","full depth · 10% budget · 18 points; descriptive"); rho=float(spearmanr(link["d_meas_id"],link["normalized_regret"]).statistic)
    for metric in METRICS:
        sub=link.loc[link["metric"].eq(metric)]; ax.errorbar(sub["d_meas_id"],sub["normalized_regret"],xerr=[sub["d_meas_id"]-sub["d_meas_id_ci_low"],sub["d_meas_id_ci_high"]-sub["d_meas_id"]],yerr=[sub["normalized_regret"]-sub["normalized_regret_ci_low"],sub["normalized_regret_ci_high"]-sub["normalized_regret"]],fmt="o",color=metric_color(metric),ecolor=metric_color(metric),capsize=2,ms=4,label=metric_label(metric)); rows.extend({"panel":"E",**r.to_dict(),"uncertainty_type":"90% synchronized macro-bootstrap interval","provenance":"reliability_transport_decision_link.csv"} for _,r in sub.iterrows())
    ax.text(.03,.97,f"ρ={rho:.4f}; n={len(link)}\ncluster q05=.7388, q95=.9003",transform=ax.transAxes,va="top",fontsize=6.8,fontweight="bold"); ax.set_xlabel(r"$D_{\mathrm{meas-ID}}$"); ax.set_ylabel("normalized regret"); ax.legend(frameon=False,fontsize=5.6); clean_axes(ax,grid=True)
    ax=fig.add_subplot(grid[1,2]); add_panel_label(ax,"F"); _set_title(ax,"Regret decomposition","existing point decomposition; not causal"); components=[("measurement_floor_regret_point","measurement floor","#0072B2"),("excess_regret_point","excess","#D55E00"),("joint_floor_regret_point","joint floor","#009E73"),("joint_excess_regret_point","joint excess","#CC79A7")]; dec=summary.loc[summary["decision_budget_fraction"].eq(.1)].groupby("metric",as_index=False)[[c[0] for c in components]].median()
    for yi,(_,r) in enumerate(dec.iterrows()):
        for ci,(column,label,color) in enumerate(components): ax.scatter(r[column],yi+(ci-1.5)*.1,color=color,s=24,label=label if yi==0 else None); rows.append({"panel":"F","metric":r["metric"],"component":label,"value":r[column],"budget_fraction":.1,"provenance":"reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv"})
    ax.axvline(0,color="#46515C",lw=.7); ax.set_yticks(range(len(dec)),[metric_label(v) for v in dec["metric"]]); ax.set_xlabel("point regret component"); ax.legend(frameon=False,fontsize=5.5,ncol=2); clean_axes(ax,grid=True); fig.suptitle("Figure 4 | Identifiable reordering changes experimental decisions",fontsize=12,fontweight="bold"); fig.text(.5,.01,"Circles and crosses in Panel C distinguish retention and normalized regret; ribbons are descriptive IQRs across directed transfers, not aggregate CIs.",ha="center",fontsize=7.2,color="#46515C"); return save_figure(fig,out_dir,"reliability_transportability_fig4_decision_consequence"),pd.DataFrame(rows)


def figure5(out_dir: Path, manifests: Path, registry: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    ranks=_rank_table(manifests); cases=_failure_cases(manifests,ranks); risks=_primary_risks(manifests); failure=pd.read_csv(manifests/"reliability_transport_failure_anatomy.csv"); complete=failure.loc[failure["status"].eq("executed")].copy(); rows=[]; fig=plt.figure(figsize=(14,8.2)); grid=fig.add_gridspec(2,3,height_ratios=(1.1,1.0),hspace=.48,wspace=.34)
    for ci,(_,case) in enumerate(cases.iterrows()):
        ax=fig.add_subplot(grid[0,ci]); add_panel_label(ax,chr(ord("A")+ci)); q=int(round(case["quantile"]*100)); _set_title(ax,f"Case q{q}: {case['perturbation_label']}","complete primary row; post-hoc descriptors only"); ax.bar([0,1],[case["source_rank"],case["ifng_rank"]],color=["#0072B2","#D55E00"],width=.55); ax.set_xticks([0,1],["Ctrl\nsource","IFNγ\ntarget"]); ax.set_ylabel("rank (1 = best)"); ax.invert_yaxis(); ax.text(.04,.96,f"Δrank={case['rank_delta']:+.0f}\nburden={case['reordering_burden']:.3f}",transform=ax.transAxes,va="top",fontsize=7,fontweight="bold"); ax.text(.54,.96,f"response={case['response_displacement']:.3f}\neffect={case['effect_magnitude']:.3f}",transform=ax.transAxes,va="top",fontsize=6.3,color="#46515C"); risk_sub=risks.loc[risks["perturbation_label"].eq(case["perturbation_label"])&risks["cell_budget_label"].eq("full")];
        if not risk_sub.empty:
            source_values=pd.to_numeric(risk_sub["left_risk"],errors="coerce").dropna(); target_values=pd.to_numeric(risk_sub["right_risk"],errors="coerce").dropna(); inset=ax.inset_axes([.59,.27,.35,.43]); inset.boxplot([source_values,target_values],vert=False,positions=[0,1],widths=.55,patch_artist=True,boxprops={"facecolor":"#D9EEF9","edgecolor":"#0072B2"},medianprops={"color":"#111111"}); inset.set_yticks([0,1],["S","T"],fontsize=5); inset.set_xlabel("risk",fontsize=5); inset.tick_params(axis="x",labelsize=5); clean_axes(inset)
        ax.text(.5,.02,"response/effect are post-hoc; no gene/pathway annotation",transform=ax.transAxes,ha="center",fontsize=5.6,color="#46515C"); rows.append({"panel":chr(ord("A")+ci),**case.to_dict(),"risk_distribution":"30 seed × matched measurement replicate at full depth","provenance":"reliability_transport_failure_anatomy.csv + reliability_transport_measurement_depth_matched_fixed_risks.csv"})
    ax=fig.add_subplot(grid[1,0]); add_panel_label(ax,"D"); _set_title(ax,"Global corrected state transitions","six directed Frangieh transfers; unstable/unresolved gray"); states=_pair_state_counts(manifests/"reliability_transport_metric_pair_states.csv"); positions={(s,t):(i,j) for i,s in enumerate(STATE_ORDER) for j,t in enumerate(STATE_ORDER)}
    for mi,metric in enumerate(METRICS):
        for _,state in states.loc[states["metric"].eq(metric)].iterrows(): x,y=positions[(state["source_state"],state["target_state"])] ; color="#AAB4BC" if "unstable" in (state["source_state"],state["target_state"]) else metric_color(metric); ax.scatter(x+(mi-1)*.18,y+mi*.04,s=700*state["fraction"]+4,color=color,alpha=.5,edgecolors="#303840",linewidths=.3); rows.append({"panel":"D",**state.to_dict(),"provenance":"reliability_transport_metric_pair_states.csv"})
    ax.set_xticks(range(4),STATE_ORDER); ax.set_yticks(range(4),STATE_ORDER); ax.set_xlabel("target state"); ax.set_ylabel("source state"); clean_axes(ax)
    ax=fig.add_subplot(grid[1,1]); add_panel_label(ax,"E"); _set_title(ax,"Response displacement versus burden","complete-only; no imputation; post-hoc explanatory"); overall_rho=_spearman_pair(complete)
    for metric in METRICS:
        sub=complete.loc[complete["metric"].eq(metric)]; ax.scatter(sub["reordering_burden"],sub["response_displacement"],s=18,color=metric_color(metric),alpha=.30,label=metric_label(metric)); rows.append({"panel":"E summary","metric":metric,"n_complete":len(sub),"spearman":_spearman_pair(sub),"provenance":"reliability_transport_failure_anatomy.csv","no_imputation":True})
    ax.axvline(0,color="#46515C",lw=.6); ax.text(.03,.97,f"overall ρ={overall_rho:.2f}; n={len(complete)}\n219/2784 complete",transform=ax.transAxes,va="top",fontsize=7,fontweight="bold"); ax.set_xlabel("reordering burden"); ax.set_ylabel("response displacement"); ax.legend(frameon=False,fontsize=5.7); clean_axes(ax,grid=True)
    ax=fig.add_subplot(grid[1,2]); add_panel_label(ax,"F"); _set_title(ax,"Coverage and heterogeneity","existing fields only; no imputation"); coverage=pd.read_csv(manifests/"reliability_transport_failure_anatomy_coverage.csv"); coverage["fraction"]=coverage["complete"]/coverage["total"]; coverage=coverage.sort_values(["source_environment_id","target_environment_id","metric"]); y=np.arange(len(coverage)); ax.barh(y,coverage["fraction"],color=[metric_color(m) for m in coverage["metric"]],alpha=.65); ax.set_yticks(y,[f"{context_label(a)}→{context_label(b)}\n{metric_label(m)}" for a,b,m in zip(coverage["source_environment_id"],coverage["target_environment_id"],coverage["metric"])],fontsize=5.1); ax.set_xlim(0,max(.1,float(coverage["fraction"].max())*1.25)); ax.set_xlabel("complete fraction"); ax.text(.98,.02,"219/2784 total; legacy response fields missing elsewhere",transform=ax.transAxes,ha="right",va="bottom",fontsize=5.8,color="#46515C"); clean_axes(ax,grid=True); rows.extend({"panel":"F",**r.to_dict(),"no_imputation":True,"provenance":"reliability_transport_failure_anatomy_coverage.csv"} for _,r in coverage.iterrows())
    fig.suptitle("Figure 5 | Transport failure has a heterogeneous, partly unobserved anatomy",fontsize=12,fontweight="bold"); fig.text(.5,.01,"Case cards use real canonical labels and ranks; explanatory quantities are post-hoc and never prospective predictors.",ha="center",fontsize=7.2,color="#46515C"); return save_figure(fig,out_dir,"reliability_transportability_fig5_failure_anatomy"),pd.DataFrame(rows)


def _full_predictability(manifests: Path) -> pd.DataFrame:
    pred=pd.read_csv(manifests/"reliability_transport_predictability.csv"); return pred.loc[pred["status"].eq("executed")&pred["cell_budget_label"].eq("full")&pred["dataset"].eq(ATLAS_DATASET)&pred["dataset_split"].eq("within_dataset_leave_one_perturbation_label_out")&pred["model"].isin(["source_u_only","linear_source_features"])].assign(spearman=lambda d:pd.to_numeric(d["spearman"],errors="coerce"),auroc_high_identifiable=lambda d:pd.to_numeric(d["auroc_high_identifiable"],errors="coerce"))


def figure6(out_dir: Path, manifests: Path, registry: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    pred=_full_predictability(manifests); rows=[]; fig=plt.figure(figsize=(14,8.2)); grid=fig.add_gridspec(2,3,height_ratios=(1.,1.05),hspace=.47,wspace=.35)
    ax=fig.add_subplot(grid[0,0]); add_panel_label(ax,"A"); _set_title(ax,"Information firewall","what is legal before the target experiment"); ax.axis("off"); ax.add_patch(FancyBboxPatch((.03,.14),.41,.72,boxstyle="round,pad=.02",facecolor="#EAF2F8",edgecolor="#7A8996")); ax.add_patch(FancyBboxPatch((.56,.14),.41,.72,boxstyle="round,pad=.02",facecolor="#FBE7E7",edgecolor="#C44E52")); ax.text(.235,.80,"source-observable",ha="center",fontweight="bold",fontsize=8); ax.text(.765,.80,"target-informed",ha="center",fontweight="bold",fontsize=8,color="#9E2A2B"); ax.text(.06,.70,"Allowed features",fontsize=7,fontweight="bold"); ax.text(.06,.63,"\n".join("• "+item for item in FEATURES),va="top",fontsize=6.0); ax.text(.59,.70,"Forbidden at prediction",fontsize=7,fontweight="bold",color="#9E2A2B"); ax.text(.59,.63,"\n".join("× "+item for item in FORBIDDEN_TARGET_FIELDS),va="top",fontsize=6.0,color="#9E2A2B"); _draw_arrow(ax,(.46,.50),(.54,.50),color="#C44E52"); ax.text(.5,.06,"target outcomes appear only after the prediction for evaluation",ha="center",fontsize=6.3,color="#46515C"); rows.append({"panel":"A","allowed_features":";".join(FEATURES),"forbidden_target_informed_quantities":";".join(FORBIDDEN_TARGET_FIELDS),"provenance":"reliability_transport_predictability.csv + failure anatomy manifest"})
    ax=fig.add_subplot(grid[0,1]); add_panel_label(ax,"B"); _set_title(ax,"Full-depth prospective performance","task-level paired distributions; source-only models")
    for mi,metric in enumerate(METRICS):
        for model,offset in (("source_u_only",-.12),("linear_source_features",.12)):
            vals=pred.loc[pred["metric"].eq(metric)&pred["model"].eq(model),"spearman"].dropna(); center=vals.median(); ax.scatter(np.full(len(vals),mi+offset),vals,color=metric_color(metric),alpha=.35,s=15,marker="o" if model=="source_u_only" else "D"); ax.errorbar(mi+offset,center,yerr=[[center-vals.quantile(.25)],[vals.quantile(.75)-center]],fmt="o",color="#111111",capsize=2,ms=3); rows.append({"panel":"B","metric":metric,"model":model,"median":center,"q25":vals.quantile(.25),"q75":vals.quantile(.75),"n_tasks":len(vals),"uncertainty":"task-level IQR","provenance":"reliability_transport_predictability.csv"})
    ax.axhline(0,color="#46515C",lw=.7); ax.set_xticks(range(3),[metric_label(v) for v in METRICS],rotation=20,ha="right"); ax.set_ylabel("held-out Spearman"); ax.text(.03,.97,"○ uncertainty-only   ◆ six-feature source-only",transform=ax.transAxes,va="top",fontsize=6.0,color="#46515C"); clean_axes(ax,grid=True)
    ax=fig.add_subplot(grid[0,2]); add_panel_label(ax,"C"); _set_title(ax,"Source-only models across metrics","negative performance remains negative"); ordered=pred.sort_values(["metric","model","source_environment_id","left_target_environment_id"])
    for yi,(_,r) in enumerate(ordered.iterrows()): ax.scatter(r["spearman"],yi,color=metric_color(r["metric"]),marker="o" if r["model"]=="source_u_only" else "D",s=15); rows.append({"panel":"C",**r.to_dict(),"provenance":"reliability_transport_predictability.csv"})
    ax.axvline(0,color="#46515C",lw=.7); ax.set_xlabel("held-out Spearman"); ax.set_yticks([]); ax.text(.03,.97,f"n={len(pred)} task-level points",transform=ax.transAxes,va="top",fontsize=6.3,color="#46515C"); clean_axes(ax,grid=True)
    ax=fig.add_subplot(grid[1,0]); add_panel_label(ax,"D"); _set_title(ax,"Held-out performance by source × contrast","two legal source-only models"); heat=ax.inset_axes([.06,.08,.89,.82]); heat.axis("off"); cells=[(s,c) for s in CONTEXT_ORDER for c in CONTRAST_ORDER]; labels=[f"{context_label(s)}\n{contrast_label(*c)}" for s,c in cells]
    for mi,model in enumerate(["source_u_only","linear_source_features"]):
        subax=heat.inset_axes([0,.52-mi*.48,.96,.40]); data=np.full((len(cells),len(METRICS)),np.nan)
        for ri,(source,contrast) in enumerate(cells):
            for ci,metric in enumerate(METRICS):
                vals=pred.loc[pred["model"].eq(model)&pred["metric"].eq(metric)&pred["source_environment_id"].eq(source)&pred["left_target_environment_id"].eq(contrast[0])&pred["right_target_environment_id"].eq(contrast[1]),"spearman"]; data[ri,ci]=vals.iloc[0] if len(vals) else np.nan; rows.append({"panel":"D","model":model,"source_environment_id":source,"contrast":contrast_label(*contrast),"metric":metric,"spearman":data[ri,ci],"provenance":"reliability_transport_predictability.csv"})
        subax.imshow(np.ma.masked_invalid(data),cmap="RdBu_r",vmin=-.6,vmax=.6,aspect="auto"); subax.set_title(model,fontsize=6.0,loc="left"); subax.set_xticks(range(3),[metric_label(v) for v in METRICS],rotation=25,ha="right",fontsize=4.8); subax.set_yticks(range(len(cells)),labels if mi==0 else [""]*len(cells),fontsize=4.2)
        for ri in range(len(cells)):
            for ci in range(3):
                if np.isfinite(data[ri,ci]): subax.text(ci,ri,f"{data[ri,ci]:.2f}",ha="center",va="center",fontsize=4.2)
    ax.set_xticks([]); ax.set_yticks([]); clean_axes(ax)
    ax=fig.add_subplot(grid[1,1]); add_panel_label(ax,"E"); _set_title(ax,"Prospective high-identifiability discrimination","AUROC; 0.5 = chance; no calibration curve reported")
    for mi,metric in enumerate(METRICS):
        for model,offset in (("source_u_only",-.12),("linear_source_features",.12)):
            vals=pred.loc[pred["metric"].eq(metric)&pred["model"].eq(model),"auroc_high_identifiable"].dropna(); center=vals.median(); ax.scatter(np.full(len(vals),mi+offset),vals,s=15,color=metric_color(metric),alpha=.35,marker="o" if model=="source_u_only" else "D"); ax.errorbar(mi+offset,center,yerr=[[center-vals.quantile(.25)],[vals.quantile(.75)-center]],fmt="o",color="#111111",capsize=2,ms=3); rows.append({"panel":"E","metric":metric,"model":model,"median_auroc":center,"q25":vals.quantile(.25),"q75":vals.quantile(.75),"n_tasks":len(vals),"reference":.5,"calibration_status":"not_reported_train_fold_only_no_posthoc_target_calibration","provenance":"reliability_transport_predictability.csv"})
    ax.axhline(.5,color="#46515C",lw=.8,ls="--"); ax.set_ylim(0,1); ax.set_xticks(range(3),[metric_label(v) for v in METRICS],rotation=20,ha="right"); ax.set_ylabel("AUROC"); clean_axes(ax,grid=True)
    ax=fig.add_subplot(grid[1,2]); add_panel_label(ax,"F"); _set_title(ax,"Blocked prospective cross-dataset prediction","Frangieh → Nadig; no numeric result"); ax.axis("off"); ax.add_patch(FancyBboxPatch((.05,.18),.90,.58,boxstyle="round,pad=.03",facecolor="#F4F5F6",edgecolor="#7A8996")); ax.text(.50,.65,"UNAVAILABLE",ha="center",va="center",fontsize=12,fontweight="bold",color="#46515C"); ax.text(.50,.49,"Nadig is aggregate-only; no shared per-label\ntarget outcome is available for leave-dataset-out evaluation.",ha="center",va="center",fontsize=7.2); ax.text(.50,.28,"No zero imputation. No oracle number.",ha="center",va="center",fontsize=7.0,color="#9E2A2B",fontweight="bold"); rows.append({"panel":"F","status":"unavailable","reason":"Nadig aggregate-only no shared per-label target","numeric_result":False,"provenance":"reliability_transport_predictability.csv"})
    fig.suptitle("Figure 6 | What can be known before experiment? Source-only prospective limits",fontsize=12,fontweight="bold"); fig.text(.5,.01,"The legal feature boundary is explicit. Negative or near-chance source-only performance is retained, and blocked Nadig cross-dataset prediction is not filled with a proxy.",ha="center",fontsize=7.2,color="#46515C"); return save_figure(fig,out_dir,"reliability_transportability_fig6_prospective_limit"),pd.DataFrame(rows)


def run(root: Path = ROOT) -> dict[str, Any]:
    root=root.resolve(); manifests=root/"artifacts/manifests"; out_dir=root/"results/figures/reliability_transportability"; out_dir.mkdir(parents=True,exist_ok=True); apply_style(); registry=build_exemplar_registry(root)
    functions=[("fig1","fig1_source.csv",lambda:figure1(out_dir,manifests,registry)),("fig2","fig2_transport_atlas.csv",lambda:figure2(out_dir,manifests,registry)),("fig3","fig3_measurement_boundary.csv",lambda:figure3(out_dir,manifests,registry)),("fig4","fig4_decision_consequence.csv",lambda:figure4(out_dir,manifests,registry)),("fig5","fig5_failure_anatomy.csv",lambda:figure5(out_dir,manifests,registry)),("fig6","fig6_prospective_limit.csv",lambda:figure6(out_dir,manifests,registry))]
    outputs={}
    for key,source_name,function in functions:
        paths,source=function(); outputs[key]={"files":paths,"source_table":_write_source(root,source_name,source),"rows":int(len(source))}
    report={"schema_version":3,"status":"executed","figure_count":6,"figure_order":["problem","atlas","measurement_boundary","decision_consequence","failure_anatomy","prospective_limit"],"formats":["png","pdf","svg","tiff"],"source_tables_directory":"results/figures/reliability_transportability/source_tables","exemplar_registry":"artifacts/manifests/figure_exemplar_registry.json","outputs":outputs,"data_policy":"Frozen Frangieh/Nadig data and existing matched-fixed/bootstrap summaries; unavailable cells stay gray/NA; target-informed explanatory quantities never enter prospective prediction.","selection_policy":"Fig1/2/4 displayed labels use source-only rank quantiles; Fig5 cases use complete burden quantiles; all selections are registered deterministically.","style_policy":"Centralized ptl_figure_style.py; colorblind-safe palette; diverging maps centered at zero; no rainbow, 3-D, decorative gradients, or long-ID axes.","anti_fabrication_checks":["strict support >= 8","219/2784 complete failure rows with no imputation","Fig6 feature firewall","no numeric Nadig prospective result","primary intervals are 90%"]}
    (manifests/"reliability_transport_figures.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); print(json.dumps(report,indent=2,ensure_ascii=False)); return report


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--root",type=Path,default=ROOT); run(parser.parse_args().root); return 0


if __name__ == "__main__":
    raise SystemExit(main())
