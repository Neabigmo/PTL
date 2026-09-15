"""Build the prespecified identifiable-reordering to decision-loss link.

The primary association is fixed before reading its value: full measurement
depth, a 0.10 decision budget, ``D_meas-ID`` on the x-axis, and normalized
regret on the y-axis.  The six directed endpoint transfers and three frozen
metrics therefore yield exactly 18 points.  All quantities are derived from
the existing canonical depth, atlas, and 2,000-draw decision summary tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.decision_analysis_common import CANONICAL_SUMMARY, add_context_ids, load_canonical_d_adj  # noqa: E402

METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")
PRIMARY_DEPTH = "full"
PRIMARY_DECISION_BUDGET = 0.10


def _safe_spearman(frame: pd.DataFrame) -> float:
    values = frame[["d_meas_id", "normalized_regret"]].dropna()
    if len(values) < 3 or values["d_meas_id"].nunique() < 2 or values["normalized_regret"].nunique() < 2:
        return float("nan")
    return float(spearmanr(values["d_meas_id"], values["normalized_regret"]).statistic)


def _block_bootstrap(frame: pd.DataFrame, block: str, *, seed: int = 20260910, draws: int = 4000) -> tuple[float, float]:
    """Bootstrap dependence-aware context blocks, carrying all associated rows."""

    clusters = np.asarray(sorted(frame[block].unique()), dtype=object)
    if len(clusters) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(draws)):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = [frame.loc[frame[block].eq(cluster)] for cluster in sampled]
        bootstrap = pd.concat(pieces, ignore_index=True)
        value = _safe_spearman(bootstrap)
        if np.isfinite(value):
            values.append(value)
    if len(values) < 20:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))


def build(root: Path = ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    depth_path = manifests / "reliability_transport_measurement_depth_matched_fixed_summary.csv"
    atlas_path = manifests / "reliability_transport_atlas.csv"
    decision_path = manifests / "reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv"
    for path in (depth_path, atlas_path, decision_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    depth = load_canonical_d_adj(root, METRICS)

    atlas = pd.read_csv(atlas_path)
    atlas = atlas.loc[atlas["dataset"].eq("FrangiehIzar2021_RNA") & atlas["metric"].isin(METRICS)].copy()
    atlas = atlas.loc[
        atlas["source_environment_id"].eq(atlas["left_target_environment_id"])
        | atlas["source_environment_id"].eq(atlas["right_target_environment_id"])
    ].copy()
    atlas["target_environment_id"] = np.where(
        atlas["source_environment_id"].eq(atlas["left_target_environment_id"]),
        atlas["right_target_environment_id"],
        atlas["left_target_environment_id"],
    )
    atlas = atlas[["source_environment_id", "target_environment_id", "metric", "joint_identifiable", "joint_identifiable_ci_low", "joint_identifiable_ci_high"]]
    if atlas.duplicated(["source_environment_id", "target_environment_id", "metric"]).any():
        raise ValueError("Frangieh atlas endpoint rows are not unique")

    decision = pd.read_csv(decision_path, dtype={"cell_budget_label": "string"})
    decision = decision.loc[decision["cell_budget_label"].eq(PRIMARY_DEPTH) & np.isclose(decision["decision_budget_fraction"].astype(float), PRIMARY_DECISION_BUDGET)].copy()
    decision = decision.loc[decision["metric"].isin(METRICS)].copy()
    if decision.duplicated(["source_environment_id", "target_environment_id", "metric"]).any():
        raise ValueError("primary decision link has duplicate source-target-metric keys")

    linked = depth.merge(atlas, on=["source_environment_id", "target_environment_id", "metric"], how="left", validate="one_to_one")
    linked = linked.merge(
        decision[["source_environment_id", "target_environment_id", "metric", "cell_budget_label", "decision_budget_fraction", "normalized_regret_point", "normalized_regret_ci_low", "normalized_regret_ci_high", "draw_count", "raw_bootstrap_sha256"]],
        on=["source_environment_id", "target_environment_id", "metric"],
        how="inner",
        validate="one_to_one",
    )
    if len(linked) != 18:
        raise ValueError(f"primary decision link must contain 18 points, found {len(linked)}")
    linked["transfer_id"] = linked["source_environment_id"] + "->" + linked["target_environment_id"]
    linked["d_joint_id"] = linked["joint_identifiable"]
    linked["d_joint_id_ci_low"] = linked["joint_identifiable_ci_low"]
    linked["d_joint_id_ci_high"] = linked["joint_identifiable_ci_high"]
    linked["normalized_regret"] = linked["normalized_regret_point"]
    linked["normalized_regret_ci_low"] = linked["normalized_regret_ci_low"]
    linked["normalized_regret_ci_high"] = linked["normalized_regret_ci_high"]
    linked = add_context_ids(linked)
    linked["analysis_unit"] = "directed source-select/target-evaluate row × metric (descriptive point-estimate surface)"
    linked["analysis_contract"] = "full depth; decision budget 0.10; D_meas-ID -> normalized_regret"
    linked["provenance"] = json.dumps(
        {
            "measurement_depth": CANONICAL_SUMMARY,
            "d_adj_source": "corrected canonical summary only",
            "atlas_joint_identifiability": atlas_path.relative_to(root).as_posix(),
            "decision_bootstrap_summary": decision_path.relative_to(root).as_posix(),
            "decision_uncertainty": "canonical 2,000-draw perturbation-label × measurement-seed summary",
        },
        sort_keys=True,
    )
    output_columns = [
        "source_environment_id", "target_environment_id", "transfer_id", "source_context_id", "unordered_context_pair_id", "metric", "cell_budget_label", "decision_budget_fraction",
        "d_meas_id", "d_meas_id_ci_low", "d_meas_id_ci_high", "d_joint_id", "d_joint_id_ci_low", "d_joint_id_ci_high",
        "normalized_regret", "normalized_regret_ci_low", "normalized_regret_ci_high", "draw_count", "raw_bootstrap_sha256",
        "analysis_unit", "analysis_contract", "provenance",
    ]
    linked = linked[output_columns].sort_values(["source_environment_id", "target_environment_id", "metric"], kind="stable").reset_index(drop=True)
    correlations: dict[str, Any] = {
        "overall": {"n": int(len(linked)), "spearman": _safe_spearman(linked)},
        "metric_specific": {},
    }
    low, high = _block_bootstrap(linked, "unordered_context_pair_id")
    correlations["overall"]["unordered_pair_bootstrap_q05"] = low
    correlations["overall"]["unordered_pair_bootstrap_q95"] = high
    correlations["leave_one_unordered_context_pair_out"] = {
        pair: _safe_spearman(linked.loc[~linked["unordered_context_pair_id"].eq(pair)])
        for pair in sorted(linked["unordered_context_pair_id"].unique())
    }
    correlations["leave_one_source_context_out"] = {
        source: _safe_spearman(linked.loc[~linked["source_context_id"].eq(source)])
        for source in sorted(linked["source_context_id"].unique())
    }
    for metric, group in linked.groupby("metric", sort=True):
        metric_low, metric_high = _block_bootstrap(group, "unordered_context_pair_id")
        correlations["metric_specific"][metric] = {
            "n": int(len(group)),
            "spearman": _safe_spearman(group),
            "unordered_pair_bootstrap_q05": metric_low,
            "unordered_pair_bootstrap_q95": metric_high,
        }
    report = {
        "schema_version": 1,
        "status": "executed",
        "primary_contract": {
            "measurement_depth": PRIMARY_DEPTH,
            "decision_budget_fraction": PRIMARY_DECISION_BUDGET,
            "x": "D_meas-ID",
            "y": "normalized_regret",
            "unit": "18 directed rows (three unordered context pairs × both directions × three metrics)",
            "independent_cluster_claim": "none; the 18 rows are retained only as a descriptive point-estimate surface",
            "interpretation": "descriptive association; not causal; sensitivity blocks are unordered context pairs or source contexts",
        },
        "correlations": correlations,
        "rows": int(len(linked)),
        "unique_keys": bool(not linked.duplicated(["source_environment_id", "target_environment_id", "metric", "cell_budget_label", "decision_budget_fraction"]).any()),
        "outputs": {
            "table": "artifacts/manifests/reliability_transport_decision_link.csv",
            "report": "artifacts/manifests/reliability_transport_decision_link.json",
        },
    }
    return linked, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifests = root / "artifacts/manifests"
    table, report = build(root)
    table_path = manifests / "reliability_transport_decision_link.csv"
    report_path = manifests / "reliability_transport_decision_link.json"
    table.to_csv(table_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
