"""Build the budget-wise decision robustness table from the primary decision surface.

The 10% random-reference association is the same canonical surface used by
``reliability_transport_decision_link.csv``.  Budget is the only varied
quantity in the robustness table; the bounded worst-oracle normalization is
retained as a secondary diagnostic from the same source-frozen decision
curves.
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

from scripts.decision_analysis_common import directed_endpoint_surface, load_canonical_d_adj  # noqa: E402
from scripts.build_reviewer_full_surface_controls import METRICS, _load_inputs  # noqa: E402
from src.evaluation.decision_theory import DEFAULT_BUDGET_FRACTIONS, replicate_decision_curve  # noqa: E402


PRIMARY_DEPTH = "full"
PRIMARY_RANDOM_BUDGET = 0.10
DECISION_BUDGETS = (0.05, 0.10, 0.20, 0.50)
STRATIFIED_BOOTSTRAP_DRAWS = 2000
STRATIFIED_BOOTSTRAP_SEED = 20260914


def _rank_within_metric(frame: pd.DataFrame, field: str) -> pd.Series:
    """Rank a decision quantity within metric, preserving the fixed 18-row surface."""

    return frame.groupby("metric", sort=False)[field].rank(method="average")


def _directed(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep endpoint sources and make the directed target explicit."""

    return directed_endpoint_surface(frame)


def _spearman(frame: pd.DataFrame, field: str) -> float:
    data = frame[["d_meas_id", field]].dropna()
    if len(data) < 3 or data["d_meas_id"].nunique() < 2 or data[field].nunique() < 2:
        return float("nan")
    return float(spearmanr(data["d_meas_id"], data[field]).statistic)


def _within_metric_spearman(frame: pd.DataFrame) -> float:
    """Spearman after removing between-metric scale by within-metric ranking."""

    data = frame[["metric", "d_meas_id", "normalized_regret"]].dropna().copy()
    if data.empty:
        return float("nan")
    data["d_rank"] = _rank_within_metric(data, "d_meas_id")
    data["regret_rank"] = _rank_within_metric(data, "normalized_regret")
    if data["d_rank"].nunique() < 2 or data["regret_rank"].nunique() < 2:
        return float("nan")
    return float(spearmanr(data["d_rank"], data["regret_rank"]).statistic)


def _metric_stratified(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the prespecified metric-confounding audit on the fixed 10% surface.

    The point estimate remains the original 18 directed rows. The bootstrap
    resamples unordered context-pair blocks, never individual metric rows, and
    re-ranks inside each metric for every draw.
    """

    frame = primary.loc[np.isclose(primary["budget_fraction"].astype(float), PRIMARY_RANDOM_BUDGET)].copy()
    frame = frame[["source_environment_id", "target_environment_id", "transfer_id", "source_context_id", "unordered_context_pair_id", "metric", "d_meas_id", "normalized_regret"]].copy()
    if len(frame) != 18 or frame["unordered_context_pair_id"].nunique() != 3 or frame["metric"].nunique() != 3:
        raise RuntimeError("metric-stratified audit requires the fixed 18-row (3 unordered pair x 2 direction x 3 metric) surface")
    if frame.duplicated(["transfer_id", "metric"]).any():
        raise RuntimeError("metric-stratified audit has duplicate transfer x metric keys")

    rows: list[dict[str, Any]] = []
    base = {
        "budget_fraction": PRIMARY_RANDOM_BUDGET,
        "regret_scale": "normalized_regret",
        "bootstrap_unit": "unordered context-pair block",
        "analysis_surface": "matched-fixed full-depth random-reference decision surface",
    }
    for metric in sorted(frame["metric"].unique()):
        sub = frame.loc[frame["metric"].eq(metric)]
        rows.append({**base, "scope": "metric_specific", "metric": metric, "omitted_transfer": "", "n": int(len(sub)), "spearman": _spearman(sub, "normalized_regret"), "ci_low": float("nan"), "ci_high": float("nan"), "bootstrap_draws": 0})

    point = _within_metric_spearman(frame)
    rows.append({**base, "scope": "within_metric_pooled", "metric": "all_metrics_ranked_within_metric", "omitted_transfer": "", "n": int(len(frame)), "spearman": point, "ci_low": float("nan"), "ci_high": float("nan"), "bootstrap_draws": 0})

    pairs = sorted(frame["unordered_context_pair_id"].unique())
    rng = np.random.default_rng(STRATIFIED_BOOTSTRAP_SEED)
    bootstrap_values: list[float] = []
    for _ in range(STRATIFIED_BOOTSTRAP_DRAWS):
        selected = rng.integers(0, len(pairs), size=len(pairs))
        pieces = [frame.loc[frame["unordered_context_pair_id"].eq(pairs[index])].copy() for index in selected]
        bootstrap_frame = pd.concat(pieces, ignore_index=True)
        value = _within_metric_spearman(bootstrap_frame)
        if np.isfinite(value):
            bootstrap_values.append(value)
    if len(bootstrap_values) < STRATIFIED_BOOTSTRAP_DRAWS * 0.95:
        raise RuntimeError("too many invalid unordered-pair bootstrap draws")
    rows.append({
        **base,
        "scope": "within_metric_pooled_unordered_pair_bootstrap",
        "metric": "all_metrics_ranked_within_metric",
        "omitted_transfer": "",
        "n": int(len(frame)),
        "spearman": point,
        "ci_low": float(np.quantile(bootstrap_values, 0.05)),
        "ci_high": float(np.quantile(bootstrap_values, 0.95)),
        "bootstrap_draws": int(len(bootstrap_values)),
    })

    for omitted in pairs:
        kept = frame.loc[~frame["unordered_context_pair_id"].eq(omitted)].copy()
        rows.append({
            **base,
            "scope": "within_metric_pooled_leave_one_unordered_context_pair_out",
            "metric": "all_metrics_ranked_within_metric",
            "omitted_transfer": "",
            "omitted_unordered_context_pair": omitted,
            "n": int(len(kept)),
            "spearman": _within_metric_spearman(kept),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_draws": 0,
        })

    output = pd.DataFrame(rows)
    report = {
        "status": "executed",
        "budget_fraction": PRIMARY_RANDOM_BUDGET,
        "depth": PRIMARY_DEPTH,
        "surface": "3 unordered context pairs x 2 directions x 3 metrics = 18 directed rows",
        "metric_specific_n": 6,
        "within_metric_transform": "rank d_meas_id and normalized_regret separately within each metric, then pool",
        "bootstrap": {
            "unit": "unordered context-pair block",
            "draws": STRATIFIED_BOOTSTRAP_DRAWS,
            "seed": STRATIFIED_BOOTSTRAP_SEED,
            "interval": "percentile 90% interval",
            "valid_draws": len(bootstrap_values),
        },
        "point_spearman": point,
        "within_metric_unordered_pair_bootstrap_ci": [float(np.quantile(bootstrap_values, 0.05)), float(np.quantile(bootstrap_values, 0.95))],
        "metric_specific_spearman": {
            row["metric"]: row["spearman"] for row in rows if row["scope"] == "metric_specific"
        },
        "leave_one_unordered_context_pair_out": {
            row["omitted_unordered_context_pair"]: row["spearman"]
            for row in rows
            if row["scope"] == "within_metric_pooled_leave_one_unordered_context_pair_out"
        },
    }
    return output, report


def _directed_curves(root: Path) -> pd.DataFrame:
    """Load the already-completed primary decision summaries for all budgets."""

    manifests = root / "artifacts/manifests"
    decision = pd.read_csv(
        manifests / "reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv",
        dtype={"cell_budget_label": "string"},
    )
    decision = _directed(decision)
    decision = decision.loc[
        decision["cell_budget_label"].eq(PRIMARY_DEPTH)
        & decision["decision_budget_fraction"].astype(float).isin(DECISION_BUDGETS)
        & decision["metric"].isin(METRICS)
    ].copy()
    if decision.duplicated(["source_environment_id", "target_environment_id", "metric", "decision_budget_fraction"]).any():
        raise RuntimeError("primary decision summary has duplicate directed keys")

    canonical = load_canonical_d_adj(root, METRICS)

    joined = decision.merge(
        canonical.drop(columns=["transfer_id", "source_context_id", "unordered_context_pair_id"]),
        on=["source_environment_id", "target_environment_id", "metric"],
        how="inner",
        validate="many_to_one",
    )
    joined["transfer_id"] = joined["source_environment_id"] + "->" + joined["target_environment_id"]
    joined["budget_fraction"] = joined["decision_budget_fraction"].astype(float)
    joined["normalized_regret"] = joined["normalized_regret_point"]
    joined["normalized_regret_ci_low"] = joined["normalized_regret_ci_low"]
    joined["normalized_regret_ci_high"] = joined["normalized_regret_ci_high"]
    joined["decision_protocol"] = "source selects; target evaluates; matched fixed full-depth decision summary"
    expected = 18 * len(DECISION_BUDGETS)
    if len(joined) != expected:
        raise RuntimeError(f"expected {expected} primary full-depth decision rows, found {len(joined)}")
    return joined.sort_values(["budget_fraction", "source_environment_id", "target_environment_id", "metric"], kind="stable").reset_index(drop=True)


def _bounded_auxiliary_curves(root: Path) -> pd.DataFrame:
    """Materialize the secondary bounded normalization from canonical arrays."""

    values = _load_inputs(root)
    rows: list[dict[str, Any]] = []
    for encoded, arrays in sorted(values.items()):
        source, left, right, metric = encoded.split("__", 3)
        if metric not in METRICS or source not in (left, right):
            continue
        target = right if source == left else left
        source_side = "left" if source == left else "right"
        target_side = "right" if source == left else "left"
        source_a = np.stack(arrays[f"cross_{source_side}_a"])
        source_b = np.stack(arrays[f"cross_{source_side}_b"])
        target_a = np.stack(arrays[f"cross_{target_side}_a"])
        target_b = np.stack(arrays[f"cross_{target_side}_b"])
        seed_rows: list[dict[str, Any]] = []
        for seed_index in range(source_a.shape[0]):
            seed_rows.extend(
                replicate_decision_curve(
                    np.stack((source_a[seed_index], source_b[seed_index])),
                    np.stack((target_a[seed_index], target_b[seed_index])),
                    budgets=DEFAULT_BUDGET_FRACTIONS,
                )
            )
        averaged = pd.DataFrame(seed_rows).groupby("budget_fraction", as_index=False).mean(numeric_only=True)
        for curve in averaged.to_dict("records"):
            rows.append({
                "source_environment_id": source,
                "target_environment_id": target,
                "transfer_id": f"{source}->{target}",
                "metric": metric,
                **curve,
            })
    return pd.DataFrame(rows)


def run(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    curves = _directed_curves(root)
    joined = curves
    # Preserve the previously materialized bounded normalization as a
    # secondary diagnostic.  The primary random-reference rows below are
    # rebuilt from the same matched-fixed surface as Fig. 4B.
    auxiliary_path = manifests / "reviewer_full_surface_decision_curves.csv"
    auxiliary = pd.DataFrame()
    if auxiliary_path.is_file():
        auxiliary = pd.read_csv(auxiliary_path)
    if auxiliary.empty or "worst_oracle_normalized_regret" not in auxiliary.columns:
        auxiliary = _bounded_auxiliary_curves(root)
    if not auxiliary.empty:
        auxiliary = auxiliary.drop(columns=["d_meas_id"], errors="ignore").merge(
            joined[["source_environment_id", "target_environment_id", "metric", "d_meas_id"]].drop_duplicates(),
            on=["source_environment_id", "target_environment_id", "metric"],
            how="inner",
            validate="many_to_one",
        )
        auxiliary["transfer_id"] = auxiliary["source_environment_id"] + "->" + auxiliary["target_environment_id"]
        auxiliary = directed_endpoint_surface(
            auxiliary.assign(
                left_target_environment_id=auxiliary["source_environment_id"],
                right_target_environment_id=auxiliary["target_environment_id"],
            )
        )
    association_rows: list[dict[str, Any]] = []
    for budget, group in joined.groupby("budget_fraction", sort=True):
        for scale in ("normalized_regret", "worst_oracle_normalized_regret"):
            scale_group = group
            if scale == "worst_oracle_normalized_regret":
                if auxiliary.empty or scale not in auxiliary:
                    continue
                scale_group = auxiliary.loc[np.isclose(auxiliary["budget_fraction"].astype(float), float(budget))].copy()
                if len(scale_group) != 18:
                    raise RuntimeError(f"expected 18 auxiliary bounded-regret rows at budget {budget}, found {len(scale_group)}")
            association_rows.append({
                "budget_fraction": float(budget), "regret_scale": scale, "scope": "all_18_metric_transfer_points",
                "n": int(len(scale_group)), "spearman": _spearman(scale_group, scale),
            })
            for omitted in sorted(scale_group["unordered_context_pair_id"].unique()):
                kept = scale_group.loc[~scale_group["unordered_context_pair_id"].eq(omitted)]
                association_rows.append({
                    "budget_fraction": float(budget), "regret_scale": scale,
                    "scope": "leave_one_unordered_context_pair_out",
                    "omitted_unordered_context_pair": omitted,
                    "n": int(len(kept)), "spearman": _spearman(kept, scale),
                })
            for omitted in sorted(scale_group["source_context_id"].unique()):
                kept = scale_group.loc[~scale_group["source_context_id"].eq(omitted)]
                association_rows.append({
                    "budget_fraction": float(budget), "regret_scale": scale,
                    "scope": "leave_one_source_context_out",
                    "omitted_source_context": omitted,
                    "n": int(len(kept)), "spearman": _spearman(kept, scale),
                })
    association = pd.DataFrame(association_rows)
    stratified, stratified_report = _metric_stratified(joined)
    curve_path = manifests / "reviewer_full_surface_decision_curves.csv"
    association_path = manifests / "reviewer_decision_budget_lopo.csv"
    compatibility_association_path = manifests / "reviewer_decision_budget_loto.csv"
    report_path = manifests / "reviewer_decision_robustness.json"
    stratified_path = manifests / "reviewer_decision_metric_stratified.csv"
    stratified_report_path = manifests / "reviewer_decision_metric_stratified.json"
    curve_output = joined.copy()
    if not auxiliary.empty and "worst_oracle_normalized_regret" in auxiliary.columns:
        curve_output = curve_output.merge(
            auxiliary[
                ["source_environment_id", "target_environment_id", "metric", "budget_fraction", "worst_oracle_normalized_regret"]
            ],
            on=["source_environment_id", "target_environment_id", "metric", "budget_fraction"],
            how="left",
            validate="one_to_one",
        )
    curve_output.to_csv(curve_path, index=False)
    association.to_csv(association_path, index=False)
    association.to_csv(compatibility_association_path, index=False)
    stratified.to_csv(stratified_path, index=False)
    stratified_report_path.write_text(json.dumps(stratified_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    primary_row = association.loc[
        association["scope"].eq("all_18_metric_transfer_points")
        & association["regret_scale"].eq("normalized_regret")
        & np.isclose(association["budget_fraction"].astype(float), PRIMARY_RANDOM_BUDGET)
    ]
    if len(primary_row) != 1:
        raise RuntimeError("primary 10% decision association is not unique")
    report = {
        "status": "executed",
        "analysis_surface": "matched-fixed full-depth decision macro summaries; source-frozen predictor",
        "directed_transfer_metric_rows": 18,
        "budget_fractions": list(DECISION_BUDGETS),
        "regret_scales": {
            "normalized_regret": "random-reference normalization; same matched-fixed canonical surface as Fig. 4B",
            "worst_oracle_normalized_regret": "bounded worst-oracle secondary diagnostic retained from the materialized source-frozen decision curves",
        },
        "primary_descriptive_association": {
            "budget_fraction": PRIMARY_RANDOM_BUDGET,
            "spearman": float(primary_row.iloc[0]["spearman"]),
            "source": "reliability_transport_decision_link.csv",
        },
        "primary_cv": {
            "method": "leave-one-unordered-context-pair-out",
            "blocks": 3,
            "held_out_unit": "both directions and all three metrics for one unordered context pair",
            "reverse_direction_leakage": False,
        },
        "source_context_sensitivity": {
            "method": "leave-one-source-context-out",
            "blocks": 3,
            "held_out_unit": "all outgoing directed rows for one source context",
        },
        "robustness": "all budgets plus unordered-pair LOPO and source-context sensitivity; descriptive association only",
        "metric_stratified": stratified_report,
        "outputs": {
            "decision_curves": curve_path.relative_to(root).as_posix(),
            "budget_lopo": association_path.relative_to(root).as_posix(),
            "budget_loto_compatibility_alias": compatibility_association_path.relative_to(root).as_posix(),
            "metric_stratified_csv": stratified_path.relative_to(root).as_posix(),
            "metric_stratified_json": stratified_report_path.relative_to(root).as_posix(),
        },
    }
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
