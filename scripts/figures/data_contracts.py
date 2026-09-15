"""Small, explicit data kill-switches for the six main figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import spearmanr

from scripts.figures.common import ATLAS_DATASET, CONTEXT_ORDER, DEPTH_ORDER, METRICS, full_rank_table, rank_table


def validate_canonical_inputs(root: Path) -> dict[str, Any]:
    """Validate counts and contracts before a FINAL figure render.

    This intentionally checks the small canonical summaries only; it does not
    replace model execution or statistical validation.
    """
    manifests = root / "artifacts/manifests"
    checks: dict[str, Any] = {}
    ranks = rank_table(manifests)
    checks["formal_reliability_labels"] = int(len(ranks))
    checks["formal_reliability_labels_expected"] = 47
    full_ranks = full_rank_table(manifests)
    checks["full_source_frozen_labels"] = int(len(full_ranks))
    checks["full_source_frozen_labels_expected"] = 243
    checks["contexts"] = list(CONTEXT_ORDER)
    atlas = pd.read_csv(manifests / "reliability_transport_atlas.csv", low_memory=False)
    atlas = atlas.loc[atlas["dataset"].eq(ATLAS_DATASET) & atlas["metric"].isin(METRICS)]
    checks["atlas_rows"] = int(len(atlas))
    checks["atlas_rows_expected"] = 27
    checks["metrics"] = sorted(atlas["metric"].dropna().unique().tolist())
    depth_values = set(pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_summary.csv", usecols=["cell_budget_label"], low_memory=False)["cell_budget_label"].astype(str).str.lower())
    checks["depths"] = [depth for depth in DEPTH_ORDER if depth in depth_values]
    checks["depths_expected"] = list(DEPTH_ORDER)
    link = pd.read_csv(manifests / "reliability_transport_decision_link.csv", low_memory=False)
    checks["decision_link_rows"] = int(len(link))
    checks["decision_link_rows_expected"] = 18
    checks["decision_link_rho"] = float(spearmanr(link["d_meas_id"], link["normalized_regret"]).statistic)
    checks["decision_link_rho_expected"] = 0.7750257997936018
    summary = pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv", usecols=["decision_budget_fraction"], low_memory=False)
    checks["decision_budgets"] = sorted(pd.to_numeric(summary["decision_budget_fraction"], errors="coerce").dropna().unique().tolist())
    checks["decision_budgets_expected"] = [0.05, 0.10, 0.20, 0.50]
    failure = pd.read_csv(manifests / "reliability_transport_failure_anatomy.csv", usecols=["status"], low_memory=False)
    checks["failure_total"] = int(len(failure))
    checks["failure_complete"] = int(failure["status"].eq("executed").sum())
    checks["failure_complete_expected"] = 219
    nadig = pd.read_csv(manifests / "reliability_transport_atlas.csv", low_memory=False)
    nadig = nadig.loc[nadig["dataset"].eq("NadigOConner2024") & nadig["metric"].isin(METRICS)]
    checks["nadig_rows"] = int(len(nadig))
    checks["nadig_rows_expected"] = 6
    checks["nadig_status"] = "full-size per-label replication; two directions × three metrics"

    failures: list[str] = []
    if checks["formal_reliability_labels"] != checks["formal_reliability_labels_expected"]: failures.append("formal reliability label count")
    if checks["full_source_frozen_labels"] != checks["full_source_frozen_labels_expected"]: failures.append("full source-frozen label count")
    if checks["atlas_rows"] != checks["atlas_rows_expected"]: failures.append("Frangieh atlas row count")
    if set(checks["metrics"]) != set(METRICS): failures.append("metric set")
    if checks["depths"] != list(DEPTH_ORDER): failures.append("depth set")
    if checks["decision_link_rows"] != checks["decision_link_rows_expected"]: failures.append("decision link row count")
    if abs(checks["decision_link_rho"] - checks["decision_link_rho_expected"]) > 1e-6: failures.append("decision rho")
    if checks["decision_budgets"] != checks["decision_budgets_expected"]: failures.append("decision budgets")
    if checks["failure_complete"] != checks["failure_complete_expected"]: failures.append("failure coverage")
    if checks["nadig_rows"] != checks["nadig_rows_expected"]: failures.append("Nadig replication rows")
    checks["status"] = "PASS" if not failures else "MISMATCH"
    checks["failures"] = failures
    return checks


__all__ = ["validate_canonical_inputs"]
