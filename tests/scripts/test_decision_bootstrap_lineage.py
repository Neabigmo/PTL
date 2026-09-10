from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "artifacts/manifests"
SUMMARY = MANIFESTS / "reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv"
SUMMARY_REPORT = MANIFESTS / "reliability_transport_measurement_depth_decision_macro_bootstrap_summary.json"
MASTER = MANIFESTS / "reliability_transport_master.csv"

KEY = [
    "source_environment_id",
    "target_environment_id",
    "left_target_environment_id",
    "right_target_environment_id",
    "metric",
    "cell_budget_label",
    "decision_budget_fraction",
    "universe_mode",
]


def test_decision_bootstrap_summary_has_explicit_point_and_inferential_layers() -> None:
    summary = pd.read_csv(SUMMARY, dtype={"cell_budget_label": "string"})
    report = json.loads(SUMMARY_REPORT.read_text(encoding="utf-8"))
    assert len(summary) == 432
    assert summary[KEY].drop_duplicates().shape[0] == 432
    assert set(summary["draw_count"]) == {2000}
    assert set(summary["valid_draw_count"]) == {2000}
    assert set(summary["seed_count"]) == {30}
    assert set(summary["status"]) == {"executed"}
    assert set(summary["bootstrap_unit"]) == {"perturbation_label_then_measurement_seed"}
    assert "raw_bootstrap_sha256" in summary.columns
    assert report["raw_rows"] == 864000
    assert report["valid_directed_groups"] == 108
    assert report["summary_rows"] == 432


def test_master_decision_intervals_trace_to_canonical_summary() -> None:
    master = pd.read_csv(MASTER)
    decision = master.loc[master["regime"].eq("decision_transport_directed")].copy()
    assert len(decision) == 432
    for column in (
        "retention_seed_q05", "retention_seed_q95", "retention_ci_low", "retention_ci_high",
        "regret_seed_q05", "regret_seed_q95", "regret_ci_low", "regret_ci_high",
        "excess_regret_seed_q05", "excess_regret_seed_q95", "excess_regret_ci_low", "excess_regret_ci_high",
    ):
        assert decision[column].notna().all()
    provenance = decision["provenance"].map(json.loads)
    assert provenance.map(lambda item: item["bootstrap_summary"].endswith("decision_macro_bootstrap_summary.csv")).all()
    assert provenance.map(lambda item: item["raw_bootstrap"].endswith("decision_macro_bootstrap.csv")).all()
    assert provenance.map(lambda item: item["risk_input"].endswith("matched_fixed_risks.csv")).all()
