"""Narrative and source-table checks for the six-figure scientific story."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "paper/iclr2027/main.tex"
SUPPLEMENT = ROOT / "paper/iclr2027/supplement.tex"
MANIFESTS = ROOT / "artifacts/manifests"
SOURCE_DIR = ROOT / "results/figures/reliability_transportability/source_tables"


def test_main_has_exactly_six_narrative_figures() -> None:
    text = MAIN.read_text(encoding="utf-8")
    assert len(re.findall(r"\\begin\{figure\*\}", text)) == 6
    stems = (
        "reliability_transportability_fig1_graphical_abstract.pdf",
        "reliability_transportability_fig2_transport_atlas.pdf",
        "reliability_transportability_fig3_measurement_boundary.pdf",
        "reliability_transportability_fig4_decision_consequence.pdf",
        "reliability_transportability_fig5_failure_anatomy.pdf",
        "reliability_transportability_fig6_prospective_limit.pdf",
    )
    assert all(stem in text for stem in stems)
    assert "fig6_prospective_limit.pdf" in text.lower()
    assert "reliability_transportability_fig2_forest" not in text


def test_source_tables_are_named_and_materialized() -> None:
    names = ("fig1_source.csv", "fig2_transport_atlas.csv", "fig3_measurement_boundary.csv", "fig4_decision_consequence.csv", "fig5_failure_anatomy.csv", "fig6_prospective_limit.csv")
    for name in names:
        path = SOURCE_DIR / name
        assert path.is_file(), name
        assert len(pd.read_csv(path)) > 0, name
    report = json.loads((MANIFESTS / "reliability_transport_figures.json").read_text(encoding="utf-8"))
    assert report["figure_count"] == 6
    assert report["source_tables_directory"].endswith("source_tables")

    registry = json.loads((MANIFESTS / "figure_exemplar_registry.json").read_text(encoding="utf-8"))
    assert registry["primary_display_context"] == {
        "source_environment_id": "frangieh_melanoma_control",
        "target_environment_id": "frangieh_melanoma_ifng",
        "metric": "delta_cosine",
        "reason": "fixed semantic exemplar, first frozen metric, canonical Ctrl→IFNγ shift; not selected by observed magnitude",
    }
    assert registry["rank_flow"]["n_labels"] == 10
    assert registry["failure_cases"]["complete_rows"] == 10
    assert "NaN" not in (MANIFESTS / "figure_exemplar_registry.json").read_text(encoding="utf-8")


def test_decision_link_contract_is_fixed_and_complete() -> None:
    report = json.loads((MANIFESTS / "reliability_transport_decision_link.json").read_text(encoding="utf-8"))
    contract = report["primary_contract"]
    assert contract["measurement_depth"] == "full"
    assert contract["decision_budget_fraction"] == 0.1
    assert contract["x"] == "D_meas-ID"
    assert contract["y"] == "normalized_regret"
    assert report["rows"] == 18
    table = pd.read_csv(MANIFESTS / "reliability_transport_decision_link.csv")
    assert len(table) == 18
    assert table[["d_meas_id", "normalized_regret"]].notna().all().all()


def test_failure_anatomy_preserves_prospective_boundary() -> None:
    report = json.loads((MANIFESTS / "reliability_transport_failure_anatomy.json").read_text(encoding="utf-8"))
    boundary = report["prospective_feature_boundary"]
    assert "prediction_norm" in boundary
    assert "response_displacement" not in boundary
    table = pd.read_csv(MANIFESTS / "reliability_transport_failure_anatomy.csv")
    assert set(table["target_informed_explanatory"].dropna().unique()) == {1}
    assert "response_displacement" in table.columns


def test_measurement_boundary_exposes_cell_thresholds_and_descriptive_iqr() -> None:
    table = pd.read_csv(SOURCE_DIR / "fig3_measurement_boundary.csv")
    required = {"resolution_90pct_full_budget", "threshold_kind", "budget", "budget_label", "uncertainty_type"}
    assert required.issubset(table.columns)
    threshold_map = table.loc[table["panel"].eq("C")]
    assert set(threshold_map["threshold_kind"].dropna()) == {"detect", "resolve"}
    assert threshold_map["budget_label"].notna().all()
    assert not table.astype(str).apply(lambda column: column.str.contains("mean of .*CI|mean of source×contrast intervals", regex=True).any()).any()
    curve = table.loc[table["panel"].eq("E")]
    assert set(curve["uncertainty_type"].dropna()) == {"descriptive IQR across source×contrast cells"}
    signed = pd.to_numeric(table["identifiable_divergence"], errors="coerce").dropna()
    assert (signed < 0).any()
    assert "ratio_to_full" not in table.columns


def test_decision_curves_use_transfer_iqr_and_keep_group_intervals() -> None:
    table = pd.read_csv(SOURCE_DIR / "fig4_decision_consequence.csv")
    assert not table.astype(str).apply(lambda column: column.str.contains("mean of canonical group-level 90% intervals", case=False).any()).any()
    assert "descriptive IQR across directed transfers" in set(table["uncertainty_type"].dropna())
    group_rows = table.loc[table["panel"].eq("E")]
    assert {"d_meas_id_ci_low", "d_meas_id_ci_high", "normalized_regret_ci_low", "normalized_regret_ci_high"}.issubset(group_rows.columns)
    contract = pd.read_csv(MANIFESTS / "reliability_transport_decision_link.csv")
    assert len(contract) == 18


def test_failure_state_transitions_and_coverage_contract() -> None:
    table = pd.read_csv(SOURCE_DIR / "fig5_failure_anatomy.csv")
    required = {"source_state", "target_state", "count", "fraction"}
    assert required.issubset(table.columns)
    state_rows = table.loc[table["panel"].eq("D")]
    assert set(state_rows["source_state"].dropna()).issubset({"-1", "tie", "+1", "unstable"})
    assert set(state_rows["target_state"].dropna()).issubset({"-1", "tie", "+1", "unstable"})
    assert state_rows["fraction"].between(0, 1).all()
    coverage = json.loads((MANIFESTS / "reliability_transport_failure_anatomy_coverage.json").read_text(encoding="utf-8"))
    assert coverage["no_imputation"] is True
    assert sum(item["total"] for item in coverage["coverage_by_source_target_metric"]) == 2784
    assert sum(item["complete"] for item in coverage["coverage_by_source_target_metric"]) == 219
    assert "complete_vs_missing_distribution" in coverage


def test_prospective_feature_firewall_and_blocked_replication() -> None:
    table = pd.read_csv(SOURCE_DIR / "fig6_prospective_limit.csv")
    firewall = table.loc[table["panel"].eq("A")].iloc[0]
    assert firewall["allowed_features"].split(";") == [
        "uq_cosine_disagreement", "uq_mean_gene_variance", "uq_effect_norm_variance",
        "prediction_norm", "prediction_sparsity", "prediction_concentration",
    ]
    for forbidden in ("response_displacement", "effect_magnitude", "target_risk", "D_meas-ID", "target failure label"):
        assert forbidden in firewall["forbidden_target_informed_quantities"]
        assert forbidden not in firewall["allowed_features"]
    calibration = table.loc[table["panel"].eq("E"), "calibration_status"].dropna()
    assert set(calibration) == {"not_reported_train_fold_only_no_posthoc_target_calibration"}
    blocked = table.loc[table["panel"].eq("F")].iloc[0]
    assert blocked["numeric_result"] is False or str(blocked["numeric_result"]).lower() == "false"
    assert "no shared per-label target" in blocked["reason"]


def test_metric_pair_states_enforce_frozen_strict_support() -> None:
    states_path = MANIFESTS / "reliability_transport_metric_pair_states.csv"
    required = {
        "source_state_discovery", "target_state_validation",
        "source_strict_count", "target_strict_count",
    }
    invalid = 0
    for chunk in pd.read_csv(states_path, usecols=sorted(required), chunksize=250_000):
        source_stable = chunk["source_state_discovery"].astype(str).str.startswith("stable_")
        target_stable = chunk["target_state_validation"].astype(str).str.startswith("stable_")
        invalid += int((source_stable & chunk["source_strict_count"].lt(8)).sum())
        invalid += int((target_stable & chunk["target_strict_count"].lt(8)).sum())
    assert invalid == 0

    report = json.loads((MANIFESTS / "reliability_transport_metric_dependence.json").read_text(encoding="utf-8"))
    assert "minimum strict support 8" in report["tie_policy"]


def test_main_reduces_engineering_detail_but_keeps_blockers() -> None:
    main = MAIN.read_text(encoding="utf-8").lower()
    supplement = SUPPLEMENT.read_text(encoding="utf-8").lower()
    engineering = ("sha256", "manifest", "checkpoint", "claim lock")
    assert sum(main.count(token) for token in engineering) < sum(supplement.count(token) for token in engineering)
    for blocker in ("state", "txpert", "scgpt"):
        assert blocker in main
    assert "source-frozen" in main
    assert "target outcomes" in supplement
    assert "deployment probe" not in main
    assert "feasibility boundary" not in main
