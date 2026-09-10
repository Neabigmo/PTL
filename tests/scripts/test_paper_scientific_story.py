"""Narrative and source-table checks for the five-figure scientific story."""

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


def test_main_has_exactly_five_narrative_figures() -> None:
    text = MAIN.read_text(encoding="utf-8")
    assert len(re.findall(r"\\begin\{figure\*\}", text)) == 5
    stems = (
        "reliability_transportability_fig1_graphical_abstract.pdf",
        "reliability_transportability_fig2_transport_atlas.pdf",
        "reliability_transportability_fig3_measurement_boundary.pdf",
        "reliability_transportability_fig4_decision_consequence.pdf",
        "reliability_transportability_fig5_failure_anatomy.pdf",
    )
    assert all(stem in text for stem in stems)
    assert "Figure 6" not in text and "fig6" not in text.lower()
    assert "reliability_transportability_fig2_forest" not in text


def test_source_tables_are_named_and_materialized() -> None:
    names = ("fig1_source.csv", "fig2_transport_atlas.csv", "fig3_measurement_boundary.csv", "fig4_decision_consequence.csv", "fig5_failure_anatomy.csv")
    for name in names:
        path = SOURCE_DIR / name
        assert path.is_file(), name
        assert len(pd.read_csv(path)) > 0, name
    report = json.loads((MANIFESTS / "reliability_transport_figures.json").read_text(encoding="utf-8"))
    assert report["figure_count"] == 5
    assert report["source_tables_directory"].endswith("source_tables")


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


def test_main_reduces_engineering_detail_but_keeps_blockers() -> None:
    main = MAIN.read_text(encoding="utf-8").lower()
    supplement = SUPPLEMENT.read_text(encoding="utf-8").lower()
    engineering = ("sha256", "manifest", "checkpoint", "claim lock")
    assert sum(main.count(token) for token in engineering) < sum(supplement.count(token) for token in engineering)
    for blocker in ("state", "txpert", "scgpt"):
        assert blocker in main
    assert "source-frozen" in main
    assert "target outcomes" in supplement
