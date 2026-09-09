"""Global checks for the canonical formal-v2 artifact surface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "artifacts/manifests"
REMOVED_IDS = {"adamson_k562", "replogle_k562_gwps"}
STABILITY_SEEDS = {20260907, 20260917, 20260927, 20261007, 20261017}


def test_canonical_environment_ids_match_registry_and_removed_surface_is_absent() -> None:
    registry = pd.read_csv(MANIFESTS / "environment_registry.csv")
    current = set(registry["environment_id"].astype(str))
    assert len(current) == 8
    paths = [
        MANIFESTS / "formal_v2_environment_transfer_matrix.csv",
        MANIFESTS / "formal_v2_deployment_utility.csv",
        MANIFESTS / "formal_v2_identity_probe_confusion.csv",
        MANIFESTS / "formal_v2_reproducibility.csv",
        MANIFESTS / "formal_v2_reliability_metrics.csv",
        MANIFESTS / "formal_v2_predictor_metrics.csv",
    ]
    for path in paths:
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8").lower()
        assert not any(removed in text for removed in REMOVED_IDS), path
        if "environment_id" in pd.read_csv(path, nrows=0).columns:
            frame = pd.read_csv(path, usecols=["environment_id"])
            observed = {value for value in frame["environment_id"].dropna().astype(str) if value not in {"all", "off_diagonal", "environment_macro"}}
            assert observed.issubset(current), (path, observed - current)


def test_canonical_split_and_gene_contracts_are_current() -> None:
    summary = json.loads((MANIFESTS / "formal_v2_reliability_summary.json").read_text(encoding="utf-8"))
    assert int(summary["split_seed"]) == 20260907
    assert int(summary["rows"]) == len(pd.read_csv(ROOT / "artifacts/source_data/formal_v2_reliability_predictions.csv"))
    predictor = json.loads((MANIFESTS / "formal_v2_predictor_summary.json").read_text(encoding="utf-8"))
    assert int(predictor["split_seed"]) == 20260907
    assert int(predictor["evaluation_gene_count"]) == 8229
    for path in MANIFESTS.glob("formal_v2_*__split_*.csv"):
        # Split-suffixed files may only be one of the five declared seeds.
        seed = int(path.stem.rsplit("_", 1)[-1])
        assert seed in STABILITY_SEEDS, path


def test_figure_manifest_sources_exist_and_conclusion_is_cautious() -> None:
    payload = json.loads((MANIFESTS / "formal_v2_figure_manifest.json").read_text(encoding="utf-8"))
    conclusion = payload["core_conclusion"].lower()
    assert "rank displacement" in conclusion
    assert "context-specific excess" in conclusion
    assert "heterogeneous" in conclusion
    for relative in payload["source_data"]:
        assert (ROOT / relative).is_file(), relative


def test_external_ladder_status_matches_materialized_summary() -> None:
    ladder = json.loads((MANIFESTS / "formal_v2_reliability_reordering_summary.json").read_text(encoding="utf-8"))
    entry = next(item for item in ladder["ladder"] if item["comparison"] == "head_to_head_crisprko_crispri")
    assert entry["status"] == "executed_external_evaluation_only"
    assert entry["canonical_summary_path"] == "artifacts/manifests/head_to_head_controlled_shift_ladder_summary.json"
    assert entry["pooled_with_formal_frangieh_predictors"] is False
    assert "evaluation-only" in entry["predictor_contract"]
    assert "not a trained perturbation model" in entry["predictor_contract"]
    external = json.loads((MANIFESTS / "head_to_head_controlled_shift_ladder_summary.json").read_text(encoding="utf-8"))
    assert external["status"] == entry["status"]
    assert external["predictor_contract"] == entry["predictor_contract"]
    assert external["pooled_with_formal_frangieh_predictors"] is False
    legacy_status = "external_data_available_" + "not" + "_ingested"
    legacy_reason = "dedicated ingestion " + "contr" + "act"
    for path in (ROOT / "scripts/run_formal_v2_reliability_reordering.py", MANIFESTS / "formal_v2_reliability_reordering_summary.json"):
        text = path.read_text(encoding="utf-8").lower()
        assert legacy_status not in text
        assert legacy_reason not in text


def test_reliability_shift_and_transport_claims_are_bounded() -> None:
    paper = (ROOT / "paper/iclr2027/main.tex").read_text(encoding="utf-8").lower()
    assert "hard limits" not in paper
    assert "deployment-available context is insufficient" not in paper
    assert "information is insufficient" not in paper
    assert "validation-only acceptance criterion" in paper
    descriptors = json.loads((MANIFESTS / "formal_v2_transport_predictor_summary.json").read_text(encoding="utf-8"))
    descriptor_columns = descriptors.get("descriptor_columns") or descriptors.get("feature_columns")
    assert descriptor_columns is not None
    assert set(descriptor_columns) == {
        "same_cell_context", "same_perturbation_modality", "same_platform", "same_condition",
        "same_dataset_family", "same_context_modality", "prediction_geometry_distance",
        "uq_distribution_distance",
    }
    systema = json.loads((MANIFESTS / "formal_v2_systema_robustness_summary.json").read_text(encoding="utf-8"))
    assert systema.get("metric_entrypoint")
    gears = json.loads((MANIFESTS / "formal_v2_gears_reliability_summary.json").read_text(encoding="utf-8"))
    assert gears["validation_predictions_used"] is True
    assert set(gears["datasets"]) == {"NormanWeissman2019_filtered", "ReplogleWeissman2022_rpe1"}
    multisplit = json.loads((MANIFESTS / "formal_v2_multisplit_reliability_summary.json").read_text(encoding="utf-8"))
    contrast = multisplit["paired_asymmetry_contrast"]
    assert contrast["contrast"] == "delta_leave_environment_out_minus_delta_leave_predictor_out"
    assert int(contrast["n_splits"]) == 5


def test_reliability_reordering_and_recalibration_audits_are_materialized() -> None:
    reordering = json.loads((MANIFESTS / "formal_v2_reliability_reordering_summary.json").read_text(encoding="utf-8"))
    assert reordering["status"] == "formal_v2_reliability_reordering_executed_frangieh_first_stage"
    reordering_detail = pd.read_csv(MANIFESTS / "formal_v2_reliability_reordering.csv")
    reordering_perturbations = pd.read_csv(MANIFESTS / "formal_v2_reliability_reordering_perturbations.csv")
    assert len(reordering_detail) == 135
    assert len(reordering_perturbations) > 1000
    assert set(reordering_perturbations["response_program_is_descriptive"]) == {1}
    assert set(reordering_perturbations["risk_and_confidence_outcomes_used_for_evaluation_only"]) == {1}
    assert {"frangieh_condition", "tian_modality", "head_to_head_crisprko_crispri"} == {
        item["comparison"] for item in reordering["ladder"]
    }

    recalibration = json.loads((MANIFESTS / "formal_v2_recalibration_counterfactual_summary.json").read_text(encoding="utf-8"))
    assert recalibration["status"] == "formal_v2_recalibration_counterfactual_executed"
    recalibration_detail = pd.read_csv(MANIFESTS / "formal_v2_recalibration_counterfactual.csv")
    assert len(recalibration_detail) == 45
    assert int(recalibration_detail["n_order_disagreements"].sum()) == 0
    assert set(recalibration_detail["calibration_fit_on"]) == {"calibration_rows_only"}


def test_controlled_shift_scientific_lock_is_materialized() -> None:
    inference = json.loads((MANIFESTS / "formal_v2_controlled_shift_inference.json").read_text(encoding="utf-8"))
    assert inference["status"] == "formal_v2_controlled_shift_scientific_lock_executed"
    assert inference["bootstrap"]["unit"] == "matched perturbation label"
    assert inference["bootstrap"]["pairwise_reconstruction"] is True
    assert inference["confidence_permutation_null"]["draws"] == 2000
    assert inference["noise_floor"]["type"] == "within_context_stochastic_model_member"
    cross = pd.read_csv(MANIFESTS / "formal_v2_controlled_shift_oof_inference.csv")
    noise = pd.read_csv(MANIFESTS / "formal_v2_controlled_shift_noise_floor.csv")
    assert len(cross) == 3
    assert len(noise) == 9
    assert set(cross["comparison"]) == {"frangieh_condition_oof"}
    assert set(noise["measurement_noise_included"]) == {False}
    assert set(cross["risk_inversion_rate_ci_low"] < cross["risk_inversion_rate"]) == {True}
    assert set(cross["risk_inversion_rate"] < cross["risk_inversion_rate_ci_high"]) == {True}
    assert set(cross["normalized_rank_displacement_mean"] > 0) == {True}
    assert set(noise["risk_inversion_rate"] > 0) == {True}


def test_claim_lock_source_frozen_and_measurement_artifacts_are_materialized() -> None:
    source = json.loads((MANIFESTS / "formal_v2_claim_lock_source_frozen.json").read_text(encoding="utf-8"))
    assert source["status"] == "source_frozen_primary_executed"
    assert int(source["common_perturbation_count"]) == 243
    assert int(source["folds"]) == 5
    assert int(source["evaluation_gene_count"]) == 8229
    npz_path = ROOT / source["prediction_npz"]
    assert npz_path.is_file()
    with np.load(npz_path, allow_pickle=False) as payload:
        assert payload["prediction"].shape == (3, 243, 3, 8229)
        assert payload["fold_id"].shape == (243,)
        assert payload["confidence"].shape == (3, 243)
    reordering = pd.read_csv(MANIFESTS / "formal_v2_claim_lock_source_frozen_reordering.csv")
    assert len(reordering) == 27
    assert set(reordering["estimand"]) == {"source_frozen_primary"}
    assert set(reordering["metric"]) == {"delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement"}
    measurement = json.loads((MANIFESTS / "formal_v2_claim_lock_measurement.json").read_text(encoding="utf-8"))
    assert measurement["status"] == "matched_budget_measurement_and_joint_floors_executed"
    assert len(measurement["split_seeds"]) == 30
    floors = pd.read_csv(MANIFESTS / "formal_v2_claim_lock_measurement_floors.csv")
    summary = pd.read_csv(MANIFESTS / "formal_v2_claim_lock_measurement_summary.csv")
    assert len(floors) == 2430
    assert len(summary) == 27
    assert set(summary["n_split_seeds"]) == {30}
    assert summary[["cross_d", "measurement_floor_d", "joint_floor_d", "delta_joint"]].notna().all().all()
    boundary = json.loads((MANIFESTS / "formal_v2_claim_lock_replication_boundary.json").read_text(encoding="utf-8"))
    assert boundary["status"] == "independent_replication_claim_lock_executed"
    assert boundary["selected_candidate"] == "nadig_hepg2_vs_jurkat"
    assert boundary["prediction_evaluated"] is True
    assert boundary["risk_evaluated"] is True
    replication = json.loads((MANIFESTS / "formal_v2_claim_lock_replication_nadig.json").read_text(encoding="utf-8"))
    assert replication["status"] == "independent_replication_claim_lock_executed"
    assert replication["candidate_id"] == "nadig_hepg2_vs_jurkat"
    assert int(replication["common_perturbation_count"]) == 2392
    assert int(replication["evaluation_gene_count"]) == 6606
    replication_summary = pd.read_csv(MANIFESTS / "formal_v2_claim_lock_replication_nadig.csv")
    replication_floors = pd.read_csv(MANIFESTS / "formal_v2_claim_lock_replication_nadig_floors.csv")
    assert len(replication_summary) == 12
    assert len(replication_floors) == 1080
    assert set(replication_summary["n_split_seeds"]) == {30}
    assert set(replication_summary["bootstrap_draws"]) == {2000}
    assert replication_summary[["cross_d", "measurement_floor_d", "joint_floor_d", "delta_joint"]].notna().all().all()
    assert np.isfinite(replication_floors.select_dtypes(include=["number"]).to_numpy()).all()
    with np.load(ROOT / replication["prediction_path"], allow_pickle=False) as payload:
        assert payload["prediction"].shape == (2, 2392, 3, 6606)
        assert np.isfinite(payload["prediction"]).all()
    sensitivity = json.loads((MANIFESTS / "formal_v2_claim_lock_measurement_sensitivity40.json").read_text(encoding="utf-8"))
    assert sensitivity["status"] == "matched_budget_measurement_and_joint_floors_sensitivity40_executed"
    assert sensitivity["sensitivity_executed"] is True
    sensitivity_detail = pd.read_csv(MANIFESTS / "formal_v2_claim_lock_measurement_sensitivity40.csv")
    assert len(sensitivity_detail) == 27
    assert set(sensitivity_detail["eligibility_min_cells"]) == {40}
    registry = json.loads((MANIFESTS / "reordering_replication_candidate_registry.json").read_text(encoding="utf-8"))
    assert registry["outcome_blind"] is True
    assert registry["prediction_evaluated"] is False
    assert registry["risk_evaluated"] is False
    assert int(registry["candidate_count"]) >= 4
    registry_detail = pd.read_csv(MANIFESTS / "reordering_replication_candidate_registry.csv")
    assert set(registry_detail["prediction_evaluated"]) == {False}
    assert set(registry_detail["risk_evaluated"]) == {False}
    guide = json.loads((MANIFESTS / "formal_v2_claim_lock_guide_id_semantics.json").read_text(encoding="utf-8"))
    assert guide["status"] == "guide_id_semantics_audited_not_single_sgrna_identity"
    assert guide["risk_evaluated"] is False
