import pandas as pd

from src.evaluation.reliability_transport_atlas import build_atlas


def test_atlas_freezes_tiers_from_metadata_and_keeps_unavailable_fields_explicit():
    registry = pd.DataFrame({
        "environment_id": ["env1", "env2"],
        "environment_key": ["source", "other"],
        "dataset_id": ["d1", "d2"],
        "role": ["common_core", "context_expansion"],
        "semantic_status": ["raw_context_verified", "raw_context_verified"],
        "status": ["ready_signature_surface", "ready_signature_surface"],
    })
    summary = pd.DataFrame({
        "source_environment_id": ["source"],
        "left_target_environment_id": ["left"],
        "right_target_environment_id": ["right"],
        "metric": ["delta_cosine"],
        "ordering_cross_disagreement": [0.2],
        "ordering_cross_disagreement_ci_low": [0.1],
        "ordering_cross_disagreement_ci_high": [0.3],
        "ordering_delta_meas_id": [0.1],
        "ordering_delta_meas_id_ci_low": [0.0],
        "ordering_delta_meas_id_ci_high": [0.2],
        "ordering_delta_joint_id": [0.05],
        "ordering_delta_joint_id_ci_low": [-0.05],
        "ordering_delta_joint_id_ci_high": [0.15],
    })
    _, atlas = build_atlas(registry=registry, frangieh_summary=summary)
    assert atlas.loc[0, "evidence_tier"] == 1
    assert pd.isna(atlas.loc[0, "stable_fraction_both"])
    assert atlas.loc[0, "availability"] == "canonical_fullsize_claim_lock"
