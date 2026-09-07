import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def test_formal_transfer_atlas_has_two_complete_eight_by_eight_surfaces() -> None:
    path = ROOT / "artifacts/manifests/formal_v2_environment_transfer_matrix.csv"
    frame = pd.read_csv(path)
    assert set(frame["baseline"]) == {"raw_scalar_uq", "u_only_rf"}
    for baseline, group in frame.groupby("baseline"):
        assert group[["source_environment_id", "target_environment_id"]].drop_duplicates().shape[0] == 64
        assert group["same_source_target"].sum() == 8


def test_formal_transform_manifest_records_non_dummy_manifold_and_feature_order() -> None:
    path = ROOT / "artifacts/manifests/formal_v2_feature_transform_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "formal_v2_feature_transforms_materialized"
    assert len(payload["records"]) == 12
    assert all(record["transform_signature"] for record in payload["records"])
    assert all(record["feature_order"]["P"][-1] == "prediction_manifold_distance" for record in payload["records"])


def test_formal_feature_artifact_contains_actual_manifold_values() -> None:
    path = ROOT / "artifacts/source_data/formal_v2_reliability_deployment_features.csv"
    frame = pd.read_csv(path)
    values = pd.to_numeric(frame["prediction_manifold_distance"], errors="coerce")
    assert values.notna().all()
    assert values.max() > 0.0
