import pandas as pd


def test_biological_instance_registry_is_canonical_and_disjoint() -> None:
    manifest = pd.read_csv("artifacts/manifests/biological_instance_registry.csv")
    registry = pd.read_csv("artifacts/manifests/environment_registry.csv")

    assert len(manifest) == 1881
    assert manifest["biological_instance_id"].is_unique
    assert set(manifest["environment_id"]) == set(registry["environment_id"])
    assert not manifest.duplicated(["environment_id", "perturbation_label", "dose", "timepoint"]).any()
    assert set(manifest["split"]) == {"train", "validation", "test"}
    assert manifest.groupby("environment_id")["split"].nunique().eq(3).all()
    assert manifest["split_seed"].eq(20260907).all()
    assert manifest["model_seeds"].eq("0,1,2").all()
    assert manifest["aggregation_rule"].eq("weighted_mean_of_batch_matched_deltas").all()
