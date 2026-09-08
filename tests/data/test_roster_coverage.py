import pandas as pd


def test_roster_coverage_matrix_is_explicit_and_planning_only() -> None:
    matrix = pd.read_csv("artifacts/manifests/predictor_environment_coverage.csv")
    assert matrix["predictor_id"].nunique() == 7
    assert matrix["environment_id"].nunique() == 8
    registry = pd.read_csv("artifacts/manifests/environment_registry.csv")
    assert set(matrix["environment_id"]) == set(registry["environment_id"])
    assert not matrix["environment_key"].is_unique
    assert len(matrix) == 56
    assert matrix["execution_status"].eq("planned_formal_run").all()
    assert matrix["coverage_policy"].notna().all()
