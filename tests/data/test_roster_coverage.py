import pandas as pd


def test_roster_coverage_matrix_is_explicit_and_planning_only() -> None:
    matrix = pd.read_csv("artifacts/manifests/predictor_environment_coverage.csv")
    assert matrix["predictor_id"].nunique() == 6
    assert matrix["environment_id"].nunique() == 8
    assert len(matrix) == 48
    assert matrix["execution_status"].eq("planned_formal_run").all()
    assert matrix["coverage_policy"].notna().all()
