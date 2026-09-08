from pathlib import Path

import pandas as pd

from scripts.build_context_benchmark import load_spec


def test_context_benchmark_has_ten_semantically_resolved_environments() -> None:
    spec = load_spec(Path("configs/context_benchmark.yaml"))
    environments = spec["environments"]
    assert len(environments) == 8
    assert len({row["environment_id"] for row in environments}) == 8
    assert all(row["perturbation_modality"] != "metadata_pending" for row in environments)
    assert all(row["platform"] != "processed_signature_table" for row in environments)
    frangieh = [row for row in environments if row["dataset_id"] == "FrangiehIzar2021_RNA"]
    assert {row["condition"] for row in frangieh} == {"Control", "Co-culture", "IFNγ"}


def test_materialized_registry_and_panel_are_consistent() -> None:
    registry = pd.read_csv("artifacts/manifests/environment_registry.csv")
    panel = pd.read_csv("artifacts/manifests/evaluation_gene_space.csv")
    assert len(registry) == 8
    assert registry["environment_key"].is_unique
    assert registry["environment_id"].is_unique
    assert len(panel) > 8000
    assert panel["gene_symbol"].is_unique
    assert panel["panel_id"].eq("ptl_context_v2_intersection").all()
