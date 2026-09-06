from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.run_perturbation_model import (
    build_gears_adata_from_prepared,
    build_gears_run_matrix,
    custom_condition_split_from_prepared,
    ensure_gears_filter_metadata,
    invalidate_incompatible_gears_coexpression_cache,
    materialize_official_gears_go_subgraph,
    materialize_delta_predictions,
    run_all,
)


def test_gears_run_matrix_filters_mandatory_dataset_split_surface(tmp_path: Path) -> None:
    audit = pd.DataFrame(
        [
            {
                "split_family": "random_split",
                "split_protocol": "random_condition_holdout",
                "perturbation_overlap": "zero",
                "track": "signature",
                "dataset_scope": "NormanWeissman2019_filtered",
                "seed": 0,
                "status": "ready",
                "output_path": "splits/norman_random.json",
            },
            {
                "split_family": "external_holdout",
                "track": "signature",
                "dataset_scope": "NormanWeissman2019_filtered",
                "seed": 0,
                "status": "ready",
                "output_path": "splits/norman_external.json",
            },
            {
                "split_family": "random_split",
                "track": "signature",
                "dataset_scope": "UnsupportedDataset",
                "seed": 0,
                "status": "ready",
                "output_path": "splits/unsupported.json",
            },
        ]
    )
    audit_path = tmp_path / "split_audit.csv"
    output_path = tmp_path / "run_matrix.csv"
    audit.to_csv(audit_path, index=False)

    matrix = build_gears_run_matrix(audit_path, output_path)

    assert output_path.exists()
    assert len(matrix) == 1
    assert matrix.iloc[0]["model"] == "gears"
    assert matrix.iloc[0]["run_id"].endswith("__gears")
    assert matrix.iloc[0]["split_protocol"] == "random_condition_holdout"
    assert matrix.iloc[0]["perturbation_overlap"] == "zero"


def test_gears_missing_dependency_creates_blocker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    matrix = pd.DataFrame(
        [
            {
                "run_id": "random_split__NormanWeissman2019_filtered__seed0__signature__gears",
                "model": "gears",
                "dataset_id": "NormanWeissman2019_filtered",
                "dataset_scope": "NormanWeissman2019_filtered",
                "split_family": "random_split",
                "seed": 0,
                "split_json_path": "split.json",
                "track": "signature",
                "gene_space_policy": "native",
                "output_dir": str(tmp_path / "gears_out"),
                "prediction_path": str(tmp_path / "gears_out" / "test_predictions.npz"),
                "metadata_path": str(tmp_path / "gears_out" / "test_metadata.parquet"),
                "genes_path": str(tmp_path / "gears_out" / "genes.txt"),
                "log_file": str(tmp_path / "gears.log"),
                "status": "planned",
            }
        ]
    )
    matrix_path = tmp_path / "perturbation_model_run_matrix.csv"
    registry_path = tmp_path / "registry.csv"
    matrix.to_csv(matrix_path, index=False)

    monkeypatch.setattr(
        "src.models.run_perturbation_model.check_gears_dependencies",
        lambda: {"available": False, "modules": {"torch": True, "torch_geometric": False, "gears": False}, "missing": ["torch_geometric", "gears"]},
    )
    manual_path = tmp_path / "manual.md"
    monkeypatch.setattr("src.models.run_perturbation_model.MANUAL_INTERVENTION", manual_path)

    with pytest.raises(RuntimeError, match="GEARS mandatory model blocked"):
        run_all(
            argparse.Namespace(
                model="gears",
                run_matrix=str(matrix_path),
                registry=str(registry_path),
                split_audit=str(tmp_path / "split_audit.csv"),
            )
        )

    blocker = tmp_path / "gears_out" / "blocked_missing_dependency.json"
    assert blocker.exists()
    payload = json.loads(blocker.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked_missing_dependency"
    assert "torch_geometric" in payload["missing_dependencies"]
    assert registry_path.exists()
    assert manual_path.exists()


def test_gears_adapter_builds_anndata_and_custom_split() -> None:
    train = pd.DataFrame(
        {
            "perturbation_label": ["control", "A"],
            "is_control": [True, False],
            "signature_id": ["d__0__control", "d__0__A"],
            "dataset_id": ["d", "d"],
        }
    )
    val = pd.DataFrame(
        {
            "perturbation_label": ["B"],
            "is_control": [False],
            "signature_id": ["d__0__B"],
            "dataset_id": ["d"],
        }
    )
    test = pd.DataFrame(
        {
            "perturbation_label": ["A_B"],
            "is_control": [False],
            "signature_id": ["d__0__A_B"],
            "dataset_id": ["d"],
        }
    )
    prepared = SimpleNamespace(
        train_frame=train,
        val_frame=val,
        test_frame=test,
        y_train=np.ones((2, 3), dtype=np.float32),
        y_val=np.ones((1, 3), dtype=np.float32) * 2,
        y_test=np.ones((1, 3), dtype=np.float32) * 3,
        ref_train=np.ones((2, 3), dtype=np.float32) * 10,
        ref_val=np.ones((1, 3), dtype=np.float32) * 10,
        ref_test=np.ones((1, 3), dtype=np.float32) * 10,
        gene_columns=["A", "B", "C"],
    )

    adata = build_gears_adata_from_prepared(prepared, "toy_dataset")
    split = custom_condition_split_from_prepared(prepared)

    assert adata.shape == (4, 3)
    assert adata.obs["condition"].tolist() == ["ctrl", "A", "B", "A_B"]
    assert adata.var["gene_name"].tolist() == ["A", "B", "C"]
    assert split["train"] == ["A", "ctrl"]
    assert split["val"] == ["B", "ctrl"]
    assert split["test"] == ["A_B"]


def test_gears_condition_split_rejects_overlapping_conditions() -> None:
    prepared = SimpleNamespace(
        train_frame=pd.DataFrame({"perturbation_label": ["A"], "is_control": [False]}),
        val_frame=pd.DataFrame({"perturbation_label": ["A"], "is_control": [False]}),
        test_frame=pd.DataFrame({"perturbation_label": ["B"], "is_control": [False]}),
    )
    with pytest.raises(AssertionError, match="condition split overlap"):
        custom_condition_split_from_prepared(prepared)


def test_gears_absolute_predictions_are_converted_to_matched_reference_delta() -> None:
    prepared = SimpleNamespace(
        y_test=np.zeros((2, 3), dtype=np.float32),
        ref_test=np.asarray([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]], dtype=np.float32),
    )
    result = materialize_delta_predictions(
        prepared,
        ["A", "B"],
        {"A": np.asarray([11.0, 12.0, 13.0]), "B": np.asarray([18.0, 21.0, 20.0])},
        prediction_space="absolute_expression",
    )
    assert np.allclose(result, [[1.0, 2.0, 3.0], [-2.0, 1.0, 0.0]])


def test_gears_filter_metadata_requires_official_preprocessing_outputs() -> None:
    prepared = SimpleNamespace(
        train_frame=pd.DataFrame({"perturbation_label": ["control"], "is_control": [True]}),
        val_frame=pd.DataFrame({"perturbation_label": ["A"], "is_control": [False]}),
        test_frame=pd.DataFrame({"perturbation_label": ["B"], "is_control": [False]}),
        y_train=np.ones((1, 2), dtype=np.float32),
        y_val=np.ones((1, 2), dtype=np.float32),
        y_test=np.ones((1, 2), dtype=np.float32),
        ref_train=np.ones((1, 2), dtype=np.float32),
        ref_val=np.ones((1, 2), dtype=np.float32),
        ref_test=np.ones((1, 2), dtype=np.float32),
        gene_columns=["A", "B"],
    )
    adata = build_gears_adata_from_prepared(prepared, "toy_dataset")

    with pytest.raises(RuntimeError, match="official GEARS preprocessing"):
        ensure_gears_filter_metadata(adata)


def test_official_go_subgraph_preserves_only_relevant_official_edges(tmp_path: Path) -> None:
    graph_dir = tmp_path / "go_essential_all"
    graph_dir.mkdir()
    source = graph_dir / "go_essential_all.csv"
    source.write_text(
        "source,target,importance\nA,B,0.9\nA,C,0.8\nC,C,0.7\n",
        encoding="utf-8",
    )

    materialize_official_gears_go_subgraph(tmp_path, {"A", "B"})

    filtered = pd.read_csv(source)
    assert filtered.to_dict("records") == [{"source": "A", "target": "B", "importance": 0.9}]
    assert (graph_dir / "go_essential_all_full.csv").exists()
    assert (graph_dir / "go_essential_all_pilot_subset.marker").exists()


def test_incompatible_coexpression_cache_is_preserved_as_stale(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    cache = dataset_path / "custom_0_None_0.4_20_co_expression_network.csv"
    cache.write_text(
        "source,target,importance\nA,B,0.9\nC,A,0.8\n",
        encoding="utf-8",
    )
    pert_data = SimpleNamespace(
        dataset_path=str(dataset_path),
        split="custom",
        seed=0,
        train_gene_set_size=None,
    )

    invalidate_incompatible_gears_coexpression_cache(pert_data, {"A", "B"})

    assert not cache.exists()
    assert (dataset_path / (cache.name + ".incompatible")).exists()

    cache.write_text(
        "source,target,importance\nA,B,0.9\nC,A,0.8\n",
        encoding="utf-8",
    )
    invalidate_incompatible_gears_coexpression_cache(pert_data, {"A", "B"})

    assert not cache.exists()
    assert (dataset_path / (cache.name + ".incompatible")).exists()
