from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

anndata = pytest.importorskip("anndata")

from data.preprocess_dataset import compute_profiles


def test_backed_csc_profiles_preserve_controls_and_reference_deltas(tmp_path):
    matrix = sparse.csc_matrix(
        np.asarray(
            [
                [1.0, 2.0, 0.0],  # ref1 control
                [2.0, 0.0, 1.0],  # ref1 perturbation
                [1.0, 1.0, 1.0],  # ref2 control
                [1.0, 2.0, 1.0],  # ref2 perturbation
            ],
            dtype=np.float32,
        )
    )
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(4)])
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    source = anndata.AnnData(X=matrix, obs=obs, var=var)
    path = tmp_path / "small_csc.h5ad"
    source.write_h5ad(path)
    backed = anndata.read_h5ad(path, backed="r")

    metadata = pd.DataFrame(
        {
            "dataset_id": "toy",
            "source_dataset": "toy-source",
            "perturbation_label": ["control", "A", "control", "A"],
            "is_control": [True, False, True, False],
            "ncounts": [3.0, 3.0, 3.0, 4.0],
            "reference_key": ["ref1", "ref1", "ref2", "ref2"],
            "group_key": ["ref1||control", "ref1||A", "ref2||control", "ref2||A"],
            "keep_for_analysis": [True, True, True, True],
        }
    )
    group_info = pd.DataFrame(
        {
            "group_key": ["ref1||control", "ref1||A", "ref2||control", "ref2||A"],
            "reference_key": ["ref1", "ref1", "ref2", "ref2"],
            "perturbation_label": ["control", "A", "control", "A"],
            "is_control": [True, False, True, False],
        }
    )
    config = SimpleNamespace(
        dataset_id="toy",
        source_dataset="toy-source",
        control_values=["control"],
        signature_target_sum=100.0,
        delta_reference_fields=[],
        var_name_column=None,
        qc={"gene_chunk_size": 2},
    )

    try:
        pseudobulk, signatures = compute_profiles(backed, metadata, group_info, config)
    finally:
        backed.file.close()

    assert pseudobulk.shape == (4, 10)
    assert signatures.shape == (4, 11)
    assert signatures["signature_id"].is_unique
    gene_columns = ["g1", "g2", "g3"]
    controls = signatures[signatures["is_control"]]
    assert np.allclose(controls[gene_columns].to_numpy(), 0.0, atol=1e-6)
    assert set(signatures["reference_key"]) == {"ref1", "ref2"}
