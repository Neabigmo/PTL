from pathlib import Path

import numpy as np
import pytest

from ptl.data.contracts import validate_feature_columns
from ptl.data.ids import biological_instance_id, environment_id, prediction_id
from ptl.data.path_resolver import WorkspacePathResolver
from ptl.uncertainty.uq import confidence_from_uq, summarize_ensemble


def test_predictors_share_biological_instance_but_not_prediction_id():
    env = environment_id("NormanWeissman2019_filtered", "K562", "RNA", "baseline")
    bio = biological_instance_id(env, "FOXA3_FOXL2")
    left = prediction_id(bio, "ridge", "split0", "seed0")
    right = prediction_id(bio, "mean_matching", "split0", "seed0")
    assert left != right
    assert bio in left and bio in right


def test_prediction_ids_are_unique_at_signature_level():
    environment = "env__norman__k562__rna__baseline"
    signatures = [biological_instance_id(environment, value) for value in ["sig_a", "sig_b"]]
    ids = [prediction_id(value, "ridge", "dataset_heldout_split", "ensemble3") for value in signatures]
    assert len(set(ids)) == len(ids)


def test_outcomes_cannot_enter_deployment_features():
    with pytest.raises(ValueError, match="outcome"):
        validate_feature_columns(["support_cells", "fidelity_delta_cosine"])
    validate_feature_columns(["support_cells", "perturbation_novelty", "cell_context_code"])


def test_path_resolver_rejects_legacy_absolute_path(tmp_path: Path):
    resolver = WorkspacePathResolver(tmp_path)
    assert resolver.resolve("configs/datasets.yaml") == tmp_path / "configs/datasets.yaml"
    with pytest.raises(ValueError, match="legacy absolute path"):
        resolver.resolve(r"H:\2026try\4.24\data\processed\old.parquet")


def test_ensemble_uq_is_target_free_and_normalizable():
    rng = np.random.default_rng(4)
    members = rng.normal(size=(3, 8, 5))
    summary = summarize_ensemble(members)
    assert summary.mean_gene_variance.shape == (8,)
    confidence = confidence_from_uq(summary.mean_gene_variance, summary.mean_gene_variance)
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))
