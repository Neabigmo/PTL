from __future__ import annotations

from pathlib import Path

import anndata as ad

from src.data.preprocess_dataset import load_config, resolve_cell_context, resolve_perturbation_label, resolve_obs_series


NEW_PHASE11_CONFIGS = [
    "adamson_weissman_2016_10x005.yaml",
    "datlinger_bock_2017.yaml",
    "datlinger_bock_2021.yaml",
    "dixit_regev_2016_k562_tfs_7_days.yaml",
    "papalexi_satija_2021_eccite_rna.yaml",
    "replogle_weissman_2022_rpe1.yaml",
]


def test_new_preprocessing_configs_resolve_required_obs_fields() -> None:
    config_dir = Path("config/preprocessing")
    for name in NEW_PHASE11_CONFIGS:
        config = load_config(config_dir / name)
        assert config.input_path.exists(), f"Raw h5ad missing for {config.dataset_id}"
        adata = ad.read_h5ad(config.input_path, backed="r")
        obs = adata.obs.head(256).copy()
        perturbation = resolve_perturbation_label(obs, config)
        context = resolve_cell_context(obs, config)
        batch = resolve_obs_series(obs, config.obs_fields.get("batch"), default_value="batch0")
        assert perturbation.notna().all(), config.dataset_id
        assert context.notna().all(), config.dataset_id
        assert batch.notna().all(), config.dataset_id
        assert perturbation.astype(str).str.len().gt(0).all(), config.dataset_id
        adata.file.close()
