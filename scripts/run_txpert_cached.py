"""Run the official TxPert inference code against an already cached bundle.

The upstream entrypoint first calls Zenodo even when its cache is complete.
This wrapper avoids that network call, applies only a memory-safe no-op to the
upstream same-width AnnData copy, and then invokes the unmodified Hydra model
construction/inference path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config-name", default="config-gat")
    args = parser.parse_args()
    root = args.root.resolve()
    txpert_root = root / "third_party" / "TxPert"
    sys.path.insert(0, str(txpert_root))
    os.chdir(txpert_root)

    import main as upstream_main
    import gspp.data.datamodule as datamodule_module
    from lightning import Trainer as LightningTrainer

    upstream_main.download_files_from_zenodo = lambda *unused_args, **unused_kwargs: None
    upstream_main.Trainer = lambda **kwargs: LightningTrainer(
        enable_progress_bar=False,
        logger=False,
        enable_model_summary=False,
        **kwargs,
    )

    def memory_safe_dimensions(self) -> None:
        """Preserve the upstream dimensions but avoid copying an equal-width view."""

        self.cell_types = np.unique(self.adata.obs[datamodule_module.cs.CELL_TYPE])
        self.number_of_cell_types = len(self.cell_types)
        dim_adjustment = 0
        self.adata_output_dim = self.adata.X.shape[1] - dim_adjustment
        if self.obsm_key != datamodule_module.cs.ObsmKey.RAW:
            self.input_dim = self.adata.obsm[self.obsm_key].shape[1]
            self.output_dim = self.adata.obsm[self.obsm_key].shape[1] - dim_adjustment
        else:
            self.input_dim = self.adata.X.shape[1]
            self.output_dim = self.adata.X.shape[1] - dim_adjustment
        datamodule_module.logger.info(f"adata_output_dim :  {self.adata_output_dim}")
        if self.adata.n_vars != self.adata_output_dim:
            self.adata = self.adata[:, : self.adata_output_dim].copy()

    datamodule_module.PertDataModule._set_input_output_dimensions = memory_safe_dimensions
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3", config_dir=str(txpert_root / "configs")):
        config = compose(config_name=args.config_name)
    upstream_main.infer(config)
    print(f"txpert_cached_inference_complete config={args.config_name}")


if __name__ == "__main__":
    main()
