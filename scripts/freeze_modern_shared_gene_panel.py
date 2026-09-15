"""Freeze the common modern-predictor evaluation panel from source controls.

The panel is selected without perturbation outcomes: only QC-passed Frangieh
control cells from the three source environments and the existing 8,229-gene
evaluation universe are used.  The resulting list is a contract artifact for
future strong-linear/GEARS/scGPT/TxPert comparisons; it does not itself claim
that every predictor has been run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


PANEL_SIZE = 1536
RAW_REL = Path("data/raw/scperturb_v1.4/FrangiehIzar2021_RNA.h5ad")
META_REL = Path("data/processed/FrangiehIzar2021_RNA_cell_metadata.parquet")
FEATURE_REL = Path("data/processed/FrangiehIzar2021_RNA_feature_metadata.parquet")
PRED_REL = Path("artifacts/source_data/frangieh_source_frozen_predictions.npz")
ENVIRONMENTS = ("Control", "Co-culture", "IFNγ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    args = parser.parse_args()
    root = args.root.resolve()
    payload = np.load(root / PRED_REL, allow_pickle=False)
    evaluation_genes = payload["evaluation_gene_symbols"].astype(str)
    metadata = pd.read_parquet(root / META_REL, columns=["row_index", "passes_qc", "is_control", "perturbation_2", "ncounts"])
    metadata["ncounts"] = pd.to_numeric(metadata["ncounts"], errors="raise").clip(lower=1.0).astype(np.float64)
    metadata_by_row = metadata.set_index("row_index")
    environment = metadata["perturbation_2"].astype(str)
    keep = metadata["passes_qc"].astype(bool) & metadata["is_control"].astype(bool) & environment.isin(ENVIRONMENTS)
    control_rows = metadata.loc[keep, "row_index"].to_numpy(dtype=np.int64)
    if len(control_rows) < 100:
        raise ValueError("too few source control rows for the common panel")
    feature = pd.read_parquet(root / FEATURE_REL, columns=["feature_name", "feature_order"])
    feature["feature_name"] = feature["feature_name"].astype(str)
    feature = feature.drop_duplicates("feature_name", keep="first").set_index("feature_name")
    missing = sorted(set(evaluation_genes).difference(feature.index))
    if missing:
        raise ValueError(f"evaluation panel genes missing from raw matrix: {missing[:5]}")
    positions = feature.loc[evaluation_genes, "feature_order"].to_numpy(dtype=np.int64)
    with h5py.File(root / RAW_REL, "r") as handle:
        x_group = handle["X"]
        shape = tuple(int(value) for value in x_group.attrs["shape"])
        if str(x_group.attrs.get("encoding-type", "")).lower() != "csc_matrix":
            raise ValueError("expected CSC raw matrix")
        data_ds, indices_ds, indptr_ds = x_group["data"], x_group["indices"], x_group["indptr"]
        sums = np.zeros(len(evaluation_genes), dtype=np.float64)
        squares = np.zeros(len(evaluation_genes), dtype=np.float64)
        order = np.argsort(positions)
        sorted_positions = positions[order]
        for start in range(0, shape[1], 256):
            end = min(start + 256, shape[1])
            mask = (sorted_positions >= start) & (sorted_positions < end)
            if not mask.any():
                continue
            data_start = int(indptr_ds[start])
            data_end = int(indptr_ds[end])
            indptr = np.asarray(indptr_ds[start:end + 1], dtype=np.int64) - data_start
            data = np.asarray(data_ds[data_start:data_end], dtype=np.float32)
            indices = np.asarray(indices_ds[data_start:data_end], dtype=np.int32)
            block = sparse.csc_matrix((data, indices, indptr), shape=(shape[0], end - start))
            local = (sorted_positions[mask] - start).astype(np.int64)
            values = block[control_rows][:, local].tocsr().astype(np.float64)
            scale = 10000.0 / metadata_by_row.loc[control_rows, "ncounts"].to_numpy(dtype=np.float64)
            values = values.multiply(scale[:, None]).tocsr()
            values.data = np.log1p(values.data)
            sums[order[mask]] = np.asarray(values.sum(axis=0)).ravel()
            squares[order[mask]] = np.asarray(values.multiply(values).sum(axis=0)).ravel()
    n = float(len(control_rows))
    variance = np.maximum(squares / n - (sums / n) ** 2, 0.0)
    ranked = sorted(range(len(evaluation_genes)), key=lambda index: (-variance[index], evaluation_genes[index]))
    selected = evaluation_genes[np.asarray(ranked[: int(args.panel_size)], dtype=np.int64)]
    output_dir = root / "artifacts" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / "modern_shared_gene_panel__source_control_1536.txt"
    panel_path.write_text("\n".join(selected.tolist()) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "panel_id": "modern_shared_gene_panel__source_control_1536",
        "selection": "top variance within existing 8,229 evaluation genes using QC-passed control cells only",
        "source_environments": list(ENVIRONMENTS),
        "n_control_cells": int(len(control_rows)),
        "raw_shape": list(shape),
        "n_evaluation_genes": int(len(evaluation_genes)),
        "n_selected_genes": int(len(selected)),
        "panel_path": str(panel_path),
        "panel_sha256": _sha256(panel_path),
        "scientific_result_claimed": False,
        "note": "Contract panel only; model-specific checkpoint, source-only training, and frozen prediction vectors remain separate gates.",
    }
    (output_dir / "modern_shared_gene_panel__source_control_1536.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
