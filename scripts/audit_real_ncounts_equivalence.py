"""Audit the CSC normalization denominator on sampled real cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


DEFAULT_DATASETS = (
    "TianKampmann2021_CRISPRa",
    "TianKampmann2021_CRISPRi",
    "FrangiehIzar2021_RNA",
)


def audit_dataset(path: Path, sample_count: int, seed: int) -> dict[str, Any]:
    data = ad.read_h5ad(path, backed="r")
    try:
        if "ncounts" not in data.obs:
            raise ValueError(f"ncounts is missing from {path}")
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(data.n_obs, size=min(sample_count, data.n_obs), replace=False))
        metadata_counts = pd.to_numeric(data.obs["ncounts"].iloc[indices], errors="coerce").to_numpy(dtype=float)
        matrix_counts = np.asarray(data.X[indices].sum(axis=1)).ravel().astype(float, copy=False)
    finally:
        data.file.close()

    denominator = np.maximum(np.abs(matrix_counts), 1e-12)
    relative_error = np.abs(metadata_counts - matrix_counts) / denominator
    return {
        "dataset_id": path.stem,
        "path": path.as_posix(),
        "sample_count": int(len(indices)),
        "seed": int(seed),
        "max_relative_error": float(np.max(relative_error)),
        "mean_relative_error": float(np.mean(relative_error)),
        "ncounts_all_finite": bool(np.isfinite(metadata_counts).all()),
        "matrix_row_sums_all_finite": bool(np.isfinite(matrix_counts).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/scperturb_v1.4"))
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output", type=Path, default=Path("artifacts/manifests/csc_ncounts_equivalence.json"))
    args = parser.parse_args()
    if args.sample_count <= 0:
        raise ValueError("sample-count must be positive")

    rows = [audit_dataset(args.raw_dir / f"{dataset}.h5ad", args.sample_count, args.seed) for dataset in DEFAULT_DATASETS]
    summary = {
        "schema_version": 1,
        "normalization_field": "obs.ncounts",
        "comparison": "absolute(metadata_ncounts - matrix_row_sum) / max(abs(matrix_row_sum), 1e-12)",
        "datasets": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
