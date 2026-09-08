"""Compute a raw-cell, within-reference split-half reproducibility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess_dataset import (  # noqa: E402
    build_contract_mask,
    build_group_metadata,
    build_qc_mask,
    build_standardized_metadata,
    compute_profiles,
    get_var_frame,
    load_adata,
    load_config,
)
from src.ptl.evaluation.reproducibility import cosine, weighted_profile  # noqa: E402


DEFAULT_CONFIGS = (
    "config/preprocessing/norman_weissman_2019.yaml",
    "config/preprocessing/replogle_weissman_2022_rpe1.yaml",
)


def _assign_split_half(metadata: pd.DataFrame, reference_fields: list[str], seed: int) -> pd.Series:
    """Assign cells within each reference/perturbation stratum deterministically."""

    result = pd.Series(-1, index=metadata.index, dtype=np.int8)
    group_columns = [field for field in reference_fields if field in metadata.columns] + ["perturbation_label"]
    rng = np.random.default_rng(seed)
    for _, group in metadata.loc[metadata["keep_for_analysis"]].groupby(group_columns, sort=True, dropna=False):
        indices = group.index.to_numpy()
        order = rng.permutation(len(indices))
        midpoint = len(indices) // 2
        if len(indices) > 1:
            midpoint = max(1, min(len(indices) - 1, midpoint))
        result.loc[indices[order[:midpoint]]] = 0
        result.loc[indices[order[midpoint:]]] = 1
    return result


def _paired_rows(delta: pd.DataFrame, config: Any, *, seed: int) -> list[dict[str, Any]]:
    if delta.empty:
        return []
    metadata_columns = {
        "dataset_id", "source_dataset", "signature_id", "reference_key", "perturbation_label", "is_control",
        "n_cells", "control_label_used", "split_half", *config.delta_reference_fields,
    }
    gene_columns = [column for column in delta.columns if column not in metadata_columns]
    if not gene_columns or "split_half" not in delta.columns:
        return []
    rows: list[dict[str, Any]] = []
    non_control = delta.loc[~delta["is_control"].astype(bool)].copy()
    for perturbation, group in non_control.groupby("perturbation_label", sort=True):
        profiles: dict[int, tuple[np.ndarray, int, int]] = {}
        for half, half_group in group.groupby("split_half", sort=True):
            half = int(half)
            values = half_group[gene_columns].to_numpy(dtype=np.float64)
            weights = pd.to_numeric(half_group["n_cells"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            if len(values) == 0 or float(weights.sum()) <= 0:
                continue
            profiles[half] = (weighted_profile(values, weights), int(len(half_group)), int(weights.sum()))
        if 0 not in profiles or 1 not in profiles:
            continue
        left, left_groups, left_cells = profiles[0]
        right, right_groups, right_cells = profiles[1]
        rows.append({
            "dataset_id": config.dataset_id,
            "perturbation_label": str(perturbation),
            "split_half_cosine": cosine(left, right),
            "left_signature_groups": left_groups,
            "right_signature_groups": right_groups,
            "left_cells": left_cells,
            "right_cells": right_cells,
            "gene_count": len(gene_columns),
            "split_seed": seed,
            "reproducibility_unit": "raw_cells_within_reference_perturbation_strata",
            "raw_cell_expression_split": 1,
        })
    return rows


def run(root: Path, output_path: Path, summary_path: Path, *, seed: int = 20260907, config_paths: list[str] | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    selected_paths = config_paths or list(DEFAULT_CONFIGS)
    for relative_path in selected_paths:
        config_path = (root / relative_path).resolve()
        config = load_config(config_path)
        adata = load_adata(config)
        try:
            metadata = build_standardized_metadata(adata, config)
            metadata["passes_contract_filter"] = build_contract_mask(metadata, config)
            metadata["passes_qc"] = build_qc_mask(metadata, config.qc)
            metadata["keep_for_analysis"] = metadata["passes_contract_filter"] & metadata["passes_qc"]
            base_reference_fields = [field for field in config.delta_reference_fields if field in metadata.columns]
            if not base_reference_fields:
                base_reference_fields = ["dataset_id"]
            metadata["split_half"] = _assign_split_half(metadata, base_reference_fields, seed + sum(map(ord, config.dataset_id)))
            raw_config = config
            raw_config.delta_reference_fields = [*base_reference_fields, "split_half"]
            kept = metadata.loc[metadata["keep_for_analysis"]].copy()
            kept["reference_key"] = kept[raw_config.delta_reference_fields].astype(str).agg("||".join, axis=1)
            kept["group_key"] = kept[raw_config.delta_reference_fields + ["perturbation_label"]].astype(str).agg("||".join, axis=1)
            metadata.loc[kept.index, "reference_key"] = kept["reference_key"]
            metadata.loc[kept.index, "group_key"] = kept["group_key"]
            group_info, eligible_mask = build_group_metadata(metadata, raw_config)
            metadata["keep_for_analysis"] = eligible_mask
            _, delta = compute_profiles(adata, metadata, group_info, raw_config)
            dataset_rows = _paired_rows(delta, raw_config, seed=seed)
            rows.extend(dataset_rows)
            audit.append({
                "dataset_id": config.dataset_id,
                "config_path": config_path.relative_to(root).as_posix(),
                "n_input_cells": int(adata.n_obs),
                "n_qc_contract_cells": int((metadata["passes_contract_filter"] & metadata["passes_qc"]).sum()),
                "n_split_cells": int(metadata["keep_for_analysis"].sum()),
                "n_delta_rows": int(len(delta)),
                "n_paired_perturbations": int(len(dataset_rows)),
                "status": "executed",
            })
        finally:
            close = getattr(adata, "close", None)
            if close is not None:
                close()
    frame = pd.DataFrame(rows)
    if frame.empty or frame["dataset_id"].nunique() < 2:
        raise ValueError("raw-cell split-half audit needs at least two datasets with paired perturbations")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["dataset_id", "perturbation_label"], kind="stable").to_csv(output_path, index=False)
    summary = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "split_seed": seed,
        "unit": "raw_cells_within_reference_perturbation_strata",
        "raw_cell_expression_split": True,
        "n_rows": int(len(frame)),
        "dataset_count": int(frame["dataset_id"].nunique()),
        "dataset_audit": audit,
        "median_split_half_cosine": float(frame["split_half_cosine"].median()),
        "output_path": output_path.relative_to(root).as_posix(),
        "status": "raw_cell_expression_split_half_executed",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--config", action="append", dest="config_paths", default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    result = run(
        root,
        (args.output or root / "artifacts/manifests/formal_v2_raw_cell_split_half.csv").resolve(),
        (args.summary or root / "artifacts/manifests/formal_v2_raw_cell_split_half.json").resolve(),
        seed=args.seed,
        config_paths=args.config_paths,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
