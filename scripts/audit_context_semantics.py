"""Audit raw biological-context fields for the planned context expansion.

Only H5AD observation metadata and categorical codes are read. The expression
matrix is never loaded. The output records the raw values supporting the
environment registry semantics for Tian CRISPRa/CRISPRi and Frangieh RNA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "TianKampmann2021_CRISPRa": {
        "filename": "TianKampmann2021_CRISPRa.h5ad",
        "cell_context": "iPSC-induced_neuron",
        "perturbation_modality": "CRISPRa",
        "condition_field": "",
        "condition_values": [],
        "semantic_status": "raw_celltype_verified",
    },
    "TianKampmann2021_CRISPRi": {
        "filename": "TianKampmann2021_CRISPRi.h5ad",
        "cell_context": "iPSC-induced_neuron",
        "perturbation_modality": "CRISPRi",
        "condition_field": "",
        "condition_values": [],
        "semantic_status": "raw_celltype_verified",
    },
    "FrangiehIzar2021_RNA": {
        "filename": "FrangiehIzar2021_RNA.h5ad",
        "cell_context": "patient-derived_melanoma",
        "perturbation_modality": "CRISPR",
        "condition_field": "perturbation_2",
        "condition_values": ["Control", "Co-culture", "IFNγ"],
        "semantic_status": "raw_condition_field_verified",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "scperturb_v1.4"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "manifests" / "context_semantics_audit.csv"))
    parser.add_argument("--details", default=str(ROOT / "artifacts" / "manifests" / "context_semantics_audit.json"))
    return parser.parse_args()


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    return "" if value is None else str(value)


def category_counts(node: Any) -> dict[str, int]:
    if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
        categories = [decode(value) for value in node["categories"][:]]
        codes = np.asarray(node["codes"])
        counts = np.bincount(codes[codes >= 0], minlength=len(categories))
        return {category: int(count) for category, count in zip(categories, counts) if count}
    values = [decode(value) for value in node[:]]
    counts = pd.Series(values, dtype="string").value_counts(dropna=False)
    return {str(index): int(value) for index, value in counts.items()}


def matrix_shape(node: Any) -> tuple[int, int]:
    shape = getattr(node, "shape", None)
    if shape is None:
        shape = node.attrs.get("shape")
    if shape is None or len(shape) != 2:
        raise ValueError("H5AD X has no two-dimensional shape")
    return int(shape[0]), int(shape[1])


def audit_target(raw_dir: Path, dataset_id: str, target: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = raw_dir / target["filename"]
    with h5py.File(path, "r") as handle:
        n_cells, n_genes = matrix_shape(handle["X"])
        obs = handle["obs"]
        values = {field: category_counts(obs[field]) for field in [
            "celltype", "tissue_type", "disease", "perturbation_type", target["condition_field"]
        ] if field and field in obs}
    row = {
        "dataset_id": dataset_id,
        "raw_file": str(path),
        "n_cells": n_cells,
        "n_genes": n_genes,
        "cell_context": target["cell_context"],
        "perturbation_modality": target["perturbation_modality"],
        "condition_field": target["condition_field"],
        "condition_values": "|".join(target["condition_values"]),
        "semantic_status": target["semantic_status"],
        "celltype_values": "|".join(values.get("celltype", {})),
        "tissue_type_values": "|".join(values.get("tissue_type", {})),
        "disease_values": "|".join(values.get("disease", {})),
        "perturbation_type_values": "|".join(values.get("perturbation_type", {})),
    }
    details = {
        "dataset_id": dataset_id,
        "raw_file": str(path),
        "n_cells": n_cells,
        "n_genes": n_genes,
        "proposed_semantics": target,
        "raw_observation_values": values,
    }
    return row, details


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    rows = []
    details = []
    for dataset_id, target in TARGETS.items():
        row, detail = audit_target(raw_dir, dataset_id, target)
        rows.append(row)
        details.append(detail)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    details_path = Path(args.details)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps({"targets": details}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} context semantic audits to {output} and {details_path}.")


if __name__ == "__main__":
    main()
