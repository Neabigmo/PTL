"""Build split-half reproducibility audits from independent reference groups."""

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

from scripts.run_formal_v2_predictors import METADATA_COLUMNS  # noqa: E402
from src.ptl.evaluation.reproducibility import cosine, split_half_indices, weighted_profile  # noqa: E402


def run(root: Path, output_path: Path, summary_path: Path, *, seed: int = 20260907) -> dict[str, Any]:
    registry = pd.read_csv(root / "artifacts/manifests/environment_registry.csv")
    rows: list[dict[str, Any]] = []
    for _, environment in registry.iterrows():
        dataset_id = str(environment["dataset_id"])
        source = root / "data/processed" / f"{dataset_id}_delta_signatures.parquet"
        if not source.is_file():
            continue
        signature = pd.read_parquet(source)
        if "is_control" in signature:
            signature = signature.loc[~signature["is_control"].astype(bool)].copy()
        condition_field = str(environment.get("condition_field", "") or "").strip()
        condition = str(environment.get("condition", "") or "").strip()
        if condition_field and condition_field in signature:
            signature = signature.loc[signature[condition_field].astype(str).eq(condition)].copy()
        non_gene_columns = set(METADATA_COLUMNS) | {
            "source_dataset", "signature_id", "reference_key", "control_label_used",
            "batch", "cell_line", "cell_type", "perturbation_2", "replicate", "time", "n_cells",
        }
        gene_columns = [column for column in signature.columns if column not in non_gene_columns]
        if not gene_columns or "reference_key" not in signature or "perturbation_label" not in signature:
            continue
        for perturbation, group in signature.groupby("perturbation_label", sort=True):
            references = sorted(group["reference_key"].astype(str).unique())
            if len(references) < 2:
                continue
            left, right = split_half_indices(len(references), seed=seed + sum(map(ord, str(perturbation))))
            reference_to_index = {reference: index for index, reference in enumerate(references)}
            values = group[gene_columns].to_numpy(dtype=np.float64)
            weights = pd.to_numeric(group["n_cells"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
            reference_values = []
            reference_weights = []
            for reference in references:
                mask = group["reference_key"].astype(str).eq(reference).to_numpy()
                reference_values.append(weighted_profile(values[mask], weights[mask]))
                reference_weights.append(float(weights[mask].sum()))
            left_profile = weighted_profile(np.asarray(reference_values)[left], np.asarray(reference_weights)[left])
            right_profile = weighted_profile(np.asarray(reference_values)[right], np.asarray(reference_weights)[right])
            rows.append({
                "environment_id": environment["environment_id"],
                "environment_key": environment["environment_key"],
                "dataset_id": dataset_id,
                "perturbation_label": perturbation,
                "n_reference_groups": len(references),
                "left_reference_groups": len(left),
                "right_reference_groups": len(right),
                "split_half_cosine": cosine(left_profile, right_profile),
                "gene_count": len(gene_columns),
                "split_seed": seed,
                "reproducibility_unit": "independent_reference_groups",
                "used_in_ptl_features": 0,
                "raw_cell_expression_split": 0,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no perturbations have at least two independent reference groups")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["environment_id", "perturbation_label"], kind="stable").to_csv(output_path, index=False)
    summary = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "split_seed": seed,
        "unit": "independent_reference_groups",
        "raw_cell_expression_split": False,
        "raw_cell_expression_note": "This audit does not enter PTL features; raw-cell split requires an expression reader and is not inferred from metadata.",
        "n_rows": int(len(frame)),
        "environment_count": int(frame["environment_id"].nunique()),
        "median_split_half_cosine": float(frame["split_half_cosine"].median()),
        "output_path": output_path.relative_to(root).as_posix(),
        "status": "reproducibility_audit_executed_reference_group_split",
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
    args = parser.parse_args()
    root = args.root.resolve()
    result = run(
        root,
        (args.output or root / "artifacts/manifests/formal_v2_reproducibility.csv").resolve(),
        (args.summary or root / "artifacts/manifests/formal_v2_reproducibility.json").resolve(),
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
