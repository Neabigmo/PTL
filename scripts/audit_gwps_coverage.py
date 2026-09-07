"""Audit GWPS coverage loss across the local preprocessing stages.

The audit reads H5AD observation metadata with h5py and reuses the staged
metadata/signature artifacts. It never loads the expression matrix ``X`` and
does not modify any processed data. Its purpose is to identify whether the
small final perturbation surface comes from label parsing, cell QC, minimum
cell eligibility, or reference matching.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = "ReplogleWeissman2022_K562_gwps"
DEFAULT_RAW = ROOT / "data" / "raw" / "scperturb_v1.4" / f"{DATASET}.h5ad"
DEFAULT_METADATA = ROOT / "data" / "processed" / f"{DATASET}_cell_metadata.parquet"
DEFAULT_SIGNATURES = ROOT / "data" / "processed" / f"{DATASET}_delta_signatures.parquet"
DEFAULT_SUMMARY = ROOT / "results" / "tables" / "preprocessing_summary.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "manifests" / "gwps_coverage_audit.csv"
DEFAULT_DETAILS = ROOT / "artifacts" / "manifests" / "gwps_coverage_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--signatures", default=str(DEFAULT_SIGNATURES))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--details", default=str(DEFAULT_DETAILS))
    parser.add_argument("--min-genes", type=int, default=700)
    parser.add_argument("--min-counts", type=float, default=1500)
    parser.add_argument("--max-percent-mito", type=float, default=25)
    parser.add_argument("--min-cells-per-perturbation", type=int, default=10)
    return parser.parse_args()


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    return "" if value is None else str(value)


def json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def iter_h5_column(node: Any, chunk_size: int = 8192) -> Iterable[str]:
    if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
        categories = [decode(value) for value in node["categories"][:]]
        codes = node["codes"]
        for start in range(0, int(codes.shape[0]), chunk_size):
            for code in codes[start : start + chunk_size]:
                index = int(code)
                yield categories[index] if 0 <= index < len(categories) else ""
        return
    size = int(node.shape[0])
    for start in range(0, size, chunk_size):
        for value in node[start : start + chunk_size]:
            yield decode(value)


def raw_metadata_summary(raw_path: Path) -> dict[str, Any]:
    with h5py.File(raw_path, "r") as handle:
        n_obs = int(handle["X"].shape[0])
        obs = handle["obs"]
        labels = Counter(value.strip() for value in iter_h5_column(obs["perturbation"]))
        type_counts = Counter(value.strip() for value in iter_h5_column(obs["perturbation_type"]))
        return {
            "n_cells": n_obs,
            "n_perturbations": sum(1 for value in labels if value and value.casefold() != "control"),
            "n_control_cells": int(labels.get("control", 0)),
            "n_unique_labels": len(labels),
            "label_counts_top50": dict(labels.most_common(50)),
            "perturbation_type_counts": dict(type_counts),
        }


def stage_row(stage: str, frame: pd.DataFrame, label_column: str = "perturbation_label") -> dict[str, Any]:
    labels = frame[label_column].astype("string")
    controls = frame["is_control"].astype(bool)
    non_control = labels[~controls].dropna()
    return {
        "stage": stage,
        "n_cells": int(len(frame)),
        "n_perturbations": int(non_control.nunique()),
        "n_control_cells": int(controls.sum()),
        "control_fraction": float(controls.mean()) if len(frame) else 0.0,
        "n_unique_labels": int(labels.nunique(dropna=True)),
    }


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw)
    metadata_path = Path(args.metadata)
    signatures_path = Path(args.signatures)
    summary_path = Path(args.summary)
    metadata_columns = [
        "passes_contract_filter",
        "passes_qc",
        "keep_for_analysis",
        "perturbation_label",
        "is_control",
        "batch",
        "reference_key",
        "group_key",
        "ngenes",
        "ncounts",
        "percent_mito",
    ]
    metadata = pd.read_parquet(metadata_path, columns=metadata_columns)
    raw = raw_metadata_summary(raw_path)
    summary = pd.read_csv(summary_path)
    summary_row = summary.loc[summary["dataset_id"].astype(str).eq(DATASET)]
    if summary_row.empty:
        raise KeyError(f"{DATASET} is missing from {summary_path}")
    expected = summary_row.iloc[0]
    if int(expected["n_cells_input"]) != raw["n_cells"] or len(metadata) != raw["n_cells"]:
        raise AssertionError("raw H5AD and staged metadata cell counts disagree")

    stages = [
        {
            **stage_row("raw_obs", pd.DataFrame({
                "perturbation_label": metadata["perturbation_label"],
                "is_control": metadata["is_control"],
            })),
            "filter_rule": "raw obs perturbation metadata",
        },
        {
            **stage_row("label_parsing", metadata),
            "filter_rule": "direct perturbation -> perturbation_label; control == control",
        },
    ]
    contract = metadata[metadata["passes_contract_filter"].astype(bool)]
    stages.append({**stage_row("contract_filter", contract), "filter_rule": "contract_filter_query (empty in GWPS config)"})
    contract_qc = contract.copy()
    ngenes = pd.to_numeric(contract_qc["ngenes"], errors="coerce").fillna(0)
    ncounts = pd.to_numeric(contract_qc["ncounts"], errors="coerce").fillna(0)
    percent_mito = pd.to_numeric(contract_qc["percent_mito"], errors="coerce").fillna(0)
    contract_qc = contract_qc.loc[ngenes >= args.min_genes]
    stages.append(
        {
            **stage_row("qc_min_genes", contract_qc),
            "filter_rule": f"ngenes >= {args.min_genes}",
        }
    )
    ncounts = pd.to_numeric(contract_qc["ncounts"], errors="coerce").fillna(0)
    contract_qc = contract_qc.loc[ncounts >= args.min_counts]
    stages.append(
        {
            **stage_row("qc_min_counts", contract_qc),
            "filter_rule": f"ncounts >= {args.min_counts:g}",
        }
    )
    percent_mito = pd.to_numeric(contract_qc["percent_mito"], errors="coerce").fillna(0)
    contract_qc = contract_qc.loc[percent_mito <= args.max_percent_mito]
    expected_qc = metadata[
        metadata["passes_contract_filter"].astype(bool) & metadata["passes_qc"].astype(bool)
    ]
    if not contract_qc.index.equals(expected_qc.index):
        raise AssertionError("staged QC flags do not match the configured QC thresholds")
    stages.append(
        {
            **stage_row("qc_max_percent_mito", contract_qc),
            "filter_rule": f"percent_mito <= {args.max_percent_mito:g}",
        }
    )
    eligible = metadata[metadata["keep_for_analysis"].astype(bool)]
    stages.append(
        {
            **stage_row("min_cells_per_perturbation", eligible),
            "filter_rule": f"grouped by batch + perturbation; non-control groups >= {args.min_cells_per_perturbation} cells; controls retained",
            "n_reference_groups": int(eligible["group_key"].nunique()),
        }
    )

    signatures = pd.read_parquet(signatures_path)
    if signatures.empty:
        raise AssertionError("GWPS delta-signature artifact is empty")
    signature_controls = signatures["is_control"].astype(bool)
    signature_labels = signatures["perturbation_label"].astype("string")
    stages.append(
        {
            "stage": "reference_matching",
            "n_cells": int(signatures["n_cells"].sum()),
            "n_perturbations": int(signature_labels[~signature_controls].nunique()),
            "n_control_cells": int(signature_controls.sum()),
            "control_fraction": float(signature_controls.mean()),
            "n_unique_labels": int(signature_labels.nunique(dropna=True)),
            "n_signatures": int(len(signatures)),
            "n_pseudobulk_rows": int(expected["n_pseudobulk_rows"]),
            "filter_rule": "reference_key must contain a control profile before delta signature is emitted",
        }
    )
    audit = pd.DataFrame(stages)
    audit.insert(0, "dataset_id", DATASET)
    audit.insert(1, "source_path", str(raw_path))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    qc_non_control = contract_qc.loc[~contract_qc["is_control"].astype(bool)]
    global_counts = qc_non_control.groupby("perturbation_label", dropna=False).size()
    batch_counts = qc_non_control.groupby(["batch", "perturbation_label"], dropna=False).size()
    retained_labels = set(
        eligible.loc[~eligible["is_control"].astype(bool), "perturbation_label"].astype(str)
    )
    batch_labels_at_threshold = {
        str(threshold): int(
            batch_counts[batch_counts >= threshold].reset_index()["perturbation_label"].astype(str).nunique()
        )
        for threshold in (1, 3, 5, 10, 25, 50, 100)
    }
    eligibility_diagnostics = {
        "n_qc_non_control_labels": int(global_counts.size),
        "global_perturbation_labels_at_least_n_cells": {
            str(threshold): int((global_counts >= threshold).sum())
            for threshold in (1, 3, 5, 10, 25, 50, 100)
        },
        "batch_perturbation_groups_at_least_n_cells": {
            str(threshold): int((batch_counts >= threshold).sum())
            for threshold in (1, 3, 5, 10, 25, 50, 100)
        },
        "batch_perturbation_labels_with_at_least_one_eligible_group": batch_labels_at_threshold,
        "global_labels_at_least_10_but_not_retained": int(
            sum(
                count >= args.min_cells_per_perturbation and str(label) not in retained_labels
                for label, count in global_counts.items()
            )
        ),
    }
    details = {
        "dataset_id": DATASET,
        "raw_path": str(raw_path),
        "processed_metadata_path": str(metadata_path),
        "processed_signatures_path": str(signatures_path),
        "qc": {
            "min_genes": args.min_genes,
            "min_counts": args.min_counts,
            "max_percent_mito": args.max_percent_mito,
            "min_cells_per_perturbation": args.min_cells_per_perturbation,
        },
        "raw": raw,
        "raw_qc_quantiles": {
            column: np.percentile(pd.to_numeric(metadata[column], errors="coerce").fillna(0), [0, 1, 5, 50, 95, 99, 100]).tolist()
            for column in ("ngenes", "ncounts", "percent_mito")
        },
        "eligibility_diagnostics": eligibility_diagnostics,
        "preprocessing_summary": {
            key: json_scalar(expected[key])
            for key in [
                "n_cells_input",
                "n_cells_contract",
                "n_cells_kept",
                "n_pseudobulk_rows",
                "n_signature_rows",
                "n_unique_perturbations",
                "n_control_cells",
            ]
        },
        "stages": stages,
        "interpretation": {
            "final_surface_is_small": bool(int(expected["n_unique_perturbations"]) < 100),
            "coverage_loss_is_qc_dominated": bool(
                int(expected["n_cells_kept"]) < int(expected["n_cells_input"]) * 0.1
            ),
            "not_a_headline_genome_wide_benchmark": True,
        },
    }
    details_path = Path(args.details)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(audit)} GWPS coverage stages to {output} and {details_path}.")


if __name__ == "__main__":
    main()
