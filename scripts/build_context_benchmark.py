"""Materialize the formal PTL context registry and shared evaluation panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ptl.data.ids import environment_id as stable_environment_id


REGISTRY_COLUMNS = [
    "environment_id",
    "environment_key",
    "dataset_id",
    "cell_context",
    "perturbation_modality",
    "readout_modality",
    "platform",
    "condition",
    "condition_field",
    "role",
    "semantic_status",
    "status",
]


def load_spec(path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema_version") != 2:
        raise ValueError(f"Expected schema_version: 2 in {path}")
    environments = spec.get("environments")
    if not isinstance(environments, list) or len(environments) < 2:
        raise ValueError("The formal context benchmark must contain at least two environments")
    ids = [str(row.get("environment_id", "")) for row in environments]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Formal environment IDs must be present and unique")
    for row in environments:
        for field in ("dataset_id", "cell_context", "perturbation_modality", "readout_modality", "platform", "condition"):
            if not str(row.get(field, "")).strip():
                raise ValueError(f"Formal environment {row.get('environment_id')} is missing {field}")
        if "metadata_pending" in str(row.get("perturbation_modality", "")).lower():
            raise ValueError(f"Formal environment {row['environment_id']} has unresolved perturbation modality")
        if str(row.get("platform", "")).lower() == "processed_signature_table":
            raise ValueError(f"Formal environment {row['environment_id']} uses a storage format as platform")
        unresolved = " ".join(str(row.get(field, "")) for field in ("role", "semantic_status", "status")).lower()
        if any(token in unresolved for token in ("pending", "candidate", "unresolved")):
            raise ValueError(f"Formal environment {row['environment_id']} is unresolved or only a candidate")
    return spec


def feature_names(path: Path, gene_id_field: str) -> list[str]:
    frame = pd.read_parquet(path)
    if gene_id_field == "feature_name" and gene_id_field in frame.columns:
        values = frame[gene_id_field].astype(str).str.strip().tolist()
    else:
        values = frame.index.astype(str).str.strip().tolist()
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError(f"Feature metadata has empty or duplicate normalized IDs: {path}")
    return values


def materialize(root: Path, spec_path: Path, registry_path: Path, gene_space_path: Path, summary_path: Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    environments = spec["environments"]
    evaluation = spec["evaluation"]
    dataset_to_genes: dict[str, set[str]] = {}
    dataset_to_source: dict[str, str] = {}
    for row in environments:
        dataset_id = str(row["dataset_id"])
        source = root / "data" / "processed" / f"{dataset_id}_feature_metadata.parquet"
        if dataset_id not in dataset_to_genes:
            genes = feature_names(source, str(evaluation.get("gene_id_field", "feature_name")))
            dataset_to_genes[dataset_id] = set(genes)
            dataset_to_source[dataset_id] = source.relative_to(root).as_posix()
    common_genes = sorted(set.intersection(*dataset_to_genes.values()))
    if not common_genes:
        raise ValueError("Formal environment feature-universe intersection is empty")

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        for row in environments:
            registry_row = {column: row.get(column, "") for column in REGISTRY_COLUMNS}
            registry_row["environment_key"] = row["environment_id"]
            registry_row["environment_id"] = stable_environment_id(
                row["dataset_id"], row["cell_context"], row["readout_modality"], row["condition"]
            )
            writer.writerow(registry_row)

    gene_space_path.parent.mkdir(parents=True, exist_ok=True)
    gene_space_digest = hashlib.sha256("\n".join(common_genes).encode("utf-8")).hexdigest()
    with gene_space_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["panel_id", "gene_index", "gene_symbol", "source_gene_universe", "source_sha256"],
        )
        writer.writeheader()
        for index, gene in enumerate(common_genes):
            writer.writerow(
                {
                    "panel_id": "ptl_context_v2_intersection",
                    "gene_index": index,
                    "gene_symbol": gene,
                    "source_gene_universe": "context_benchmark_v2",
                    "source_sha256": gene_space_digest,
                }
            )

    summary = {
        "schema_version": 2,
        "benchmark_id": spec["benchmark_id"],
        "environment_count": len(environments),
        "environment_keys": [row["environment_id"] for row in environments],
        "environment_ids": [
            stable_environment_id(row["dataset_id"], row["cell_context"], row["readout_modality"], row["condition"])
            for row in environments
        ],
        "canonical_environment_ids": [
            stable_environment_id(row["dataset_id"], row["cell_context"], row["readout_modality"], row["condition"])
            for row in environments
        ],
        "dataset_count": len(dataset_to_genes),
        "dataset_feature_counts": {key: len(value) for key, value in sorted(dataset_to_genes.items())},
        "shared_gene_count": len(common_genes),
        "panel_id": "ptl_context_v2_intersection",
        "gene_id_normalization": evaluation.get("gene_id_normalization"),
        "panel_policy": evaluation.get("panel_policy"),
        "predictor_native_space_separate": bool(evaluation.get("predictor_native_space_separate", True)),
        "source_feature_metadata": dataset_to_source,
        "registry_path": registry_path.relative_to(root).as_posix(),
        "gene_space_path": gene_space_path.relative_to(root).as_posix(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--gene-space", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    config = (args.config or root / "configs/context_benchmark.yaml").resolve()
    registry = (args.registry or root / "artifacts/manifests/environment_registry.csv").resolve()
    gene_space = (args.gene_space or root / "artifacts/manifests/evaluation_gene_space.csv").resolve()
    summary = (args.summary or root / "artifacts/manifests/evaluation_gene_space.json").resolve()
    result = materialize(root, config, registry, gene_space, summary)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
