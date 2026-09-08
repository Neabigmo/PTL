"""Build condition-level ground truth and a frozen biological-instance split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ptl.data.ids import biological_instance_id, environment_id as stable_environment_id


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def load_registry(path: Path) -> pd.DataFrame:
    registry = pd.read_csv(path)
    required = {
        "environment_id",
        "environment_key",
        "dataset_id",
        "condition",
        "condition_field",
    }
    missing = required.difference(registry.columns)
    if missing:
        raise ValueError(f"Environment registry is missing columns: {sorted(missing)}")
    if registry["environment_id"].duplicated().any() or registry["environment_key"].duplicated().any():
        raise ValueError("Environment registry IDs and keys must be unique")
    return registry


def _same_value(left: object, right: object) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return str(left).strip().casefold() == str(right).strip().casefold()


def assign_environment(row: pd.Series, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if len(candidates) == 1:
        return candidates[0]
    fields = {str(candidate.get("condition_field", "")).strip() for candidate in candidates}
    fields.discard("")
    if len(fields) != 1:
        raise ValueError("Multiple environments for one dataset need one condition_field")
    field = fields.pop()
    if field not in row.index:
        raise ValueError(f"Condition field {field} is absent from the processed signature table")
    matches = [candidate for candidate in candidates if _same_value(row[field], candidate["condition"])]
    if len(matches) != 1:
        raise ValueError(f"Could not map condition value {row[field]!r} to one formal environment")
    return matches[0]


def aggregate_dataset(root: Path, records: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    dataset_id = str(records[0]["dataset_id"])
    source = root / "data" / "processed" / f"{dataset_id}_delta_signatures.parquet"
    frame = pd.read_parquet(source)
    required = {"perturbation_label", "is_control", "n_cells"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    frame = frame.loc[~frame["is_control"].astype(bool)].copy()
    labels = frame["perturbation_label"].astype(str).str.strip()
    frame = frame.loc[labels.ne("") & ~labels.str.casefold().isin({"nan", "none", "null"})].copy()
    frame["n_cells"] = pd.to_numeric(frame["n_cells"], errors="coerce")
    frame = frame.loc[frame["n_cells"].gt(0)].copy()
    assignments = [assign_environment(row, records) for _, row in frame.iterrows()]
    frame["environment_id"] = [item["environment_id"] for item in assignments]
    frame["environment_key"] = [item["environment_key"] for item in assignments]
    frame["condition"] = [item["condition"] for item in assignments]
    frame["condition_field"] = [item.get("condition_field", "") for item in assignments]

    metadata_columns = {
        "dataset_id", "source_dataset", "signature_id", "reference_key", "perturbation_label",
        "is_control", "n_cells", "control_label_used", "batch", "perturbation_2",
        "cell_line", "celltype", "cell_context", "target", "guide_id", "perturbation_type",
        "environment_id", "environment_key", "condition", "condition_field", "dose", "timepoint",
    }
    gene_columns = [column for column in frame.columns if column not in metadata_columns]
    if not gene_columns:
        raise ValueError(f"No gene columns found in {source}")

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_values: list[np.ndarray] = []
    optional_grain = [column for column in ("dose", "timepoint") if column in frame.columns]
    group_fields = ["environment_id", "perturbation_label", *optional_grain]
    for group_key, group in frame.groupby(group_fields, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        environment, perturbation = group_key[:2]
        dose = group_key[2] if "dose" in optional_grain else None
        timepoint_index = 2 + ("dose" in optional_grain)
        timepoint = group_key[timepoint_index] if "timepoint" in optional_grain else None
        weights = group["n_cells"].to_numpy(dtype=float)
        values = group[gene_columns].to_numpy(dtype=np.float32, copy=False)
        aggregate_values.append(np.average(values, axis=0, weights=weights).astype(np.float32))
        first = group.iloc[0]
        aggregate_rows.append(
            {
                "biological_instance_id": biological_instance_id(environment, perturbation, dose, timepoint),
                "environment_id": environment,
                "environment_key": first["environment_key"],
                "dataset_id": dataset_id,
                "condition": first["condition"],
                "condition_field": first["condition_field"],
                "perturbation_label": perturbation,
                "dose": "" if dose is None or pd.isna(dose) else str(dose),
                "timepoint": "" if timepoint is None or pd.isna(timepoint) else str(timepoint),
                "n_reference_groups": int(len(group)),
                "total_cells": int(group["n_cells"].sum()),
                "source_reference_keys": ";".join(sorted(group["reference_key"].astype(str).unique())),
                "aggregation_rule": "weighted_mean_of_batch_matched_deltas",
            }
        )

    output = pd.concat(
        [pd.DataFrame(aggregate_rows), pd.DataFrame(np.vstack(aggregate_values), columns=gene_columns)],
        axis=1,
    )
    output_path = output_dir / f"{dataset_id}_condition_ground_truth.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    for row in aggregate_rows:
        row["ground_truth_path"] = output_path.relative_to(root).as_posix()
    return aggregate_rows


def assign_splits(rows: list[dict[str, Any]], split_spec: dict[str, Any], *, split_seed: int | None = None) -> None:
    seed = int(split_spec["split_seed"] if split_seed is None else split_seed)
    fractions = split_spec["split_fractions"]
    if abs(sum(float(value) for value in fractions.values()) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to one")
    frame = pd.DataFrame(rows)
    frame["split"] = ""
    for environment_index, (environment, group) in enumerate(frame.groupby("environment_id", sort=True)):
        order = np.random.default_rng(seed + environment_index).permutation(group.index.to_numpy())
        n = len(order)
        n_train = max(1, int(np.floor(n * float(fractions["train"]))))
        n_validation = max(1, int(np.floor(n * float(fractions["validation"]))))
        if n_train + n_validation >= n:
            n_validation = max(1, n - n_train - 1)
        frame.loc[order[:n_train], "split"] = "train"
        frame.loc[order[n_train:n_train + n_validation], "split"] = "validation"
        frame.loc[order[n_train + n_validation:], "split"] = "test"
    for index, value in frame["split"].items():
        rows[index]["split"] = value
        rows[index]["split_seed"] = seed
        rows[index]["model_seeds"] = ",".join(str(seed_value) for seed_value in split_spec["model_seeds"])


def materialize(
    root: Path,
    benchmark_path: Path,
    registry_path: Path,
    split_path: Path,
    manifest_path: Path,
    summary_path: Path,
    split_seed: int | None = None,
) -> dict[str, Any]:
    benchmark = load_yaml(benchmark_path)
    split_spec = load_yaml(split_path)
    if split_seed is not None:
        split_spec = dict(split_spec)
        split_spec["split_seed"] = int(split_seed)
    registry = load_registry(registry_path)
    benchmark_keys = {str(row["environment_id"]) for row in benchmark["environments"]}
    registry_keys = set(registry["environment_key"].astype(str))
    if benchmark_keys != registry_keys:
        raise ValueError("Benchmark keys and registry environment_key values differ")
    for row in benchmark["environments"]:
        canonical = stable_environment_id(row["dataset_id"], row["cell_context"], row["readout_modality"], row["condition"])
        actual = registry.loc[registry["environment_key"].eq(row["environment_id"]), "environment_id"].iloc[0]
        if canonical != actual:
            raise ValueError(f"Registry canonical ID mismatch for {row['environment_id']}")

    records_by_dataset = {
        dataset: group.to_dict(orient="records")
        for dataset, group in registry.groupby("dataset_id", sort=True)
    }
    rows: list[dict[str, Any]] = []
    ground_truth_paths: dict[str, str] = {}
    for dataset, records in records_by_dataset.items():
        dataset_rows = aggregate_dataset(root, records, root / "data" / "processed")
        rows.extend(dataset_rows)
        ground_truth_paths[dataset] = dataset_rows[0]["ground_truth_path"] if dataset_rows else ""
    assign_splits(rows, split_spec, split_seed=split_seed)

    manifest_columns = [
        "biological_instance_id", "environment_id", "environment_key", "dataset_id", "condition",
        "condition_field", "perturbation_label", "dose", "timepoint", "n_reference_groups", "total_cells",
        "source_reference_keys", "aggregation_rule", "ground_truth_path", "split", "split_seed", "model_seeds",
    ]
    manifest = pd.DataFrame(rows)[manifest_columns].sort_values(
        ["environment_id", "perturbation_label"], kind="stable"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "schema_version": 1,
        "split_id": split_spec["split_id"],
        "benchmark_id": benchmark["benchmark_id"],
        "unit": split_spec["unit"],
        "environment_count": int(manifest["environment_id"].nunique()),
        "biological_instance_count": int(len(manifest)),
        "instances_per_environment": {
            str(key): int(value) for key, value in manifest.groupby("environment_id").size().items()
        },
        "split_counts": {str(key): int(value) for key, value in manifest["split"].value_counts().items()},
        "split_seed": int(split_spec["split_seed"]),
        "model_seeds": [int(value) for value in split_spec["model_seeds"]],
        "aggregation": split_spec["aggregation"],
        "reliability_target": split_spec["reliability_target"],
        "ground_truth_paths": ground_truth_paths,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    selected_seed = args.split_seed
    manifest_default = root / "artifacts/manifests/biological_instance_registry.csv"
    summary_default = root / "artifacts/manifests/biological_instance_registry.json"
    if selected_seed is not None and selected_seed != 20260907:
        manifest_default = root / f"artifacts/manifests/biological_instance_registry__split_{selected_seed}.csv"
        summary_default = root / f"artifacts/manifests/biological_instance_registry__split_{selected_seed}.json"
    result = materialize(
        root,
        (args.benchmark or root / "configs/context_benchmark.yaml").resolve(),
        (args.registry or root / "artifacts/manifests/environment_registry.csv").resolve(),
        (args.split or root / "configs/biological_instance_split.yaml").resolve(),
        (args.manifest or manifest_default).resolve(),
        (args.summary or summary_default).resolve(),
        split_seed=selected_seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
