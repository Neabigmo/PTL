"""Build the small, reproducible GEARS scientific-pilot split manifest.

The pilot randomly holds out perturbation conditions, keeping every condition
in exactly one of train, validation, or test. The split JSONs contain only
signature IDs; all expression/reference data continue to be resolved by
DataRepository. ``split_family`` remains ``random_split`` for compatibility
with the existing run registry, while explicit protocol fields prevent it from
being confused with a familiar/random signature-level evaluation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "NormanWeissman2019_filtered",
    "ReplogleWeissman2022_K562_gwps",
    "ReplogleWeissman2022_rpe1",
)
DEFAULT_SEEDS = (0, 1, 2)
PILOT_DATASETS = {
    "NormanWeissman2019_filtered",
    "ReplogleWeissman2022_K562_gwps",
    "ReplogleWeissman2022_rpe1",
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--summary", default=str(ROOT / "results" / "tables" / "preprocessing_summary.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed" / "splits"))
    parser.add_argument("--audit", default=str(ROOT / "results" / "tables" / "gears_scientific_pilot_split_audit.csv"))
    parser.add_argument("--min-replicates", type=int, default=2)
    parser.add_argument(
        "--official-gene2go",
        default=str(ROOT / "data" / "processed" / "official_gears" / "gene2go_all.pkl"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(str(value))
    if path.is_file():
        return path
    text = str(value).replace("/", "\\")
    marker = "\\data\\processed\\"
    if marker in text:
        candidate = ROOT / "data" / "processed" / text.split(marker, 1)[1]
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(value)


def condition_components(label: object, is_control: object) -> list[str]:
    text = str(label)
    if str(is_control).lower() in {"true", "1", "yes"} or text.lower() in {"ctrl", "control"}:
        return []
    return [part for part in text.split("_") if part]


def build_condition_disjoint_split(
    signature: pd.DataFrame,
    seed: int,
) -> tuple[pd.Index, pd.Index, pd.Index, dict[str, list[str]]]:
    """Split perturbation conditions, keeping every condition in one role."""

    controls = signature["is_control"].astype(bool)
    non_control = signature.loc[~controls]
    conditions = np.asarray(sorted(non_control["perturbation_label"].astype(str).unique()))
    if len(conditions) < 3:
        raise ValueError("condition-disjoint GEARS split needs at least three perturbation conditions")
    shuffled = conditions[np.random.default_rng(seed).permutation(len(conditions))]
    n_test = max(1, int(round(len(conditions) * 0.20)))
    n_val = max(1, int(round(len(conditions) * 0.20)))
    if n_test + n_val >= len(conditions):
        n_val = max(1, len(conditions) - n_test - 1)
    test_conditions = sorted(shuffled[:n_test].tolist())
    val_conditions = sorted(shuffled[n_test:n_test + n_val].tolist())
    train_conditions = sorted(shuffled[n_test + n_val:].tolist())
    label_values = signature["perturbation_label"].astype(str)
    train_mask = controls | label_values.isin(train_conditions)
    val_mask = label_values.isin(val_conditions)
    test_mask = label_values.isin(test_conditions)
    if int(train_mask.sum() + val_mask.sum() + test_mask.sum()) != len(signature):
        raise AssertionError("condition-disjoint split did not assign every signature row")
    return (
        signature.index[train_mask],
        signature.index[val_mask],
        signature.index[test_mask],
        {
            "train": train_conditions,
            "val": val_conditions,
            "test": test_conditions,
        },
    )


def build_payload(
    dataset: str,
    signature: pd.DataFrame,
    seed: int,
    summary_row: pd.Series,
    output_dir: Path,
    min_replicates: int,
    official_perturbation_genes: set[str],
) -> tuple[dict, dict]:
    condition_counts = signature["perturbation_label"].astype(str).value_counts()
    signature = signature.loc[
        signature["perturbation_label"].astype(str).map(condition_counts).ge(min_replicates)
        & signature.apply(
            lambda row: set(condition_components(row["perturbation_label"], row["is_control"])).issubset(
                official_perturbation_genes
            ),
            axis=1,
        )
    ].reset_index(drop=True)
    if signature.empty:
        raise ValueError(f"No GEARS-eligible signatures remain for {dataset}.")
    train_idx, val_idx, test_idx, conditions = build_condition_disjoint_split(signature, seed=seed)
    split_path = output_dir / f"gears_condition_split__{dataset}__seed{seed}__signature.json"
    payload = {
        "split_family": "random_split",
        "split_protocol": "random_condition_holdout",
        "perturbation_overlap": "zero",
        "track": "signature",
        "dataset_scope": dataset,
        "seed": int(seed),
        "status": "ready",
        "holdout_variable": "perturbation_condition",
        "heldout_target": "",
        "unsupported_reason": "",
        "input_paths": {
            "cell_metadata_path": str(resolve_path(summary_row["cell_metadata_path"])),
            "qc_manifest_path": str(resolve_path(summary_row["qc_manifest_path"])),
            "feature_metadata_path": str(resolve_path(summary_row["feature_metadata_path"])),
            "signature_path": str(resolve_path(summary_row["signature_path"])),
        },
        "train_ids": signature.loc[train_idx, "signature_id"].astype(str).tolist(),
        "validation_ids": signature.loc[val_idx, "signature_id"].astype(str).tolist(),
        "test_ids": signature.loc[test_idx, "signature_id"].astype(str).tolist(),
        "pilot_scope": "official_gears_scientific_pilot_condition_disjoint",
    }
    parts = {"train": signature.loc[train_idx], "validation": signature.loc[val_idx], "test": signature.loc[test_idx]}
    audit = {
        "split_family": "random_split",
        "split_protocol": "random_condition_holdout",
        "perturbation_overlap": "zero",
        "track": "signature",
        "dataset_scope": dataset,
        "seed": int(seed),
        "status": "ready",
        "holdout_variable": "perturbation_condition",
        "heldout_target": "",
        "unsupported_reason": "",
        "output_path": str(split_path),
        "train_units": len(parts["train"]),
        "validation_units": len(parts["validation"]),
        "test_units": len(parts["test"]),
        "train_perturbations": parts["train"]["perturbation_label"].astype(str).nunique(),
        "validation_perturbations": parts["validation"]["perturbation_label"].astype(str).nunique(),
        "test_perturbations": parts["test"]["perturbation_label"].astype(str).nunique(),
        "train_controls": int(parts["train"]["is_control"].sum()),
        "validation_controls": int(parts["validation"]["is_control"].sum()),
        "test_controls": int(parts["test"]["is_control"].sum()),
        "train_conditions": "|".join(conditions["train"]),
        "validation_conditions": "|".join(conditions["val"]),
        "test_conditions": "|".join(conditions["test"]),
        "train_validation_condition_overlap": len(set(conditions["train"]) & set(conditions["val"])),
        "train_test_condition_overlap": len(set(conditions["train"]) & set(conditions["test"])),
        "validation_test_condition_overlap": len(set(conditions["val"]) & set(conditions["test"])),
        "pilot_scope": "official_gears_scientific_pilot_condition_disjoint",
    }
    return payload, audit


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    with Path(args.official_gene2go).open("rb") as handle:
        official_perturbation_genes = set(pickle.load(handle))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for dataset in args.datasets:
        row = summary.loc[summary["dataset_id"].astype(str).eq(dataset)]
        if row.empty:
            raise KeyError(f"Dataset {dataset!r} is missing from preprocessing summary.")
        summary_row = row.iloc[0]
        signature = pd.read_parquet(resolve_path(summary_row["signature_path"]))
        for seed in args.seeds:
            payload, audit = build_payload(
                dataset,
                signature,
                int(seed),
                summary_row,
                output_dir,
                args.min_replicates,
                official_perturbation_genes,
            )
            path = output_dir / f"gears_condition_split__{dataset}__seed{int(seed)}__signature.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            audit["output_path"] = str(path)
            audit_rows.append(audit)
    combined = pd.DataFrame(audit_rows)
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(audit_path, index=False)
    print(f"Wrote {len(audit_rows)} GEARS pilot rows to {audit_path}.")


if __name__ == "__main__":
    main()
