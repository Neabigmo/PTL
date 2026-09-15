"""Read-only metadata audit for local scPerturb H5AD files.

The script never loads ``X`` into memory.  It reads H5AD dataframe metadata
and categorical codes with h5py, which keeps the G1 audit safe for multi-GB
matrices and makes its output independent of an optional scanpy installation.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


CONTROL_RE = re.compile(
    r"control|ctrl|non[-_ ]?target|ntc|scramble|negative|unperturbed|untreated|vehicle|mock|none",
    re.IGNORECASE,
)
PERT_RE = re.compile(r"pert|guide|grna|sg.?rna|target|gene", re.IGNORECASE)
CELL_RE = re.compile(r"cell|tissue|organ|disease|cancer", re.IGNORECASE)
CONDITION_RE = re.compile(r"condition|time|day|batch|sample|replicate|donor", re.IGNORECASE)


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ""
    return str(value)


def h5_shape(node: Any) -> tuple[int, int]:
    shape = getattr(node, "shape", None)
    if shape is None:
        shape = node.attrs.get("shape")
    if shape is None or len(shape) != 2:
        raise ValueError("H5AD X has no two-dimensional shape")
    return int(shape[0]), int(shape[1])


def obs_keys(handle: h5py.File) -> list[str]:
    group = handle["obs"]
    index_name = decode(group.attrs.get("_index", ""))
    return sorted(key for key in group.keys() if key != index_name)


def iter_column(node: Any, chunk_size: int = 8192) -> Iterable[str]:
    """Yield decoded values without loading a large H5AD column at once."""

    if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
        categories = [decode(x) for x in node["categories"][:]]
        codes = node["codes"]
        for start in range(0, int(codes.shape[0]), chunk_size):
            for code in codes[start : start + chunk_size]:
                index = int(code)
                yield categories[index] if 0 <= index < len(categories) else ""
        return

    if not isinstance(node, h5py.Dataset):
        return
    size = int(node.shape[0]) if node.shape else 0
    for start in range(0, size, chunk_size):
        for value in node[start : start + chunk_size]:
            yield decode(value)


def is_control_label(value: str) -> bool:
    normalized = value.strip().casefold()
    return normalized in {"*", "nt", "ntc", "nc", "gfp", "non-targeting"} or bool(
        CONTROL_RE.search(value)
    )


def is_combination_label(value: str) -> bool:
    return bool(re.search(r"(?:\+|,|;|\\band\\b)", value, re.IGNORECASE))


def column_summary(group: h5py.Group, key: str) -> tuple[int, int, list[str], int, int, list[str]]:
    """Return nonmissing, unique, sample, controls, combinations, and labels."""

    node = group[key]
    counter: Counter[str] = Counter()
    nonmissing = 0
    controls = 0
    combinations = 0
    for value in iter_column(node):
        value = value.strip()
        if not value or value.lower() in {"nan", "na", "null", "none"}:
            continue
        nonmissing += 1
        if len(counter) < 10000 or value in counter:
            counter[value] += 1
        if is_control_label(value):
            controls += 1
        if is_combination_label(value):
            combinations += 1
    sample = sorted(counter)[:12]
    control_values = sorted(value for value in counter if is_control_label(value))
    return nonmissing, len(counter), sample, controls, combinations, control_values


def choose_primary_perturbation(keys: list[str]) -> str:
    priority = [
        "perturbation",
        "condition",
        "target_gene",
        "target",
        "gene",
        "guide_id",
        "sgrna",
        "grna",
    ]
    lowered = {key.casefold(): key for key in keys}
    for candidate in priority:
        if candidate in lowered:
            return lowered[candidate]
    candidates = [key for key in keys if PERT_RE.search(key)]
    return sorted(candidates)[0] if candidates else ""


def source_study(filename: str) -> str:
    stem = Path(filename).stem
    mapping = {
        "AdamsonWeissman2016": "Adamson et al. 2016",
        "NormanWeissman2019": "Norman et al. 2019",
        "ReplogleWeissman2022": "Replogle et al. 2022",
        "TianKampmann2021": "Tian et al. 2021",
        "FrangiehIzar2021": "Frangieh et al. 2021",
        "PapalexiSatija2021": "Papalexi et al. 2021",
        "DatlingerBock": "Datlinger et al.",
        "XuCao2023": "Xu/Cao et al. 2023",
        "WesselsSatija2023": "Wessels/Satija et al. 2023",
        "JoungZhang2023": "Joung/Zhang et al. 2023",
    }
    for token, label in mapping.items():
        if token in stem:
            return label
    return "unknown/local scPerturb record"


def modality(filename: str, keys: list[str]) -> str:
    text = f"{filename} {' '.join(keys)}".lower()
    values = []
    if "crispra" in text or "crispr_a" in text:
        values.append("CRISPRa")
    if "crispri" in text or "crispr_i" in text:
        values.append("CRISPRi")
    if "protein" in text or "eccite" in text:
        values.append("protein/multimodal")
    if "rna" in text or "perturb" in text:
        values.append("RNA")
    return "+".join(dict.fromkeys(values)) or "unknown"


def eligibility(filename: str) -> tuple[str, str, str]:
    name = Path(filename).stem
    common = ""
    context = ""
    external = ""
    if name == "NormanWeissman2019_filtered":
        common = "common_core"
    elif name == "ReplogleWeissman2022_K562_gwps":
        common = "common_core"
    elif name == "ReplogleWeissman2022_rpe1":
        common = "common_core"
    elif name.startswith("AdamsonWeissman2016_"):
        common = "common_core_candidate"
    if "TianKampmann2021" in name or "FrangiehIzar2021_RNA" in name:
        context = "context_expansion_candidate"
    elif "PapalexiSatija2021_eccite_RNA" in name or name == "XuCao2023":
        context = "context_expansion_candidate"
    elif "DatlingerBock" in name:
        context = "context_expansion_supplementary"
    if name.startswith("WesselsSatija2023") or name.startswith("JoungZhang2023"):
        external = "external_candidate"
    return common, context, external


def audit_file(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        n_obs, n_vars = h5_shape(handle["X"])
        obs = handle["obs"]
        keys = obs_keys(handle)
        primary = choose_primary_perturbation(keys)
        primary_nonmissing = 0
        n_unique = 0
        control_labels: list[str] = []
        n_controls = 0
        n_combinations = 0
        if primary:
            primary_nonmissing, n_unique, values, n_controls, n_combinations, control_values = column_summary(
                obs, primary
            )
            control_labels = control_values

        perturbation_cols = [key for key in keys if PERT_RE.search(key)]
        cell_cols = [key for key in keys if CELL_RE.search(key)]
        condition_cols = [key for key in keys if CONDITION_RE.search(key)]
        lower_keys = {key.casefold() for key in keys}
        has_control = bool(
            {"control", "is_control", "control_status", "control_condition"} & lower_keys
        ) or bool(control_labels)
        has_cell = bool(cell_cols)
        has_modality = "perturbation_type" in lower_keys or "crispr" in " ".join(lower_keys)
        has_condition = bool(condition_cols)
        has_timepoint = any("time" in key.casefold() or "day" in key.casefold() for key in keys)
        has_batch = any("batch" in key.casefold() or "sample" in key.casefold() for key in keys)
        common, context, external = eligibility(path.name)

        return {
            "dataset_id": path.stem,
            "filename": path.name,
            "source_study": source_study(path.name),
            "size_bytes": path.stat().st_size,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "obs_columns": ";".join(keys),
            "candidate_perturbation_columns": ";".join(perturbation_cols),
            "primary_perturbation_column": primary,
            "candidate_control_labels": ";".join(control_labels),
            "cell_line_type_columns": ";".join(cell_cols),
            "condition_timepoint_batch_columns": ";".join(condition_cols),
            "perturbation_modality": modality(path.name, keys),
            "n_controls": n_controls,
            "n_unique_perturbations": n_unique,
            "combination_support": f"rows={n_combinations};present={n_combinations > 0}",
            "metadata_completeness_flags": ";".join(
                [
                    f"perturbation={bool(primary and primary_nonmissing)}",
                    f"control={has_control}",
                    f"cell_context={has_cell}",
                    f"modality={has_modality}",
                    f"condition={has_condition}",
                    f"timepoint={has_timepoint}",
                    f"batch={has_batch}",
                ]
            ),
            "eligible_common_core": common,
            "eligible_context_expansion": context,
            "eligible_external": external,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "artifacts/manifests/dataset_audit.csv"
    files = sorted((root / "data/raw/scperturb_v1.4").glob("*.h5ad"))
    if not files:
        raise SystemExit("No local H5AD files found")
    rows = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path.name}", flush=True)
        try:
            rows.append(audit_file(path))
        except Exception as exc:  # keep the audit complete and actionable
            rows.append(
                {
                    "dataset_id": path.stem,
                    "filename": path.name,
                    "source_study": source_study(path.name),
                    "size_bytes": path.stat().st_size,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
