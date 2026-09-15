"""Audit outcome-blind within-file scPerturb3 context replication.

Only AnnData observation metadata and the variable index are read. The
expression matrix and layers are deliberately never accessed. The audit
reports whether the registered provenance criteria are met; it does not run a
predictor when the strict criteria fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/raw/scperturb_v1.4/SrivatsanTrapnell2020_sciplex3.h5ad"
CONTEXTS = ("A549", "MCF7", "K562")
MIN_SHARED_PERTURBATIONS = 100
MIN_MEDIAN_CELLS_PER_SHARED_UNIT = 20


def _norm_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("<NA>").str.strip()


def _read_observations() -> tuple[pd.DataFrame, int, list[str]]:
    if not DATASET.is_file():
        raise FileNotFoundError(DATASET)
    adata = ad.read_h5ad(DATASET, backed="r")
    try:
        # Explicitly use only obs and var metadata; never touch adata.X/layers.
        obs = adata.obs.copy()
        n_vars = int(adata.n_vars)
        var_columns = [str(column) for column in adata.var.columns]
    finally:
        adata.file.close()
    return obs, n_vars, var_columns


def _context_summary(obs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for context in CONTEXTS:
        part = obs.loc[obs["cell_line"] == context]
        perturbations = part["perturbation"].replace("<NA>", np.nan).dropna().nunique()
        controls = part["perturbation"].str.lower().isin(
            {"control", "vehicle", "dmso", "untreated"}
        )
        rows.append(
            {
                "context": context,
                "cells": int(len(part)),
                "perturbations": int(perturbations),
                "control_cells": int(controls.sum()),
                "control_units": int(
                    part.loc[controls, ["dose_value", "time"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "replicates": int(part["replicate"].replace("<NA>", np.nan).dropna().nunique()),
                "plates": int(part["plate"].replace("<NA>", np.nan).dropna().nunique()),
            }
        )
    return pd.DataFrame(rows)


def _matched_units(obs: pd.DataFrame) -> pd.DataFrame:
    work = obs.loc[obs["cell_line"].isin(CONTEXTS)].copy()
    work["perturbation"] = work["perturbation"].replace("<NA>", np.nan)
    work["dose_unit"] = work["dose_unit"].replace("<NA>", np.nan)
    work["time"] = pd.to_numeric(work["time"].replace("<NA>", np.nan), errors="coerce")
    work["dose_value"] = pd.to_numeric(
        work["dose_value"].replace("<NA>", np.nan), errors="coerce"
    )
    work = work.dropna(subset=["perturbation", "dose_value", "time", "dose_unit"])
    keys = ["perturbation", "dose_value", "dose_unit", "time"]
    counts = (
        work.groupby(["cell_line", *keys], observed=True)
        .size()
        .rename("cells")
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(CONTEXTS):
        for right in CONTEXTS[left_index + 1 :]:
            left_counts = counts.loc[counts["cell_line"] == left].drop(columns="cell_line")
            right_counts = counts.loc[counts["cell_line"] == right].drop(columns="cell_line")
            joined = left_counts.merge(
                right_counts,
                on=keys,
                how="inner",
                suffixes=("_left", "_right"),
            )
            if joined.empty:
                median_min = float("nan")
                shared_perturbations = 0
            else:
                joined["min_cells"] = joined[["cells_left", "cells_right"]].min(axis=1)
                median_min = float(joined["min_cells"].median())
                shared_perturbations = int(joined["perturbation"].nunique())
            rows.append(
                {
                    "context_left": left,
                    "context_right": right,
                    "matched_dose_time_units": int(len(joined)),
                    "shared_perturbations": shared_perturbations,
                    "median_min_cells_per_shared_unit": median_min,
                    "units_meeting_20_cells_each": int(
                        (joined["min_cells"] >= MIN_MEDIAN_CELLS_PER_SHARED_UNIT).sum()
                    )
                    if not joined.empty
                    else 0,
                }
            )
    return pd.DataFrame(rows)


def _plate_audit(obs: pd.DataFrame) -> dict[str, object]:
    work = obs.loc[obs["cell_line"].isin(CONTEXTS)].copy()
    work["plate"] = work["plate"].replace("<NA>", np.nan)
    work = work.dropna(subset=["plate"])
    plate_contexts = work.groupby("plate", observed=True)["cell_line"].nunique()
    unique = plate_contexts == 1
    return {
        "plates_total": int(len(plate_contexts)),
        "plates_shared_by_at_least_two_contexts": int((plate_contexts >= 2).sum()),
        "plates_unique_to_one_context": int(unique.sum()),
        "fraction_cells_on_context_unique_plates": float(
            work["plate"].isin(plate_contexts.index[unique]).mean()
        )
        if len(work)
        else float("nan"),
        "no_plate_uniquely_confounded": bool(not unique.any()),
    }


def run() -> dict[str, object]:
    obs, n_vars, var_columns = _read_observations()
    obs["cell_line"] = _norm_text(obs["cell_line"])
    for column in ["perturbation", "dose_unit", "replicate", "plate", "time", "dose_value"]:
        obs[column] = _norm_text(obs[column])
    context_obs = obs.loc[obs["cell_line"].isin(CONTEXTS)].copy()
    summary = _context_summary(context_obs)
    matched = _matched_units(context_obs)
    plates = _plate_audit(context_obs)
    all_pairs_match = bool(
        (matched["shared_perturbations"] >= MIN_SHARED_PERTURBATIONS).all()
        and (
            matched["matched_dose_time_units"] >= MIN_SHARED_PERTURBATIONS
        ).all()
    )
    all_depth = bool(
        (
            matched["median_min_cells_per_shared_unit"]
            >= MIN_MEDIAN_CELLS_PER_SHARED_UNIT
        ).all()
    )
    controls_each = bool((summary["control_cells"] > 0).all())
    same_rna_readout = bool(len(context_obs) > 0 and n_vars > 0)
    fixed_gene_panel = bool(n_vars > 0)
    checks = {
        "shared_perturbations_at_least_100_for_every_pair": all_pairs_match,
        "matched_dose_time_units_at_least_100_for_every_pair": all_pairs_match,
        "median_min_cells_at_least_20_for_every_pair": all_depth,
        "same_rna_readout": same_rna_readout,
        "controls_present_in_each_context": controls_each,
        "no_plate_uniquely_confounded": plates["no_plate_uniquely_confounded"],
        "fixed_gene_panel": fixed_gene_panel,
    }
    strict_gate = bool(all(checks.values()))
    return {
        "schema_version": 2,
        "status": "eligible" if strict_gate else "blocked",
        "predictor_run": False,
        "outcome_blind": True,
        "dataset": DATASET.relative_to(ROOT).as_posix(),
        "contexts": list(CONTEXTS),
        "shape": {"cells_in_file": int(len(obs)), "genes_in_file": n_vars},
        "observation_fields_used": [
            "cell_line",
            "perturbation",
            "dose_value",
            "dose_unit",
            "time",
            "replicate",
            "plate",
        ],
        "outcome_fields_not_used": ["X", "layers", "obsm", "obsp"],
        "variable_metadata_columns": var_columns,
        "registered_thresholds": {
            "minimum_shared_perturbations": MIN_SHARED_PERTURBATIONS,
            "minimum_median_cells_per_shared_unit": MIN_MEDIAN_CELLS_PER_SHARED_UNIT,
            "matched_dose_time_required": True,
            "same_rna_readout_required": True,
            "controls_each_context_required": True,
            "no_unique_plate_confounding_required": True,
            "fixed_gene_panel_required": True,
        },
        "context_summary": summary.to_dict(orient="records"),
        "matched_pairs": matched.to_dict(orient="records"),
        "plate_audit": plates,
        "strict_gate_checks": checks,
        "decision": (
            "Run the source-only predictor only if every strict gate check is "
            "true. This audit does not pass data to a predictor."
        ),
    }


def main() -> int:
    report = run()
    outdir = ROOT / "artifacts/manifests"
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "sciplex3_context_replication_audit_v2.json"
    csv_path = outdir / "sciplex3_context_replication_audit_v2.csv"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    rows = []
    rows.extend(
        {"section": "context_summary", **row}
        for row in report["context_summary"]
    )
    rows.extend({"section": "matched_pair", **row} for row in report["matched_pairs"])
    rows.append({"section": "plate_audit", **report["plate_audit"]})
    rows.append({"section": "strict_gate_checks", **report["strict_gate_checks"]})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report["status"] == "eligible" else 2


if __name__ == "__main__":
    raise SystemExit(main())
