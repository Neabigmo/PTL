"""Audit whether Frangieh ``guide_id`` is a single-guide identity.

This audit uses QC-passed cell metadata only.  It does not read expression,
predictions, risks, or any outcome-derived quantity.  The purpose is to stop a
cell-level combinatorial barcode from being mistaken for a perturbation-level
sgRNA label in a sensitivity analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "data/processed/FrangiehIzar2021_RNA_cell_metadata.parquet"
ENVIRONMENT_BY_CONDITION = {
    "Control": "frangieh_melanoma_control",
    "Co-culture": "frangieh_melanoma_coculture",
    "IFNγ": "frangieh_melanoma_ifng",
}


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _tokens(value: object) -> tuple[str, ...]:
    text = _clean(value)
    if not text or text.casefold() in {"nan", "none", "na"}:
        return ()
    return tuple(token for token in re.split(r"[;|,]", text) if token)


def summarize_group(label: str, group: pd.DataFrame) -> dict[str, Any]:
    raw_values = group["guide_id"].tolist()
    token_lists = [_tokens(value) for value in raw_values]
    observed = [tokens for tokens in token_lists if tokens]
    guide_strings = {";".join(tokens) for tokens in observed}
    guide_tokens = {token for tokens in observed for token in tokens}
    token_counts = np.asarray([len(tokens) for tokens in token_lists], dtype=float)
    target_prefix = f"{label}_".casefold()
    contains_target = [
        any(token.casefold().startswith(target_prefix) for token in tokens)
        for tokens in token_lists
    ]
    return {
        "perturbation_label": label,
        "n_cells_qc": int(len(group)),
        "n_cells_with_nonmissing_guide_id": int(len(observed)),
        "n_unique_guide_strings": int(len(guide_strings)),
        "n_unique_guide_tokens": int(len(guide_tokens)),
        "median_guide_tokens_per_cell": float(np.median(token_counts)) if len(token_counts) else np.nan,
        "fraction_multi_token_cells": float(np.mean(token_counts > 1)) if len(token_counts) else np.nan,
        "fraction_cells_containing_target_token": float(np.mean(contains_target)) if contains_target else np.nan,
        "guide_id_missing_fraction": float(np.mean(token_counts == 0)) if len(token_counts) else np.nan,
    }


def audit(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root.resolve()
    columns = ["perturbation_label", "guide_id", "is_control", "passes_qc", "perturbation_2"]
    frame = pd.read_parquet(root / METADATA_PATH.relative_to(ROOT), columns=columns)
    frame = frame.loc[frame["passes_qc"].astype(bool)].copy()
    frame["condition"] = frame["perturbation_2"].astype("string").fillna("").str.strip()
    frame["environment_key"] = frame["condition"].map(ENVIRONMENT_BY_CONDITION)
    frame["perturbation_label"] = frame["perturbation_label"].astype("string").fillna("").str.strip()
    frame = frame.loc[frame["environment_key"].notna() & frame["perturbation_label"].ne("")].copy()
    rows: list[dict[str, Any]] = []
    for (environment, label), group in frame.groupby(["environment_key", "perturbation_label"], sort=True):
        summary = summarize_group(str(label), group)
        summary["environment_key"] = str(environment)
        summary["status"] = "audit_only_not_primary"
        rows.append(summary)
    result = pd.DataFrame(rows).sort_values(["environment_key", "perturbation_label"], kind="stable").reset_index(drop=True)
    noncontrol = result.loc[~result["perturbation_label"].eq("control")]
    report = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2_frangieh_guide_id_semantics_audit_v1",
        "status": "guide_id_semantics_audited_not_single_sgrna_identity",
        "outcome_blind": True,
        "prediction_evaluated": False,
        "risk_evaluated": False,
        "source": "data/processed/FrangiehIzar2021_RNA_cell_metadata.parquet",
        "qc_scope": "passes_qc rows only; no expression matrix loaded",
        "n_qc_cells": int(len(frame)),
        "n_perturbation_groups": int(len(noncontrol)),
        "median_unique_guide_strings_per_group": float(noncontrol["n_unique_guide_strings"].median()),
        "max_unique_guide_strings_per_group": int(noncontrol["n_unique_guide_strings"].max()),
        "median_multi_token_cell_fraction": float(noncontrol["fraction_multi_token_cells"].median()),
        "representative_groups": result.loc[result["perturbation_label"].isin(["A2M", "ACSL3", "control"])]
        .to_dict("records"),
        "interpretation": (
            "guide_id is a cell-level combinatorial barcode string: values can contain multiple "
            "semicolon-delimited guide tokens and non-gene/no-site tokens. It is not a unique "
            "single-sgRNA identity for perturbation_label."
        ),
        "guide_stratified_sensitivity_executed": False,
        "guide_stratified_sensitivity_policy": (
            "do not use guide_id as a single-guide stratification key; the existing guide coverage "
            "table remains audit-only and is not evidence for a completed sensitivity"
        ),
        "csv": "artifacts/manifests/formal_v2_claim_lock_guide_id_semantics.csv",
    }
    return result, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result, report = audit(args.root)
    output_dir = args.root.resolve() / "artifacts/manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "formal_v2_claim_lock_guide_id_semantics.csv"
    json_path = output_dir / "formal_v2_claim_lock_guide_id_semantics.json"
    result.to_csv(csv_path, index=False)
    report["csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
