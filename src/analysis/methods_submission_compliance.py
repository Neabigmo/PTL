from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
LOG_DIR = ROOT / "results" / "logs" / "analysis"


PUBLIC_HOLDOUTS = [
    "GSE284197_screen",
    "ReplogleWeissman2022_rpe1",
    "PapalexiSatija2021_eccite_RNA",
    "DatlingerBock2021",
]

GSE264667_CANDIDATES = [
    {
        "candidate_id": "GSE264667_hepg2_raw_singlecell_01",
        "dataset_label": "NadigOConner2024_hepg2",
        "cell_line": "HepG2",
        "geo_url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_hepg2_raw_singlecell_01.h5ad",
        "geo_size_reported": "5.2G",
        "local_proxy": ROOT / "data" / "raw" / "scperturb_v1.4" / "NadigOConner2024_hepg2.h5ad",
    },
    {
        "candidate_id": "GSE264667_jurkat_raw_singlecell_01",
        "dataset_label": "NadigOConner2024_jurkat",
        "cell_line": "Jurkat",
        "geo_url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_jurkat_raw_singlecell_01.h5ad",
        "geo_size_reported": "8.7G",
        "local_proxy": ROOT / "data" / "raw" / "scperturb_v1.4" / "NadigOConner2024_jurkat.h5ad",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Methods submission compliance tables.")
    parser.add_argument("--tables-dir", default=str(TABLES))
    parser.add_argument("--log-file", default=str(LOG_DIR / "methods_submission_compliance.log"))
    return parser.parse_args()


def log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def build_public_screen_validation(tables_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(tables_dir / "all_metrics.csv")
    prep = pd.read_csv(tables_dir / "preprocessing_summary.csv")
    rows: list[dict[str, Any]] = []
    for dataset_id in PUBLIC_HOLDOUTS:
        subset = metrics[
            metrics["split_family"].eq("dataset_heldout_split")
            & metrics["dataset_scope"].eq("all_datasets")
            & metrics["heldout_target"].eq(dataset_id)
        ].copy()
        prep_row = prep[prep["dataset_id"].eq(dataset_id)]
        if subset.empty:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "validation_role": "additional_public_screen_holdout",
                    "status": "no_completed_dataset_holdout_rows",
                    "n_runs": 0,
                    "n_models": 0,
                    "n_seeds": 0,
                    "mean_cosine_non_control": float("nan"),
                    "best_model": "",
                    "best_mean_cosine_non_control": float("nan"),
                    "n_signature_rows": int(prep_row["n_signature_rows"].iloc[0]) if not prep_row.empty else 0,
                    "n_unique_perturbations": int(prep_row["n_unique_perturbations"].iloc[0]) if not prep_row.empty else 0,
                    "n_cells_kept": int(prep_row["n_cells_kept"].iloc[0]) if not prep_row.empty else 0,
                    "manuscript_claim": "excluded from completed validation summary",
                }
            )
            continue
        by_model = (
            subset.groupby("model", as_index=False)
            .agg(
                n_runs=("run_id", "nunique"),
                mean_cosine_non_control=("mean_cosine_non_control", "mean"),
                median_cosine_non_control=("median_cosine_non_control", "median"),
                mean_false_transportability_at_0p8=("false_transportability_rate_at_0p8", "mean"),
            )
            .sort_values("mean_cosine_non_control", ascending=False)
        )
        best = by_model.iloc[0]
        role = "independent_public_experimental_screen" if dataset_id == "GSE284197_screen" else "additional_public_screen_holdout"
        rows.append(
            {
                "dataset_id": dataset_id,
                "validation_role": role,
                "status": "completed_from_existing_dataset_holdout",
                "n_runs": int(subset["run_id"].nunique()),
                "n_models": int(subset["model"].nunique()),
                "n_seeds": int(subset["seed"].nunique()),
                "mean_cosine_non_control": _safe_mean(subset["mean_cosine_non_control"]),
                "best_model": str(best["model"]),
                "best_mean_cosine_non_control": float(best["mean_cosine_non_control"]),
                "n_signature_rows": int(prep_row["n_signature_rows"].iloc[0]) if not prep_row.empty else 0,
                "n_unique_perturbations": int(prep_row["n_unique_perturbations"].iloc[0]) if not prep_row.empty else 0,
                "n_cells_kept": int(prep_row["n_cells_kept"].iloc[0]) if not prep_row.empty else 0,
                "manuscript_claim": "public experimental screen validation; no newly generated wet-lab data",
            }
        )
    return pd.DataFrame(rows)


def inspect_h5ad_proxy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "local_proxy_exists": False,
            "local_proxy_size_bytes": 0,
            "n_cells": 0,
            "n_genes": 0,
            "has_perturbation_field": False,
            "has_control_cells": False,
            "n_control_cells": 0,
            "n_unique_perturbations": 0,
            "obs_fields_detected": "",
        }
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs
    perturbation = obs["perturbation"].astype(str) if "perturbation" in obs.columns else pd.Series([], dtype=str)
    control_mask = perturbation.str.lower().isin({"control", "non-targeting", "ntc", "negative_control"})
    return {
        "local_proxy_exists": True,
        "local_proxy_size_bytes": int(path.stat().st_size),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "has_perturbation_field": "perturbation" in obs.columns,
        "has_control_cells": bool(control_mask.any()),
        "n_control_cells": int(control_mask.sum()),
        "n_unique_perturbations": int(perturbation.nunique()) if not perturbation.empty else 0,
        "obs_fields_detected": ";".join(map(str, obs.columns[:30])),
    }


def build_external_candidate_audit() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in GSE264667_CANDIDATES:
        local = inspect_h5ad_proxy(Path(candidate["local_proxy"]))
        contract_pass = bool(local["local_proxy_exists"] and local["has_perturbation_field"] and local["has_control_cells"])
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "dataset_label": candidate["dataset_label"],
                "cell_line": candidate["cell_line"],
                "geo_accession": "GSE264667",
                "geo_url": candidate["geo_url"],
                "geo_size_reported": candidate["geo_size_reported"],
                "local_proxy_path": str(Path(candidate["local_proxy"]).relative_to(ROOT)) if Path(candidate["local_proxy"]).is_relative_to(ROOT) else str(candidate["local_proxy"]),
                **local,
                "contract_audit_status": "contract_passed_proxy_available" if contract_pass else "contract_not_ready",
                "adoption_status": "candidate_not_claimed_in_main_results",
                "reason": (
                    "GEO raw h5ad is publicly available but very large; local scPerturb proxy exposes perturbation and control fields. "
                    "This is retained as an external candidate until raw-source processing and baseline rows are completed."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_methods_comparison_table() -> pd.DataFrame:
    rows = [
        {
            "approach": "Random-split benchmark",
            "input_requirements": "Predictions and random train/test membership",
            "keep_abstain_decision": "No",
            "detects_novelty_failure": "Weak",
            "strength": "Simple low-stress anchor",
            "limitation": "Does not test perturbation, support, or dataset transfer",
        },
        {
            "approach": "Naive confidence filtering",
            "input_requirements": "Predictor confidence or fidelity proxy",
            "keep_abstain_decision": "Yes",
            "detects_novelty_failure": "Weak",
            "strength": "Easy to deploy",
            "limitation": "Can accept confident but non-transportable predictions",
        },
        {
            "approach": "Support/novelty heuristic",
            "input_requirements": "Training support and perturbation novelty fields",
            "keep_abstain_decision": "Yes",
            "detects_novelty_failure": "Moderate",
            "strength": "Transparent operational rule",
            "limitation": "Ignores model output and calibration",
        },
        {
            "approach": "Calibrated logistic reliability model",
            "input_requirements": "Deployment features and transportability labels",
            "keep_abstain_decision": "Yes",
            "detects_novelty_failure": "Moderate",
            "strength": "Calibrated, interpretable baseline",
            "limitation": "Linear decision surface",
        },
        {
            "approach": "Model-specific calibration",
            "input_requirements": "One predictor family and its confidence outputs",
            "keep_abstain_decision": "Yes",
            "detects_novelty_failure": "Variable",
            "strength": "Can exploit model-specific uncertainty",
            "limitation": "Hard to compare across predictor families",
        },
        {
            "approach": "External-only validation",
            "input_requirements": "Independent experimental screen",
            "keep_abstain_decision": "No",
            "detects_novelty_failure": "Strong at dataset level",
            "strength": "Direct public experimental transfer test",
            "limitation": "Does not yield per-signature deployment decisions",
        },
        {
            "approach": "PTL workflow",
            "input_requirements": "Stress splits, signatures, predictor outputs, labels, deployment features",
            "keep_abstain_decision": "Yes",
            "detects_novelty_failure": "Strong",
            "strength": "Model-agnostic reliability audit with failure atlas",
            "limitation": "Requires labeled benchmark outputs and recalibration for new collections",
        },
    ]
    return pd.DataFrame(rows)


def write_latex_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Approach & Keep/abstain & Novelty failure & Main limitation \\\\",
        "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        vals = [row.approach, row.keep_abstain_decision, row.detects_novelty_failure, row.limitation]
        vals = [re.sub(r"([_%&])", r"\\\1", str(v)) for v in vals]
        lines.append(" & ".join(vals) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    log_file = Path(args.log_file)
    tables_dir.mkdir(parents=True, exist_ok=True)

    public_validation = build_public_screen_validation(tables_dir)
    public_validation.to_csv(tables_dir / "additional_public_screen_validation_summary.csv", index=False)
    log(log_file, f"Wrote {len(public_validation)} public-screen validation rows.")

    candidate_audit = build_external_candidate_audit()
    candidate_audit.to_csv(tables_dir / "external_candidate_contract_audit.csv", index=False)
    log(log_file, f"Wrote {len(candidate_audit)} external candidate audit rows.")

    comparison = build_methods_comparison_table()
    comparison.to_csv(tables_dir / "methods_comparison_table.csv", index=False)
    write_latex_table(comparison, tables_dir / "methods_comparison_table.tex")
    log(log_file, f"Wrote {len(comparison)} method comparison rows.")

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "public_screen_rows": len(public_validation),
        "external_candidate_rows": len(candidate_audit),
        "comparison_rows": len(comparison),
    }
    (tables_dir / "methods_submission_compliance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
