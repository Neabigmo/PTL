from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
BASELINES = ROOT / "results" / "baselines"
LOGS = ROOT / "results" / "logs" / "analysis"


CASE_QUERIES = [
    (
        "retained_high_transportability",
        "Transportable prediction retained for biological interpretation",
        lambda df: df[
            df["transportable"].eq(True)
            & df["is_control"].eq(False)
            & df["split_family"].isin(["unseen_perturbation_split", "unseen_combination_split", "low_support_split"])
            & df["cosine_similarity"].ge(0.45)
        ].sort_values(["cosine_similarity", "deg_direction_consistency_at_50"], ascending=False),
    ),
    (
        "confidence_failure_rejected",
        "High-confidence prediction rejected as a transportability risk",
        lambda df: df[
            df["transportable"].eq(False)
            & df["is_control"].eq(False)
            & df["confidence"].ge(0.75)
            & df["failure_mode"].isin(["severe_failure", "high_risk_error"])
        ].sort_values(["confidence", "risk"], ascending=False),
    ),
    (
        "transfer_boundary_failure",
        "Dataset or external transfer boundary with degraded prediction fidelity",
        lambda df: df[
            df["is_control"].eq(False)
            & df["split_family"].isin(["dataset_heldout_split", "external_holdout"])
            & df["cosine_similarity"].le(0.12)
        ].sort_values(["risk", "train_test_centroid_l2"], ascending=False),
    ),
]


def _output_dir_for(row: pd.Series) -> Path:
    model = str(row["model"])
    run_id = str(row["run_id"])
    suffix = f"__{model}"
    run_stem = run_id[: -len(suffix)] if run_id.endswith(suffix) else run_id
    return BASELINES / model / run_stem


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _case_from_output(case_type: str, description: str, row: pd.Series, pathway: pd.DataFrame) -> tuple[dict, dict | None]:
    outdir = _output_dir_for(row)
    pred_path = outdir / "test_predictions.npz"
    meta_path = outdir / "test_metadata.parquet"
    genes_path = outdir / "genes.txt"
    if not pred_path.exists() or not meta_path.exists() or not genes_path.exists():
        return {}, {
            "case_type": case_type,
            "run_id": row["run_id"],
            "signature_id": row["signature_id"],
            "reason": "missing output contract files",
            "output_dir": str(outdir.relative_to(ROOT)) if outdir.exists() else str(outdir),
        }

    meta = pd.read_parquet(meta_path)
    matches = meta.index[meta["signature_id"].astype(str).eq(str(row["signature_id"]))].tolist()
    if not matches:
        return {}, {
            "case_type": case_type,
            "run_id": row["run_id"],
            "signature_id": row["signature_id"],
            "reason": "signature_id not found in test_metadata.parquet",
            "output_dir": str(outdir.relative_to(ROOT)),
        }
    idx = int(matches[0])
    arrays = np.load(pred_path)
    y_true = np.asarray(arrays["y_true"], dtype=float)[idx]
    y_pred = np.asarray(arrays["y_pred"], dtype=float)[idx]
    genes = genes_path.read_text(encoding="utf-8").splitlines()
    if len(genes) != y_true.shape[0]:
        return {}, {
            "case_type": case_type,
            "run_id": row["run_id"],
            "signature_id": row["signature_id"],
            "reason": "gene list length does not match prediction matrix",
            "output_dir": str(outdir.relative_to(ROOT)),
        }

    true_order = np.argsort(np.abs(y_true))[::-1][:8]
    pred_order = np.argsort(np.abs(y_pred))[::-1][:8]
    top_true = [genes[i] for i in true_order]
    top_pred = [genes[i] for i in pred_order]
    shared = sorted(set(top_true).intersection(top_pred))
    direction = float(np.mean(np.sign(y_true[true_order]) == np.sign(y_pred[true_order])))
    path_row = pathway[pathway["run_id"].astype(str).eq(str(row["run_id"]))]
    mean_pathway_cosine = float(path_row["mean_pathway_cosine"].iloc[0]) if not path_row.empty else np.nan
    mean_pathway_direction = (
        float(path_row["mean_pathway_direction_consistency"].iloc[0]) if not path_row.empty else np.nan
    )

    return {
        "case_type": case_type,
        "case_description": description,
        "run_id": row["run_id"],
        "signature_id": row["signature_id"],
        "dataset_id": row["dataset_id"],
        "perturbation_label": row["perturbation_label"],
        "model": row["model"],
        "split_family": row["split_family"],
        "seed": int(row["seed"]),
        "cosine_similarity": float(row["cosine_similarity"]),
        "recomputed_cosine": _cosine(y_pred, y_true),
        "transportable": bool(row["transportable"]),
        "confidence": float(row["confidence"]),
        "risk": float(row["risk"]),
        "failure_mode": row["failure_mode"],
        "n_cells": int(row["n_cells"]),
        "support_signatures": float(row.get("perturbation_train_signature_count", np.nan)),
        "support_cells": float(row.get("perturbation_train_cell_count", np.nan)),
        "top_true_genes": ";".join(top_true),
        "top_pred_genes": ";".join(top_pred),
        "shared_top_genes": ";".join(shared),
        "top8_direction_consistency": direction,
        "mean_pathway_cosine_for_run": mean_pathway_cosine,
        "mean_pathway_direction_for_run": mean_pathway_direction,
        "output_dir": str(outdir.relative_to(ROOT)),
    }, None


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    examples = pd.read_parquet(TABLES / "ptl_examples.parquet")
    pathway = pd.read_csv(TABLES / "pathway_metric_summary.csv")
    examples = examples[examples["is_control"].eq(False)].copy()

    selected: list[dict] = []
    excluded: list[dict] = []
    used_signatures: set[str] = set()
    for case_type, description, query in CASE_QUERIES:
        candidates = query(examples)
        for _, row in candidates.head(200).iterrows():
            if str(row["signature_id"]) in used_signatures:
                continue
            case, exclusion = _case_from_output(case_type, description, row, pathway)
            if exclusion is not None:
                excluded.append(exclusion)
                continue
            selected.append(case)
            used_signatures.add(str(row["signature_id"]))
            break

    case_df = pd.DataFrame(selected)
    exclusion_df = pd.DataFrame(excluded)
    case_df.to_csv(TABLES / "cbac_case_examples.csv", index=False)
    exclusion_df.to_csv(TABLES / "cbac_case_exclusion.csv", index=False)
    summary = {
        "completed_case_count": int(len(case_df)),
        "excluded_candidate_count": int(len(exclusion_df)),
        "case_types": case_df["case_type"].tolist() if not case_df.empty else [],
    }
    (TABLES / "cbac_submission_compliance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (LOGS / "cbac_case_examples.log").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
