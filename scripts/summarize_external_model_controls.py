"""Summarize the comparable public-cohort condition-holdout model layer.

Only GEARS and baseline predictions evaluated on each cohort's frozen GEARS
gene panel are collected here.  Native full-gene baselines are intentionally
excluded: they are useful references but not a fair model-family comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT / "results" / "external_controls"
GEARS_ROOT = ROOT / "results" / "formal_v2" / "gears_remote"
OUT_DIR = ROOT / "results" / "tables"


def metric_record(path: Path, model_override: str | None = None) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    details = payload.get("model_details", {})
    return {
        "analysis_layer": "external_public_cohort_condition_holdout_shared_gene_panel",
        "model": model_override or str(payload["model"]),
        "dataset_id": str(payload.get("dataset_id", payload.get("dataset_scope", ""))),
        "seed": int(payload["seed"]),
        "gene_count": int(payload["gene_count"]),
        "gene_space_policy": str(payload["gene_space_policy"]),
        "n_test_noncontrol": int(metrics["n_test_noncontrol"]),
        "mean_cosine_similarity_non_control_test": float(metrics["mean_cosine_similarity_non_control_test"]),
        "mse_non_control_test": float(metrics["mse_non_control_test"]),
        "mae_non_control_test": float(metrics["mae_non_control_test"]),
        "n_test_conditions": int(details.get("n_test_conditions", 0)),
        "run_metrics_path": path.relative_to(ROOT).as_posix(),
    }


def main() -> None:
    # The verified GEARS campaign is stored under the formal-v2 provenance
    # root, whereas this script's baseline outputs live under the current
    # external-controls root.  Keep the two locations explicit so a stale
    # or orphaned summary can never be mistaken for a completed run.
    gears_paths = sorted(GEARS_ROOT.glob("*/run_metrics.json"))
    baseline_paths = sorted((EXTERNAL_ROOT / "shared_panel_baselines").glob("*/*/run_metrics.json"))
    if not gears_paths:
        raise RuntimeError("No completed GEARS metrics found.")
    if not baseline_paths:
        raise RuntimeError("No completed shared-panel baseline metrics found.")
    gears_records = [metric_record(path, model_override="gears") for path in gears_paths]
    if len(gears_records) != 9:
        raise ValueError("Canonical formal-v2 GEARS layer must contain exactly 9 runs.")
    for path in gears_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        details = payload.get("model_details", {})
        if int(details.get("epochs", 0)) != 20 or str(details.get("device")) != "cuda":
            raise ValueError(f"Non-formal GEARS output discovered under canonical root: {path}")
    records = gears_records
    records.extend(metric_record(path) for path in baseline_paths)
    metrics = pd.DataFrame(records).sort_values(["dataset_id", "seed", "model"]).reset_index(drop=True)
    if metrics["gene_count"].nunique() != 1:
        raise ValueError("Comparable external model summary received mixed gene counts.")
    expected = set(metrics.loc[metrics["model"].eq("gears"), ["dataset_id", "seed"]].itertuples(index=False, name=None))
    for model, group in metrics.groupby("model"):
        observed = set(group[["dataset_id", "seed"]].itertuples(index=False, name=None))
        if observed != expected:
            raise ValueError(f"{model} lacks shared-panel cohort/seed rows.")
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            n_runs=("model", "size"),
            n_cohorts=("dataset_id", "nunique"),
            gene_count=("gene_count", "first"),
            cosine_mean=("mean_cosine_similarity_non_control_test", "mean"),
            cosine_median=("mean_cosine_similarity_non_control_test", "median"),
            mse_mean=("mse_non_control_test", "mean"),
            mae_mean=("mae_non_control_test", "mean"),
        )
        .sort_values("cosine_mean", ascending=False)
        .reset_index(drop=True)
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_DIR / "external_condition_holdout_shared_panel_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "external_condition_holdout_shared_panel_summary.csv", index=False)
    print(f"wrote {len(metrics)} comparable runs across {summary['n_cohorts'].iloc[0]} cohorts")


if __name__ == "__main__":
    main()
