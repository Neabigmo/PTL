"""Link measured split-half reproducibility to selective reliability.

The reproducibility score is joined only for evaluation and stratification; it
is never placed in the deployment feature matrix or used to fit PTL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_reliability import _aurc_only  # noqa: E402

METHODS = ("raw_normalized_uq", "u_only_rf", "ptl_rf")
RAW_METHOD = "raw_normalized_uq"


def _stratum(values: pd.Series) -> pd.Series:
    lower = float(values.quantile(1 / 3))
    upper = float(values.quantile(2 / 3))
    return pd.Series(
        np.where(values <= lower, "low", np.where(values >= upper, "high", "middle")),
        index=values.index,
    )


def _metrics(group: pd.DataFrame, method: str) -> dict[str, float]:
    risk = group["continuous_risk"].to_numpy(dtype=float)
    confidence = group[method].to_numpy(dtype=float)
    aurc = _aurc_only(risk, confidence)
    return {"n": int(len(group)), "aurc": float(aurc), "mean_risk": float(risk.mean())}


def run(root: Path, *, input_path: Path | None = None, reproducibility_path: Path | None = None) -> dict[str, Any]:
    predictions = pd.read_csv(input_path or root / "artifacts/source_data/formal_v2_reliability_predictions.csv")
    reproducibility = pd.read_csv(reproducibility_path or root / "artifacts/manifests/formal_v2_raw_cell_split_half.csv")
    required_repro = {"dataset_id", "perturbation_label", "split_half_cosine"}
    if required_repro.difference(reproducibility.columns):
        raise ValueError("raw-cell reproducibility artifact is missing required join fields")
    frame = predictions.loc[predictions["scenario"].eq("in_domain")].copy()
    joined = frame.merge(
        reproducibility[["dataset_id", "perturbation_label", "split_half_cosine"]],
        on=["dataset_id", "perturbation_label"],
        how="inner",
        validate="many_to_one",
    )
    if joined.empty:
        raise ValueError("no reliability rows joined to measured raw-cell reproducibility")
    joined["reproducibility_stratum"] = _stratum(joined["split_half_cosine"])
    rows: list[dict[str, Any]] = []
    for (dataset_id, predictor, stratum), group in joined.groupby(["dataset_id", "predictor", "reproducibility_stratum"], sort=True):
        for method in METHODS:
            values = _metrics(group, method)
            rows.append({
                "dataset_id": dataset_id,
                "predictor": predictor,
                "reproducibility_stratum": stratum,
                "method": method,
                "reproducibility_min": float(group["split_half_cosine"].min()),
                "reproducibility_max": float(group["split_half_cosine"].max()),
                "reproducibility_mean": float(group["split_half_cosine"].mean()),
                **values,
                "used_in_ptl_features": 0,
                "evaluation_only": 1,
            })
    continuous_rows: list[dict[str, Any]] = []
    for (dataset_id, predictor), group in joined.groupby(["dataset_id", "predictor"], sort=True):
        risk = pd.to_numeric(group["continuous_risk"], errors="coerce")
        for method in METHODS:
            score = pd.to_numeric(group[method], errors="coerce")
            continuous_rows.append({
                "dataset_id": dataset_id,
                "predictor": predictor,
                "method": method,
                "n": int(len(group)),
                "spearman_reproducibility_risk": float(group["split_half_cosine"].corr(risk, method="spearman")),
                "spearman_reproducibility_confidence": float(group["split_half_cosine"].corr(score, method="spearman")),
            })
    result_frame = pd.DataFrame(rows).sort_values(["dataset_id", "predictor", "reproducibility_stratum", "method"], kind="stable")
    continuous_frame = pd.DataFrame(continuous_rows).sort_values(["dataset_id", "predictor", "method"], kind="stable")
    manifest_dir = root / "artifacts/manifests"
    output_path = manifest_dir / "formal_v2_reproducibility_stratified_reliability.csv"
    summary_path = manifest_dir / "formal_v2_reproducibility_stratified_reliability.json"
    continuous_path = manifest_dir / "formal_v2_reproducibility_risk_relationship.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output_path, index=False)
    continuous_frame.to_csv(continuous_path, index=False)
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "join": "dataset_id + perturbation_label",
        "reproducibility_source": "formal_v2_raw_cell_split_half.csv",
        "strata": ["low", "middle", "high"],
        "methods": list(METHODS),
        "evaluation_only": True,
        "raw_cell_rows_joined": int(len(joined)),
        "datasets": sorted(joined["dataset_id"].unique().tolist()),
        "output_path": output_path.relative_to(root).as_posix(),
        "relationship_path": continuous_path.relative_to(root).as_posix(),
        "status": "formal_v2_reproducibility_stratified_reliability_executed",
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve()), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
