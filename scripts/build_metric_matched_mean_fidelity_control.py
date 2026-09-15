"""Build the metric-matched mean-fidelity control for the PTL decision link.

This control uses the already materialized matched-fixed full-depth risk surface.
For every directed source-to-target transfer and metric, the mean-risk shift is
computed over the exact perturbation-label universe used by the corresponding
measurement-adjusted ordering estimand. No target data are used to refit or
select the source predictor; target risks are read only from the canonical
source-frozen risk artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.decision_analysis_common import load_canonical_d_adj  # noqa: E402

METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")
FULL_LABEL = "full"
PRIMARY_BUDGET = 0.10


def _sha256_labels(labels: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(labels)).encode("utf-8")).hexdigest()


def _directed_decision_surface(root: Path) -> pd.DataFrame:
    path = root / "artifacts/manifests/reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv"
    frame = pd.read_csv(path, dtype={"cell_budget_label": "string"})
    frame = frame.loc[
        frame["cell_budget_label"].eq(FULL_LABEL)
        & np.isclose(frame["decision_budget_fraction"].astype(float), PRIMARY_BUDGET)
        & frame["metric"].isin(METRICS)
    ].copy()
    frame["budget_fraction"] = frame["decision_budget_fraction"].astype(float)
    frame["normalized_regret"] = pd.to_numeric(frame["normalized_regret_point"], errors="coerce")
    if frame.duplicated(["source_environment_id", "target_environment_id", "metric"]).any():
        raise RuntimeError("decision surface has duplicate directed transfer x metric keys")
    if len(frame) != 18:
        raise RuntimeError(f"expected 18 primary decision rows, found {len(frame)}")
    return frame[[
        "source_environment_id", "target_environment_id", "metric", "cell_budget_label",
        "decision_budget_fraction", "budget_fraction", "normalized_regret",
        "retention_point", "regret_point",
    ]].copy()


def _risk_rows(root: Path) -> pd.DataFrame:
    path = root / "artifacts/manifests/reliability_transport_measurement_depth_matched_fixed_risks.csv"
    usecols = [
        "source_environment_id", "left_target_environment_id", "right_target_environment_id",
        "metric", "perturbation_label", "cell_budget_label", "left_risk", "right_risk",
    ]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    frame = frame.loc[frame["cell_budget_label"].astype(str).eq(FULL_LABEL) & frame["metric"].isin(METRICS)].copy()
    frame["perturbation_label"] = frame["perturbation_label"].astype(str)
    frame["left_risk"] = pd.to_numeric(frame["left_risk"], errors="coerce")
    frame["right_risk"] = pd.to_numeric(frame["right_risk"], errors="coerce")
    if frame[["left_risk", "right_risk"]].isna().any().any():
        raise RuntimeError("canonical matched-fixed risk surface contains non-finite full-depth risks")
    return frame


def _build_rows(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    risks = _risk_rows(root)
    decision = _directed_decision_surface(root)
    canonical = load_canonical_d_adj(root, METRICS)
    decision = decision.merge(
        canonical.drop(columns=["transfer_id", "source_context_id", "unordered_context_pair_id"]),
        on=["source_environment_id", "target_environment_id", "metric"],
        how="inner",
        validate="one_to_one",
    )
    coverage = pd.read_csv(root / "artifacts/manifests/reliability_transport_measurement_depth_matched_fixed_coverage.csv")
    coverage = coverage.loc[coverage["cell_budget_label"].astype(str).eq(FULL_LABEL)].copy()

    rows: list[dict[str, Any]] = []
    for (source, left, right, metric), group in risks.groupby(
        ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"], sort=True
    ):
        if source not in (left, right):
            continue
        per_label = group.groupby("perturbation_label", sort=True).agg(
            left_risk=("left_risk", "mean"), right_risk=("right_risk", "mean")
        ).reset_index()
        if per_label.empty or per_label["perturbation_label"].duplicated().any():
            raise RuntimeError(f"invalid label aggregation for {source} / {left} / {right} / {metric}")
        cov = coverage.loc[
            coverage["left_target_environment_id"].eq(left)
            & coverage["right_target_environment_id"].eq(right)
        ]
        cov = cov.loc[cov["universe_mode"].eq("matched_fixed")]
        if len(cov) != 1:
            raise RuntimeError(f"missing unique full-depth coverage record for {left} / {right}")
        expected_n = int(cov.iloc[0]["n_labels_matched"])
        labels = per_label["perturbation_label"].tolist()
        if len(labels) != expected_n or _sha256_labels(labels) != str(cov.iloc[0]["matched_universe_sha256"]):
            raise RuntimeError(f"risk labels do not match the canonical fixed universe for {left} / {right}")

        source_is_left = source == left
        target = right if source_is_left else left
        source_mean = float(per_label["left_risk" if source_is_left else "right_risk"].mean())
        target_mean = float(per_label["right_risk" if source_is_left else "left_risk"].mean())
        rows.append({
            "source_environment_id": source,
            "target_environment_id": target,
            "transfer_id": f"{source}->{target}",
            "source_context_id": source,
            "unordered_context_pair_id": "<->".join(sorted((source, target))),
            "left_target_environment_id": left,
            "right_target_environment_id": right,
            "metric": metric,
            "cell_budget_label": FULL_LABEL,
            "n_labels": len(labels),
            "matched_universe_sha256": str(cov.iloc[0]["matched_universe_sha256"]),
            "mean_risk_source": source_mean,
            "mean_risk_target": target_mean,
            "absolute_mean_risk_shift": abs(target_mean - source_mean),
            "risk_aggregation": "mean over perturbation labels after averaging the canonical full-depth measurement replicates",
            "risk_artifact": "artifacts/manifests/reliability_transport_measurement_depth_matched_fixed_risks.csv",
        })

    out = pd.DataFrame(rows)
    if out.duplicated(["source_environment_id", "target_environment_id", "metric"]).any() or len(out) != 18:
        raise RuntimeError(f"expected 18 directed metric-matched mean-fidelity rows, found {len(out)}")
    merged = out.merge(
        decision[
            ["source_environment_id", "target_environment_id", "metric", "d_meas_id", "d_meas_id_ci_low", "d_meas_id_ci_high",
             "normalized_regret", "retention_point", "regret_point"]
        ],
        on=["source_environment_id", "target_environment_id", "metric"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 18:
        raise RuntimeError("metric-matched control did not join one-to-one to the primary decision surface")
    merged["d_meas_id"] = pd.to_numeric(merged["d_meas_id"], errors="coerce")
    merged["d_meas_id_ci_low"] = pd.to_numeric(merged["d_meas_id_ci_low"], errors="coerce")
    merged["d_meas_id_ci_high"] = pd.to_numeric(merged["d_meas_id_ci_high"], errors="coerce")
    merged["normalized_regret"] = pd.to_numeric(merged["normalized_regret"], errors="coerce")
    merged["retention_point"] = pd.to_numeric(merged["retention_point"], errors="coerce")
    merged["regret_point"] = pd.to_numeric(merged["regret_point"], errors="coerce")
    merged = merged.sort_values(["transfer_id", "metric"], kind="stable").reset_index(drop=True)
    report = {
        "status": "executed",
        "surface": "3 unordered context pairs x 2 directions x 3 metrics = 18 directed rows",
        "depth": FULL_LABEL,
        "decision_budget_fraction": PRIMARY_BUDGET,
        "mean_risk_definition": "absolute difference between target and source mean metric risk over the exact matched-fixed perturbation universe",
        "d_adj_source": "corrected canonical summary only",
        "metric_policy": "metric-specific risk vectors; no mean-risk value is replicated across metrics",
        "universe_policy": "the same full-depth matched-fixed label universe used by D_meas-ID for each transfer",
        "outputs": {
            "csv": "artifacts/manifests/metric_matched_mean_fidelity_control.csv",
            "json": "artifacts/manifests/metric_matched_mean_fidelity_control.json",
        },
        "rows": int(len(merged)),
    }
    return merged, report


def run(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    out, report = _build_rows(root)
    manifests = root / "artifacts/manifests"
    csv_path = manifests / "metric_matched_mean_fidelity_control.csv"
    json_path = manifests / "metric_matched_mean_fidelity_control.json"
    out.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
