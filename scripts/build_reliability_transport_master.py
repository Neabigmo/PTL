"""Materialize the single master table and story claim registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MASTER_COLUMNS = (
    "dataset", "evidence_tier", "source_environment_id", "left_target_environment_id", "right_target_environment_id",
    "predictor", "regime", "metric", "cell_budget", "cell_budget_label", "label_universe", "n",
    "observed_D", "observed_D_ci_low", "observed_D_ci_high", "measurement_identifiable", "measurement_identifiable_ci_low",
    "measurement_identifiable_ci_high", "joint_identifiable", "joint_identifiable_ci_low", "joint_identifiable_ci_high",
    "stable_fraction_both", "stable_inversion_fraction", "top_k_retention", "normalized_regret", "excess_regret",
    "checksums", "provenance", "status",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty_record() -> dict:
    return {column: np.nan for column in MASTER_COLUMNS}


def _canonical_rows(manifests: Path) -> pd.DataFrame:
    atlas = pd.read_csv(manifests / "reliability_transport_atlas.csv")
    ordering = pd.read_csv(manifests / "formal_v2_claim_lock_measurement_fullsize_ordering.csv")
    rows: list[dict] = []
    ordering_key = ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"]
    ordering = ordering[ordering_key + ["stable_order_inversion_fraction"]]
    merged = atlas.merge(ordering, on=ordering_key, how="left")
    for record in merged.to_dict("records"):
        row = _empty_record()
        row.update({
            "dataset": record.get("dataset"),
            "evidence_tier": record.get("evidence_tier"),
            "source_environment_id": record.get("source_environment_id"),
            "left_target_environment_id": record.get("left_target_environment_id"),
            "right_target_environment_id": record.get("right_target_environment_id"),
            "predictor": "source_frozen_ensemble",
            "regime": record.get("availability"),
            "metric": record.get("metric"),
            "cell_budget": "full",
            "cell_budget_label": "full",
            "label_universe": "canonical matched perturbation universe",
            "n": record.get("n_perturbations_primary_min"),
            "observed_D": record.get("observed_D"),
            "observed_D_ci_low": record.get("observed_D_ci_low"),
            "observed_D_ci_high": record.get("observed_D_ci_high"),
            "measurement_identifiable": record.get("measurement_identifiable"),
            "measurement_identifiable_ci_low": record.get("measurement_identifiable_ci_low"),
            "measurement_identifiable_ci_high": record.get("measurement_identifiable_ci_high"),
            "joint_identifiable": record.get("joint_identifiable"),
            "joint_identifiable_ci_low": record.get("joint_identifiable_ci_low"),
            "joint_identifiable_ci_high": record.get("joint_identifiable_ci_high"),
            "stable_fraction_both": record.get("stable_fraction_both"),
            "stable_inversion_fraction": record.get("stable_order_inversion_fraction"),
            "checksums": _sha256(manifests / str(record.get("provenance")).split("/")[-1]) if (manifests / str(record.get("provenance")).split("/")[-1]).is_file() else "unavailable",
            "provenance": record.get("provenance"),
            "status": record.get("status", "available"),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _depth_rows(manifests: Path) -> pd.DataFrame:
    summary_path = manifests / "reliability_transport_measurement_depth_summary.csv"
    if not summary_path.is_file():
        return pd.DataFrame(columns=MASTER_COLUMNS)
    frame = pd.read_csv(summary_path)
    rows: list[dict] = []
    for record in frame.to_dict("records"):
        row = _empty_record()
        row.update({
            "dataset": "FrangiehIzar2021_RNA",
            "evidence_tier": 2,
            "source_environment_id": record.get("source_environment_id"),
            "left_target_environment_id": record.get("left_target_environment_id"),
            "right_target_environment_id": record.get("right_target_environment_id"),
            "predictor": "source_frozen_ensemble",
            "regime": "measurement_depth",
            "metric": record.get("metric"),
            "cell_budget": record.get("cell_budget"),
            "cell_budget_label": record.get("cell_budget_label", record.get("cell_budget")),
            "label_universe": "fixed matched universe at cell budget",
            "n": record.get("n_perturbations_mean"),
            "observed_D": record.get("cross_disagreement"),
            "observed_D_ci_low": record.get("cross_disagreement_ci_low"),
            "observed_D_ci_high": record.get("cross_disagreement_ci_high"),
            "measurement_identifiable": record.get("identifiable_divergence"),
            "measurement_identifiable_ci_low": record.get("identifiable_divergence_ci_low"),
            "measurement_identifiable_ci_high": record.get("identifiable_divergence_ci_high"),
            "stable_fraction_both": record.get("stable_pair_fraction"),
            "checksums": _sha256(summary_path),
            "provenance": summary_path.relative_to(ROOT).as_posix(),
            "status": "available",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build(root: Path = ROOT) -> tuple[pd.DataFrame, dict]:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    master = pd.concat([_canonical_rows(manifests), _depth_rows(manifests)], ignore_index=True, sort=False)
    for column in MASTER_COLUMNS:
        if column not in master:
            master[column] = np.nan
    master = master.loc[:, list(MASTER_COLUMNS)].sort_values(
        ["dataset", "source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "regime", "cell_budget_label"],
        kind="stable",
    ).reset_index(drop=True)
    claims = {
        "schema_version": 1,
        "status": "locked_to_master_table",
        "selection_policy": "claim text may cite only rows with status available and a non-unavailable provenance/checksum",
        "claims": [
            {
                "claim_id": "C1_transport_reordering_is_observed",
                "claim": "Cross-context ordering disagreement is measurable under the prespecified pairwise-order estimand.",
                "support_filter": {"regime": "canonical_fullsize_claim_lock", "status": "available"},
                "scope": "descriptive transportability claim; not a universal biological law",
            },
            {
                "claim_id": "C2_measurement_depth_is_a_boundary",
                "claim": "The identifiable component is reported only after subtracting U-statistic measurement floors, with depth-dependent resolution made explicit.",
                "support_filter": {"regime": "measurement_depth", "status": "available"},
                "scope": "Frangieh source-frozen prediction and raw-cell contract",
            },
            {
                "claim_id": "C3_decision_consequence_is_budget_specific",
                "claim": "Transporting a source ordering has budget-specific retention and regret consequences; a global ordering statistic is not itself a deployment utility.",
                "support_filter": {"regime": "measurement_depth", "status": "available"},
                "scope": "top-k budgets only; no decision claim outside evaluated universes",
            },
            {
                "claim_id": "C4_near_chance_is_valid",
                "claim": "Source-only predictability may remain weak after nested calibration; weak prediction is reported as a result rather than converted into a positive mechanistic claim.",
                "support_filter": {"regime": "prospective_source_only", "status": "available"},
                "scope": "only if a source-only held-out manifest is present",
            },
        ],
    }
    return master, claims


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    manifests = args.root.resolve() / "artifacts/manifests"
    master, claims = build(args.root)
    master_path = manifests / "reliability_transport_master.csv"
    claims_path = manifests / "story_claims.json"
    master.to_csv(master_path, index=False)
    claims_path.write_text(json.dumps(claims, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"master_rows": len(master), "claim_count": len(claims["claims"]), "master": master_path.as_posix(), "claims": claims_path.as_posix()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
