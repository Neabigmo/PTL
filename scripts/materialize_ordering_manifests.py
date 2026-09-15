"""Materialize canonical Frangieh and synthetic ordering-identifiability manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _frangieh(root: Path) -> dict[str, object]:
    manifest_dir = root / "artifacts/manifests"
    summary = pd.read_csv(manifest_dir / "formal_v2_claim_lock_measurement_fullsize_summary.csv")
    ordering = pd.read_csv(manifest_dir / "formal_v2_claim_lock_measurement_fullsize_ordering.csv")
    key = ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"]
    stable_columns = [column for column in ordering.columns if column.startswith("stable_") or column.startswith("crossfit_") or column in {"n_items", "n_pairs", "n_replicates_left", "n_replicates_right", "minimum_strict_support", "ordering_estimand"}]
    merged = summary.merge(ordering[key + stable_columns], on=key, how="left", suffixes=("", "_ordering"))
    merged["bootstrap_uncertainty_scope"] = "raw-cell/pseudoreplicate sampling uncertainty; not all biological variation"
    merged["rank_displacement_role"] = "secondary magnitude diagnostic"
    out = manifest_dir / "ordering_identifiability_frangieh.csv"
    merged.to_csv(out, index=False)
    report = {
        "schema_version": 1,
        "purpose": "Frangieh ordering-identifiability manifest",
        "primary_estimand": "pairwise_order_disagreement",
        "secondary_estimand": "normalized_rank_displacement",
        "rows": int(len(merged)),
        "summary_path": out.relative_to(root).as_posix(),
        "macro_path": "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_macro.csv",
        "uncertainty_scope": "raw-cell/pseudoreplicate sampling uncertainty; not all biological variation",
    }
    (manifest_dir / "ordering_identifiability_frangieh.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    print(json.dumps({"frangieh": _frangieh(root)}, indent=2))


if __name__ == "__main__":
    main()
