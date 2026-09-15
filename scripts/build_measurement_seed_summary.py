"""Aggregate matched-fixed measurement-depth items to seed-level estimates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    cols = [
        "source_environment_id", "left_target_environment_id", "right_target_environment_id",
        "metric", "split_seed", "cell_budget_label", "perturbation_label", "identifiable_divergence",
    ]
    pieces = []
    for chunk in pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_items.csv", usecols=cols, low_memory=False, chunksize=250_000):
        keep = chunk.loc[
            (chunk["source_environment_id"] == "frangieh_melanoma_control")
            & (chunk["left_target_environment_id"] == "frangieh_melanoma_control")
            & (chunk["right_target_environment_id"] == "frangieh_melanoma_ifng")
        ].copy()
        if not keep.empty:
            pieces.append(keep)
    if not pieces:
        raise RuntimeError("No primary matched-fixed measurement rows found")
    data = pd.concat(pieces, ignore_index=True)
    data["identifiable_divergence"] = pd.to_numeric(data["identifiable_divergence"], errors="coerce")
    out = data.groupby(["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "cell_budget_label"], as_index=False).agg(
        identifiable_divergence=("identifiable_divergence", "mean"),
        n_perturbations=("perturbation_label", "nunique"),
    )
    out["cell_budget_label"] = out["cell_budget_label"].astype(str).str.lower()
    order = {"10": 1, "20": 2, "40": 3, "80": 4, "160": 5, "full": 6}
    out["_order"] = out["cell_budget_label"].map(order)
    out = out.sort_values(["metric", "split_seed", "_order"], kind="stable").drop(columns="_order")
    path = manifests / "measurement_seed_summary.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path.relative_to(root)} ({len(out)} rows)")
    return {"status": "executed", "rows": int(len(out)), "seeds": int(out["split_seed"].nunique()), "output": path.relative_to(root).as_posix()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
