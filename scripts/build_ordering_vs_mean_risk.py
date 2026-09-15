"""Build a data-gated comparison of mean risk shift and ordering transport.

The output is exploratory and is not inserted into the manuscript unless its
relationship adds information beyond the primary adjusted-ordering result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    oof = pd.read_csv(manifests / "formal_v2_controlled_shift_oof_perturbations.csv")
    oof["risk_left"] = pd.to_numeric(oof["risk_left"], errors="coerce")
    oof["risk_right"] = pd.to_numeric(oof["risk_right"], errors="coerce")
    means = oof.groupby(["left_environment_id", "right_environment_id"], as_index=False).agg(
        mean_risk_left=("risk_left", "mean"), mean_risk_right=("risk_right", "mean"), n_labels=("perturbation_label", "nunique")
    )
    means["absolute_mean_risk_shift"] = (means["mean_risk_right"] - means["mean_risk_left"]).abs()
    atlas = pd.read_csv(manifests / "reliability_transport_atlas.csv")
    atlas = atlas.loc[atlas["dataset"] == "FrangiehIzar2021_RNA"].copy()
    atlas = atlas.rename(columns={"left_target_environment_id": "left_environment_id", "right_target_environment_id": "right_environment_id"})
    out = atlas[["source_environment_id", "left_environment_id", "right_environment_id", "metric", "measurement_identifiable", "measurement_identifiable_ci_low", "measurement_identifiable_ci_high"]].merge(means, on=["left_environment_id", "right_environment_id"], how="left")
    out["data_gate"] = np.where(out["absolute_mean_risk_shift"].notna() & out["measurement_identifiable"].notna(), "computed_for_review", "incomplete")
    path = manifests / "ordering_vs_mean_risk.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path.relative_to(root)} ({len(out)} rows); exploratory only")
    return {"status": "executed", "rows": int(len(out)), "paper_inclusion": "data-gated; not included automatically", "output": path.relative_to(root).as_posix()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
