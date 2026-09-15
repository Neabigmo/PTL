"""Summarize strict-support stable inversions from the canonical ordering table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    source = pd.read_csv(manifests / "formal_v2_claim_lock_measurement_fullsize_ordering.csv")
    source = source.loc[
        source["source_environment_id"].astype(str).str.startswith("frangieh_")
        & (pd.to_numeric(source["minimum_strict_support"], errors="coerce") >= 8)
    ].copy()
    keep = [
        "source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric",
        "n_items", "minimum_strict_support", "stable_fraction_both", "stable_order_inversion_fraction",
        "crossfit_heldout_inversion_fraction", "crossfit_heldout_evaluable_fraction",
    ]
    out = source[keep].sort_values(["source_environment_id", "metric"], kind="stable").reset_index(drop=True)
    path = manifests / "stable_inversion_summary.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path.relative_to(root)} ({len(out)} rows)")
    return {"status": "executed", "rows": int(len(out)), "minimum_strict_support": 8, "output": path.relative_to(root).as_posix()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
