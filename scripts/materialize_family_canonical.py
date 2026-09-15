"""Materialize a family canonical surface from a merged measurement report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_frangieh_frozen_family_measurement import _canonical_surface  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--family", required=True)
    parser.add_argument("--measurement-report", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    report_path = root / args.measurement_report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    surface, same_context = _canonical_surface(root, family=args.family, measurement_report=report)
    outdir = root / "artifacts/manifests/predictor_families_v2"
    outdir.mkdir(parents=True, exist_ok=True)
    family = args.family.strip().lower().replace("-", "_").replace(" ", "_")
    surface_path = outdir / f"{family}_d_adj_fullsize.csv"
    floor_path = outdir / f"{family}_same_context_floor.csv"
    output_report = outdir / f"{family}_d_adj_fullsize.json"
    surface.to_csv(surface_path, index=False)
    same_context.to_csv(floor_path, index=False)
    result = {
        "schema_version": 1,
        "status": "executed",
        "family": family,
        "canonical_contract": {
            "rows": int(len(surface)),
            "directed_transfer_count": 6,
            "metric_count": 3,
            "minimum_strict_support": int(surface["minimum_strict_support"].min()),
            "measurement_seeds": 30,
            "measurement_replicates_per_seed": 2,
            "target_outcomes_used_for_fit_or_tuning": False,
        },
        "measurement_report": report,
        "outputs": {
            "d_adj_fullsize": surface_path.relative_to(root).as_posix(),
            "same_context_floor": floor_path.relative_to(root).as_posix(),
        },
    }
    output_report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
