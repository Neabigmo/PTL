"""Build the canonical reliability transportability atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.reliability_transport_atlas import atlas_report, build_atlas  # noqa: E402


def run(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    registry_path = manifests / "environment_registry.csv"
    frangieh_path = manifests / "formal_v2_claim_lock_measurement_fullsize_summary.csv"
    ordering_path = manifests / "formal_v2_claim_lock_measurement_fullsize_ordering.csv"
    nadig_path = manifests / "formal_v2_claim_lock_replication_nadig_fullsize.csv"
    registry = pd.read_csv(registry_path)
    frangieh = pd.read_csv(frangieh_path)
    nadig = pd.read_csv(nadig_path) if nadig_path.is_file() else None
    frozen, atlas = build_atlas(registry=registry, frangieh_summary=frangieh, nadig_summary=nadig, ordering_path=ordering_path)
    registry_out = manifests / "reliability_transport_atlas_registry.csv"
    atlas_out = manifests / "reliability_transport_atlas.csv"
    report_out = manifests / "reliability_transport_atlas.json"
    frozen.to_csv(registry_out, index=False)
    atlas.to_csv(atlas_out, index=False)
    report = atlas_report(atlas, frozen)
    report["outputs"] = {
        "registry": registry_out.relative_to(root).as_posix(),
        "atlas": atlas_out.relative_to(root).as_posix(),
    }
    report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
