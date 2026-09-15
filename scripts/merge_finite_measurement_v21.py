"""Merge completed, non-overlapping v2.1 simulation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def merge(*, mode: str = "primary", shard_count: int = 4, root: Path = ROOT) -> dict[str, object]:
    root = root.resolve()
    outdir = root / "artifacts/manifests/simulation_v21"
    csv_parts = [
        outdir / f"finite_measurement_{mode}_v21_part{index:02d}of{shard_count:02d}.csv"
        for index in range(1, shard_count + 1)
    ]
    json_parts = [path.with_suffix(".json") for path in csv_parts]
    if any(not path.is_file() for path in csv_parts + json_parts):
        missing = [str(path) for path in csv_parts + json_parts if not path.is_file()]
        raise FileNotFoundError("incomplete v2.1 shard set: " + ", ".join(missing))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in json_parts]
    expected_count = int(reports[0]["expected_condition_count"])
    if any(int(report["shard_count"]) != shard_count for report in reports):
        raise ValueError("shard_count metadata mismatch")
    intervals = [
        (int(report["condition_start_index"]), int(report["condition_end_index_exclusive"]))
        for report in reports
    ]
    if intervals != sorted(intervals) or intervals[0][0] != 0 or intervals[-1][1] != expected_count:
        raise ValueError(f"shard intervals are not contiguous: {intervals}")
    if any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError(f"shard intervals overlap or have a gap: {intervals}")
    frames = [pd.read_csv(path) for path in csv_parts]
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != expected_count:
        raise ValueError(f"expected {expected_count} rows, found {len(frame)}")
    condition_columns = ["law", "n_items", "depth", "resolution", "noise_scale", "inversion_fraction"]
    if frame.duplicated(condition_columns).any():
        raise ValueError("duplicate simulation condition detected")
    csv_path = outdir / f"finite_measurement_{mode}_v21.csv"
    frame.to_csv(csv_path, index=False)
    report = {
        "schema_version": 2,
        "status": "executed",
        "mode": mode,
        "condition_count": int(len(frame)),
        "expected_condition_count": expected_count,
        "trials_per_condition": int(frame["trials"].iloc[0]),
        "truth_draws_per_condition": int(frame["truth_draws"].iloc[0]),
        "shard_count": int(shard_count),
        "shards": [path.relative_to(root).as_posix() for path in csv_parts],
        "csv": csv_path.relative_to(root).as_posix(),
        "truth_definition": reports[0]["truth_definition"],
        "stable_rule": reports[0]["stable_rule"],
        "interpretation": reports[0]["interpretation"],
    }
    json_path = outdir / f"finite_measurement_{mode}_v21.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**report, "json": json_path.relative_to(root).as_posix()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mode", default="primary")
    parser.add_argument("--shard-count", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(merge(mode=args.mode, shard_count=args.shard_count, root=args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
