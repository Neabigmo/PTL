"""Aggregate the executed source-to-target atlas across frozen split seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SEED = 20260907


def configured_seeds(path: Path) -> list[int]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    seeds = [int(payload["split_seed"]), *(int(value) for value in payload.get("stability_seeds", []))]
    if seeds[0] != PRIMARY_SEED or len(seeds) != len(set(seeds)):
        raise ValueError("split configuration must keep the primary seed and unique stability seeds")
    return seeds


def build(root: Path, seeds: list[int]) -> dict[str, Any]:
    manifest_dir = root / "artifacts/manifests"
    frames: list[pd.DataFrame] = []
    expected_pairs: set[tuple[str, str, str]] | None = None
    for seed in seeds:
        suffix = "" if seed == PRIMARY_SEED else f"__split_{seed}"
        path = manifest_dir / f"formal_v2_environment_transfer_matrix{suffix}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing executed transfer atlas for split_seed={seed}: {path}")
        frame = pd.read_csv(path)
        required = {"source_environment_id", "target_environment_id", "baseline", "transfer_gain"}
        if required.difference(frame.columns):
            raise ValueError(f"transfer atlas is missing columns: {sorted(required.difference(frame.columns))}")
        keys = set(zip(
            frame["baseline"].astype(str),
            frame["source_environment_id"].astype(str),
            frame["target_environment_id"].astype(str),
        ))
        if expected_pairs is None:
            expected_pairs = keys
        elif keys != expected_pairs:
            raise AssertionError(f"split_seed={seed} has a different source-target atlas surface")
        if "split_seed" in frame.columns:
            observed_seeds = set(pd.to_numeric(frame["split_seed"], errors="raise").astype(int).unique())
            if observed_seeds != {int(seed)}:
                raise AssertionError(
                    f"split_seed={seed} matrix carries inconsistent split_seed values: {sorted(observed_seeds)}"
                )
            frame["split_seed"] = int(seed)
        else:
            frame.insert(0, "split_seed", int(seed))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    pair_count = int(combined[["source_environment_id", "target_environment_id"]].drop_duplicates().shape[0])
    group_columns = ["baseline", "source_environment_id", "target_environment_id"]
    stability = combined.groupby(group_columns, as_index=False).agg(
        split_seed_count=("split_seed", "nunique"),
        transfer_gain_mean=("transfer_gain", "mean"),
        transfer_gain_std=("transfer_gain", "std"),
        transfer_gain_median=("transfer_gain", "median"),
        negative_transfer_rate=("transfer_gain", lambda values: float((values < 0).mean())),
    )
    stability["transfer_gain_std"] = stability["transfer_gain_std"].fillna(0.0)
    output = manifest_dir / "formal_v2_environment_transfer_stability.csv"
    combined_output = manifest_dir / "formal_v2_environment_transfer_stability_raw.csv"
    stability.to_csv(output, index=False)
    combined.to_csv(combined_output, index=False)
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "split_seeds": seeds,
        "atlas_pair_count_per_baseline": pair_count,
        "raw_rows": int(len(combined)),
        "summary_rows": int(len(stability)),
        "matrix_path": output.relative_to(root).as_posix(),
        "raw_path": combined_output.relative_to(root).as_posix(),
        "status": "formal_v2_environment_transfer_stability_executed",
    }
    (manifest_dir / "formal_v2_environment_transfer_stability.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-config", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    config = (args.split_config or root / "configs/biological_instance_split.yaml").resolve()
    print(json.dumps(build(root, configured_seeds(config)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
