"""Fit and audit the outcome-free reliability transport predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ptl.reliability.transport import leave_target_environment_out, PAIR_DESCRIPTOR_COLUMNS  # noqa: E402


def run(root: Path, atlas_path: Path, output_path: Path, summary_path: Path, *, permutation_repeats: int = 1000) -> dict[str, Any]:
    atlas = pd.read_csv(atlas_path)
    pairs = atlas.loc[atlas["baseline"].eq("u_only_rf")].copy()
    pair_keys = ["source_environment_id", "target_environment_id"]
    if "split_seed" in pairs.columns:
        pair_keys.append("split_seed")
    if pairs.duplicated(pair_keys).any():
        raise ValueError("U-only transport atlas must contain one row per source-target pair and split")
    if (pairs["source_environment_id"].astype(str) == pairs["target_environment_id"].astype(str)).any():
        raise ValueError("transport predictor cannot train or select a self-source pair")
    required = set(PAIR_DESCRIPTOR_COLUMNS) | set(pair_keys) | {"transfer_gain"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"transport atlas is missing required fields: {sorted(missing)}")
    scored, summary = leave_target_environment_out(pairs, permutation_repeats=permutation_repeats)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.sort_values(["target_environment_id", "source_environment_id"], kind="stable").to_csv(output_path, index=False)
    selection = pd.DataFrame(summary.pop("selection"))
    selection_path = output_path.with_name("formal_v2_transport_source_selection.csv")
    selection.to_csv(selection_path, index=False)
    selection_detail = pd.DataFrame(summary.pop("source_selection_detail"))
    selection_detail_path = output_path.with_name("formal_v2_transport_source_selection_metrics.csv")
    selection_detail.sort_values(
        [column for column in ["target_environment_id", "split_seed"] if column in selection_detail.columns],
        kind="stable",
    ).to_csv(selection_detail_path, index=False)
    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "descriptor_policy": "deployment features only; strict target-unseen source and target exclusion",
        "gain_baseline": "target-excluded pooled U-only RF",
        "descriptor_columns": list(PAIR_DESCRIPTOR_COLUMNS),
        "source_atlas": atlas_path.relative_to(root).as_posix(),
        "prediction_path": output_path.relative_to(root).as_posix(),
        "selection_path": selection_path.relative_to(root).as_posix(),
        "selection_detail_path": selection_detail_path.relative_to(root).as_posix(),
        "status": "outcome_free_transport_predictor_executed",
        **summary,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    return result


def combine_split_atlases(root: Path, split_config: Path) -> Path:
    """Materialize the audited five-split atlas used by transport evaluation."""

    payload = yaml.safe_load(split_config.read_text(encoding="utf-8"))
    seeds = [int(payload["split_seed"]), *(int(value) for value in payload.get("stability_seeds", []))]
    if len(seeds) != len(set(seeds)):
        raise ValueError("transport split configuration contains duplicate seeds")
    manifest_dir = root / "artifacts/manifests"
    frames: list[pd.DataFrame] = []
    expected_keys: set[tuple[int, str, str, str]] | None = None
    for seed in seeds:
        suffix = "" if seed == 20260907 else f"__split_{seed}"
        path = manifest_dir / f"formal_v2_environment_transfer_matrix{suffix}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing transfer atlas for split_seed={seed}: {path}")
        frame = pd.read_csv(path)
        if "split_seed" not in frame.columns:
            frame.insert(0, "split_seed", seed)
        observed = set(zip(
            pd.to_numeric(frame["split_seed"], errors="raise").astype(int),
            frame["baseline"].astype(str),
            frame["source_environment_id"].astype(str),
            frame["target_environment_id"].astype(str),
        ))
        if any(key[0] != seed for key in observed):
            raise AssertionError(f"atlas {path} contains inconsistent split_seed values")
        current_keys = {(seed, baseline, source, target) for _, baseline, source, target in observed}
        if expected_keys is None:
            expected_keys = {(seed, baseline, source, target) for _, baseline, source, target in observed}
        else:
            prior_surface = {(baseline, source, target) for _, baseline, source, target in expected_keys}
            current_surface = {(baseline, source, target) for _, baseline, source, target in current_keys}
            if current_surface != prior_surface:
                raise AssertionError(f"split_seed={seed} has a different transport atlas surface")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    output = manifest_dir / "formal_v2_environment_transfer_matrix_all_splits.csv"
    combined.to_csv(output, index=False)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--atlas", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--permutation-repeats", type=int, default=1000)
    parser.add_argument("--all-splits", action="store_true", help="combine primary and configured stability split atlases")
    parser.add_argument("--split-config", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    atlas_path = (args.atlas or root / "artifacts/manifests/formal_v2_environment_transfer_matrix.csv").resolve()
    if args.all_splits:
        if args.atlas is not None:
            raise ValueError("--all-splits and --atlas are mutually exclusive")
        split_config = (args.split_config or root / "configs/biological_instance_split.yaml").resolve()
        atlas_path = combine_split_atlases(root, split_config)
    result = run(
        root,
        atlas_path,
        (args.output or root / "artifacts/manifests/formal_v2_transport_predictor_predictions.csv").resolve(),
        (args.summary or root / "artifacts/manifests/formal_v2_transport_predictor_summary.json").resolve(),
        permutation_repeats=args.permutation_repeats,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
