from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.evaluate_phase06 import build_run_selection


def test_build_run_selection_uses_manifest_universe_and_latest_registry_status(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "run_id": "run_a",
                "model": "m1",
                "split_json_path": "split_a.json",
                "split_family": "random_split",
                "dataset_scope": "ds",
                "heldout_target": "",
                "seed": 0,
                "gene_space_policy": "native",
                "expected_gene_count": 10,
                "output_dir": str(tmp_path / "run_a"),
                "log_file": str(tmp_path / "run_a.log"),
            },
            {
                "run_id": "run_b",
                "model": "m2",
                "split_json_path": "split_b.json",
                "split_family": "low_support_split",
                "dataset_scope": "ds",
                "heldout_target": "",
                "seed": 0,
                "gene_space_policy": "native",
                "expected_gene_count": 10,
                "output_dir": str(tmp_path / "run_b"),
                "log_file": str(tmp_path / "run_b.log"),
            },
        ]
    )
    registry = pd.DataFrame(
        [
            {
                "run_id": "run_a",
                "timestamp": "2026-04-26T01:00:00+08:00",
                "phase": "Phase 05",
                "dataset": "ds",
                "split": "random_split",
                "model": "m1",
                "seed": 0,
                "command": "cmd",
                "status": "failed",
                "main_metric": None,
                "output_dir": "x",
                "log_file": "x",
                "notes": "first",
            },
            {
                "run_id": "run_a",
                "timestamp": "2026-04-26T02:00:00+08:00",
                "phase": "Phase 05",
                "dataset": "ds",
                "split": "random_split",
                "model": "m1",
                "seed": 0,
                "command": "cmd",
                "status": "cached",
                "main_metric": 0.1,
                "output_dir": "x",
                "log_file": "x",
                "notes": "latest",
            },
            {
                "run_id": "run_b",
                "timestamp": "2026-04-26T02:00:00+08:00",
                "phase": "Phase 05",
                "dataset": "ds",
                "split": "low_support_split",
                "model": "m2",
                "seed": 0,
                "command": "cmd",
                "status": "failed",
                "main_metric": None,
                "output_dir": "x",
                "log_file": "x",
                "notes": "latest failed",
            },
            {
                "run_id": "smoke__extra",
                "timestamp": "2026-04-26T03:00:00+08:00",
                "phase": "Phase 05",
                "dataset": "ds",
                "split": "random_split",
                "model": "m1",
                "seed": 0,
                "command": "cmd",
                "status": "success",
                "main_metric": 0.2,
                "output_dir": "x",
                "log_file": "x",
                "notes": "",
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    registry_path = tmp_path / "registry.csv"
    manifest.to_csv(manifest_path, index=False)
    registry.to_csv(registry_path, index=False)

    selected, audit = build_run_selection(manifest_path, registry_path)

    assert [run.run_id for run in selected] == ["run_a"]
    assert set(audit["run_id"]) == {"run_a", "run_b", "smoke__extra"}
    assert bool(audit.loc[audit["run_id"] == "run_b", "included_in_phase06"].iloc[0]) is False
    assert bool(audit.loc[audit["run_id"] == "smoke__extra", "included_in_phase06"].iloc[0]) is False
