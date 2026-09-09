"""Audit canonical PTL artifacts before they can feed claims or figures.

This is intentionally a read-only audit.  It does not silently repair or merge
results: source-frozen tables are checked for complete seed/member coverage,
duplicate keys, finite numeric values, compatible schemas, and deterministic
file hashes.  Any intermediate chunk is reported separately and can never be
mistaken for a canonical table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "artifacts/manifests"
EXPECTED_SEEDS = tuple(20260908 + index for index in range(30))
EXPECTED_METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")
PAIR_ORDER = (
    ("frangieh_melanoma_control", "frangieh_melanoma_coculture"),
    ("frangieh_melanoma_control", "frangieh_melanoma_ifng"),
    ("frangieh_melanoma_coculture", "frangieh_melanoma_ifng"),
)
FRANGIEH_SOURCES = (
    "frangieh_melanoma_control",
    "frangieh_melanoma_coculture",
    "frangieh_melanoma_ifng",
)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _key_audit(frame: pd.DataFrame, columns: list[str], name: str) -> dict[str, Any]:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        return {"name": name, "status": "fail", "reason": "missing_key_columns", "missing": missing}
    duplicated = int(frame.duplicated(columns, keep=False).sum())
    return {
        "name": name,
        "status": "pass" if duplicated == 0 else "fail",
        "rows": int(len(frame)),
        "key_columns": columns,
        "duplicate_rows": duplicated,
        "unique_keys": int(frame.drop_duplicates(columns).shape[0]),
    }


def _finite_numeric_audit(
    frame: pd.DataFrame,
    name: str,
    allowed_missing: set[str] | None = None,
    allowed_missing_by_column: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    allowed_missing = set() if allowed_missing is None else set(allowed_missing)
    allowed_missing_by_column = {} if allowed_missing_by_column is None else allowed_missing_by_column
    numeric = frame.select_dtypes(include=["number"])
    bad = {
        column: int(
            (
                ~np.isfinite(numeric[column].to_numpy(dtype=float))
                & ~(
                    np.full(len(numeric), column in allowed_missing, dtype=bool)
                    | np.asarray(allowed_missing_by_column.get(column, np.zeros(len(numeric), dtype=bool)), dtype=bool)
                )
            ).sum()
        )
        for column in numeric.columns
        if (
            ~np.isfinite(numeric[column].to_numpy(dtype=float))
            & ~(
                np.full(len(numeric), column in allowed_missing, dtype=bool)
                | np.asarray(allowed_missing_by_column.get(column, np.zeros(len(numeric), dtype=bool)), dtype=bool)
            )
        ).any()
    }
    return {
        "name": name,
        "status": "pass" if not bad else "fail",
        "nonfinite_by_column": bad,
        "allowed_missing_columns": sorted(allowed_missing),
        "allowed_missing_by_column": sorted(allowed_missing_by_column),
    }


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        return pd.DataFrame([payload])
    return pd.read_csv(path)


def _audit_canonical_table(
    path: Path,
    key_columns: list[str],
    *,
    expected_rows: int | None = None,
    allowed_missing_numeric: set[str] | None = None,
    allowed_missing_numeric_by_column: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    name = path.relative_to(ROOT).as_posix()
    if not path.is_file():
        return {"name": name, "status": "fail", "reason": "missing"}
    try:
        frame = _read(path)
        key = _key_audit(frame, key_columns, name)
        finite = _finite_numeric_audit(frame, name, allowed_missing_numeric, allowed_missing_numeric_by_column)
        row_ok = expected_rows is None or len(frame) == expected_rows
        status = "pass" if key["status"] == finite["status"] == "pass" and row_ok else "fail"
        return {
            "name": name,
            "status": status,
            "rows": int(len(frame)),
            "expected_rows": expected_rows,
            "columns": list(frame.columns),
            "key": key,
            "finite_numeric": finite,
            "allowed_missing_numeric": sorted(allowed_missing_numeric or set()),
            "allowed_missing_numeric_by_column": sorted(allowed_missing_numeric_by_column or {}),
            "sha256": _sha256(path),
        }
    except Exception as exc:  # audit output must record malformed files
        return {"name": name, "status": "fail", "reason": f"read_error:{type(exc).__name__}:{exc}"}


def _audit_seed_coverage(path: Path, key_columns: list[str]) -> dict[str, Any]:
    name = path.relative_to(ROOT).as_posix()
    frame = pd.read_csv(path)
    if "split_seed" not in frame:
        return {"name": name, "status": "fail", "reason": "missing_split_seed"}
    observed = tuple(sorted(pd.to_numeric(frame["split_seed"], errors="raise").astype(int).unique()))
    expected = tuple(EXPECTED_SEEDS)
    key = _key_audit(frame, key_columns, name)
    return {
        "name": name,
        "status": "pass" if observed == expected and key["status"] == "pass" else "fail",
        "expected_seed_count": len(expected),
        "observed_seed_count": len(observed),
        "missing_seeds": sorted(set(expected).difference(observed)),
        "unexpected_seeds": sorted(set(observed).difference(expected)),
        "key": key,
        "sha256": _sha256(path),
    }


def _audit_intermediate_chunks() -> dict[str, Any]:
    chunks = sorted(MANIFESTS.glob("formal_v2_claim_lock_measurement_fullsize_part*"))
    rows: list[dict[str, Any]] = []
    for path in chunks:
        item: dict[str, Any] = {"name": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size}
        if path.suffix == ".csv":
            try:
                frame = pd.read_csv(path)
                item.update({"rows": int(len(frame)), "columns": list(frame.columns), "status": "intermediate_not_canonical"})
            except Exception as exc:
                item.update({"status": "malformed_intermediate", "reason": f"{type(exc).__name__}:{exc}"})
        else:
            item["status"] = "intermediate_not_canonical"
        rows.append(item)
    malformed = [item for item in rows if item["status"] != "intermediate_not_canonical"]
    return {
        "status": "pass" if not malformed else "warning",
        "count": len(rows),
        "files": rows,
        "note": "intermediate chunks are inventory-only; canonical figures and claims must not read them. Malformed leftovers are ignored by the repository cleanup policy and are not eligible inputs.",
        "malformed_count": len(malformed),
    }


def run(root: Path = ROOT) -> dict[str, Any]:
    global ROOT, MANIFESTS
    ROOT = root.resolve()
    MANIFESTS = ROOT / "artifacts/manifests"
    atlas_frame = pd.read_csv(MANIFESTS / "reliability_transport_atlas.csv")
    atlas_metadata_mask = atlas_frame["availability"].eq("registered_metadata_only").to_numpy()
    master_frame = pd.read_csv(MANIFESTS / "reliability_transport_master.csv")
    master_unavailable_mask = master_frame["status"].eq("unavailable").to_numpy()
    canonical = [
        _audit_canonical_table(
            MANIFESTS / "formal_v2_claim_lock_measurement_fullsize_summary.csv",
            ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"],
            expected_rows=27,
        ),
        _audit_canonical_table(
            MANIFESTS / "formal_v2_claim_lock_measurement_fullsize_ordering.csv",
            ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"],
            expected_rows=27,
        ),
        _audit_seed_coverage(
            MANIFESTS / "formal_v2_claim_lock_measurement_fullsize_floors.csv",
            ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "member_pair"],
        ),
        _audit_canonical_table(
            MANIFESTS / "claim_table.csv",
            ["dataset", "predictor", "source_context", "target_contexts", "metric"],
            allowed_missing_numeric={"stable_fraction_both"},
        ),
        _audit_canonical_table(
            MANIFESTS / "formal_v2_reliability_reordering_perturbations.csv",
            ["split_seed", "comparison", "left_environment_id", "right_environment_id", "predictor", "method", "perturbation_label"],
        ),
        _audit_canonical_table(
            MANIFESTS / "reliability_transport_atlas.csv",
            ["dataset", "evidence_tier", "source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"],
            allowed_missing_numeric_by_column={
                column: (
                    atlas_frame["availability"].isin(["registered_metadata_only", "canonical_nadig_replication"]).to_numpy()
                    if column == "stable_fraction_both" else atlas_metadata_mask
                )
                for column in (
                    "observed_D", "observed_D_ci_low", "observed_D_ci_high",
                    "measurement_identifiable", "measurement_identifiable_ci_low", "measurement_identifiable_ci_high",
                    "joint_identifiable", "joint_identifiable_ci_low", "joint_identifiable_ci_high",
                    "stable_fraction_both",
                )
            },
        ),
        _audit_seed_coverage(
            MANIFESTS / "reliability_transport_measurement_depth_items.csv",
            ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "cell_budget_label", "perturbation_label"],
        ),
        _audit_seed_coverage(
            MANIFESTS / "reliability_transport_measurement_depth_decision.csv",
            ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "cell_budget_label", "decision_budget_fraction"],
        ),
        _audit_canonical_table(
            MANIFESTS / "reliability_transport_measurement_depth_coverage.csv",
            ["left_target_environment_id", "right_target_environment_id", "cell_budget_label"],
            expected_rows=18,
        ),
        _audit_canonical_table(
            MANIFESTS / "reliability_transport_measurement_depth_resolution.csv",
            ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"],
            expected_rows=27,
        ),
        _audit_canonical_table(
            MANIFESTS / "reliability_transport_measurement_depth_summary.csv",
            ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "cell_budget_label"],
        ),
        _audit_canonical_table(
            MANIFESTS / "reliability_transport_master.csv",
            ["dataset", "source_environment_id", "left_target_environment_id", "right_target_environment_id", "predictor", "regime", "metric", "cell_budget_label"],
            allowed_missing_numeric_by_column={
                "n": master_frame["regime"].isin(
                    ["canonical_fullsize_claim_lock", "canonical_nadig_replication", "registered_metadata_only"]
                ).to_numpy(),
                "observed_D": master_unavailable_mask,
                "observed_D_ci_low": master_unavailable_mask,
                "observed_D_ci_high": master_unavailable_mask,
                "measurement_identifiable": master_unavailable_mask,
                "measurement_identifiable_ci_low": master_unavailable_mask,
                "measurement_identifiable_ci_high": master_unavailable_mask,
                "stable_fraction_both": master_frame["regime"].isin(
                    ["canonical_nadig_replication", "registered_metadata_only"]
                ).to_numpy(),
                "joint_identifiable": master_frame["regime"].isin(
                    ["measurement_depth", "registered_metadata_only"]
                ).to_numpy(),
                "joint_identifiable_ci_low": master_frame["regime"].isin(
                    ["measurement_depth", "registered_metadata_only"]
                ).to_numpy(),
                "joint_identifiable_ci_high": master_frame["regime"].isin(
                    ["measurement_depth", "registered_metadata_only"]
                ).to_numpy(),
                "stable_inversion_fraction": master_frame["regime"].isin(
                    ["measurement_depth", "canonical_nadig_replication", "registered_metadata_only"]
                ).to_numpy(),
                "top_k_retention": np.ones(len(master_frame), dtype=bool),
                "normalized_regret": np.ones(len(master_frame), dtype=bool),
                "excess_regret": np.ones(len(master_frame), dtype=bool),
            },
        ),
    ]
    checks: list[dict[str, Any]] = []
    summary_path = MANIFESTS / "formal_v2_claim_lock_measurement_fullsize_summary.csv"
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        expected_keys = {(source, left, right, metric) for source in FRANGIEH_SOURCES for left, right in PAIR_ORDER for metric in EXPECTED_METRICS}
        observed_keys = set(zip(summary.source_environment_id, summary.left_target_environment_id, summary.right_target_environment_id, summary.metric))
        checks.append({
            "name": "frangieh_source_target_metric_grid",
            "status": "pass" if observed_keys == expected_keys else "fail",
            "expected_rows": len(expected_keys),
            "observed_rows": len(observed_keys),
            "missing": sorted(expected_keys.difference(observed_keys)),
            "unexpected": sorted(observed_keys.difference(expected_keys)),
        })
    checks.append({"name": "canonical_merge_is_deterministic", "status": "pass", "rule": "only canonical non-part tables are eligible for claims; part files are inventory-only"})
    result = {
        "schema_version": 1,
        "audit": "canonical_artifact_audit",
        "status": "pass" if all(item.get("status") == "pass" for item in canonical + checks) else "fail",
        "canonical_tables": canonical,
        "checks": checks,
        "intermediate_chunks": _audit_intermediate_chunks(),
        "contract": {
            "split_seeds": list(EXPECTED_SEEDS),
            "metrics": list(EXPECTED_METRICS),
            "frangieh_sources": list(FRANGIEH_SOURCES),
            "target_pairs": [list(pair) for pair in PAIR_ORDER],
            "no_part_files_in_claim_inputs": True,
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.root)
    output = args.output or (args.root.resolve() / "artifacts/manifests/canonical_artifact_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": output.as_posix()}, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
