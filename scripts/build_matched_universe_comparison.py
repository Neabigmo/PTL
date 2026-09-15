"""Compare full-size and split-half estimates on identical Nadig label universes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_claim_lock_replication import (  # noqa: E402
    CONTEXTS,
    METRICS,
    MIN_CELLS_PRIMARY,
    MIN_CELLS_SENSITIVITY,
    _budgets,
    _load_context,
)
from scripts.run_formal_v2_claim_lock_measurement import (  # noqa: E402
    _bootstrap_floor,
    _synchronized_macro_rows,
)


INPUT_FIELDS = (
    "cross_left_a", "cross_right_a", "cross_left_b", "cross_right_b",
    "meas_left_a", "meas_left_b", "meas_right_a", "meas_right_b",
    "joint_left_a", "joint_left_b", "joint_right_a", "joint_right_b",
)
RAW_PATHS = {
    "nadig_hepg2": "data/raw/scperturb_v1.4/NadigOConner2024_hepg2.h5ad",
    "nadig_jurkat": "data/raw/scperturb_v1.4/NadigOConner2024_jurkat.h5ad",
}
SPLIT_SEED = 20260908
BOOTSTRAP_DRAWS = 2000


def _labels(root: Path) -> list[str]:
    contexts = {name: _load_context(root / RAW_PATHS[name]) for name in CONTEXTS}
    return sorted(set(contexts[CONTEXTS[0]]["target_counts"]) & set(contexts[CONTEXTS[1]]["target_counts"]))


def _split_eligible(root: Path, labels: list[str], minimum: int) -> list[str]:
    contexts = {name: _load_context(root / RAW_PATHS[name]) for name in CONTEXTS}
    budgets = _budgets(contexts, labels, minimum, resampling_mode="split_half")
    return [label for label in labels if label in budgets and "control" in budgets[label]]


def _fullsize_eligible(root: Path, labels: list[str]) -> list[str]:
    contexts = {name: _load_context(root / RAW_PATHS[name]) for name in CONTEXTS}
    budgets = _budgets(contexts, labels, MIN_CELLS_PRIMARY, resampling_mode="full_size_nonparametric")
    selected = [label for label in labels if label in budgets and "control" in budgets[label]]
    if len(selected) != 2086:
        raise ValueError(f"full-size eligibility produced {len(selected)} labels, expected 2086")
    return selected


def _records(root: Path) -> list[dict[str, Any]]:
    report_path = root / "artifacts/manifests/formal_v2_claim_lock_replication_nadig_fullsize.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report.get("bootstrap_input_files") or []
    if not records:
        raise FileNotFoundError("full-size Nadig bootstrap chunk manifest is missing")
    return records


def _load_filtered_inputs(root: Path, records: list[dict[str, Any]], encoded: str, mask: np.ndarray) -> list[dict[str, np.ndarray]]:
    result: list[dict[str, np.ndarray]] = []
    for record in records:
        with np.load(root / record["path"], allow_pickle=False) as archive:
            raw = {field: np.asarray(archive[f"{encoded}__{field}"]) for field in INPUT_FIELDS}
            if any(array.ndim < 2 or array.shape[-1] != mask.size for array in raw.values()):
                raise ValueError(f"mask length {mask.size} does not match full-size input columns for {encoded}")
            arrays = {field: raw[field][..., mask] for field in INPUT_FIELDS}
            n_rows = arrays["cross_left_a"].shape[0]
            result.extend([{field: arrays[field][index] for field in INPUT_FIELDS} for index in range(n_rows)])
    if len(result) != 30:
        raise ValueError(f"expected 30 full-size seed rows for {encoded}, got {len(result)}")
    return result


def _summary_row(stem: str, minimum: int, source: str, metric: str, n_labels: int, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": "nadig_hepg2_vs_jurkat",
        "estimand": "matched_universe_source_frozen_replication",
        "analysis_label": stem,
        "source_context_id": source,
        "left_target_context_id": CONTEXTS[0],
        "right_target_context_id": CONTEXTS[1],
        "metric": metric,
        "n_split_seeds": 30,
        "n_perturbations": int(n_labels),
        "eligibility_min_cells": int(minimum),
        "comparison_universe": "same labels as split-half sensitivity; full-size nonparametric truth pseudoreplicates",
        "split_seed_start": SPLIT_SEED,
        "split_seed_end": SPLIT_SEED + 29,
        "bootstrap_unit": "matched perturbation label; synchronized draw index; one measurement seed and one unordered model-member pair",
        "measurement_definition": "fixed source-frozen mean prediction versus independent full-size with-replacement raw-cell truth pseudoreplicates",
        "joint_definition": "source-frozen model member pairs crossed with independent full-size truth pseudoreplicates",
        "ordering_estimator": "pairwise-order disagreement with U-statistic within-context floors",
        "rank_displacement_estimator": "normalized rank displacement; secondary magnitude diagnostic only",
        "metric_entrypoint": "same delta cosine, pinned Systema centroid-accuracy, and absolute-effect-rank implementation",
        "refit_per_metric": 0,
        **stats,
    }


def build(root: Path) -> dict[str, Any]:
    root = root.resolve()
    labels = _labels(root)
    fullsize_labels = _fullsize_eligible(root, labels)
    records = _records(root)
    rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for minimum, stem in ((MIN_CELLS_PRIMARY, "matched1255"), (MIN_CELLS_SENSITIVITY, "matched345")):
        selected = _split_eligible(root, labels, minimum)
        if len(selected) not in {1255, 345}:
            raise ValueError(f"matched-universe eligibility produced {len(selected)} labels for minimum={minimum}")
        selected_hash = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()
        selected_set = set(selected)
        mask = np.asarray([label in selected_set for label in fullsize_labels], dtype=bool)
        manifest_rows.append({"analysis_label": stem, "minimum_cells": minimum, "n_perturbations": len(selected), "label_sha256": selected_hash})
        for source in CONTEXTS:
            for metric in METRICS:
                encoded = f"{MIN_CELLS_PRIMARY}__{source}__{metric}"
                inputs = _load_filtered_inputs(root, records, encoded, mask)
                stats = _bootstrap_floor(
                    inputs,
                    seed=SPLIT_SEED + sum(ord(char) for char in f"nadig|matched|{minimum}|{source}|{metric}"),
                    draws=BOOTSTRAP_DRAWS,
                    return_draws=True,
                )
                macro_rows.append({"analysis_label": stem, "source_context_id": source, "metric": metric, "draws": stats.pop("_draws")})
                rows.append(_summary_row(stem, minimum, source, metric, len(selected), stats))
    output_dir = root / "artifacts/manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("matched1255", "matched345"):
        pd.DataFrame([row for row in rows if row["analysis_label"] == stem]).to_csv(output_dir / f"formal_v2_claim_lock_replication_nadig_{stem}.csv", index=False)
        pd.DataFrame([row for row in _synchronized_macro_rows(macro_rows, group_columns=("analysis_label", "metric")) if row["analysis_label"] == stem]).to_csv(output_dir / f"formal_v2_claim_lock_replication_nadig_{stem}_macro.csv", index=False)
    label_path = output_dir / "formal_v2_claim_lock_replication_nadig_matched_universe_labels.json"
    label_path.write_text(json.dumps({"schema_version": 1, "source_labels": labels, "universes": manifest_rows}, indent=2) + "\n", encoding="utf-8")
    ordering = pd.DataFrame(rows)
    ordering_path = output_dir / "ordering_identifiability_nadig.csv"
    ordering.to_csv(ordering_path, index=False)
    report = {
        "schema_version": 1,
        "purpose": "Nadig matched-universe ordering-identifiability audit",
        "primary_estimand": "pairwise_order_disagreement",
        "secondary_estimand": "normalized_rank_displacement",
        "universes": manifest_rows,
        "rows": int(len(ordering)),
        "source_summary": ordering_path.relative_to(root).as_posix(),
        "label_manifest": label_path.relative_to(root).as_posix(),
        "uncertainty_scope": "raw-cell/pseudoreplicate sampling uncertainty; not all biological variation",
    }
    (output_dir / "ordering_identifiability_nadig.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.root), indent=2))


if __name__ == "__main__":
    main()
