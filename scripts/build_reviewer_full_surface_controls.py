"""Materialize full-surface and same-context negative-control evidence.

This script intentionally reuses the completed 30-seed full-size raw-cell
pseudoreplicates.  It does not retrain a predictor or re-read the raw h5ad:
the biological comparison and the negative control therefore share the exact
source-frozen prediction and measurement protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.ordering_estimands import (  # noqa: E402
    pairwise_disagreement,
    pairwise_order_probabilities,
    within_disagreement_u,
)


METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")
PART_PATTERN = re.compile(r"formal_v2_claim_lock_measurement_fullsize_part\d{2}\.json$")


def _estimate(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    _, left_prob = pairwise_order_probabilities(left)
    _, right_prob = pairwise_order_probabilities(right)
    cross = pairwise_disagreement(left_prob, right_prob)
    # This is the same U-statistic floor used by the merged primary analysis,
    # calculated from pairwise direction counts rather than re-enumerating all
    # replicate pairs on every bootstrap draw.
    left_floor = within_disagreement_u(left_prob, n_replicates=left.shape[0])
    right_floor = within_disagreement_u(right_prob, n_replicates=right.shape[0])
    return {
        "cross_pairwise_disagreement": float(cross),
        "within_floor_left": float(left_floor),
        "within_floor_right": float(right_floor),
        "measurement_identifiable": float(cross - .5 * (left_floor + right_floor)),
    }


def _load_inputs(root: Path) -> dict[str, dict[str, list[np.ndarray]]]:
    manifests = root / "artifacts/manifests"
    paths = sorted(path for path in manifests.glob("formal_v2_claim_lock_measurement_fullsize_part*.json") if PART_PATTERN.search(path.name))
    if len(paths) != 10:
        raise RuntimeError(f"expected ten completed full-size chunks, found {len(paths)}")
    merged: dict[str, dict[str, list[np.ndarray]]] = {}
    for report_path in paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        archive_path = root / report["bootstrap_input_path"]
        with np.load(archive_path, allow_pickle=False) as archive:
            for encoded in report["bootstrap_input_keys"]:
                target = merged.setdefault(encoded, {"cross_left_a": [], "cross_left_b": [], "cross_right_a": [], "cross_right_b": []})
                for field in target:
                    target[field].extend(np.asarray(archive[f"{encoded}__{field}"], dtype=np.float64))
    if any(len(values["cross_left_a"]) != 30 for values in merged.values()):
        raise RuntimeError("completed full-size inputs do not contain 30 seeds per comparison")
    return merged


def _bootstrap(left: np.ndarray, right: np.ndarray, *, seed: int, draws: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {key: [] for key in _estimate(left, right)}
    for _ in range(draws):
        seed_index = rng.integers(0, left.shape[0], size=left.shape[0])
        # The eligible label surface is fixed by the locked support rule.  Do
        # not resample labels with replacement here: duplicate labels create
        # artificial ties in a pairwise-order estimand.  Resample only the
        # independent completed measurement seeds.
        estimate = _estimate(left[seed_index], right[seed_index])
        for key, value in estimate.items():
            values[key].append(value)
    return {key: (float(np.quantile(value, .05)), float(np.quantile(value, .95))) for key, value in values.items()}


def _same_context_rows(root: Path, *, draws: int) -> pd.DataFrame:
    merged = _load_inputs(root)
    rows: list[dict[str, Any]] = []
    environments = sorted({encoded.split("__", 3)[0] for encoded in merged})
    for environment in environments:
        for metric in METRICS:
            candidates: list[tuple[str, str, str, str]] = []
            for encoded in merged:
                source, left, right, candidate_metric = encoded.split("__", 3)
                if source == environment and candidate_metric == metric and environment in (left, right):
                    candidates.append((encoded, left, right, "left" if environment == left else "right"))
            if not candidates:
                raise RuntimeError(f"missing same-context input for {environment} / {metric}")
            encoded, left_env, right_env, side = sorted(candidates)[0]
            values = merged[encoded]
            left = np.stack(values[f"cross_{side}_a"], axis=0)
            right = np.stack(values[f"cross_{side}_b"], axis=0)
            estimate = _estimate(left, right)
            row: dict[str, Any] = {
                "control_type": "same_context_independent_fullsize_pseudoreplicates",
                "source_environment_id": environment,
                "target_environment_id": environment,
                "metric": metric,
                "reference_pair": f"{left_env}__{right_env}",
                "n_items": int(left.shape[1]),
                "n_seeds": int(left.shape[0]),
                "interval_status": "point_estimate_only; formal uncertainty remains the completed 30-seed biological comparison",
                "measurement_definition": "same target context; independent A/B full-size with-replacement raw-cell pseudoreplicates; fixed source-frozen mean prediction",
            }
            for key, value in estimate.items():
                row[key] = value
            if draws:
                interval = _bootstrap(left, right, seed=20260912 + sum(ord(char) for char in encoded), draws=draws)
                row["bootstrap_draws"] = int(draws)
                row["interval_method"] = "90% seed-resampling interval over the fixed eligible label surface"
                for key in estimate:
                    row[f"{key}_ci_low"], row[f"{key}_ci_high"] = interval[key]
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["source_environment_id", "metric"], kind="stable").reset_index(drop=True)


def run(root: Path = ROOT, *, draws: int = 0) -> dict[str, Any]:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    full = pd.read_csv(manifests / "formal_v2_claim_lock_measurement_fullsize_summary.csv")
    directed = full.loc[
        (full["source_environment_id"].eq(full["left_target_environment_id"]))
        | (full["source_environment_id"].eq(full["right_target_environment_id"]))
    ].copy()
    directed["target_environment_id"] = np.where(
        directed["source_environment_id"].eq(directed["left_target_environment_id"]),
        directed["right_target_environment_id"],
        directed["left_target_environment_id"],
    )
    directed["shared_prediction_label_count"] = 243
    directed["eligible_label_count"] = directed["n_perturbations_primary_min"].astype(int)
    directed["analysis_scope"] = "full shared prediction surface after locked per-transfer min-cell eligibility"
    directed = directed.sort_values(["source_environment_id", "target_environment_id", "metric"], kind="stable").reset_index(drop=True)
    pseudo = _same_context_rows(root, draws=draws)
    directed_path = manifests / "reviewer_full_surface_directed.csv"
    pseudo_path = manifests / "reviewer_pseudocontext_negative_control.csv"
    report_path = manifests / "reviewer_full_surface_controls.json"
    directed.to_csv(directed_path, index=False)
    pseudo.to_csv(pseudo_path, index=False)
    report = {
        "status": "executed",
        "shared_prediction_label_count": 243,
        "eligible_label_count_range": [int(directed["eligible_label_count"].min()), int(directed["eligible_label_count"].max())],
        "directed_biological_rows": int(len(directed)),
        "same_context_negative_control_rows": int(len(pseudo)),
        "negative_control": "same context A/B pseudoreplicates reuse completed source-frozen full-size raw-cell resampling inputs; no predictor retraining or raw-data recomputation",
        "outputs": {
            "directed_full_surface": directed_path.relative_to(root).as_posix(),
            "pseudocontext_negative_control": pseudo_path.relative_to(root).as_posix(),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--draws", type=int, default=0)
    args = parser.parse_args()
    run(args.root, draws=args.draws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
