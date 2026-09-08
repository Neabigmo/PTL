"""Run the pinned Systema perturbation-specific centroid metric on saved outputs.

The formal arrays store delta responses.  Systema's centroid-accuracy
entrypoint uses pairwise distances, so the common control translation cancels;
the audit records this delta-space adaptation explicitly.  PTL/U-only scores
are reused from the saved reliability table and are never refit here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_reliability import (  # noqa: E402
    _aurc_only,
    _prediction_rows,
    load_manifest,
    load_panel,
    load_registry,
    read_ground_truth,
)

SYSTEMA_ROOT = ROOT / "third_party/systema"
if str(SYSTEMA_ROOT / "evaluation") not in sys.path:
    sys.path.insert(0, str(SYSTEMA_ROOT / "evaluation"))
from centroid_accuracy import calculate_centroid_accuracies  # noqa: E402


METHODS = ("u_only_rf", "ptl_rf")


def run(root: Path, *, split_seed: int = 20260907) -> dict[str, Any]:
    reliability = pd.read_csv(root / "artifacts/source_data/formal_v2_reliability_predictions.csv")
    reliability = reliability.loc[reliability["scenario"].eq("in_domain")].copy()
    score_columns = ["biological_instance_id", "environment_id", "predictor", *METHODS]
    reliability_scores = reliability[score_columns].drop_duplicates(
        ["biological_instance_id", "environment_id", "predictor"]
    )
    prediction_rows = _prediction_rows(root, split_seed=split_seed)
    prediction_rows = prediction_rows.loc[prediction_rows["split"].eq("test")].copy()
    prediction_rows = prediction_rows.merge(
        reliability_scores,
        on=["biological_instance_id", "environment_id", "predictor"],
        how="left",
        validate="many_to_one",
    )
    if prediction_rows[list(METHODS)].isna().any().any():
        raise ValueError("Systema audit could not join saved PTL/U-only reliability scores")
    registry = load_registry(root)
    manifest = load_manifest(root, split_seed=split_seed)
    panel = load_panel(root)
    rows: list[dict[str, Any]] = []
    for (environment_id, predictor), group in prediction_rows.groupby(["environment_id", "predictor"], sort=True):
        env_manifest = manifest.loc[manifest["environment_id"].eq(environment_id)]
        if env_manifest.empty:
            raise ValueError(f"Systema audit cannot find manifest rows for {environment_id}")
        ground_truth_path = str(env_manifest["ground_truth_path"].iloc[0])
        truth = read_ground_truth(root, ground_truth_path, panel).set_index("biological_instance_id")
        group = group.loc[group["biological_instance_id"].isin(truth.index)].copy()
        if group.empty:
            continue
        ids = group["biological_instance_id"].astype(str).tolist()
        pred_matrix = np.stack(group["prediction_vector"].to_numpy())
        truth_matrix = truth.loc[ids, panel].to_numpy(dtype=float)
        pred_frame = pd.DataFrame(pred_matrix, index=pd.MultiIndex.from_tuples([(identifier, "formal_prediction") for identifier in ids], names=["condition", "method"]), columns=panel)
        truth_frame = pd.DataFrame(truth_matrix, index=ids, columns=panel)
        centroid_scores = calculate_centroid_accuracies(pred_frame, truth_frame)
        systema_score = centroid_scores["formal_prediction"].reindex(ids).to_numpy(dtype=float)
        joined = group[["biological_instance_id", "environment_key", "predictor", "continuous_risk", *METHODS]].copy()
        joined["systema_centroid_accuracy"] = systema_score
        joined["systema_risk"] = 1.0 - joined["systema_centroid_accuracy"]
        for method in METHODS:
            rows.append({
                "split_seed": split_seed,
                "environment_id": environment_id,
                "environment_key": str(group["environment_key"].iloc[0]),
                "predictor": predictor,
                "method": method,
                "n_rows": int(len(joined)),
                "systema_centroid_aurc": float(_aurc_only(joined["systema_risk"].to_numpy(dtype=float), joined[method].to_numpy(dtype=float))),
                "systema_mean_risk": float(joined["systema_risk"].mean()),
                "primary_delta_cosine_mean_risk": float(joined["continuous_risk"].mean()),
                "metric_entrypoint": "third_party/systema/evaluation/centroid_accuracy.py::calculate_centroid_accuracies",
                "prediction_contract": "formal_v2_saved_delta_prediction",
                "ptl_scores_reused_without_refit": 1,
                "target_outcomes_used_for_score_construction": 0,
            })
    result = pd.DataFrame(rows).sort_values(["environment_id", "predictor", "method"], kind="stable")
    if result.empty:
        raise ValueError("Systema robustness produced no rows")
    macro = result.groupby("method", as_index=False).agg(
        n_rows=("n_rows", "sum"),
        systema_centroid_aurc=("systema_centroid_aurc", "mean"),
        systema_mean_risk=("systema_mean_risk", "mean"),
        primary_delta_cosine_mean_risk=("primary_delta_cosine_mean_risk", "mean"),
    )
    macro.insert(0, "aggregation_level", "environment_predictor_macro")
    manifest_dir = root / "artifacts/manifests"
    output_path = manifest_dir / "formal_v2_systema_robustness.csv"
    summary_path = manifest_dir / "formal_v2_systema_robustness_summary.json"
    result.to_csv(output_path, index=False)
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "split_seed": split_seed,
        "metric": "Systema centroid accuracy / perturbation-specific distance",
        "metric_entrypoint": "third_party/systema/evaluation/centroid_accuracy.py::calculate_centroid_accuracies",
        "delta_space_adaptation": "pairwise centroid distances are translation-invariant under a shared control offset",
        "scores_reused": ["u_only_rf", "ptl_rf"],
        "refit": False,
        "rows": result.to_dict("records"),
        "macro": macro.to_dict("records"),
        "output_path": output_path.relative_to(root).as_posix(),
        "status": "formal_v2_systema_robustness_executed_pinned_entrypoint",
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-seed", type=int, default=20260907)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), split_seed=args.split_seed), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
