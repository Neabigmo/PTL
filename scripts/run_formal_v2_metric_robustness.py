"""Evaluate existing learned predictions under multiple fidelity definitions."""

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

from scripts.run_formal_v2_predictors import load_manifest, load_panel, read_ground_truth  # noqa: E402
from scripts.run_formal_v2_reliability import _selective_metrics  # noqa: E402
from src.ptl.evaluation.fidelity import (  # noqa: E402
    absolute_effect_rank_agreement,
    centered_delta_cosine,
    delta_cosine,
    evaluate_fidelity_definitions,
)


PREDICTORS = ("mean_matching", "strong_linear", "slim_string")


def run(root: Path, output_path: Path, summary_path: Path) -> dict[str, Any]:
    panel = load_panel(root)
    manifest = load_manifest(root)
    rows: list[dict[str, Any]] = []
    instance_metric_rows: list[dict[str, Any]] = []
    for environment_key, group in manifest.groupby("environment_key", sort=True):
        for predictor in PREDICTORS:
            array_path = root / "results/formal_v2/predictors" / f"{environment_key}__{predictor}.npz"
            if not array_path.is_file():
                raise FileNotFoundError(array_path)
            payload = np.load(array_path, allow_pickle=True)
            ids = payload["test_biological_instance_ids"].astype(str)
            environment_rows = group.loc[group["split"].eq("test")].copy()
            truth_path = str(environment_rows["ground_truth_path"].iloc[0])
            truth = read_ground_truth(root, truth_path, genes=panel).set_index("biological_instance_id")
            missing = sorted(set(ids).difference(truth.index.astype(str)))
            if missing:
                raise ValueError(f"missing test truth rows for {environment_key}/{predictor}: {missing[:3]}")
            y_true = truth.loc[list(ids), panel].to_numpy(dtype=np.float64)
            y_pred = payload["test_predictions"].mean(axis=0).astype(np.float64)
            values = evaluate_fidelity_definitions(y_true, y_pred)
            primary_values = delta_cosine(y_true, y_pred)
            centered_values = centered_delta_cosine(y_true, y_pred)
            rank_values = absolute_effect_rank_agreement(y_true, y_pred)
            for row_index, biological_instance_id in enumerate(ids):
                instance_metric_rows.append({
                    "environment_key": environment_key,
                    "predictor": predictor,
                    "biological_instance_id": str(biological_instance_id),
                    "delta_cosine": float(primary_values[row_index]),
                    "centered_delta_cosine": float(centered_values[row_index]),
                    "absolute_effect_rank_agreement": float(rank_values[row_index]),
                })
            rows.append({
                "environment_key": environment_key,
                "predictor": predictor,
                "split": "test",
                "n_rows": int(len(ids)),
                "evaluation_gene_count": len(panel),
                "refit_per_metric": 0,
                **values,
            })
    frame = pd.DataFrame(rows).sort_values(["environment_key", "predictor"], kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    selective_path = output_path.with_name("formal_v2_metric_robustness_selective.csv")
    score_path = root / "artifacts/source_data/formal_v2_reliability_predictions.csv"
    selective_rows: list[dict[str, Any]] = []
    if score_path.is_file():
        scores = pd.read_csv(score_path)
        scores = scores.loc[scores["split"].eq("test")].copy()
        instance_metrics = pd.DataFrame(instance_metric_rows)
        scores = scores.merge(
            instance_metrics,
            on=["environment_key", "predictor", "biological_instance_id"],
            how="inner",
            validate="many_to_one",
            suffixes=("", "_recomputed"),
        )
        method_columns = [
            "raw_normalized_uq", "best_scalar_uq", "u_only_rf", "u_p_rf", "u_ps_rf", "u_psn_rf", "ptl_rf"
        ]
        group_columns = ["environment_key", "predictor", "scenario", "heldout_environment_id", "heldout_predictor"]
        metric_columns = ["delta_cosine", "centered_delta_cosine", "absolute_effect_rank_agreement"]
        for group_key, group in scores.groupby(group_columns, sort=True, dropna=False):
            group_values = dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,)))
            for metric in metric_columns:
                risk = 1.0 - group[metric].to_numpy(dtype=float)
                reliable = group["reliable_label"].to_numpy(dtype=int)
                for method in method_columns:
                    if method not in group.columns:
                        continue
                    summary = _selective_metrics(risk, reliable, group[method].to_numpy(dtype=float))
                    selective_rows.append({
                        "analysis": "selective_risk_under_alternative_fidelity",
                        "metric": metric,
                        "method": method,
                        **group_values,
                        **summary,
                    })
    selective_frame = pd.DataFrame(selective_rows)
    selective_frame.to_csv(selective_path, index=False)
    primary = frame.pivot(index="environment_key", columns="predictor", values="delta_cosine")
    agreement_rows: list[dict[str, Any]] = []
    for metric in ("centered_delta_cosine", "environment_centroid_cosine", "absolute_effect_rank_agreement"):
        comparison = frame.pivot(index="environment_key", columns="predictor", values=metric)
        agreement_rows.append({
            "metric": metric,
            "atlas_spearman_vs_primary": float(primary.stack().corr(comparison.stack(), method="spearman")),
        })
    result = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "definitions": ["delta_cosine", "centered_delta_cosine", "environment_centroid_cosine", "absolute_effect_rank_agreement"],
        "protocol_source": "local_diagnostic_only; no Systema or scPertEval claim",
        "refit_per_metric": False,
        "prediction_source": "existing_formal_v2_test_prediction_arrays",
        "output_path": output_path.relative_to(root).as_posix(),
        "selective_output_path": selective_path.relative_to(root).as_posix(),
        "selective_rows": int(len(selective_frame)),
        "selective_metric_policy": "reuse_saved_reliability_scores; recompute row-level fidelity risks without refitting",
        "prediction_surface_spearman_vs_primary": agreement_rows,
        "status": "metric_robustness_executed_real_predictions",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    result = run(
        root,
        (args.output or root / "artifacts/manifests/formal_v2_metric_robustness.csv").resolve(),
        (args.summary or root / "artifacts/manifests/formal_v2_metric_robustness.json").resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
