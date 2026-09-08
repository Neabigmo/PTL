"""Separate monotonic confidence recalibration from reliability ranking.

The formal-v2 artifact already contains a calibration-side scalar score and
its monotonic logistic/isotonic recalibrations.  This diagnostic compares them
on held-out Frangieh condition rows.  It is a counterfactual audit, not a new
deployment model: calibration is fit by the existing train-only pipeline and
test outcomes are used only for evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_grouping_loss import artifact_paths, declared_seeds  # noqa: E402


BASE = "best_scalar_uq"
CALIBRATED = "platt_logistic"
ISOTONIC = "isotonic"


def _finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 4 or np.unique(left[mask]).size < 2 or np.unique(right[mask]).size < 2:
        return float("nan")
    return float(spearmanr(left[mask], right[mask]).statistic)


def _aurc(risk: np.ndarray, confidence: np.ndarray) -> float:
    order = np.argsort(-np.asarray(confidence, dtype=float), kind="mergesort")
    ordered_risk = np.asarray(risk, dtype=float)[order]
    return float((np.cumsum(ordered_risk) / np.arange(1, len(ordered_risk) + 1)).mean())


def _brier(labels: np.ndarray, confidence: np.ndarray) -> float:
    return float(np.mean((np.asarray(confidence, dtype=float) - np.asarray(labels, dtype=int)) ** 2))


def _log_loss(labels: np.ndarray, confidence: np.ndarray) -> float:
    return float(log_loss(labels, np.clip(confidence, 1e-5, 1.0 - 1e-5), labels=[0, 1]))


def _max_abs_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return float(np.nanmax(np.abs(numeric))) if np.isfinite(numeric).any() else float("nan")


def _order_audit(base: np.ndarray, recalibrated: np.ndarray) -> dict[str, Any]:
    base = np.asarray(base, dtype=float)
    recalibrated = np.asarray(recalibrated, dtype=float)
    mask = np.isfinite(base) & np.isfinite(recalibrated)
    base = base[mask]
    recalibrated = recalibrated[mask]
    if len(base) < 2:
        return {
            "n_comparable_pairs": 0,
            "n_order_disagreements": 0,
            "rank_order_preserved": True,
            "n_base_tied_pairs": 0,
            "n_recalibrated_tied_pairs": 0,
        }
    i, j = np.triu_indices(len(base), k=1)
    base_delta = base[i] - base[j]
    calibrated_delta = recalibrated[i] - recalibrated[j]
    comparable = (base_delta != 0.0) & (calibrated_delta != 0.0)
    disagreements = comparable & (np.sign(base_delta) != np.sign(calibrated_delta))
    return {
        "n_comparable_pairs": int(comparable.sum()),
        "n_order_disagreements": int(disagreements.sum()),
        "rank_order_preserved": bool(disagreements.sum() == 0),
        "n_base_tied_pairs": int((base_delta == 0.0).sum()),
        "n_recalibrated_tied_pairs": int((calibrated_delta == 0.0).sum()),
    }


def _metrics(frame: pd.DataFrame, method: str) -> dict[str, float]:
    risk = frame["continuous_risk"].to_numpy(dtype=float)
    labels = frame["reliable_label"].to_numpy(dtype=int)
    confidence = frame[method].to_numpy(dtype=float)
    return {
        "aurc": _aurc(risk, confidence),
        "brier": _brier(labels, confidence),
        "log_loss": _log_loss(labels, confidence),
        "rank_spearman_confidence_vs_negative_risk": _finite_spearman(confidence, -risk),
    }


def run(root: Path, *, seeds: list[int] | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds or declared_seeds(root):
        prediction_path, _ = artifact_paths(root, seed)
        predictions = pd.read_csv(prediction_path)
        selected = predictions.loc[
            predictions["scenario"].eq("in_domain")
            & predictions["environment_key"].astype(str).str.startswith("frangieh_"),
        ].copy()
        for (predictor, environment_key), frame in selected.groupby(
            ["predictor", "environment_key"], sort=True
        ):
            base_metrics = _metrics(frame, BASE)
            platt_metrics = _metrics(frame, CALIBRATED)
            isotonic_metrics = _metrics(frame, ISOTONIC)
            order_audit = _order_audit(
                frame[BASE].to_numpy(dtype=float), frame[CALIBRATED].to_numpy(dtype=float)
            )
            rows.append({
                "split_seed": int(seed),
                "scenario": "in_domain",
                "predictor": str(predictor),
                "environment_key": str(environment_key),
                "n_test": int(len(frame)),
                **{f"base_{key}": value for key, value in base_metrics.items()},
                **{f"platt_{key}": value for key, value in platt_metrics.items()},
                **{f"isotonic_{key}": value for key, value in isotonic_metrics.items()},
                "platt_delta_aurc": platt_metrics["aurc"] - base_metrics["aurc"],
                "platt_delta_brier": platt_metrics["brier"] - base_metrics["brier"],
                "platt_delta_log_loss": platt_metrics["log_loss"] - base_metrics["log_loss"],
                "platt_delta_rank_spearman": platt_metrics["rank_spearman_confidence_vs_negative_risk"] - base_metrics["rank_spearman_confidence_vs_negative_risk"] if np.isfinite(base_metrics["rank_spearman_confidence_vs_negative_risk"]) and np.isfinite(platt_metrics["rank_spearman_confidence_vs_negative_risk"]) else float("nan"),
                "isotonic_delta_aurc": isotonic_metrics["aurc"] - base_metrics["aurc"],
                "isotonic_delta_brier": isotonic_metrics["brier"] - base_metrics["brier"],
                "isotonic_delta_log_loss": isotonic_metrics["log_loss"] - base_metrics["log_loss"],
                "isotonic_delta_rank_spearman": isotonic_metrics["rank_spearman_confidence_vs_negative_risk"] - base_metrics["rank_spearman_confidence_vs_negative_risk"] if np.isfinite(base_metrics["rank_spearman_confidence_vs_negative_risk"]) and np.isfinite(isotonic_metrics["rank_spearman_confidence_vs_negative_risk"]) else float("nan"),
                **order_audit,
                "calibration_fit_on": "calibration_rows_only",
                "test_outcomes_used_for_evaluation_only": 1,
            })

    detail = pd.DataFrame(rows).sort_values(
        ["split_seed", "predictor", "environment_key"], kind="stable"
    )
    summary = detail.groupby("predictor", as_index=False).agg(
        n_groups=("environment_key", "size"),
        mean_platt_delta_aurc=("platt_delta_aurc", "mean"),
        mean_platt_delta_brier=("platt_delta_brier", "mean"),
        mean_platt_delta_log_loss=("platt_delta_log_loss", "mean"),
        mean_platt_delta_rank_spearman=("platt_delta_rank_spearman", "mean"),
        mean_isotonic_delta_aurc=("isotonic_delta_aurc", "mean"),
        mean_isotonic_delta_brier=("isotonic_delta_brier", "mean"),
        mean_isotonic_delta_log_loss=("isotonic_delta_log_loss", "mean"),
        mean_isotonic_delta_rank_spearman=("isotonic_delta_rank_spearman", "mean"),
        max_abs_platt_delta_aurc=("platt_delta_aurc", lambda values: float(np.nanmax(np.abs(values)))),
        max_abs_platt_delta_rank_spearman=("platt_delta_rank_spearman", _max_abs_or_nan),
        n_platt_order_disagreements=("n_order_disagreements", "sum"),
        n_platt_rank_order_preserved=("rank_order_preserved", "sum"),
        n_groups_brier_improved=("platt_delta_brier", lambda values: int((values < 0).sum())),
    )
    manifest_dir = root / "artifacts/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    detail_path = manifest_dir / "formal_v2_recalibration_counterfactual.csv"
    summary_path = manifest_dir / "formal_v2_recalibration_counterfactual_summary.json"
    detail.to_csv(detail_path, index=False)
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "counterfactual": "train-only Platt recalibration of the selected scalar confidence score",
        "interpretation": {
            "calibration": "Brier/log loss may change after a monotonic mapping",
            "ranking": "a strictly monotonic mapping preserves the base order and therefore cannot repair AURC or rank ordering",
            "isotonic_caveat": "isotonic regression is monotone but may introduce ties; its ranking changes are reported separately",
        },
        "detail_path": detail_path.relative_to(root).as_posix(),
        "summary": summary.to_dict("records"),
        "status": "formal_v2_recalibration_counterfactual_executed",
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seeds", type=int, nargs="*")
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), seeds=args.seeds), indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
