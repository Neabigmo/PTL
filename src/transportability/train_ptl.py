from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transportability.ptl import build_feature_audit, build_ptl_examples, infer_feature_spec, run_ptl_ablation_suite, write_json_summary


RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
TRAINING_LOG_DIR = RESULTS_DIR / "logs" / "training"
PHASE_NAME = "Phase 07"


class PhaseLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Phase 07 PTL reliability models.")
    parser.add_argument("--per-signature-metrics", default=str(TABLES_DIR / "per_signature_metrics.parquet"))
    parser.add_argument("--metrics", default=str(TABLES_DIR / "all_metrics.csv"))
    parser.add_argument("--manifest", default=str(TABLES_DIR / "baseline_run_matrix.csv"))
    parser.add_argument("--split-audit", default=str(TABLES_DIR / "split_audit.csv"))
    parser.add_argument("--preprocessing-summary", default=str(TABLES_DIR / "preprocessing_summary.csv"))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--registry", default=str(TABLES_DIR / "experiment_registry.csv"))
    parser.add_argument("--log-file", default=str(TRAINING_LOG_DIR / "phase07_ptl.log"))
    parser.add_argument("--max-examples", type=int, default=0, help="Optional stratified cap for smoke runs.")
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--include-biology-features", action="store_true", help="Reserve optional biological support columns when external annotations are available.")
    parser.add_argument("--feature-mode", choices=["deployment", "all"], default="deployment", help="Feature eligibility mode for PTL training; deployment excludes labels and evaluation-only columns.")
    parser.add_argument("--write-feature-audit", action="store_true", help="Write results/tables/ptl_feature_audit.csv with leakage annotations for every candidate column.")
    parser.add_argument("--run-label", default="phase07_ptl")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_registry_rows(registry_path: Path, rows: list[dict[str, Any]]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if registry_path.exists():
        frame.to_csv(registry_path, mode="a", header=False, index=False)
    else:
        frame.to_csv(registry_path, index=False)


def build_registry_rows(metrics: pd.DataFrame, output_dir: Path, log_file: Path, command: str, run_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timestamp = now_iso()
    for row in metrics.itertuples(index=False):
        estimator = str(row.estimator)
        ablation = str(row.ablation)
        if estimator == "naive_confidence":
            continue
        rows.append(
            {
                "run_id": f"{run_label}__{ablation}__{estimator}",
                "timestamp": timestamp,
                "phase": PHASE_NAME,
                "dataset": "signature_track_stress",
                "split": ablation,
                "model": estimator,
                "seed": "",
                "command": command,
                "status": "success",
                "main_metric": getattr(row, "mean_selective_risk_gain_vs_naive", float("nan")),
                "output_dir": str(output_dir),
                "log_file": str(log_file),
                "notes": "PTL reliability-layer ablation run",
            }
        )
    return rows


def run_phase07(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logger = PhaseLogger(Path(args.log_file))
    command = " ".join(sys.argv)

    logger.log("Starting Phase 07 PTL.")
    logger.log("Skill gate checked through SKILLS_INDEX.md; scikit-learn skill is the active implementation guide.")
    if args.include_biology_features:
        logger.log("Optional biology feature columns requested; no standalone external annotation client is available in this CLI, so columns are reserved as unavailable indicators.")
    else:
        logger.log("Optional Life Science Research annotations are not required for the core PTL run and are skipped.")

    max_examples = args.max_examples if args.max_examples and args.max_examples > 0 else None
    examples = build_ptl_examples(
        per_signature_metrics_path=Path(args.per_signature_metrics),
        all_metrics_path=Path(args.metrics),
        manifest_path=Path(args.manifest),
        split_audit_path=Path(args.split_audit),
        preprocessing_summary_path=Path(args.preprocessing_summary),
        max_examples=max_examples,
        random_state=args.random_state,
        include_biology_features=args.include_biology_features,
    )
    logger.log(
        f"PTL examples built: rows={len(examples)}, transportable_rate={examples['target_transportable'].mean():.4f}, runs={examples['run_id'].nunique()}."
    )

    feature_spec = infer_feature_spec(examples, feature_mode=args.feature_mode)
    feature_audit = build_feature_audit(examples, feature_mode=args.feature_mode, spec=feature_spec)
    metrics, ablation, failure, predictions = run_ptl_ablation_suite(
        examples,
        random_state=args.random_state,
        feature_mode=args.feature_mode,
    )
    metrics_path = tables_dir / "ptl_metrics.csv"
    ablation_path = tables_dir / "ptl_ablation.csv"
    failure_path = tables_dir / "failure_mode_summary.csv"
    predictions_path = tables_dir / "ptl_oof_predictions.csv"
    feature_audit_path = tables_dir / "ptl_feature_audit.csv"
    examples_path = tables_dir / "ptl_examples.parquet"
    summary_path = tables_dir / "ptl_run_summary.json"

    metrics.to_csv(metrics_path, index=False)
    ablation.to_csv(ablation_path, index=False)
    failure.to_csv(failure_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    if args.write_feature_audit:
        feature_audit.to_csv(feature_audit_path, index=False)
    examples.to_parquet(examples_path, index=False)
    write_json_summary(
        summary_path,
        {
            "timestamp": now_iso(),
            "n_examples": int(len(examples)),
            "n_runs": int(examples["run_id"].nunique()),
            "transportable_rate": float(examples["target_transportable"].mean()),
            "metrics_path": str(metrics_path),
            "ablation_path": str(ablation_path),
            "failure_path": str(failure_path),
            "predictions_path": str(predictions_path),
            "feature_mode": args.feature_mode,
            "feature_count": int(len(feature_spec.all_columns)),
            "feature_audit_path": str(feature_audit_path) if args.write_feature_audit else "",
            "optional_biology_features_requested": bool(args.include_biology_features),
        },
    )
    logger.log(f"Wrote {metrics_path}.")
    logger.log(f"Wrote {ablation_path}.")
    logger.log(f"Wrote {failure_path}.")
    logger.log(f"Wrote {predictions_path}.")
    if args.write_feature_audit:
        logger.log(f"Wrote {feature_audit_path}.")

    registry_rows = build_registry_rows(metrics, output_dir, Path(args.log_file), command, args.run_label)
    append_registry_rows(Path(args.registry), registry_rows)
    logger.log(f"Appended Phase 07 registry rows: {len(registry_rows)}.")

    completed = ablation[ablation["status"] == "completed"].copy()
    if not completed.empty:
        best = completed.sort_values(
            ["mean_false_transportability_rate_gain_vs_naive", "best_mean_false_transportability_rate"],
            ascending=[False, True],
        ).iloc[0]
        logger.log(
            "Best completed PTL ablation: "
            f"{best['ablation']} / {best['best_estimator']} with false-transportability gain vs naive="
            f"{float(best['mean_false_transportability_rate_gain_vs_naive']):.6f}."
        )
    logger.log("Phase 07 PTL completed.")


def main() -> None:
    run_phase07(parse_args())


if __name__ == "__main__":
    main()
