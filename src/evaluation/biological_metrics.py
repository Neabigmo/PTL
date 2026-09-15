from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.baselines.run_baseline import read_gene_list
from src.evaluation.metrics import compute_pathway_fidelity_metrics, compute_perturbation_discrimination_metrics


TABLES_DIR = ROOT / "results" / "tables"
LOG_DIR = ROOT / "results" / "logs" / "evaluation"
DEFAULT_GENE_SET_PATH = ROOT / "data" / "external" / "gene_sets" / "transportability_gene_sets.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute pathway-level and perturbation-discrimination summaries.")
    parser.add_argument("--run-manifest", default=str(TABLES_DIR / "baseline_run_matrix.csv"))
    parser.add_argument("--registry", default=str(TABLES_DIR / "experiment_registry.csv"))
    parser.add_argument("--gene-sets", default=str(DEFAULT_GENE_SET_PATH))
    parser.add_argument("--pathway-output", default=str(TABLES_DIR / "pathway_metric_summary.csv"))
    parser.add_argument("--retrieval-output", default=str(TABLES_DIR / "perturbation_discrimination_summary.csv"))
    parser.add_argument("--log-file", default=str(LOG_DIR / "biological_metrics.log"))
    return parser.parse_args()


def log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def load_gene_sets(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Gene-set cache is required for pathway metrics and was not found: {path}. "
            "Create a Reactome/GO/Hallmark-style JSON mapping pathway names to gene symbols."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(name): {str(gene) for gene in genes} for name, genes in raw.items()}


def latest_success_registry(registry_path: Path) -> pd.DataFrame:
    registry = pd.read_csv(registry_path)
    registry["timestamp"] = pd.to_datetime(registry["timestamp"], errors="coerce")
    latest = registry.sort_values(["run_id", "timestamp"]).groupby("run_id", as_index=False).tail(1)
    return latest[latest["status"].isin(["success", "cached"])].copy()


def iter_run_rows(manifest_path: Path, registry_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    latest = latest_success_registry(registry_path)
    return manifest.merge(latest[["run_id", "status"]], on="run_id", how="inner")


def load_run_arrays(output_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    metadata = pd.read_parquet(output_dir / "test_metadata.parquet")
    arrays = np.load(output_dir / "test_predictions.npz")
    genes = read_gene_list(output_dir / "genes.txt")
    return metadata, arrays["y_true"].astype(np.float32), arrays["y_pred"].astype(np.float32), genes


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_file)
    log_path.write_text("", encoding="utf-8")
    log(log_path, "Starting biological metric computation.")

    try:
        gene_sets = load_gene_sets(Path(args.gene_sets))
    except Exception as exc:
        manual = ROOT / "docs" / "manual_intervention_needed.md"
        with manual.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n\n## Pathway gene-set cache required\n"
                f"- Timestamp: {datetime.now().isoformat(timespec='seconds')}\n"
                f"- Required file: `{Path(args.gene_sets)}`\n"
                f"- Error: {exc}\n"
                "- Action: provide a JSON mapping Reactome/GO/Hallmark pathway names to gene symbols.\n"
            )
        log(log_path, f"FAILED: {exc}")
        raise

    rows = iter_run_rows(Path(args.run_manifest), Path(args.registry))
    pathway_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        output_dir = Path(row.output_dir)
        try:
            metadata, y_true, y_pred, genes = load_run_arrays(output_dir)
            non_control = ~metadata["is_control"].astype(bool).to_numpy()
            if not non_control.any():
                continue
            y_true_nc = y_true[non_control]
            y_pred_nc = y_pred[non_control]
            labels = metadata.loc[non_control, "perturbation_label"].astype(str).tolist()
            pathway = compute_pathway_fidelity_metrics(y_true_nc, y_pred_nc, genes, gene_sets)
            retrieval = compute_perturbation_discrimination_metrics(y_true_nc, y_pred_nc, labels)
            base = {
                "run_id": row.run_id,
                "model": row.model,
                "split_family": row.split_family,
                "dataset_scope": row.dataset_scope,
                "heldout_target": row.heldout_target,
                "seed": int(row.seed),
                "n_non_control": int(non_control.sum()),
            }
            pathway_rows.append(
                {
                    **base,
                    "pathway_count": int(pathway["pathway_count"]),
                    "mean_pathway_cosine": float(pathway["mean_pathway_cosine"]),
                    "mean_pathway_direction_consistency": float(pathway["mean_pathway_direction_consistency"]),
                }
            )
            retrieval_rows.append({**base, **retrieval})
        except Exception as exc:
            log(log_path, f"Run skipped with explicit error: {row.run_id}: {exc}")
            raise

    pd.DataFrame(pathway_rows).to_csv(args.pathway_output, index=False)
    pd.DataFrame(retrieval_rows).to_csv(args.retrieval_output, index=False)
    log(log_path, f"Wrote {len(pathway_rows)} pathway rows and {len(retrieval_rows)} retrieval rows.")


if __name__ == "__main__":
    main()
