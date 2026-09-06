"""Build the compact declared evaluation-gene-space manifest from pilot outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    candidates = sorted((root / "results/baselines").glob("*/dataset_heldout_split__all_datasets__holdout-NormanWeissman2019_filtered__seed0__signature/genes.txt"))
    if not candidates:
        raise SystemExit("pilot gene list not found")
    source = candidates[0]
    genes = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    panel_id = "legacy_global_intersection_6257"
    manifest = root / "artifacts/manifests/evaluation_gene_space.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel_id", "gene_index", "gene_symbol", "source_gene_list", "source_sha256"])
        writer.writeheader()
        for index, gene in enumerate(genes):
            writer.writerow({"panel_id": panel_id, "gene_index": index, "gene_symbol": gene, "source_gene_list": source.relative_to(root).as_posix(), "source_sha256": digest})
    metadata = {
        "panel_id": panel_id,
        "gene_count": len(genes),
        "source_gene_list": source.relative_to(root).as_posix(),
        "source_sha256": digest,
        "policy": "predictor-native gene space remains separate; this panel is used only for declared common metrics",
        "generated_by": "scripts/build_evaluation_gene_space.py",
        "status": "provisional_legacy_panel_until_v2_metric_audit",
    }
    summary = root / "artifacts/manifests/evaluation_gene_space.json"
    summary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
