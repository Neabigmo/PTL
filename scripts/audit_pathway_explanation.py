"""Audit the fixed pathway layer as explanation-only metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "data/external/gene_sets/transportability_gene_sets.json"
    result = {
        "schema_version": 1,
        "status": "executed" if source.is_file() else "blocked_missing_fixed_resource",
        "resource": source.relative_to(root).as_posix(),
        "resource_sha256": None,
        "pathway_count": 0,
        "role": "explanation_only_not_primary_claim_lock",
        "fdr_policy": "Benjamini-Hochberg applied within each declared pathway family; no pathway statistic can select a primary metric, context, or claim",
    }
    if source.is_file():
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        payload = json.loads(source.read_text(encoding="utf-8"))
        result["resource_sha256"] = digest
        result["pathway_count"] = len(payload)
        result["gene_set_min_size"] = min(map(len, payload.values())) if payload else 0
        result["gene_set_max_size"] = max(map(len, payload.values())) if payload else 0
    output = root / "artifacts/manifests/pathway_explanation_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "executed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
