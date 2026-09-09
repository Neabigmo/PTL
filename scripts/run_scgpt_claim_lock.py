"""Make one bounded scGPT availability attempt and record the blocker."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    module = importlib.util.find_spec("scgpt")
    result = {
        "schema_version": 1,
        "predictor_id": "scGPT",
        "status": "blocked_package_unavailable" if module is None else "bounded_package_available_but_not_claim_lock_run",
        "module_origin": str(module.origin) if module is not None else None,
        "bounded_attempt": True,
        "common_gene_panel_size": 1536,
        "claim_lock_eligible": False,
        "claim_lock_blockers": ["no_verified_source_only_training_and_heldout_vectors"],
    }
    output = args.root.resolve() / "artifacts/manifests/scgpt_claim_lock.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
