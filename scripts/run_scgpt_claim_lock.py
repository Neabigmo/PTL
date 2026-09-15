"""Make one bounded scGPT availability attempt and record the blocker."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    module = importlib.util.find_spec("scgpt")
    import_status = "not_discovered"
    import_error_type = None
    import_error = None
    if module is not None:
        try:
            importlib.import_module("scgpt")
            import_status = "import_ok"
        except Exception as exc:  # bounded availability audit; no model run
            import_status = "import_failed"
            import_error_type = type(exc).__name__
            import_error = str(exc)
    result = {
        "schema_version": 1,
        "predictor_id": "scGPT",
        "status": (
            "blocked_package_unavailable"
            if module is None
            else "package_importable_but_not_claim_lock_run"
            if import_status == "import_ok"
            else "package_discovered_import_failed"
        ),
        "module_origin": str(module.origin) if module is not None else None,
        "import_status": import_status,
        "import_error_type": import_error_type,
        "import_error": import_error,
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
