"""Attempt the official TxPert contract without relabelling its K562 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.txpert_adapter import TxPertAdapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    checkout = args.root.resolve() / "third_party/TxPert"
    adapter = TxPertAdapter(checkout)
    result = adapter.validate_source_only_claim_lock(
        source_context="nadig_hepg2",
        target_contexts=("nadig_hepg2", "nadig_jurkat"),
        target_outcome_blind=True,
    )
    result["status"] = "eligible" if result["claim_lock_eligible"] else "blocked_external_contract"
    output = args.root.resolve() / "artifacts/manifests/txpert_claim_lock.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
