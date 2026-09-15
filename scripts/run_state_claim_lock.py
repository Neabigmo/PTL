"""Perform the bounded State claim-lock feasibility attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.state_adapter import StateAdapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--panel-file", type=Path, default=None)
    args = parser.parse_args()
    result = StateAdapter(args.root / "third_party/State").validate_claim_lock(
        source_context="nadig_hepg2",
        target_contexts=("nadig_hepg2", "nadig_jurkat"),
        panel_file=args.panel_file,
    )
    checkout = (args.root / "third_party/State").resolve()
    result["official_source"] = "https://github.com/ArcInstitute/state"
    result["official_source_commit"] = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    result["official_cli_help"] = subprocess.run(
        [sys.executable, "-m", "state", "tx", "--help"], cwd=checkout, env={**__import__("os").environ, "PYTHONPATH": str(checkout / "src")}, capture_output=True, text=True, check=False,
    ).returncode == 0
    preprocess = args.root / "artifacts/modern/state_preprocessed_frangieh.h5ad"
    result["official_preprocess_artifact"] = preprocess.relative_to(args.root.resolve()).as_posix() if preprocess.is_file() else None
    result["semantic_blocker"] = "State's official set-transition contract is not the frozen global five-fold source-prediction vector contract; no source-only held-out vector and exact three-metric target-risk bridge is materialized"
    result["status"] = "blocked_semantic_contract" if not result["claim_lock_eligible"] else result["status"]
    output = args.root.resolve() / "artifacts/manifests/state_claim_lock.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
