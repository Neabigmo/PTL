"""Contract-only adapter for State-style modern-model validation.

State is not promoted into the primary result unless its source-only training,
held-out vectors, panel mapping, and target-outcome blindness are all
verified.  A missing package is a recorded blocker, not a reason to substitute
another checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StateClaimLockContract:
    source_context: str
    target_contexts: tuple[str, ...]
    common_gene_panel_size: int = 1536
    source_only_training_verified: bool = False
    heldout_prediction_vector_verified: bool = False
    panel_mapping_verified: bool = False
    target_outcome_blind_verified: bool = False

    def audit(self) -> dict[str, Any]:
        checks = {
            "source_only_training_verified": self.source_only_training_verified,
            "heldout_prediction_vector_verified": self.heldout_prediction_vector_verified,
            "panel_mapping_verified": self.panel_mapping_verified,
            "target_outcome_blind_verified": self.target_outcome_blind_verified,
        }
        blockers = [name for name, passed in checks.items() if not passed]
        return {
            "predictor_id": "state",
            "source_context": self.source_context,
            "target_contexts": list(self.target_contexts),
            "common_gene_panel_size": int(self.common_gene_panel_size),
            **checks,
            "claim_lock_eligible": not blockers,
            "claim_lock_blockers": blockers,
        }


class StateAdapter:
    """Detect State availability and emit a strict claim-lock audit."""

    def __init__(self, checkout: str | Path | None = None):
        self.checkout = Path(checkout).resolve() if checkout else None

    def availability(self) -> dict[str, Any]:
        module = importlib.util.find_spec("state")
        return {
            "package_importable": module is not None,
            "module_origin": str(module.origin) if module is not None else None,
            "checkout": self.checkout.as_posix() if self.checkout else None,
            "checkout_present": bool(self.checkout and self.checkout.is_dir()),
        }

    def validate_claim_lock(self, *, source_context: str, target_contexts: tuple[str, ...], panel_file: Path | None = None) -> dict[str, Any]:
        panel_verified = panel_file is not None and panel_file.is_file()
        contract = StateClaimLockContract(
            source_context=source_context,
            target_contexts=target_contexts,
            panel_mapping_verified=panel_verified,
        )
        result = contract.audit()
        result["availability"] = self.availability()
        result["panel_file"] = panel_file.as_posix() if panel_file else None
        result["status"] = "eligible" if result["claim_lock_eligible"] else "blocked"
        return result
