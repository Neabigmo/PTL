"""Scientific contract adapter for the official TxPert checkout.

The adapter deliberately separates *upstream inference reproducibility* from
eligibility for the PTL Claim Lock.  A public K562 checkpoint is useful for an
external contract check, but it cannot be relabelled as a source-only
HepG2/Jurkat predictor without a frozen training and held-out-vector record.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TxPertClaimLockContract:
    """Immutable provenance needed before a TxPert vector can enter Claim Lock."""

    checkpoint_training_context: str
    source_context: str
    target_contexts: tuple[str, ...]
    source_only_training_verified: bool = False
    heldout_prediction_vector_verified: bool = False
    target_outcome_blind_verified: bool = False

    def audit(self) -> dict[str, Any]:
        blockers: list[str] = []
        if not self.source_only_training_verified:
            blockers.append("source_only_training_not_verified")
        if not self.heldout_prediction_vector_verified:
            blockers.append("heldout_prediction_vector_not_verified")
        if not self.target_outcome_blind_verified:
            blockers.append("target_outcome_blindness_not_verified")
        if self.checkpoint_training_context.casefold() != self.source_context.casefold():
            blockers.append("checkpoint_training_context_does_not_match_source_context")
        return {
            "predictor_id": "txpert",
            "checkpoint_training_context": self.checkpoint_training_context,
            "source_context": self.source_context,
            "target_contexts": list(self.target_contexts),
            "source_only_training_verified": self.source_only_training_verified,
            "heldout_prediction_vector_verified": self.heldout_prediction_vector_verified,
            "target_outcome_blind_verified": self.target_outcome_blind_verified,
            "claim_lock_eligible": not blockers,
            "claim_lock_blockers": blockers,
        }


class TxPertAdapter:
    """Adapter that records official TxPert assets and validates Claim Lock inputs."""

    def __init__(self, checkout: str | Path):
        self.checkout = Path(checkout).resolve()
        self.checkpoint_dir = self.checkout / "cache" / "checkpoints"
        self.data_dir = self.checkout / "cache" / "K562_single_cell_line"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def asset_audit(self) -> dict[str, Any]:
        checkpoints = sorted(self.checkpoint_dir.glob("*.ckpt"))
        data_files = sorted(path for path in self.data_dir.rglob("*") if path.is_file())
        return {
            "official_source_checkout": self.checkout.is_dir(),
            "official_checkpoint_verified": bool(checkpoints),
            "official_checkpoint_files": [
                {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": self._sha256(path)}
                for path in checkpoints
            ],
            "cached_data_files": [
                {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": self._sha256(path)}
                for path in data_files
            ],
            "upstream_training_context": "K562",
            "upstream_cached_inference_is_external_contract": True,
        }

    def validate_source_only_claim_lock(
        self,
        *,
        source_context: str,
        target_contexts: tuple[str, ...],
        prediction_vector: Path | None = None,
        target_outcome_blind: bool = False,
    ) -> dict[str, Any]:
        """Return an auditable eligibility decision; never silently promote K562."""

        vector_verified = prediction_vector is not None and prediction_vector.is_file()
        contract = TxPertClaimLockContract(
            checkpoint_training_context="K562",
            source_context=source_context,
            target_contexts=target_contexts,
            source_only_training_verified=False,
            heldout_prediction_vector_verified=vector_verified,
            target_outcome_blind_verified=target_outcome_blind,
        )
        result = contract.audit()
        result["asset_audit"] = self.asset_audit()
        result["prediction_vector"] = prediction_vector.as_posix() if prediction_vector else None
        return result
