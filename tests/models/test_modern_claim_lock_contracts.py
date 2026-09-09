from pathlib import Path

from src.models.state_adapter import StateClaimLockContract
from src.models.txpert_adapter import TxPertClaimLockContract


def test_state_contract_does_not_promote_without_all_claim_lock_checks():
    result = StateClaimLockContract("HepG2", ("Jurkat",)).audit()
    assert result["claim_lock_eligible"] is False
    assert "source_only_training_verified" in result["claim_lock_blockers"]


def test_txpert_contract_rejects_checkpoint_context_mismatch():
    result = TxPertClaimLockContract("K562", "HepG2", ("Jurkat",)).audit()
    assert result["claim_lock_eligible"] is False
    assert "checkpoint_training_context_does_not_match_source_context" in result["claim_lock_blockers"]
