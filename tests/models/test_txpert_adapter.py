from pathlib import Path

from src.models.txpert_adapter import TxPertAdapter, TxPertClaimLockContract


def test_k562_checkpoint_cannot_be_promoted_to_nadig_source_only():
    result = TxPertClaimLockContract(
        checkpoint_training_context="K562",
        source_context="nadig_hepg2",
        target_contexts=("nadig_jurkat",),
    ).audit()
    assert result["claim_lock_eligible"] is False
    assert "checkpoint_training_context_does_not_match_source_context" in result["claim_lock_blockers"]


def test_txpert_adapter_records_external_assets_without_claim_lock(tmp_path: Path):
    checkout = tmp_path / "TxPert"
    (checkout / "cache" / "checkpoints").mkdir(parents=True)
    (checkout / "cache" / "K562_single_cell_line").mkdir(parents=True)
    (checkout / "cache" / "checkpoints" / "K562_unseen_pert_gat.ckpt").write_bytes(b"checkpoint")
    adapter = TxPertAdapter(checkout)
    result = adapter.validate_source_only_claim_lock(
        source_context="nadig_hepg2",
        target_contexts=("nadig_jurkat",),
    )
    assert result["claim_lock_eligible"] is False
    assert result["asset_audit"]["official_checkpoint_verified"] is True
    assert result["asset_audit"]["upstream_training_context"] == "K562"
