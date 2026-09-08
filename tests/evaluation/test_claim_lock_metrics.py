from __future__ import annotations

import numpy as np

from scripts.run_formal_v2_claim_lock_measurement import _systema_risk_batch, _systema_risk_chunked


def test_batched_systema_matches_reference_formula() -> None:
    rng = np.random.default_rng(20260908)
    prediction_versions = rng.normal(size=(4, 9, 17)).astype(np.float32)
    truths = rng.normal(size=(3, 9, 17)).astype(np.float32)
    expected = np.stack([
        np.stack([_systema_risk_chunked(version, truths[seed]) for version in prediction_versions])
        for seed in range(truths.shape[0])
    ])
    observed = _systema_risk_batch(prediction_versions, truths)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
