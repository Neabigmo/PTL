"""Combine three strict source-only GEARS runs into the PTL predictor contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/source_data/gears_frangieh_source_frozen"
OUTPUT = ROOT / "artifacts/source_data/frangieh_gears_source_frozen_predictions.npz"
SOURCES = (
    "frangieh_melanoma_control",
    "frangieh_melanoma_coculture",
    "frangieh_melanoma_ifng",
)


def main() -> None:
    payloads = [np.load(INPUT / f"{source}.npz", allow_pickle=False) for source in SOURCES]
    labels = payloads[0]["perturbation_label"].astype(str)
    genes = payloads[0]["evaluation_gene_symbols"].astype(str)
    folds = payloads[0]["fold_id"].astype(np.int16)
    for source, payload in zip(SOURCES, payloads, strict=True):
        if str(payload["source_environment"]) != source:
            raise RuntimeError(f"source mismatch for {source}")
        if not np.array_equal(payload["perturbation_label"].astype(str), labels):
            raise RuntimeError(f"label mismatch for {source}")
        if not np.array_equal(payload["evaluation_gene_symbols"].astype(str), genes):
            raise RuntimeError(f"gene mismatch for {source}")
        if not np.array_equal(payload["fold_id"].astype(np.int16), folds):
            raise RuntimeError(f"fold mismatch for {source}")
    prediction = np.stack([payload["prediction"].astype(np.float32) for payload in payloads])[:, :, None, :]
    confidence = np.ones((len(SOURCES), len(labels)), dtype=np.float32)
    np.savez_compressed(
        OUTPUT,
        source_environment=np.asarray(SOURCES),
        perturbation_label=labels,
        fold_id=folds,
        model_seed=np.asarray([20260916], dtype=np.int32),
        prediction=prediction,
        confidence=confidence,
        evaluation_gene_symbols=genes,
    )
    report = {
        "status": "executed",
        "model": "official_GEARS",
        "information_contract": "strict source-frozen fivefold cross-fitting; target cells and outcomes absent from fit, tuning, and graph construction",
        "sources": list(SOURCES),
        "labels": int(len(labels)),
        "genes": int(len(genes)),
        "prediction_shape": list(prediction.shape),
        "output": OUTPUT.relative_to(ROOT).as_posix(),
    }
    (INPUT / "combined.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
