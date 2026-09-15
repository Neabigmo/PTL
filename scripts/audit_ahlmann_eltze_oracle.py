"""Run a deterministic oracle check for the published Ahlmann--Eltze solver.

The source implementation is R, so this audit mirrors its ``solve_y_axb``
closed form and its ``prcomp(X)`` orientation in an independent NumPy
reference. It is deliberately small: the audit proves the Python port's
matrix orientation and centering without pretending that the unavailable R
runtime is a scientific predictor dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_predictors import fit_ahlmann_eltze_bilinear_ridge  # noqa: E402


SOURCE_URL = "https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/main/benchmark/src/run_linear_pretrained_model.R"


def official_reference(
    train_labels: np.ndarray,
    train_change: np.ndarray,
    query_labels: np.ndarray,
    post_expression: np.ndarray,
    gene_names: list[str],
    alpha: float,
    n_components: int,
) -> np.ndarray:
    """Independent translation of the published PCA + two-sided ridge path."""

    post_gene_by_condition = post_expression.T
    post_centered = post_gene_by_condition - post_gene_by_condition.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(post_centered, full_matrices=False)
    rank = min(n_components, u.shape[0], u.shape[1])
    gene_embedding = u[:, :rank] * singular_values[:rank]
    gene_index = {gene: index for index, gene in enumerate(gene_names)}

    def parts(label: object) -> list[str]:
        return [part for part in str(label).split("_") if part]

    def embedding(label: object) -> np.ndarray:
        rows = [gene_embedding[gene_index[part]] for part in parts(label) if part in gene_index]
        return np.mean(rows, axis=0) if rows else np.zeros(rank, dtype=np.float64)

    train_p = np.stack([embedding(label) for label in train_labels])
    query_p = np.stack([embedding(label) for label in query_labels])
    bias = train_change.mean(axis=0)
    centered_change = train_change - bias
    eye = np.eye(rank, dtype=np.float64)
    left = np.linalg.solve(
        gene_embedding.T @ gene_embedding + alpha * eye,
        gene_embedding.T @ centered_change.T @ train_p,
    )
    weights = np.linalg.solve(
        train_p.T @ train_p + alpha * eye,
        left.T,
    ).T
    return (gene_embedding @ weights @ query_p.T).T + bias


def run(root: Path = ROOT) -> dict[str, object]:
    genes = ["A", "B", "C", "D", "E"]
    labels = np.asarray(["A", "B", "A_B", "C"])
    query_labels = np.asarray(["D", "A_C", "Z"])
    change = np.asarray(
        [
            [1.2, -0.3, 0.4, 0.2, -0.1],
            [-0.4, 1.1, 0.2, -0.1, 0.6],
            [0.3, 0.4, 0.8, 0.5, -0.4],
            [0.2, -0.2, 1.3, 0.7, 0.9],
        ],
        dtype=np.float64,
    )
    post_expression = np.asarray(
        [
            [4.0, 1.0, 0.5, 2.0, 1.5],
            [2.0, 5.0, 0.5, 1.0, 2.5],
            [3.0, 2.0, 4.0, 6.0, 0.5],
            [1.0, 3.0, 2.0, 5.0, 4.0],
        ],
        dtype=np.float64,
    )
    kwargs = {"gene_names": genes, "post_train_values": post_expression, "alpha": 0.1, "n_components": 10}
    observed = fit_ahlmann_eltze_bilinear_ridge(labels, change, query_labels, **kwargs)
    expected = official_reference(labels, change, query_labels, post_expression, genes, 0.1, 10)
    max_abs_error = float(np.max(np.abs(observed - expected)))
    if not np.allclose(observed, expected, rtol=1e-6, atol=1e-6):
        raise AssertionError(f"Ahlmann--Eltze oracle mismatch: max_abs_error={max_abs_error}")
    result = {
        "schema_version": 1,
        "status": "pass",
        "source_url": SOURCE_URL,
        "pca_surface": "post_expression_X_train_only",
        "response_surface": "change_train_only",
        "pca_components": 10,
        "ridge_penalty": 0.1,
        "max_abs_error": max_abs_error,
        "note": "R runtime unavailable; independent NumPy oracle mirrors the published solve_y_axb path.",
    }
    output = root / "artifacts/manifests/formal_v2_ahlmann_oracle.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
