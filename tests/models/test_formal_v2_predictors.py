import numpy as np

from scripts.run_formal_v2_predictors import (
    fit_ahlmann_eltze_bilinear_ridge,
    fit_matching_mean,
)


def test_matching_mean_uses_components_then_global_fallback() -> None:
    labels = np.asarray(["A", "B", "A_B"])
    values = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    prediction = fit_matching_mean(labels, values, np.ones(3), np.asarray(["A_B_C", "Z"]))
    assert np.allclose(prediction[0], [0.5, 0.5])
    assert np.allclose(prediction[1], values.mean(axis=0))


def test_ahlmann_eltze_bilinear_ridge_is_finite_and_shape_preserving() -> None:
    labels = np.asarray(["A", "B", "A_B", "C"])
    values = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    prediction = fit_ahlmann_eltze_bilinear_ridge(
        labels[:3], values[:3], labels[3:], gene_names=["A", "B", "C"]
    )
    assert prediction.shape == (1, 3)
    assert np.isfinite(prediction).all()


def test_ahlmann_eltze_bilinear_ridge_matches_two_sided_ridge_reference() -> None:
    labels = np.asarray(["A", "B", "A_B", "C"])
    genes = ["A", "B", "C", "D"]
    values = np.asarray(
        [
            [1.2, -0.3, 0.4, 0.2],
            [-0.4, 1.1, 0.2, -0.1],
            [0.3, 0.4, 0.8, 0.5],
            [0.2, -0.2, 1.3, 0.7],
        ],
        dtype=np.float64,
    )
    alpha = 0.1
    prediction = fit_ahlmann_eltze_bilinear_ridge(
        labels[:3], values[:3], labels[3:], gene_names=genes, alpha=alpha, n_components=10
    )

    center = values[:3].mean(axis=0)
    _, _, vt = np.linalg.svd(values[:3] - center, full_matrices=False)
    rank = min(10, vt.shape[0], vt.shape[1])
    gene_embedding = vt[:rank].T
    component_index = {gene: index for index, gene in enumerate(genes)}

    def perturbation_embedding(label: str) -> np.ndarray:
        parts = [part for part in label.split("_") if part]
        return np.mean([gene_embedding[component_index[part]] for part in parts], axis=0)

    train_p = np.stack([perturbation_embedding(label) for label in labels[:3]])
    query_p = np.stack([perturbation_embedding(label) for label in labels[3:]])
    left_gram = gene_embedding.T @ gene_embedding + alpha * np.eye(rank)
    right_gram = train_p.T @ train_p + alpha * np.eye(rank)
    left = np.linalg.solve(left_gram, gene_embedding.T @ (values[:3] - center).T @ train_p)
    weights = np.linalg.solve(right_gram, left.T).T
    expected = (gene_embedding @ weights @ query_p.T).T + center

    assert np.allclose(prediction, expected, atol=1e-6)
