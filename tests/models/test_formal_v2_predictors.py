import numpy as np

from scripts.run_formal_v2_predictors import (
    fit_ahlmann_eltze_bilinear_ridge,
    fit_matching_mean,
    fit_slim_string_bilinear_ridge,
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
        labels[:3], values[:3], labels[3:], gene_names=["A", "B", "C"],
        post_train_values=np.asarray([[4.0, 1.0, 0.5], [2.0, 5.0, 0.5], [3.0, 2.0, 4.0]], dtype=np.float32),
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
    post_values = np.asarray(
        [
            [4.0, 1.0, 0.5, 2.0],
            [2.0, 5.0, 0.5, 1.0],
            [3.0, 2.0, 4.0, 6.0],
        ],
        dtype=np.float64,
    )
    alpha = 0.1
    prediction = fit_ahlmann_eltze_bilinear_ridge(
        labels[:3], values[:3], labels[3:], gene_names=genes, post_train_values=post_values,
        alpha=alpha, n_components=10
    )

    center = values[:3].mean(axis=0)
    post_centered = post_values.T - post_values.T.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(post_centered, full_matrices=False)
    rank = min(10, u.shape[0], u.shape[1])
    gene_embedding = u[:, :rank] * singular_values[:rank]
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


def test_slim_string_bilinear_ridge_is_finite_and_zero_fills_unknown_queries() -> None:
    labels = np.asarray(["A", "B", "A_B"])
    post_values = np.asarray(
        [[4.0, 1.0, 0.5, 2.0], [2.0, 5.0, 0.5, 1.0], [3.0, 2.0, 4.0, 6.0]],
        dtype=np.float64,
    )
    control = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    embeddings = {
        "A": np.asarray([1.0, 0.0, 0.0]),
        "B": np.asarray([0.0, 1.0, 0.0]),
        "C": np.asarray([0.0, 0.0, 1.0]),
        "D": np.asarray([1.0, 1.0, 1.0]),
    }
    prediction = fit_slim_string_bilinear_ridge(
        labels,
        post_values,
        control,
        np.asarray(["A_B", "Z"]),
        ["A", "B", "C", "D"],
        embeddings,
        alpha=0.1,
        n_components=2,
    )
    assert prediction.shape == (2, 4)
    assert np.isfinite(prediction).all()
    assert np.allclose(prediction[1], post_values.mean(axis=0) - control, atol=1e-6)
