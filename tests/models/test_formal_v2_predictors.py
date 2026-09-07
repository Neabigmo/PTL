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
    prediction = fit_ahlmann_eltze_bilinear_ridge(labels[:3], values[:3], labels[3:])
    assert prediction.shape == (1, 3)
    assert np.isfinite(prediction).all()
