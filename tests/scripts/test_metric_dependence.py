"""Regression checks for the frozen metric pair-state definition."""

from __future__ import annotations

import numpy as np

from scripts.run_metric_dependence import MIN_STRICT_SUPPORT, _posterior_state


def test_stable_state_requires_minimum_strict_support() -> None:
    below = np.full(MIN_STRICT_SUPPORT - 1, -1.0)
    at_threshold = np.full(MIN_STRICT_SUPPORT, -1.0)

    assert _posterior_state(below)[0] == "unresolved"
    assert _posterior_state(below)[4] == MIN_STRICT_SUPPORT - 1
    assert _posterior_state(at_threshold)[0] == "stable_p<q"
    assert _posterior_state(-at_threshold)[0] == "stable_p>q"
