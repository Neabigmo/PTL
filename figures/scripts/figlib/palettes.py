from __future__ import annotations

WHITE = "#FFFFFF"
BLACK = "#1F1F1F"
TEXT = "#242424"
MUTED = "#626262"
RULE = "#9B9B9B"
GRID = "#E4E1DC"
LIGHT_GRAY = "#F3F1ED"
MID_GRAY = "#B7B2AB"
GRAY = "#716F6B"

# Muted journal palette. Keep these semantic rather than decorative.
BLUE = "#426A8C"
LIGHT_BLUE = "#DCE6EE"
ORANGE = "#B8874A"
LIGHT_ORANGE = "#EFE2CE"
TEAL = "#4F8179"
LIGHT_TEAL = "#DDEAE6"
RED = "#A55A4E"
LIGHT_RED = "#EBD8D4"
GREEN = "#6E8B65"
LIGHT_GREEN = "#E0E8DC"
PURPLE = "#6E6887"
LIGHT_PURPLE = "#E3E0EA"
YELLOW = "#D3B761"
LIGHT_YELLOW = "#EFE8CE"

HEATMAP_NEG = "#4F7895"
HEATMAP_ZERO = "#F0EEE9"
HEATMAP_POS = "#B78355"
HEATMAP_FAIL = "#9F5148"
HEATMAP_AUDIT = "#557A95"

SPLIT_ORDER = [
    "random_split",
    "unseen_perturbation_split",
    "unseen_combination_split",
    "low_support_split",
    "dataset_heldout_split",
    "external_holdout",
]

SPLIT_LABELS = {
    "random_split": "Random\nanchor",
    "unseen_perturbation_split": "Held-out\nperturbation",
    "unseen_combination_split": "Held-out\ncombination",
    "low_support_split": "Low\nsupport",
    "dataset_heldout_split": "Dataset\nheldout",
    "external_holdout": "External\nholdout",
}

SPLIT_SHORT = {
    "random_split": "Rand.",
    "unseen_perturbation_split": "Pert.",
    "unseen_combination_split": "Comb.",
    "low_support_split": "Low\nsupp.",
    "dataset_heldout_split": "Data.\nhold.",
    "external_holdout": "Ext.\nhold.",
}

SPLIT_COLORS = {
    "random_split": BLUE,
    "unseen_perturbation_split": ORANGE,
    "unseen_combination_split": PURPLE,
    "low_support_split": GREEN,
    "dataset_heldout_split": TEAL,
    "external_holdout": RED,
}

MODEL_LABELS = {
    "ridge_regression_baseline": "Ridge",
    "global_delta_baseline": "Global delta",
    "perturbation_mean_delta_baseline": "Perturb. mean",
    "cell_context_knn_delta_baseline": "Context kNN",
    "control_mean_baseline": "Control mean",
}

MODEL_ORDER = [
    "ridge_regression_baseline",
    "perturbation_mean_delta_baseline",
    "cell_context_knn_delta_baseline",
    "global_delta_baseline",
    "control_mean_baseline",
]

MODEL_COLORS = {
    "ridge_regression_baseline": BLUE,
    "global_delta_baseline": PURPLE,
    "perturbation_mean_delta_baseline": ORANGE,
    "cell_context_knn_delta_baseline": GREEN,
    "control_mean_baseline": GRAY,
}

BASELINE_LABELS = {
    "naive_confidence": "Naive confidence",
    "ptl_logistic_calibrated": "PTL logistic",
    "full_PTL_random_forest": "Full PTL RF",
    "no_context_distance": "No context distance",
    "support_only": "Support only",
    "split_family_only": "Split-family only",
    "novelty_only": "Novelty only",
    "support_plus_novelty": "Support + novelty",
    "calibrated_logistic_deployment": "Calibrated logistic",
    "conformal_style_confidence_proxy": "Conformal proxy",
}
