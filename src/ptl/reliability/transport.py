"""Outcome-free reliability transport and conservative source selection.

The transport label is defined against the reliability model that would be
available without target outcomes: a pooled, target-excluded U-only RF. The
outer evaluation excludes every pair touching the held-out target, including
the target as a source. A deployment decision is made for one target and
split at a time; if no historical source passes the validation-selected safety
rule, the pooled target-excluded U-only score is deployed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau, rankdata


PAIR_DESCRIPTOR_COLUMNS = (
    "same_cell_context",
    "same_perturbation_modality",
    "same_platform",
    "same_condition",
    "same_dataset_family",
    "same_context_modality",
    "prediction_geometry_distance",
    "uq_distribution_distance",
)
FORBIDDEN_DESCRIPTOR_COLUMNS = {
    "fidelity",
    "continuous_risk",
    "reliable_label",
    "baseline_aurc",
    "ptl_aurc",
    "baseline_excess_aurc",
    "ptl_excess_aurc",
    "transfer_gain",
    "excess_aurc_gain",
    "risk_at_80_gain",
    "ftr_at_80_gain",
    "target_test_rows",
    "calibration_threshold",
    "target_reproducibility",
}


def validate_descriptor_columns(columns: Iterable[str]) -> None:
    columns = set(columns)
    forbidden = sorted(columns.intersection(FORBIDDEN_DESCRIPTOR_COLUMNS))
    if forbidden:
        raise ValueError(f"transport descriptors contain outcome/test fields: {forbidden}")
    missing = sorted(set(PAIR_DESCRIPTOR_COLUMNS).difference(columns))
    if missing:
        raise ValueError(f"transport descriptors are missing allowed pair fields: {missing}")


def descriptor_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Materialize only the fixed deployment-safe descriptor block."""

    validate_descriptor_columns(PAIR_DESCRIPTOR_COLUMNS)
    values = frame.loc[:, list(PAIR_DESCRIPTOR_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("transport descriptor matrix contains missing/non-numeric values")
    return values.to_numpy(dtype=np.float64)


@dataclass
class TransportPrediction:
    predicted_gain: np.ndarray
    positive_probability: np.ndarray


def _spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Compute the same average-tie Spearman correlation without pandas overhead."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(rankdata(left, method="average"), rankdata(right, method="average"))[0, 1])


def source_selection_metrics(
    scored: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    permutation_repeats: int = 1000,
    random_state: int = 20260907,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Measure source-selection quality within each target and split.

    The pooled AUROC can be distorted by target-level class composition.  This
    stratified audit therefore computes source ranking and sign metrics inside
    each target-by-split candidate set, followed by a macro-average.  The
    permutation p-value shuffles realized gains within each same target group
    while keeping deployment predictions fixed.
    """

    group_columns = ["target_environment_id"]
    if "split_seed" in scored.columns:
        group_columns.append("split_seed")
    rows: list[dict[str, object]] = []
    selection_keys = [column for column in group_columns if column in selection.columns]
    for group_key, group in scored.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        key = {column: group_key[index] for index, column in enumerate(group_columns)}
        prediction = pd.to_numeric(group["predicted_gain"], errors="coerce").to_numpy(dtype=float)
        realized = pd.to_numeric(group["transfer_gain"], errors="coerce").to_numpy(dtype=float)
        spearman = _spearman_correlation(prediction, realized) if len(group) >= 3 else float("nan")
        kendall = float(kendalltau(prediction, realized, nan_policy="omit").statistic) if len(group) >= 3 and np.std(prediction) > 0 and np.std(realized) > 0 else float("nan")
        ordered = group.sort_values(["predicted_gain", "positive_transfer_probability", "source_environment_id"], ascending=[False, False, True], kind="stable")
        selection_filter = np.ones(len(selection), dtype=bool)
        for column in selection_keys:
            selection_filter &= selection[column].astype(str).to_numpy() == str(key[column])
        selected_group = selection.loc[selection_filter]
        selected_source = str(selected_group["selected_source_environment_id"].iloc[0]) if not selected_group.empty else "u_only_fallback"
        selected_rank = float("nan") if selected_source == "u_only_fallback" else float(np.flatnonzero(ordered["source_environment_id"].astype(str).to_numpy() == selected_source)[0] + 1) if selected_source in set(ordered["source_environment_id"].astype(str)) else float("nan")
        selected_positive = float(selected_group["selected_source_positive"].iloc[0]) if not selected_group.empty else float("nan")
        oracle_regret = float(selected_group["selection_regret"].iloc[0]) if not selected_group.empty else float("nan")
        positive_fraction = float(np.mean(realized > 0.0)) if len(realized) else float("nan")
        within_auroc = float(roc_auc_score((realized > 0.0).astype(int), prediction)) if len(np.unique(realized > 0.0)) == 2 else float("nan")
        rows.append({
            **key,
            "candidate_pair_count": int(len(group)),
            "spearman_predicted_realized_gain": spearman,
            "kendall_source_rank_concordance": kendall,
            "selected_source_rank": selected_rank,
            "oracle_regret": oracle_regret,
            "selected_source_positive": selected_positive,
            "candidate_positive_source_fraction": positive_fraction,
            "within_target_positive_negative_auroc": within_auroc,
            "selected_source": selected_source,
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("source-selection audit produced no target groups")
    observed = float(detail["spearman_predicted_realized_gain"].dropna().mean())
    rng = np.random.default_rng(random_state)
    null: list[float] = []
    grouped_permutation_arrays = [
        (
            pd.to_numeric(group["predicted_gain"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(group["transfer_gain"], errors="coerce").to_numpy(dtype=float),
        )
        for _, group in scored.groupby(group_columns, sort=True, dropna=False)
    ]
    for _ in range(permutation_repeats):
        values: list[float] = []
        for predicted, realized in grouped_permutation_arrays:
            shuffled = rng.permutation(realized)
            if len(predicted) >= 3:
                correlation = _spearman_correlation(predicted, shuffled)
                if np.isfinite(correlation):
                    values.append(correlation)
        if values:
            null.append(float(np.mean(values)))
    p_value = float((1 + np.sum(np.asarray(null) >= observed)) / (1 + len(null))) if null and np.isfinite(observed) else float("nan")
    aggregate = {
        "target_split_group_count": float(len(detail)),
        "macro_spearman_predicted_realized_gain": float(detail["spearman_predicted_realized_gain"].mean()),
        "macro_kendall_source_rank_concordance": float(detail["kendall_source_rank_concordance"].mean()),
        "macro_selected_source_rank": float(detail["selected_source_rank"].mean()),
        "macro_oracle_regret": float(detail["oracle_regret"].mean()),
        "macro_selected_source_positive_rate": float(detail["selected_source_positive"].mean()),
        "macro_candidate_positive_source_fraction": float(detail["candidate_positive_source_fraction"].mean()),
        "macro_within_target_positive_negative_auroc": float(detail["within_target_positive_negative_auroc"].mean()),
        "macro_spearman_permutation_p_value": p_value,
        "source_selection_permutation_repeats": float(permutation_repeats),
    }
    return detail, aggregate


def _regressor() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])


def _classifier(y: np.ndarray, *, random_state: int = 0) -> Pipeline | DummyClassifier:
    if np.unique(y).size < 2:
        return DummyClassifier(strategy="constant", constant=int(y[0]) if len(y) else 0)
    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=1.0, solver="liblinear", random_state=random_state)),
    ])


def fit_predictor(train: pd.DataFrame, query: pd.DataFrame, *, random_state: int = 0) -> TransportPrediction:
    """Fit gain and positive-transfer models on outcome-labelled training pairs."""

    if train.empty or query.empty:
        raise ValueError("transport training and query frames must be non-empty")
    x_train = descriptor_matrix(train)
    x_query = descriptor_matrix(query)
    y_gain = pd.to_numeric(train["transfer_gain"], errors="raise").to_numpy(dtype=float)
    y_positive = (y_gain > 0.0).astype(np.int8)
    gain_model: object = _regressor()
    if len(y_gain) < 3 or np.isclose(np.var(y_gain), 0.0):
        gain_model = DummyRegressor(strategy="mean")
    gain_model.fit(x_train, y_gain)  # type: ignore[union-attr]
    classifier = _classifier(y_positive, random_state=random_state)
    classifier.fit(x_train, y_positive)
    probabilities = classifier.predict_proba(x_query)
    if probabilities.shape[1] == 1:
        positive_probability = np.full(
            len(x_query), float(getattr(classifier, "classes_", np.asarray([0]))[0] == 1), dtype=float
        )
    else:
        positive_probability = probabilities[:, 1]
    return TransportPrediction(
        predicted_gain=np.asarray(gain_model.predict(x_query), dtype=float),  # type: ignore[union-attr]
        positive_probability=np.asarray(positive_probability, dtype=float),
    )


def _target_group_columns(pairs: pd.DataFrame) -> list[str]:
    return ["target_environment_id", "split_seed"] if "split_seed" in pairs.columns else ["target_environment_id"]


def _training_pairs_for_target(pairs: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return pairs that do not touch the outer held-out target."""

    return pairs.loc[
        pairs["source_environment_id"].astype(str).ne(str(target))
        & pairs["target_environment_id"].astype(str).ne(str(target))
    ].copy()


def _select_probability_threshold(train_pairs: pd.DataFrame, *, gain_threshold: float) -> float:
    """Select a conservative gate using only non-target validation pair labels."""

    targets = sorted(train_pairs["target_environment_id"].astype(str).unique())
    validation_rows: list[dict[str, float | str]] = []
    for validation_target in targets:
        fit = _training_pairs_for_target(train_pairs, validation_target)
        query = train_pairs.loc[
            train_pairs["target_environment_id"].astype(str).eq(validation_target)
            & train_pairs["source_environment_id"].astype(str).ne(validation_target)
        ].copy()
        if fit.empty or query.empty:
            continue
        prediction = fit_predictor(fit, query)
        for index, (_, row) in enumerate(query.iterrows()):
            validation_rows.append({
                "target_environment_id": validation_target,
                "source_environment_id": str(row["source_environment_id"]),
                "transfer_gain": float(row["transfer_gain"]),
                "predicted_gain": float(prediction.predicted_gain[index]),
                "positive_transfer_probability": float(prediction.positive_probability[index]),
            })
    if not validation_rows:
        return 0.5
    validation = pd.DataFrame(validation_rows)
    candidates = np.linspace(0.5, 0.9, 9)
    scores: list[tuple[float, float]] = []
    for threshold in candidates:
        selected_gains: list[float] = []
        for _, group in validation.groupby("target_environment_id", sort=True):
            accepted = group.loc[
                group["predicted_gain"].gt(gain_threshold)
                & group["positive_transfer_probability"].ge(threshold)
            ]
            if accepted.empty:
                selected_gains.append(0.0)
            else:
                selected_gains.append(float(accepted.sort_values(
                    ["predicted_gain", "positive_transfer_probability", "source_environment_id"],
                    ascending=[False, False, True], kind="stable",
                ).iloc[0]["transfer_gain"]))
        scores.append((float(np.mean(selected_gains)), float(threshold)))
    return max(scores, key=lambda item: (item[0], item[1]))[1]


def leave_target_environment_out(
    pairs: pd.DataFrame,
    *,
    gain_threshold: float = 0.0,
    probability_threshold: float | None = None,
    permutation_repeats: int = 1000,
    random_state: int = 20260907,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate strict target-unseen transport and the deployed safeguard.

    All split observations for an outer target are held out from transport
    training. Queries are off-diagonal only, and one source/fallback choice is
    evaluated per target and split observation.
    """

    required = {"source_environment_id", "target_environment_id", "transfer_gain"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"transport pair table is missing: {sorted(missing)}")
    descriptor_matrix(pairs)
    pairs = pairs.copy()
    pairs["source_environment_id"] = pairs["source_environment_id"].astype(str)
    pairs["target_environment_id"] = pairs["target_environment_id"].astype(str)
    if (pairs["source_environment_id"] == pairs["target_environment_id"]).any():
        raise AssertionError("strict target-unseen transport requires an off-diagonal atlas")

    rows: list[dict[str, object]] = []
    selection_keys = _target_group_columns(pairs)
    for group_key, query in pairs.groupby(selection_keys, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        target = str(group_key[0])
        train = _training_pairs_for_target(pairs, target)
        if train.empty:
            raise ValueError(f"no strict target-unseen training pairs for {target}")
        prediction = fit_predictor(train, query)
        threshold = float(probability_threshold) if probability_threshold is not None else _select_probability_threshold(
            train, gain_threshold=gain_threshold
        )
        for index, (_, row) in enumerate(query.iterrows()):
            safe = bool(
                prediction.predicted_gain[index] > gain_threshold
                and prediction.positive_probability[index] >= threshold
            )
            payload: dict[str, object] = {
                "source_environment_id": str(row["source_environment_id"]),
                "target_environment_id": target,
                "transfer_gain": float(row["transfer_gain"]),
                "predicted_gain": float(prediction.predicted_gain[index]),
                "positive_transfer_probability": float(prediction.positive_probability[index]),
                "safeguard_accept": int(safe),
                "fallback_to_u_only": int(not safe),
                "strict_target_unseen": 1,
                "selected_probability_threshold": threshold,
            }
            if "split_seed" in row.index:
                payload["split_seed"] = int(row["split_seed"])
            rows.append(payload)

    scored = pd.DataFrame(rows)
    if scored.empty:
        raise ValueError("strict target-unseen transport produced no off-diagonal scores")
    selected_rows: list[dict[str, object]] = []
    for group_key, group in scored.groupby(selection_keys, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        target = str(group_key[0])
        accepted = group.loc[group["safeguard_accept"].eq(1)]
        if accepted.empty:
            selected = None
            realized_gain = 0.0
            fallback = 1
            selected_source = "u_only_fallback"
            selected_prediction = np.nan
            selected_probability = np.nan
        else:
            selected = accepted.sort_values(
                ["predicted_gain", "positive_transfer_probability", "source_environment_id"],
                ascending=[False, False, True], kind="stable",
            ).iloc[0]
            realized_gain = float(selected["transfer_gain"])
            fallback = 0
            selected_source = str(selected["source_environment_id"])
            selected_prediction = float(selected["predicted_gain"])
            selected_probability = float(selected["positive_transfer_probability"])
        oracle_gain = float(group["transfer_gain"].max())
        payload: dict[str, object] = {
            "target_environment_id": target,
            "selected_source_environment_id": selected_source,
            "selected_predicted_gain": selected_prediction,
            "selected_positive_transfer_probability": selected_probability,
            "realized_gain_vs_target_excluded_u_only": realized_gain,
            "oracle_gain": oracle_gain,
            "selection_regret": max(oracle_gain, 0.0) - max(realized_gain, 0.0),
            "fallback_to_u_only": fallback,
            "selected_source_positive": int(realized_gain > 0),
            "selected_source_negative_transfer": int(fallback == 0 and realized_gain < 0),
            "accepted_pair_count": int(len(accepted)),
            "accepted_pair_negative_transfer_count": int((accepted["transfer_gain"] < 0).sum()),
        }
        if "split_seed" in group.columns:
            payload["split_seed"] = int(group["split_seed"].iloc[0])
        selected_rows.append(payload)
    selection = pd.DataFrame(selected_rows)

    positive_true = (scored["transfer_gain"] > gain_threshold).astype(int)
    positive_score = scored["positive_transfer_probability"].to_numpy(dtype=float)
    if positive_true.nunique() > 1:
        auroc = float(roc_auc_score(positive_true, positive_score))
        balanced = float(balanced_accuracy_score(positive_true, (positive_score >= 0.5).astype(int)))
    else:
        auroc = float("nan")
        balanced = float("nan")

    # The null keeps the observed deployment predictions fixed and shuffles
    # realized gains within each target/split group. This is the same null used
    # by the source-selection audit and avoids refitting the transport model for
    # every permutation.
    permutation_aurocs: list[float] = []
    rng = np.random.default_rng(random_state)
    if permutation_repeats:
        for _ in range(permutation_repeats):
            permuted_true_parts: list[np.ndarray] = []
            for _, group in pairs.groupby(selection_keys, sort=True, dropna=False):
                values = group["transfer_gain"].to_numpy(dtype=float)
                permuted_true_parts.append((rng.permutation(values) > gain_threshold).astype(int))
            permuted_true = np.concatenate(permuted_true_parts) if permuted_true_parts else np.empty(0, dtype=int)
            if np.unique(permuted_true).size > 1:
                permutation_aurocs.append(float(roc_auc_score(permuted_true, scored["positive_transfer_probability"])))

    selected_nonfallback = selection.loc[selection["fallback_to_u_only"].eq(0)]
    positive_oracle = np.maximum(selection["oracle_gain"].to_numpy(dtype=float), 0.0)
    positive_realized = np.maximum(
        selection["realized_gain_vs_target_excluded_u_only"].to_numpy(dtype=float), 0.0
    )
    selection_detail, selection_metrics = source_selection_metrics(
        scored,
        selection,
        permutation_repeats=permutation_repeats,
        random_state=random_state + 101,
    )
    summary: dict[str, object] = {
        "pair_count": int(len(scored)),
        "target_environment_count": int(scored["target_environment_id"].nunique()),
        "deployment_decision_count": int(len(selection)),
        "predicted_gain_spearman": float(scored[["transfer_gain", "predicted_gain"]].corr(method="spearman").iloc[0, 1]),
        "positive_transfer_auroc": auroc,
        "positive_transfer_balanced_accuracy": balanced,
        "permutation_auroc_mean": float(np.mean(permutation_aurocs)) if permutation_aurocs else float("nan"),
        "permutation_auroc_std": float(np.std(permutation_aurocs)) if permutation_aurocs else float("nan"),
        "permutation_repeats": int(permutation_repeats),
        "permutation_null": "within_target_split_gain_shuffle_with_observed_predictions_fixed",
        "fallback_frequency": float(selection["fallback_to_u_only"].mean()),
        "accepted_pair_negative_transfer_rate": float(
            (scored.loc[scored["safeguard_accept"].eq(1), "transfer_gain"] < 0).mean()
        ) if scored["safeguard_accept"].any() else float("nan"),
        "selected_source_negative_transfer_rate": float(
            selected_nonfallback["realized_gain_vs_target_excluded_u_only"].lt(0).mean()
        ) if not selected_nonfallback.empty else float("nan"),
        "negative_transfer_rate_ungated": float((scored["transfer_gain"] < 0).mean()),
        "mean_realized_gain_vs_target_excluded_u_only": float(
            selection["realized_gain_vs_target_excluded_u_only"].mean()
        ),
        "worst_target_regret": float(selection["selection_regret"].max()),
        "mean_selection_regret": float(selection["selection_regret"].mean()),
        "oracle_gain_recovered_fraction": float(positive_realized.sum() / positive_oracle.sum()) if positive_oracle.sum() > 0 else float("nan"),
        "targets_helped": int((selection["realized_gain_vs_target_excluded_u_only"] > 0).sum()),
        "targets_unchanged": int((selection["realized_gain_vs_target_excluded_u_only"] == 0).sum()),
        "targets_harmed": int((selection["realized_gain_vs_target_excluded_u_only"] < 0).sum()),
        "gain_threshold": float(gain_threshold),
        "probability_threshold_policy": "validation_selected_target_excluded" if probability_threshold is None else "fixed_explicit",
        "source_selection_metrics": selection_metrics,
        "source_selection_detail": selection_detail.to_dict("records"),
    }
    return scored, {**summary, "selection": selection.to_dict("records")}  # type: ignore[return-value]
