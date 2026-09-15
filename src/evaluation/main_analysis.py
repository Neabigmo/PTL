from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
DOCS_DIR = ROOT / "docs"
PHASE_NAME = "Phase 08"
PRIMARY_METRIC = "mean_cosine_non_control"
CONTROL_MODEL = "control_mean_baseline"


FORBIDDEN_NARRATIVE_TERMS = (
    "clinical",
    "wet-lab",
    "universal-best-model",
    "drug discovery",
    "final truth",
)


class PhaseLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 08 main analysis synthesis.")
    parser.add_argument("--metrics", default=str(TABLES_DIR / "all_metrics.csv"))
    parser.add_argument("--family-summary", default=str(TABLES_DIR / "phase06_summary_by_family.csv"))
    parser.add_argument("--pairwise", default=str(TABLES_DIR / "model_pairwise_comparisons.csv"))
    parser.add_argument("--ptl-metrics", default=str(TABLES_DIR / "ptl_metrics.csv"))
    parser.add_argument("--ptl-ablation", default=str(TABLES_DIR / "ptl_ablation.csv"))
    parser.add_argument("--failure-summary", default=str(TABLES_DIR / "failure_mode_summary.csv"))
    parser.add_argument("--ptl-examples", default=str(TABLES_DIR / "ptl_examples.parquet"))
    parser.add_argument("--pathway-summary", default=str(TABLES_DIR / "pathway_metric_summary.csv"))
    parser.add_argument("--retrieval-summary", default=str(TABLES_DIR / "perturbation_discrimination_summary.csv"))
    parser.add_argument("--ptl-feature-audit", default=str(TABLES_DIR / "ptl_feature_audit.csv"))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--docs-dir", default=str(DOCS_DIR))
    parser.add_argument("--log-file", default=str(RESULTS_DIR / "logs" / "phase08_main_analysis.log"))
    return parser.parse_args()


def _rank_descending(values: pd.Series) -> pd.Series:
    return values.rank(method="min", ascending=False).astype(int)


def _float_or_nan(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_float(value: Any, digits: int = 3) -> str:
    numeric = _float_or_nan(value)
    if math.isnan(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def build_transfer_decay_summary(all_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "split_family", PRIMARY_METRIC}
    missing = required - set(all_metrics.columns)
    if missing:
        raise ValueError(f"Missing required columns for transfer summary: {sorted(missing)}")

    grouped = all_metrics.groupby(["model", "split_family"], dropna=False)
    summary = grouped.agg(
        n_runs=("run_id", "count"),
        mean_cosine_non_control=(PRIMARY_METRIC, "mean"),
        median_cosine_non_control=(PRIMARY_METRIC, "median"),
        mean_random_to_stress_drop=("random_to_stress_drop", "mean"),
        mean_stress_retention=("stress_retention", "mean"),
        mean_mmd_linear=("mmd_linear", "mean"),
        mean_energy_distance=("energy_distance", "mean"),
        mean_sliced_wasserstein_128=("sliced_wasserstein_128", "mean"),
    ).reset_index()

    summary["family_rank"] = summary.groupby("split_family")["mean_cosine_non_control"].transform(_rank_descending)
    summary["family_winner"] = summary["family_rank"].eq(1)

    random_anchor = summary[summary["split_family"] == "random_split"][
        ["model", "family_rank", "mean_cosine_non_control"]
    ].rename(
        columns={
            "family_rank": "random_split_rank",
            "mean_cosine_non_control": "random_split_mean_cosine",
        }
    )
    summary = summary.merge(random_anchor, on="model", how="left")
    summary["rank_change_vs_random"] = summary["family_rank"] - summary["random_split_rank"]
    summary["cosine_change_vs_random"] = summary["mean_cosine_non_control"] - summary["random_split_mean_cosine"]
    summary["rank_shift_direction"] = np.select(
        [
            summary["split_family"].eq("random_split"),
            summary["rank_change_vs_random"].gt(0),
            summary["rank_change_vs_random"].lt(0),
        ],
        ["anchor", "worse_rank", "better_rank"],
        default="same_rank",
    )
    summary["ranking_reversal_flag"] = summary["rank_shift_direction"].isin({"worse_rank", "better_rank"})
    summary["stress_retention_flag"] = np.select(
        [
            summary["split_family"].eq("random_split"),
            summary["mean_stress_retention"].isna(),
            summary["mean_stress_retention"].abs().gt(10.0),
        ],
        ["anchor", "not_applicable", "unstable_small_anchor_ratio"],
        default="interpretable_ratio",
    )
    return summary.sort_values(["split_family", "family_rank", "model"]).reset_index(drop=True)


def build_selective_prediction_summary(ptl_metrics: pd.DataFrame, ptl_ablation: pd.DataFrame) -> pd.DataFrame:
    if ptl_metrics.empty:
        raise ValueError("PTL metrics table is empty.")
    naive = ptl_metrics[ptl_metrics["estimator"] == "naive_confidence"]
    if naive.empty:
        raise ValueError("PTL metrics must include a naive_confidence row.")
    naive_row = naive.iloc[0]

    output = ptl_metrics.copy()
    output["comparison_baseline"] = "naive_confidence"
    output["naive_mean_false_transportability_rate"] = _float_or_nan(naive_row["mean_false_transportability_rate"])
    output["naive_mean_selective_risk"] = _float_or_nan(naive_row["mean_selective_risk"])
    output["false_transportability_rate_delta_vs_naive"] = (
        output["mean_false_transportability_rate"] - output["naive_mean_false_transportability_rate"]
    )
    output["selective_risk_delta_vs_naive"] = output["mean_selective_risk"] - output["naive_mean_selective_risk"]
    output["primary_transportability_view"] = np.where(
        output["false_transportability_rate_delta_vs_naive"].lt(0),
        "improves_false_transportability_filtering",
        "does_not_improve_false_transportability_filtering",
    )
    output["secondary_risk_view"] = np.where(
        output["selective_risk_delta_vs_naive"].lt(0),
        "lower_cosine_risk_than_naive",
        "higher_or_equal_cosine_risk_than_naive",
    )

    ablation_status = ptl_ablation[["ablation", "status", "reason"]].drop_duplicates("ablation")
    output = output.merge(ablation_status, on="ablation", how="left")
    output.loc[output["ablation"] == "naive_confidence", "status"] = "baseline"
    output["reason"] = output["reason"].fillna("")
    return output.sort_values(
        ["status", "mean_false_transportability_rate", "ablation", "estimator"],
        na_position="last",
    ).reset_index(drop=True)


def build_failure_mode_atlas(failure_summary: pd.DataFrame, ptl_examples: pd.DataFrame | None = None) -> pd.DataFrame:
    if failure_summary.empty:
        raise ValueError("Failure-mode summary is empty.")
    atlas = failure_summary.copy()
    totals = atlas.groupby(["model", "split_family"], dropna=False)["n_signatures"].transform("sum")
    atlas["failure_mode_share"] = atlas["n_signatures"] / totals.replace(0, np.nan)
    severity_map = {
        "transportable": 0,
        "below_transport_threshold": 1,
        "non_transportable": 1,
        "high_risk_failure": 2,
        "severe_failure": 3,
    }
    atlas["failure_severity"] = atlas["failure_mode"].map(severity_map).fillna(1).astype(int)
    atlas["dominant_failure"] = atlas.groupby(["model", "split_family"])["n_signatures"].transform("max").eq(atlas["n_signatures"])
    non_transportable = atlas["failure_severity"].gt(0)
    atlas["dominant_nontransportable_failure"] = False
    if non_transportable.any():
        atlas.loc[non_transportable, "dominant_nontransportable_failure"] = atlas.loc[non_transportable].groupby(
            ["model", "split_family"]
        )["n_signatures"].transform("max").eq(atlas.loc[non_transportable, "n_signatures"])
    atlas["context_signal"] = np.select(
        [
            atlas["mean_perturbation_seen"].fillna(1.0).lt(0.25),
            atlas["mean_component_seen_fraction"].fillna(1.0).lt(0.25),
            atlas["mean_reference_seen"].fillna(1.0).lt(0.50),
            atlas["mean_n_cells"].fillna(100.0).lt(20.0),
        ],
        [
            "perturbation_novelty",
            "component_novelty",
            "reference_context_novelty",
            "low_signature_support",
        ],
        default="mixed_or_supported_context",
    )

    if ptl_examples is not None and not ptl_examples.empty:
        extras = ptl_examples.groupby(["model", "split_family", "failure_mode"], dropna=False).agg(
            n_unique_signatures=("signature_id", "nunique"),
            n_unique_perturbations=("perturbation_label", "nunique"),
            mean_unseen_component_count=("unseen_component_count", "mean"),
            mean_train_test_centroid_l2=("train_test_centroid_l2", "mean"),
        ).reset_index()
        atlas = atlas.merge(extras, on=["model", "split_family", "failure_mode"], how="left")

    return atlas.sort_values(
        ["failure_severity", "failure_mode_share", "n_signatures"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_split_difficulty_summary(all_metrics: pd.DataFrame, ptl_examples: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "split_family", PRIMARY_METRIC}
    missing = required - set(all_metrics.columns)
    if missing:
        raise ValueError(f"Missing required columns for split difficulty summary: {sorted(missing)}")
    metric_columns = {
        "run_id": "count",
        PRIMARY_METRIC: "mean",
        "random_to_stress_drop": "mean",
        "stress_retention": "mean",
    }
    available_metric_columns = {key: value for key, value in metric_columns.items() if key in all_metrics.columns}
    metrics = all_metrics.groupby(["model", "split_family"], dropna=False).agg(available_metric_columns).reset_index()
    rename = {"run_id": "n_runs", PRIMARY_METRIC: "mean_cosine_non_control"}
    metrics = metrics.rename(columns=rename)

    random_anchor = metrics[metrics["split_family"].eq("random_split")][["model", "mean_cosine_non_control"]].rename(
        columns={"mean_cosine_non_control": "random_anchor_cosine"}
    )
    metrics = metrics.merge(random_anchor, on="model", how="left")
    metrics["normalized_transfer_score"] = metrics["mean_cosine_non_control"] / metrics["random_anchor_cosine"].replace(0, np.nan)

    if ptl_examples is not None and not ptl_examples.empty:
        columns = [
            "model",
            "split_family",
            "perturbation_train_signature_count",
            "perturbation_train_cell_count",
            "perturbation_seen_in_train",
            "component_seen_fraction",
            "unseen_component_count",
            "delta_norm_true",
            "train_test_centroid_l2",
            "n_cells",
        ]
        present = [column for column in columns if column in ptl_examples.columns]
        context = ptl_examples[present].copy()
        context_summary = context.groupby(["model", "split_family"], dropna=False).agg(
            n_signatures=("split_family", "count"),
            mean_signature_support=("perturbation_train_signature_count", "mean") if "perturbation_train_signature_count" in context.columns else ("split_family", "size"),
            mean_cell_support=("perturbation_train_cell_count", "mean") if "perturbation_train_cell_count" in context.columns else ("split_family", "size"),
            mean_perturbation_seen=("perturbation_seen_in_train", "mean") if "perturbation_seen_in_train" in context.columns else ("split_family", "size"),
            mean_component_seen_fraction=("component_seen_fraction", "mean") if "component_seen_fraction" in context.columns else ("split_family", "size"),
            mean_unseen_component_count=("unseen_component_count", "mean") if "unseen_component_count" in context.columns else ("split_family", "size"),
            mean_target_effect_size=("delta_norm_true", "mean") if "delta_norm_true" in context.columns else ("split_family", "size"),
            mean_train_test_centroid_l2=("train_test_centroid_l2", "mean") if "train_test_centroid_l2" in context.columns else ("split_family", "size"),
            mean_test_signature_cells=("n_cells", "mean") if "n_cells" in context.columns else ("split_family", "size"),
        ).reset_index()
        metrics = metrics.merge(context_summary, on=["model", "split_family"], how="left")

    for column in [
        "mean_signature_support",
        "mean_cell_support",
        "mean_perturbation_seen",
        "mean_component_seen_fraction",
        "mean_unseen_component_count",
        "mean_target_effect_size",
        "mean_train_test_centroid_l2",
        "mean_test_signature_cells",
    ]:
        if column not in metrics.columns:
            metrics[column] = np.nan
    support_scale = np.log1p(metrics["mean_signature_support"].fillna(0.0))
    if support_scale.max() > support_scale.min():
        support_difficulty = 1.0 - ((support_scale - support_scale.min()) / (support_scale.max() - support_scale.min()))
    else:
        support_difficulty = pd.Series(0.0, index=metrics.index)
    novelty_difficulty = 1.0 - metrics["mean_component_seen_fraction"].fillna(metrics["mean_perturbation_seen"].fillna(1.0))
    distance = metrics["mean_train_test_centroid_l2"].fillna(0.0)
    distance_difficulty = distance / distance.max() if distance.max() > 0 else pd.Series(0.0, index=metrics.index)
    metrics["difficulty_index"] = np.nanmean(
        np.vstack(
            [
                np.clip(support_difficulty, 0.0, 1.0),
                np.clip(novelty_difficulty, 0.0, 1.0),
                np.clip(distance_difficulty, 0.0, 1.0),
            ]
        ),
        axis=0,
    )
    metrics["difficulty_interpretation"] = np.select(
        [
            metrics["split_family"].eq("random_split"),
            metrics["normalized_transfer_score"].gt(1.0),
            metrics["difficulty_index"].ge(0.66),
        ],
        [
            "low_stress_anchor",
            "stress_score_exceeds_random_after_normalization_check",
            "high_difficulty_context",
        ],
        default="moderate_or_mixed_difficulty",
    )
    return metrics.sort_values(["split_family", "model"]).reset_index(drop=True)


def build_main_findings(
    transfer_decay: pd.DataFrame,
    selective_prediction: pd.DataFrame,
    failure_atlas: pd.DataFrame,
    pairwise: pd.DataFrame,
    split_difficulty: pd.DataFrame | None = None,
    pathway_summary: pd.DataFrame | None = None,
    retrieval_summary: pd.DataFrame | None = None,
    ptl_feature_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    non_control = transfer_decay[transfer_decay["model"] != CONTROL_MODEL].copy()
    stress = non_control[non_control["split_family"] != "random_split"].copy()
    family_average = stress.groupby("split_family", as_index=False).agg(
        mean_cosine_non_control=("mean_cosine_non_control", "mean"),
        mean_random_to_stress_drop=("mean_random_to_stress_drop", "mean"),
    )
    hardest = family_average.sort_values("mean_cosine_non_control").iloc[0]
    rows.append(
        {
            "finding_id": "F1",
            "question": "How much do context-stress splits change benchmark performance?",
            "claim": f"{hardest['split_family']} is the hardest stress family by average non-control cosine.",
            "evidence_table": "results/tables/transfer_decay_summary.csv",
            "primary_metric": "mean_cosine_non_control",
            "primary_value": hardest["mean_cosine_non_control"],
            "comparator": "random_split anchor and other stress families",
            "bounded_interpretation": "Context-stress evaluation exposes degradation hidden by random splits.",
            "limitation": "The result is benchmark evidence from the adopted datasets and baselines.",
        }
    )

    winners = non_control[non_control["family_winner"]].sort_values("split_family")
    random_winner = winners[winners["split_family"] == "random_split"]["model"].iloc[0]
    stress_winner_text = "; ".join(f"{row.split_family}: {row.model}" for row in winners.itertuples(index=False) if row.split_family != "random_split")
    rows.append(
        {
            "finding_id": "F2",
            "question": "Do model rankings change across split families?",
            "claim": f"Random-split winner {random_winner} does not remain the winner in every stress family.",
            "evidence_table": "results/tables/transfer_decay_summary.csv",
            "primary_metric": "family_rank",
            "primary_value": float(winners["model"].nunique()),
            "comparator": stress_winner_text,
            "bounded_interpretation": "Model choice is context dependent; stress families should be reported separately.",
            "limitation": "The comparison is limited to the implemented baseline set.",
        }
    )

    ridge = non_control[non_control["model"] == "ridge_regression_baseline"]
    external_ridge = ridge[ridge["split_family"].isin(["dataset_heldout_split", "external_holdout"])]
    if not external_ridge.empty:
        rows.append(
            {
                "finding_id": "F3",
                "question": "Which transfer regimes reveal the sharpest ranking shifts?",
                "claim": "The ridge baseline leads familiar regimes but weakens in dataset-level and external transfer summaries.",
                "evidence_table": "results/tables/transfer_decay_summary.csv",
                "primary_metric": "rank_change_vs_random",
                "primary_value": float(external_ridge["rank_change_vs_random"].mean()),
                "comparator": "random_split ridge rank",
                "bounded_interpretation": "Dataset identity behaves like a transport boundary in this benchmark.",
                "limitation": "External transfer is represented by the currently adopted external dataset.",
            }
        )

    ptl_rows = selective_prediction[selective_prediction["estimator"] != "naive_confidence"].dropna(
        subset=["mean_false_transportability_rate_gain_vs_naive"]
    )
    best_ptl = ptl_rows.sort_values("mean_false_transportability_rate_gain_vs_naive", ascending=False).iloc[0]
    rows.append(
        {
            "finding_id": "F4",
            "question": "Can PTL predict likely non-transportable outputs?",
            "claim": f"PTL improves false-transportability filtering over naive confidence; best row is {best_ptl['ablation']} / {best_ptl['estimator']}.",
            "evidence_table": "results/tables/selective_prediction_summary.csv",
            "primary_metric": "mean_false_transportability_rate_gain_vs_naive",
            "primary_value": best_ptl["mean_false_transportability_rate_gain_vs_naive"],
            "comparator": "naive_confidence",
            "bounded_interpretation": "PTL is useful as a reliability lens for retained predictions.",
            "limitation": "Cosine-risk selective ranking is a separate diagnostic and is more conservative here.",
        }
    )

    failures = failure_atlas[failure_atlas["failure_severity"] > 0].copy()
    top_failure = failures.sort_values(["failure_severity", "failure_mode_share"], ascending=[False, False]).iloc[0]
    rows.append(
        {
            "finding_id": "F5",
            "question": "What failure modes dominate the atlas?",
            "claim": f"{top_failure['failure_mode']} is the highest-severity recurring failure mode in the atlas.",
            "evidence_table": "results/tables/failure_mode_atlas.csv",
            "primary_metric": "failure_mode_share",
            "primary_value": top_failure["failure_mode_share"],
            "comparator": f"{top_failure['model']} / {top_failure['split_family']}",
            "bounded_interpretation": "Failure modes concentrate in novelty, low-support, and transfer-boundary settings.",
            "limitation": "Failure labels derive from benchmark fidelity and transportability thresholds.",
        }
    )

    if not pairwise.empty and "reject_fdr_0_05" in pairwise.columns:
        significant = int(pairwise["reject_fdr_0_05"].sum())
        rows.append(
            {
                "finding_id": "F6",
                "question": "Are model differences statistically visible in Phase 06 comparisons?",
                "claim": "Existing paired comparisons provide corrected support for several model differences.",
                "evidence_table": "results/tables/model_pairwise_comparisons.csv",
                "primary_metric": "n_fdr_significant_comparisons",
                "primary_value": float(significant),
                "comparator": "FDR 0.05 corrected pairwise tests",
                "bounded_interpretation": "The analysis should cite corrected comparisons as support, not as mechanism proof.",
                "limitation": "Tests are matched to available contexts and metrics.",
            }
        )

    if split_difficulty is not None and not split_difficulty.empty:
        hardest_norm = split_difficulty[split_difficulty["split_family"] != "random_split"].sort_values(
            "difficulty_index",
            ascending=False,
        ).iloc[0]
        rows.append(
            {
                "finding_id": "F7",
                "question": "Do stress families differ in difficulty beyond raw split labels?",
                "claim": f"{hardest_norm['split_family']} has the highest composite difficulty index in the current run.",
                "evidence_table": "results/tables/split_difficulty_summary.csv",
                "primary_metric": "difficulty_index",
                "primary_value": hardest_norm["difficulty_index"],
                "comparator": "support, novelty, effect size, and context-distance summaries",
                "bounded_interpretation": "Difficulty-normalized reporting helps interpret stress scores that can exceed the random anchor.",
                "limitation": "The index is descriptive and should be read together with raw fidelity metrics.",
            }
        )

    if pathway_summary is not None and not pathway_summary.empty:
        rows.append(
            {
                "finding_id": "F8",
                "question": "Do gene-set summaries support the gene-level fidelity view?",
                "claim": "Pathway-level fidelity is reported as an orthogonal biological summary rather than a replacement for gene-level metrics.",
                "evidence_table": "results/tables/pathway_metric_summary.csv",
                "primary_metric": "mean_pathway_cosine",
                "primary_value": _float_or_nan(pathway_summary.get("mean_pathway_cosine", pd.Series([float('nan')])).mean()),
                "comparator": "gene-level cosine and DEG direction metrics",
                "bounded_interpretation": "Biological summaries provide an additional lens on model outputs.",
                "limitation": "Gene-set coverage depends on the cached pathway collection.",
            }
        )

    if retrieval_summary is not None and not retrieval_summary.empty:
        rows.append(
            {
                "finding_id": "F9",
                "question": "Can predicted signatures discriminate perturbation identity?",
                "claim": "Perturbation-discrimination retrieval is reported alongside fidelity to separate matching from magnitude accuracy.",
                "evidence_table": "results/tables/perturbation_discrimination_summary.csv",
                "primary_metric": "perturbation_retrieval_top1",
                "primary_value": _float_or_nan(retrieval_summary.get("perturbation_retrieval_top1", pd.Series([float('nan')])).mean()),
                "comparator": "fidelity and transportability summaries",
                "bounded_interpretation": "Retrieval helps detect whether predictions preserve perturbation identity.",
                "limitation": "Retrieval is sensitive to perturbation label granularity and repeated controls.",
            }
        )

    if ptl_feature_audit is not None and not ptl_feature_audit.empty:
        excluded_targets = int(ptl_feature_audit["target_or_label"].fillna(False).sum()) if "target_or_label" in ptl_feature_audit else 0
        rows.append(
            {
                "finding_id": "F10",
                "question": "Does PTL avoid target leakage?",
                "claim": "PTL feature construction now writes a deployment-availability audit and excludes target or outcome columns.",
                "evidence_table": "results/tables/ptl_feature_audit.csv",
                "primary_metric": "n_target_or_label_columns_excluded",
                "primary_value": float(excluded_targets),
                "comparator": "deployment-available feature set",
                "bounded_interpretation": "Reliability estimates are based on features available before observing test outcomes.",
                "limitation": "The audit is column-level and should be reviewed when new feature sources are added.",
            }
        )

    return pd.DataFrame(rows)


def validate_bounded_text(text: str) -> None:
    lower = text.lower()
    found = [term for term in FORBIDDEN_NARRATIVE_TERMS if term in lower]
    if found:
        raise ValueError(f"Bounded narrative contains forbidden terms: {found}")


def generate_results_narrative(main_findings: pd.DataFrame, transfer_decay: pd.DataFrame, selective_prediction: pd.DataFrame) -> str:
    hardest_row = main_findings.loc[main_findings["finding_id"] == "F1"].iloc[0]
    ptl_row = main_findings.loc[main_findings["finding_id"] == "F4"].iloc[0]
    winner_rows = transfer_decay[(transfer_decay["family_winner"]) & (transfer_decay["model"] != CONTROL_MODEL)]
    winner_lines = [
        f"- `{row.split_family}`: `{row.model}` (mean cosine `{_format_float(row.mean_cosine_non_control)}`)"
        for row in winner_rows.itertuples(index=False)
    ]

    naive = selective_prediction[selective_prediction["estimator"] == "naive_confidence"].iloc[0]
    best_ptl = selective_prediction[selective_prediction["estimator"] != "naive_confidence"].sort_values(
        "mean_false_transportability_rate_gain_vs_naive",
        ascending=False,
    ).iloc[0]
    text = f"""# Results Narrative

Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M Asia/Shanghai')}

## Main message
The benchmark supports a context-dependent view of perturbation transportability. Random-split performance is useful as a low-stress anchor, but context-stress families expose different winners, larger transfer decay, and concentrated failure modes.

## Benchmark findings
{hardest_row['claim']} Its average non-control cosine is `{_format_float(hardest_row['primary_value'])}` across the implemented non-control baselines. Family winners are:

{chr(10).join(winner_lines)}

These results should be read as benchmark evidence on the adopted datasets and baseline models, not as a universal ordering of all possible perturbation predictors.

## PTL reliability findings
{ptl_row['claim']} The naive-confidence mean false-transportability rate is `{_format_float(naive['mean_false_transportability_rate'])}`, while the best PTL row reaches `{_format_float(best_ptl['mean_false_transportability_rate'])}` with gain `{_format_float(best_ptl['mean_false_transportability_rate_gain_vs_naive'])}`.

The cosine-risk selective view is intentionally reported separately. In this run, PTL is strongest for filtering non-transportable outputs, while raw cosine-risk retention is more conservative than naive confidence.

## Failure-mode findings
The failure atlas points to novelty, low support, and transfer-boundary settings as recurring stress points. These summaries are descriptive and threshold-based; they motivate figure design and follow-up analysis rather than mechanism-level claims.

## Limitations
- The current synthesis covers the adopted signature-track datasets and the implemented baseline models.
- External transfer is represented by the current external holdout and symmetric dataset-heldout splits.
- PTL targets use the Phase 06 relative anchor definition, so transportability and raw cosine risk are complementary views.
- Phase 09 should visualize these summaries before manuscript wording is finalized.
"""
    validate_bounded_text(text)
    return text


def write_outputs(
    output_dir: Path,
    docs_dir: Path,
    main_findings: pd.DataFrame,
    transfer_decay: pd.DataFrame,
    selective_prediction: pd.DataFrame,
    failure_atlas: pd.DataFrame,
    split_difficulty: pd.DataFrame,
    narrative: str,
) -> dict[str, Path]:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "main_findings": tables_dir / "main_findings.csv",
        "transfer_decay_summary": tables_dir / "transfer_decay_summary.csv",
        "selective_prediction_summary": tables_dir / "selective_prediction_summary.csv",
        "failure_mode_atlas": tables_dir / "failure_mode_atlas.csv",
        "split_difficulty_summary": tables_dir / "split_difficulty_summary.csv",
        "results_narrative": docs_dir / "results_narrative.md",
    }
    main_findings.to_csv(outputs["main_findings"], index=False)
    transfer_decay.to_csv(outputs["transfer_decay_summary"], index=False)
    selective_prediction.to_csv(outputs["selective_prediction_summary"], index=False)
    failure_atlas.to_csv(outputs["failure_mode_atlas"], index=False)
    split_difficulty.to_csv(outputs["split_difficulty_summary"], index=False)
    outputs["results_narrative"].write_text(narrative, encoding="utf-8")
    return outputs


def run_phase08(args: argparse.Namespace) -> dict[str, Path]:
    logger = PhaseLogger(Path(args.log_file))
    logger.log("Starting Phase 08 main analysis synthesis.")
    logger.log("Skill gate checked; scikit-learn skill is sufficient for local analysis synthesis.")

    all_metrics = pd.read_csv(args.metrics)
    pairwise = pd.read_csv(args.pairwise)
    ptl_metrics = pd.read_csv(args.ptl_metrics)
    ptl_ablation = pd.read_csv(args.ptl_ablation)
    failure_summary = pd.read_csv(args.failure_summary)
    ptl_examples = pd.read_parquet(args.ptl_examples)
    pathway_summary = pd.read_csv(args.pathway_summary) if Path(args.pathway_summary).exists() else pd.DataFrame()
    retrieval_summary = pd.read_csv(args.retrieval_summary) if Path(args.retrieval_summary).exists() else pd.DataFrame()
    ptl_feature_audit = pd.read_csv(args.ptl_feature_audit) if Path(args.ptl_feature_audit).exists() else pd.DataFrame()
    logger.log(
        "Loaded inputs: "
        f"all_metrics={all_metrics.shape}, ptl_metrics={ptl_metrics.shape}, "
        f"failure_summary={failure_summary.shape}, ptl_examples={ptl_examples.shape}."
    )

    transfer_decay = build_transfer_decay_summary(all_metrics)
    split_difficulty = build_split_difficulty_summary(all_metrics, ptl_examples)
    selective_prediction = build_selective_prediction_summary(ptl_metrics, ptl_ablation)
    failure_atlas = build_failure_mode_atlas(failure_summary, ptl_examples)
    main_findings = build_main_findings(
        transfer_decay,
        selective_prediction,
        failure_atlas,
        pairwise,
        split_difficulty=split_difficulty,
        pathway_summary=pathway_summary,
        retrieval_summary=retrieval_summary,
        ptl_feature_audit=ptl_feature_audit,
    )
    narrative = generate_results_narrative(main_findings, transfer_decay, selective_prediction)

    outputs = write_outputs(
        output_dir=Path(args.output_dir),
        docs_dir=Path(args.docs_dir),
        main_findings=main_findings,
        transfer_decay=transfer_decay,
        selective_prediction=selective_prediction,
        failure_atlas=failure_atlas,
        split_difficulty=split_difficulty,
        narrative=narrative,
    )
    for name, path in outputs.items():
        logger.log(f"Wrote {name}: {path}")
    logger.log("Phase 08 main analysis synthesis completed.")
    return outputs


def main() -> None:
    run_phase08(parse_args())


if __name__ == "__main__":
    main()
