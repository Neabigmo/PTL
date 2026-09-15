"""Build compact supplementary tables from canonical manuscript artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "iclr2027"


METRIC_NAMES = {
    "delta_cosine": "Delta cosine",
    "systema_centroid_accuracy": "Systema centroid",
    "absolute_effect_rank_agreement": "Absolute-effect rank",
}


def tex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def build_nadig_table() -> None:
    # These artifacts are the prespecified matched-universe sensitivity
    # summaries. They aggregate the two Nadig directions within each
    # eligibility universe, so they must not be relabelled as direction-level
    # full-size replication rows.
    source_files = [
        (
            "min20 / 1255",
            ROOT / "artifacts" / "manifests" / "formal_v2_claim_lock_replication_nadig_matched1255_macro.csv",
        ),
        (
            "min40 / 345",
            ROOT / "artifacts" / "manifests" / "formal_v2_claim_lock_replication_nadig_matched345_macro.csv",
        ),
    ]
    frames = []
    for universe, path in source_files:
        frame = pd.read_csv(path)
        assert len(frame) == 3, f"expected three metric rows in {path.name}, found {len(frame)}"
        frame = frame.copy()
        frame["universe"] = universe
        frame["metric_name"] = frame["metric"].map(METRIC_NAMES)
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    rows["metric_order"] = rows["metric"].map(
        {name: index for index, name in enumerate(METRIC_NAMES)}
    )
    rows = rows.sort_values(["universe", "metric_order"])
    body = "\n".join(
        "{} & {} & {:.3f} [{:.3f}, {:.3f}] \\\\".format(
            row.universe,
            row.metric_name,
            row.ordering_delta_meas_id_macro,
            row.ordering_delta_meas_id_macro_ci_low,
            row.ordering_delta_meas_id_macro_ci_high,
        )
        for row in rows.itertuples()
    )
    text = """\\begin{table}[H]
\\centering
\\caption{Nadig matched-universe sensitivity: measurement-adjusted ordering disagreement ($D_{\\rm adj}$; 90\\% bootstrap interval) at two minimum-support thresholds.}
\\label{tab:nadig}
\\small
\\begin{tabular}{lll}
\\toprule
Universe & Metric & $D_{\\rm adj}$ [90\\% CI] \\\\
\\midrule
""" + body + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    (PAPER / "supplement_nadig_table.tex").write_text(text, encoding="utf-8")


def build_decision_table() -> None:
    table = pd.read_csv(ROOT / "artifacts" / "manifests" / "reviewer_decision_budget_loto.csv")
    all_rows = table.loc[table["scope"].eq("all_18_metric_transfer_points")].copy()
    loto = table.loc[table["scope"].eq("leave_one_unordered_context_pair_out")].copy()
    assert len(all_rows) == 8, f"expected eight aggregate decision rows, found {len(all_rows)}"
    budget = {0.05: "5\\%", 0.10: "10\\%", 0.20: "20\\%", 0.50: "50\\%"}
    scale = {
        "normalized_regret": "random-reference normalized regret",
        "worst_oracle_normalized_regret": "worst-oracle-normalized regret",
    }
    rows = all_rows.merge(
        loto.groupby(["budget_fraction", "regret_scale"], as_index=False)["spearman"].agg(["min", "max"]).reset_index(),
        on=["budget_fraction", "regret_scale"],
        how="left",
    ).sort_values(["budget_fraction", "regret_scale"])
    primary_report = json.loads(
        (ROOT / "artifacts" / "manifests" / "reliability_transport_decision_link.json").read_text(encoding="utf-8")
    )
    primary_rho = float(primary_report["correlations"]["overall"]["spearman"])
    table_rho = float(
        rows.loc[
            rows["regret_scale"].eq("normalized_regret")
            & rows["budget_fraction"].astype(float).eq(0.10),
            "spearman",
        ].iloc[0]
    )
    if abs(primary_rho - table_rho) > 1e-12:
        raise ValueError(f"10% decision robustness rho {table_rho} does not match Fig. 4B rho {primary_rho}")
    body = "\n".join(
        "{} & {} & {:.3f} & [{:.3f}, {:.3f}] \\\\".format(
            budget[float(row[rows.columns.get_loc("budget_fraction")])],
            scale[row[rows.columns.get_loc("regret_scale")]],
            row[rows.columns.get_loc("spearman")],
            row[rows.columns.get_loc("min")],
            row[rows.columns.get_loc("max")],
        )
        for row in rows.itertuples(index=False, name=None)
    )
    text = """\\begin{table}[H]
\\centering
\\caption{Decision-link robustness across shortlist budgets and regret normalizations. The random-reference rows use the same canonical full-depth decision surface as Fig.~4B; the bounded worst-oracle rows are a secondary source-frozen diagnostic. The last column is the leave-one-unordered-context-pair-out range.}
\\label{tab:decision-robustness}
\\small
\\begin{tabular}{llcc}
\\toprule
Budget & Regret scale & $\\rho$ & Unordered-pair LOPO range \\\\
\\midrule
""" + body + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    (PAPER / "supplement_decision_robustness_table.tex").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    build_nadig_table()
    build_decision_table()
