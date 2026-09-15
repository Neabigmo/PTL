"""Materialize the complete outcome-blind registry as a compact LaTeX table."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def tex(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_\allowbreak{}",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "≥": r"$\geq$",
        "—": "--",
        "…": r"\ldots{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def compact(value: str, limit: int = 120) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def main() -> None:
    source = ROOT / "artifacts/manifests/reordering_replication_candidate_registry.csv"
    output = ROOT / "paper/iclr2027/registry_table.tex"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{longtable}{@{}p{0.20in}p{0.55in}p{0.95in}r p{0.55in}p{0.55in}p{0.50in}p{0.62in}p{0.65in}p{1.05in}@{}}",
        r"\caption{Complete outcome-blind candidate registry (33 candidate pairs). `B/P/R` denotes outcome-blind selection, prediction evaluated, and risk evaluated; all rows were selected with $B=1$, $P=0$, and $R=0$. The full unabridged fields and candidate identifiers are in the canonical CSV.}\\",
        r"\toprule",
        r"\# & Study & Contexts & Shared & Raw cells & Controls & Assay & Coverage & Provenance & Decision / exclusion \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\# & Study & Contexts & Shared & Raw cells & Controls & Assay & Coverage & Provenance & Decision / exclusion \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{10}{r@{}}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for index, row in enumerate(rows, start=1):
        contexts = f"{row['context_a']} / {row['context_b']}"
        raw = f"{row['raw_cells_a'] or '—'} / {row['raw_cells_b'] or '—'}"
        controls = f"{row['control_cells_a'] or '—'} / {row['control_cells_b'] or '—'}"
        if row["same_perturbation_modality"] == "True" and row["same_readout_modality"] == "True":
            assay = f"{row['perturbation_modality_a']} / {row['readout_modality_a']}"
        else:
            assay = "—"
        median = row["median_min_cells_per_shared_perturbation"]
        if median:
            coverage = f"med {float(median):.0f}; ≥20 {float(row['fraction_shared_ge20_cells']):.2f}; ≥40 {float(row['fraction_shared_ge40_cells']):.2f}"
        else:
            coverage = "—"
        flags = f"B/P/R={row['outcome_blind']}/{row['prediction_evaluated']}/{row['risk_evaluated']}"
        decision = f"{flags}; {'eligible' if row['eligible'] == 'True' else 'excluded'}: {compact(row['reason'])}"
        lines.append(
            " & ".join([
                tex(index),
                tex(row["study"]),
                tex(contexts),
                tex(row["exact_shared_perturbation_count"]),
                tex(raw),
                tex(controls),
                tex(assay),
                tex(coverage),
                tex(row["batch_context_status"]),
                tex(decision),
            ]) + r" \\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
