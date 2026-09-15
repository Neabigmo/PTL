"""Orchestrate four main and five supplementary PTL paper figures.

Figure-specific implementations live under ``scripts/figures``. This entry
point applies the shared style, runs the builders in the prescribed order,
writes source tables, and records the artifact manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ptl_figure_style import apply_style  # noqa: E402
from scripts.figures.common import build_exemplar_registry, write_source  # noqa: E402
from scripts.figures.data_contracts import validate_canonical_inputs  # noqa: E402
from scripts.figures.figure_data_manifest import write_data_map  # noqa: E402
from scripts.figures.figure1_problem import figure1  # noqa: E402
from scripts.figures.figure2_atlas import figure2  # noqa: E402
from scripts.figures.figure3_boundary import figure3  # noqa: E402
from scripts.figures.figure4_decision import figure4  # noqa: E402
from scripts.figures.supp_source_only_forecasting import figure_s1  # noqa: E402
from scripts.figures.supp_reviewer_proofing import figure_s2, figure_s3, figure_s4  # noqa: E402
from scripts.build_paper_claim_summary import run as build_paper_claim_summary  # noqa: E402
from scripts.build_ordering_vs_mean_risk import run as build_ordering_vs_mean_risk  # noqa: E402
from scripts.build_metric_matched_mean_fidelity_control import run as build_metric_matched_mean_fidelity_control  # noqa: E402
from scripts.build_stable_inversion_summary import run as build_stable_inversion_summary  # noqa: E402
from scripts.build_measurement_seed_summary import run as build_measurement_seed_summary  # noqa: E402
from scripts.run_reviewer_prospective_ladder import run as run_reviewer_prospective_ladder  # noqa: E402
from scripts.run_ordering_estimand_synthetic_validation import run as run_ordering_estimand_synthetic_validation  # noqa: E402
from scripts.build_transport_decomposition import build as build_transport_decomposition  # noqa: E402
from scripts.build_rank_comparator_benchmark import build as build_rank_comparator_benchmark  # noqa: E402
from scripts.build_reviewer_decision_incremental_value import build as build_reviewer_decision_incremental_value  # noqa: E402
from scripts.run_target_audit_simulation import build as build_target_audit_simulation  # noqa: E402
from scripts.build_scientific_upgrade_tables import build as build_scientific_upgrade_tables  # noqa: E402
from scripts.figures.supp_upgrade_v2 import build as build_supp_upgrade_v2  # noqa: E402


def run(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    out_dir = root / "results/figures/reliability_transportability"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    validation = validate_canonical_inputs(root)
    if validation["status"] != "PASS":
        mismatch = root / "paper/iclr2027/FIGURE_DATA_MISMATCH.md"
        mismatch.write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise RuntimeError("Figure data mismatch: " + ", ".join(validation["failures"]))
    build_ordering_vs_mean_risk(root)
    build_metric_matched_mean_fidelity_control(root)
    run_ordering_estimand_synthetic_validation(root, with_figure=False)
    build_stable_inversion_summary(root)
    build_measurement_seed_summary(root)
    run_reviewer_prospective_ladder(root)
    build_paper_claim_summary(root)
    build_transport_decomposition(root)
    build_rank_comparator_benchmark(root)
    build_reviewer_decision_incremental_value(root)
    build_target_audit_simulation(root)
    build_scientific_upgrade_tables(root)
    write_data_map(root)
    registry = build_exemplar_registry(root)
    builders = [
        ("fig4", "fig4_decision_consequence.csv", figure4),
        ("fig1", "fig1_source.csv", figure1),
        ("fig3", "fig3_measurement_boundary.csv", figure3),
        ("fig2", "fig2_transport_atlas.csv", figure2),
        ("supp_fig1", "supp_fig1_source_only_forecasting.csv", figure_s1),
        ("supp_fig2", "supp_fig2_metric_stratified_decision.csv", figure_s2),
        ("supp_fig3", "supp_fig3_mean_fidelity_control.csv", figure_s3),
        ("supp_fig4", "supp_fig4_estimand_validation.csv", figure_s4),
    ]
    outputs = {}
    for key, source_name, builder in builders:
        paths, source = builder(out_dir, manifests, registry)
        outputs[key] = {"files": paths, "source_table": write_source(root, source_name, source), "rows": int(len(source))}
    # The V2 neural-family/simulation figure is supplementary evidence and is
    # generated only when its explicit result artifacts are present.  This
    # keeps the canonical four-figure build runnable before that optional
    # extension is materialized on a fresh checkout.
    if (manifests / "neural_family_transport_profiles.csv").is_file() and (manifests / "simulation_v2/finite_measurement_primary.csv").is_file():
        outputs["supp_fig5"] = {"files": build_supp_upgrade_v2(), "source_table": "artifacts/manifests/neural_family_transport_profiles.csv", "rows": 216}
        figure_order = ["problem", "atlas", "measurement_boundary", "decision_consequence", "supp_source_only_forecasting", "supp_metric_stratified_decision", "supp_mean_fidelity_control", "supp_estimand_validation", "supp_upgrade_v2"]
        render_order = ["fig4", "fig1", "fig3", "fig2", "supp_fig1", "supp_fig2", "supp_fig3", "supp_fig4", "supp_fig5"]
    else:
        figure_order = ["problem", "atlas", "measurement_boundary", "decision_consequence", "supp_source_only_forecasting", "supp_metric_stratified_decision", "supp_mean_fidelity_control", "supp_estimand_validation"]
        render_order = ["fig4", "fig1", "fig3", "fig2", "supp_fig1", "supp_fig2", "supp_fig3", "supp_fig4"]
    report = {
        "schema_version": 5,
        "status": "executed",
        "figure_count": len(outputs),
        "figure_order": figure_order,
        "render_order": render_order,
        "formats": ["png", "pdf", "svg", "tiff"],
        "source_tables_directory": "results/figures/reliability_transportability/source_tables",
        "exemplar_registry": "artifacts/manifests/figure_exemplar_registry.json",
        "visual_architecture": "paper/iclr2027/FIGURE_ARCHITECTURE.md",
        "outputs": outputs,
        "data_policy": "Frozen Frangieh/Nadig data and existing matched-fixed/bootstrap summaries; unavailable prospective cells stay gray/NA; target-informed explanatory quantities never enter prospective prediction.",
        "selection_policy": "Fig1B uses source-rank quantile exemplars from the full 243-label source-frozen surface; Fig1D is a compact count preview from the complete 243-label delta-cosine risk surface and retains the full source table; Fig4A is the unique detailed alluvial view. Supplementary Fig. S1 uses the registered 144-row source-only feature ladder U→U+G→U+G+S→U+G+S+N. S2 uses the fixed 18-point decision surface; S3 uses metric-matched full-depth risk vectors on the same fixed universes; S4 uses the canonical synthetic estimand validation; S5 uses the four-family descriptive profile and the 60-condition finite-measurement primary simulation.",
        "style_policy": "Centralized ptl_figure_style.py; metric colors are stable; biological context uses position/line style; no equal-grid heatmap dominance.",
        "anti_fabrication_checks": ["strict support >= 8", "Supplementary Fig. S1 uses 144 source-only rows", "Supplementary Fig. S2 uses exactly 18 fixed decision rows", "Supplementary Fig. S3 uses metric-specific risk rows and exact matched universes", "Supplementary Fig. S4 uses stored synthetic trials", "target_feature_leakage=False", "Nadig replication uses full-size per-label artifact", "no numeric cross-dataset prospective result", "S5 uses executed neural-family profiles and simulation-v2 summaries", "primary intervals are 90%"],
        "data_kill_switch": validation,
    }
    (manifests / "reliability_transport_figures.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
