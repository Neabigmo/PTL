"""Orchestrate the six canonical PTL paper figures.

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
from scripts.figures.figure1_problem import figure1  # noqa: E402
from scripts.figures.figure2_atlas import figure2  # noqa: E402
from scripts.figures.figure3_boundary import figure3  # noqa: E402
from scripts.figures.figure4_decision import figure4  # noqa: E402
from scripts.figures.figure5_anatomy import figure5  # noqa: E402
from scripts.figures.figure6_prospective import figure6  # noqa: E402


def run(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    out_dir = root / "results/figures/reliability_transportability"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    registry = build_exemplar_registry(root)
    builders = [
        ("fig4", "fig4_decision_consequence.csv", figure4),
        ("fig5", "fig5_failure_anatomy.csv", figure5),
        ("fig1", "fig1_source.csv", figure1),
        ("fig3", "fig3_measurement_boundary.csv", figure3),
        ("fig2", "fig2_transport_atlas.csv", figure2),
        ("fig6", "fig6_prospective_limit.csv", figure6),
    ]
    outputs = {}
    for key, source_name, builder in builders:
        paths, source = builder(out_dir, manifests, registry)
        outputs[key] = {"files": paths, "source_table": write_source(root, source_name, source), "rows": int(len(source))}
    report = {
        "schema_version": 4,
        "status": "executed",
        "figure_count": 6,
        "figure_order": ["problem", "atlas", "measurement_boundary", "decision_consequence", "failure_anatomy", "prospective_limit"],
        "render_order": ["fig4", "fig5", "fig1", "fig3", "fig2", "fig6"],
        "formats": ["png", "pdf", "svg", "tiff"],
        "source_tables_directory": "results/figures/reliability_transportability/source_tables",
        "exemplar_registry": "artifacts/manifests/figure_exemplar_registry.json",
        "visual_architecture": "paper/iclr2027/FIGURE_ARCHITECTURE.md",
        "outputs": outputs,
        "data_policy": "Frozen Frangieh/Nadig data and existing matched-fixed/bootstrap summaries; unavailable cells stay gray/NA; target-informed explanatory quantities never enter prospective prediction.",
        "selection_policy": "Fig1/2/4 displayed labels use source-only rank quantiles; Fig5 cases use complete burden quantiles; all selections are registered deterministically.",
        "style_policy": "Centralized ptl_figure_style.py; metric colors are stable; biological context uses position/line style; no equal-grid heatmap dominance.",
        "anti_fabrication_checks": ["strict support >= 8", "219/2784 complete failure rows with no imputation", "Fig6 feature firewall", "no numeric Nadig prospective result", "primary intervals are 90%"],
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
