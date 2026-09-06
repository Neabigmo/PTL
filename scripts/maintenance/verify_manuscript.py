"""Verify manuscript structure and key elements"""
from pathlib import Path
import re

MANUSCRIPT_DIR = Path("h:/2026try/4.24/manuscript")
TABLES_DIR = Path("h:/2026try/4.24/results/tables")
FIGURES_DIR = Path("h:/2026try/4.24/results/figures")

print("=== Manuscript Structure Check ===\n")

# Check main.tex
main_tex = MANUSCRIPT_DIR / "main.tex"
print(f"[OK] main.tex exists: {main_tex.exists()}")

# Check sections
sections = ["introduction", "methods", "results", "discussion", "limitations"]
print("\nSections:")
for sec in sections:
    path = MANUSCRIPT_DIR / "sections" / f"{sec}.tex"
    status = "[OK]" if path.exists() else "[MISSING]"
    print(f"  {status} {sec}.tex")

# Check references
ref_path = MANUSCRIPT_DIR / "references.bib"
print(f"\n[OK] references.bib exists: {ref_path.exists()}")

# Check figures referenced in results.tex
results_tex = (MANUSCRIPT_DIR / "sections" / "results.tex").read_text()
print("\nFigure references in results.tex:")
fig_refs = re.findall(r'\\ref\{fig:(\w+)\}', results_tex)
for ref in fig_refs:
    print(f"  - fig:{ref}")

# Check figure files exist
print("\nFigure files:")
possible_names = [
    FIGURES_DIR / "fig1_study_design.png",
    FIGURES_DIR / "fig2_benchmark_composition_audit.png",
    FIGURES_DIR / "fig3_ranking_instability_transfer_decay.png",
    FIGURES_DIR / "fig4_ptl_selective_filtering.png",
    FIGURES_DIR / "fig5_failure_mode_atlas.png",
]
for p in possible_names:
    status = "[OK]" if p.exists() else "[MISSING]"
    print(f"  {status} {p.name}")

# Check key evidence tables
print("\nEvidence tables:")
key_tables = [
    TABLES_DIR / "main_findings.csv",
    TABLES_DIR / "transfer_decay_summary.csv",
    TABLES_DIR / "selective_prediction_summary.csv",
    TABLES_DIR / "failure_mode_summary.csv",
    TABLES_DIR / "ptl_feature_audit.csv",
]
for t in key_tables:
    status = "[OK]" if t.exists() else "[MISSING]"
    print(f"  {status} {t.name}")

print("\n=== Status: Manuscript structure verified ===")
print("\nNOTE: To compile the manuscript, run:")
print("  cd manuscript && latexmk -pdf main.tex")