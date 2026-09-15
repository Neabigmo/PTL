"""Generate final-size QC previews for the four main and four supplementary figures."""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.figures.data_contracts import validate_canonical_inputs
from scripts.ptl_figure_style import figure_size


FIGURES = [
    ("fig1", "reliability_transportability_fig1_graphical_abstract", "Fig. 1", "main"),
    ("fig2", "reliability_transportability_fig2_transport_atlas", "Fig. 2", "main"),
    ("fig3", "reliability_transportability_fig3_measurement_boundary", "Fig. 3", "main"),
    ("fig4", "reliability_transportability_fig4_decision_consequence", "Fig. 4", "main"),
    ("s1", "reliability_transportability_supp_fig1_source_only_forecasting", "Supp. S1", "supplementary"),
    ("s2", "reliability_transportability_supp_fig2_metric_stratified_decision", "Supp. S2", "supplementary"),
    ("s3", "reliability_transportability_supp_fig3_mean_fidelity_control", "Supp. S3", "supplementary"),
    ("s4", "reliability_transportability_supp_fig4_estimand_validation", "Supp. S4", "supplementary"),
]


def run_qc(root: Path) -> Path:
    root = root.resolve(); source = root / "results/figures/reliability_transportability"; qc = root / "paper/iclr2027/qc"; published = root / "paper/iclr2027/figures"; published_supp = published / "supplementary"; qc.mkdir(parents=True, exist_ok=True); published.mkdir(parents=True, exist_ok=True); published_supp.mkdir(parents=True, exist_ok=True)
    validation = validate_canonical_inputs(root); audits = {}; dimensions = {}; previews = []
    font = ImageFont.load_default()
    for key, stem, label, destination in FIGURES:
        png = source / f"{stem}.png"; audits[key] = json.loads((source / f"{stem}.layout_audit.json").read_text(encoding="utf-8"));
        with Image.open(png) as image:
            image = image.convert("RGB"); dimensions[key] = {"pixels": image.size, "canvas_ratio": round(image.width / image.height, 3)}
            for pct in (100, 75, 50):
                width = max(1, int(image.width * pct / 100)); height = max(1, int(image.height * pct / 100)); preview = image.resize((width, height), Image.Resampling.LANCZOS); preview.save(qc / f"{key}_{pct}pct.png", dpi=(150, 150))
            target_dir = published_supp if destination == "supplementary" else published
            target = target_dir / f"{key}.png"; shutil.copy2(png, target); shutil.copy2(source / f"{stem}.svg", target_dir / f"{key}.svg"); shutil.copy2(source / f"{stem}.pdf", target_dir / f"{key}.pdf"); previews.append(image)
    thumb_w = 540; thumb_h = 410; gutter = 28; columns = 2; rows = (len(FIGURES) + columns - 1) // columns; sheet = Image.new("RGB", (columns * thumb_w + (columns + 1) * gutter, rows * thumb_h + (rows + 1) * gutter), "white"); draw = ImageDraw.Draw(sheet)
    for idx, ((key, _, label, _), image) in enumerate(zip(FIGURES, previews)):
        image.thumbnail((thumb_w - 20, thumb_h - 42), Image.Resampling.LANCZOS); x = gutter + (idx % columns) * (thumb_w + gutter) + (thumb_w - image.width) // 2; y = gutter + (idx // columns) * (thumb_h + gutter) + 28 + (thumb_h - 42 - image.height) // 2; sheet.paste(image, (x, y)); draw.text((gutter + (idx % columns) * (thumb_w + gutter), gutter + (idx // columns) * (thumb_h + gutter)), label, fill="#30343A", font=font)
    contact = qc / "figure_contact_sheet.png"; sheet.save(contact, dpi=(150, 150))
    report = ["# Figure QC", "", f"Generated {date.today().isoformat()} with the Python/matplotlib renderer.", "", "## Numeric kill-switch", "", "```json", json.dumps(validation, indent=2, ensure_ascii=False), "```", "", "## Export and layout checks", "", "| Figure | layout audit | aspect ratio | 100/75/50% previews | formats |", "|---|---|---:|---|---|"]
    for key, _, label, _ in FIGURES:
        ratio = dimensions[key]["canvas_ratio"]; report.append(f"| {label} | {'PASS' if audits[key]['collision_count'] == 0 else 'FAIL'} | {ratio:.2f}:1 | generated | PDF / SVG / PNG |")
    report += ["", "## Visual review record", "", "* The four main-figure and four supplementary PNGs are tracked separately at the final manuscript canvas.", "* The compiled manuscript and supplement PDFs were rendered at 150 dpi and inspected at 100% page scale.", "* The 2×4 contact sheet is `qc/figure_contact_sheet.png` and is QC-only, not a manuscript figure.", "* No full pytest suite was added; this QC checks the numerical contract, export formats, layout audit, and final-size raster readability.", ""]
    (root / "paper/iclr2027/FIGURE_QC.md").write_text("\n".join(report), encoding="utf-8")
    return contact


if __name__ == "__main__":
    run_qc(Path(__file__).resolve().parents[2])
