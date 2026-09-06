from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_PATHS = (
    PROJECT_ROOT / "configs",
    PROJECT_ROOT / "src" / "ptl",
    PROJECT_ROOT / "paper" / "iclr2027",
)
LEGACY_ROOT = "H:\\2026try\\4.24"


def test_v2_sources_do_not_embed_legacy_absolute_paths() -> None:
    offenders = []
    for base in V2_PATHS:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() in {".pyc", ".png", ".pdf"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if LEGACY_ROOT.lower() in text.lower():
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []
