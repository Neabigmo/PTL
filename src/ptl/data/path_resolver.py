"""Portable workspace path resolution for the active PTL-v2 tree."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


LEGACY_ROOT_RE = re.compile(r"[A-Za-z]:[\\/]2026try[\\/]4\.24(?:[\\/]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class WorkspacePathResolver:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())

    def resolve(self, relative_path: str | Path) -> Path:
        raw = str(relative_path)
        if LEGACY_ROOT_RE.search(raw):
            raise ValueError(f"legacy absolute path is not valid for PTL-v2: {raw}")
        candidate = Path(relative_path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"path escapes PTL-v2 workspace: {relative_path}")
        return candidate

    def relative(self, path: str | Path) -> str:
        candidate = self.resolve(path)
        return candidate.relative_to(self.root).as_posix()
