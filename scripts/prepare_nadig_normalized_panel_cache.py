"""Materialize the exact normalized shared panel once for Nadig workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_claim_lock_replication import (  # noqa: E402
    CONTEXTS,
    ROW_CHUNK_SIZE,
    TARGET_SUM,
    _attach_panel_caches,
    _load_context,
    _panel_cache_path,
    _panel_for_contexts,
    _read_dense_block,
)


def prepare(root: Path) -> dict[str, object]:
    root = root.resolve()
    contexts = {
        context_id: _load_context(root / path)
        for context_id, path in {
            "nadig_hepg2": "data/raw/scperturb_v1.4/NadigOConner2024_hepg2.h5ad",
            "nadig_jurkat": "data/raw/scperturb_v1.4/NadigOConner2024_jurkat.h5ad",
        }.items()
    }
    panel = _panel_for_contexts(root, contexts)
    panel_positions_by_context = {
        context_id: np.asarray([contexts[context_id]["var_names"].index(gene) for gene in panel], dtype=np.int64)
        for context_id in CONTEXTS
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "panel_gene_count": len(panel),
        "contexts": {},
        "target_sum": TARGET_SUM,
        "normalization": "raw dense X / ncounts * 10000, then log1p, float32",
    }
    for context_id in CONTEXTS:
        output = _panel_cache_path(root, context_id, panel)
        output.parent.mkdir(parents=True, exist_ok=True)
        expected_shape = (contexts[context_id]["shape"][0], len(panel))
        if output.is_file():
            cached = np.load(output, mmap_mode="r")
            if tuple(cached.shape) != expected_shape or cached.dtype != np.dtype(np.float32):
                raise ValueError(f"existing cache has unexpected shape/dtype: {output}")
        else:
            temporary = output.with_name(output.stem + ".partial.npy")
            if temporary.exists():
                raise FileExistsError(f"stale partial cache exists; inspect before retrying: {temporary}")
            cached = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=expected_shape)
            positions = panel_positions_by_context[context_id]
            with h5py.File(contexts[context_id]["path"], "r") as handle:
                x_dataset = handle["X"]
                if str(x_dataset.attrs.get("encoding-type", "")) != "array":
                    raise ValueError("replication H5AD must retain a dense raw X array")
                for start in range(0, expected_shape[0], ROW_CHUNK_SIZE):
                    end = min(start + ROW_CHUNK_SIZE, expected_shape[0])
                    block = _read_dense_block(x_dataset, start, end, positions)
                    scales = TARGET_SUM / contexts[context_id]["obs"].iloc[start:end]["ncounts"].to_numpy(dtype=np.float32)
                    block *= scales[:, None]
                    np.log1p(block, out=block)
                    if not np.isfinite(block).all():
                        raise ValueError(f"non-finite normalized panel block in {context_id}: {start}:{end}")
                    cached[start:end] = block
            cached.flush()
            del cached
            temporary.replace(output)
            cached = np.load(output, mmap_mode="r")
        manifest["contexts"][context_id] = {
            "path": output.relative_to(root).as_posix(),
            "shape": list(cached.shape),
            "dtype": str(cached.dtype),
        }
    manifest_path = root / "artifacts/manifests/nadig_normalized_panel_cache.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), **manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
