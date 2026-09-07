"""PTL-side bridge for pinned Systema ``PertData`` consumers.

Systema is kept as a pinned, non-vendored dependency. Its ``data.py`` does
not expose a Replogle K562-essential entry, so adapters can use this small
bridge after they have constructed a PTL-approved AnnData object. The bridge
does not alter Systema source or silently substitute another dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


SYSTEMA_DATASET_ALIASES = {
    "ReplogleWeissman2022_K562_essential": "ReplogleK562_essential",
    "replogle_k562_essential": "ReplogleK562_essential",
}


def systema_dataset_name(dataset_id: str) -> str:
    """Return the explicit Systema name for a PTL dataset ID."""

    try:
        return SYSTEMA_DATASET_ALIASES[dataset_id]
    except KeyError as exc:
        raise ValueError(
            f"No PTL-side Systema mapping is frozen for dataset {dataset_id!r}; "
            "do not fall back to a different cell context."
        ) from exc


def build_pert_data_from_adata(
    adata: Any,
    dataset_id: str,
    seed: int,
    data_dir: str | Path,
    *,
    split: str = "simulation",
    default_pert_graph: bool = True,
) -> Any:
    """Create Systema/GEARS ``PertData`` from a PTL-prepared AnnData.

    The caller owns preprocessing and must provide the already validated raw
    expression surface. This function only wires the missing dataset entry,
    creates a dataset-local cache directory, and asks the pinned API to build
    its standard split.
    """

    systema_name = systema_dataset_name(dataset_id)
    if adata is None:
        raise ValueError("K562-essential bridge requires an explicit PTL-prepared AnnData object")
    try:
        from gears import PertData
    except ImportError as exc:  # pragma: no cover - depends on optional stack
        raise RuntimeError("The K562-essential bridge requires the installed GEARS PertData API") from exc

    cache_dir = Path(data_dir) / "ptl_systema" / systema_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    pert_data = PertData(str(cache_dir), default_pert_graph=default_pert_graph)
    pert_data.new_data_process(
        dataset_name=f"ptl_{systema_name.lower()}_seed{int(seed)}",
        adata=adata,
        skip_calc_de=False,
    )
    pert_data.prepare_split(split=split, seed=int(seed))
    return pert_data
