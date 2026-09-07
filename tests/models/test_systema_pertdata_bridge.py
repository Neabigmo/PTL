import sys
import types
from pathlib import Path

import pytest

from src.models.systema_pertdata_bridge import build_pert_data_from_adata, systema_dataset_name


def test_k562_essential_has_an_explicit_systema_bridge_name() -> None:
    assert systema_dataset_name("ReplogleWeissman2022_K562_essential") == "ReplogleK562_essential"


def test_bridge_rejects_unfrozen_dataset_aliases() -> None:
    with pytest.raises(ValueError, match="No PTL-side Systema mapping"):
        systema_dataset_name("ReplogleWeissman2022_K562_gwps")


def test_bridge_freezes_split_seed_independently_of_model_seed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []

    class FakePertData:
        def __init__(self, cache_dir: str, default_pert_graph: bool) -> None:
            self.cache_dir = cache_dir
            self.default_pert_graph = default_pert_graph

        def new_data_process(self, **kwargs: object) -> None:
            self.process_kwargs = kwargs

        def prepare_split(self, *, split: str, seed: int) -> None:
            calls.append((split, seed))

    monkeypatch.setitem(sys.modules, "gears", types.SimpleNamespace(PertData=FakePertData))
    for model_seed in (0, 1, 2):
        build_pert_data_from_adata(
            object(),
            "ReplogleWeissman2022_K562_essential",
            model_seed,
            tmp_path,
            split_seed=20260907,
        )

    assert calls == [("simulation", 20260907)] * 3
