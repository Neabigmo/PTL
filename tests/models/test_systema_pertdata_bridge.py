import pytest

from src.models.systema_pertdata_bridge import systema_dataset_name


def test_k562_essential_has_an_explicit_systema_bridge_name() -> None:
    assert systema_dataset_name("ReplogleWeissman2022_K562_essential") == "ReplogleK562_essential"


def test_bridge_rejects_unfrozen_dataset_aliases() -> None:
    with pytest.raises(ValueError, match="No PTL-side Systema mapping"):
        systema_dataset_name("ReplogleWeissman2022_K562_gwps")
