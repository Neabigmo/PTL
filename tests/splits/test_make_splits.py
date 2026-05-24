from src.splits.make_splits import is_combination_label


def test_combination_parser_handles_guide_suffixes_and_adamson_singletons():
    assert is_combination_label("A_B")
    assert is_combination_label("ATF6_PERK_IRE1_pMJ158")
    assert not is_combination_label("ATF6_only_pMJ145")
    assert not is_combination_label("ZAP70_1")
    assert not is_combination_label("Tcrlibrary_NFKB1_1")
    assert not is_combination_label("MULTI_TARGET")
