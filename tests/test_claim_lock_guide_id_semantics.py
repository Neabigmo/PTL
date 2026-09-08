"""Unit tests for the guide-id semantics audit."""

from __future__ import annotations

import pandas as pd

from scripts.audit_formal_v2_guide_id_semantics import summarize_group


def test_combinatorial_guide_strings_are_counted_as_cell_tokens() -> None:
    group = pd.DataFrame({"guide_id": ["A2M_1;STAT1_1", "A2M_2", None]})
    summary = summarize_group("A2M", group)
    assert summary["n_cells_qc"] == 3
    assert summary["n_cells_with_nonmissing_guide_id"] == 2
    assert summary["n_unique_guide_strings"] == 2
    assert summary["n_unique_guide_tokens"] == 3
    assert summary["fraction_multi_token_cells"] == 1 / 3
    assert summary["fraction_cells_containing_target_token"] == 2 / 3
    assert summary["guide_id_missing_fraction"] == 1 / 3
