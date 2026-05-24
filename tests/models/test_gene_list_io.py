from __future__ import annotations

from src.baselines.run_baseline import read_gene_list, write_gene_list


def test_gene_list_io_preserves_embedded_newlines(tmp_path) -> None:
    path = tmp_path / "genes.txt"
    genes = ["A", "ENSG00000229425    LOC105369302\nENSG00000229425    LOC101927745\nName: symbol, dtype: object", "B"]

    write_gene_list(path, genes)

    assert read_gene_list(path) == genes
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_gene_list_reader_accepts_legacy_plain_text(tmp_path) -> None:
    path = tmp_path / "genes.txt"
    path.write_text("A\nB\n", encoding="utf-8")

    assert read_gene_list(path) == ["A", "B"]
