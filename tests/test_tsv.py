from __future__ import annotations

from pathlib import Path

from conftest import CITATION_A, paper_a_rows, write_ets

from bioparser_ft.schema import DOCUMENT_COLUMNS, MODEL_COLUMNS, MODEL_HEADER
from bioparser_ft.tsv import normalize_cell, parse_model_tsv, read_ets_tsv, to_ets_tsv, to_model_tsv


def test_normalize_cell() -> None:
    assert normalize_cell("mm ") == "mm"
    assert normalize_cell(" 0.1 mm") == "0.1 mm"
    assert normalize_cell("a\tb") == "a b"
    assert normalize_cell("Ref A\r\nRef B\r\n") == "Ref A | Ref B"
    assert normalize_cell("x​y") == "xy"
    assert normalize_cell("mm²") == "mm²"  # no NFKC
    assert normalize_cell(None) == ""


def test_read_ets_handles_excel_quoting(tmp_path: Path) -> None:
    table = read_ets_tsv(write_ets(tmp_path / "a.tsv", paper_a_rows()))
    assert not table.warnings
    r0, r1, r2 = table.records
    assert r0["verbatimTraitUnit"] == "mm"  # trailing space stripped
    assert r0["taxonRank"] == "Species" and r0["sex"] == "Male" and r0["lifeStage"] == "Adult"
    assert r0["statisticalMethod"] == "SD"
    assert r0["verbatimLocality"] == "Cave X, Region Y"  # NBSP
    assert r1["verbatimTraitName"] == "Greatest Length of Skull /Breadth Across Molars"  # quoted TAB
    assert r1["associatedReferences"] == "Darwin et al., 1830 | Wallace, 1858"  # quoted CRLF
    assert r1["sex"] == "Female"  # "Famale" typo
    assert r2["sex"] == "Mixed" and r2["measurementAccuracy"] == "0.1 g"


def test_model_tsv_format(tmp_path: Path) -> None:
    records = read_ets_tsv(write_ets(tmp_path / "a.tsv", paper_a_rows())).records
    text = to_model_tsv(records)
    lines = text.split("\n")
    assert lines[0] == MODEL_HEADER and not text.endswith("\n")
    assert all(line.count("\t") == len(MODEL_COLUMNS) - 1 for line in lines)
    ref = MODEL_COLUMNS.index("references")
    assert lines[1].split("\t")[ref] == CITATION_A
    assert all(line.split("\t")[ref] == "" for line in lines[2:])  # written once
    assert "author" not in lines[0].split("\t")


def test_round_trip(tmp_path: Path) -> None:
    records = read_ets_tsv(write_ets(tmp_path / "a.tsv", paper_a_rows())).records
    parsed = parse_model_tsv(to_model_tsv(records))
    assert parsed.header_ok and not parsed.warnings and parsed.malformed_rows == 0
    assert parsed.records == [{c: r[c] for c in MODEL_COLUMNS} for r in records]


def test_ets_export_has_all_columns(tmp_path: Path) -> None:
    records = read_ets_tsv(write_ets(tmp_path / "a.tsv", paper_a_rows())).records
    lines = to_ets_tsv(records).splitlines()
    assert len(lines) == 4 and lines[0].split("\t")[-2] == "author"


def _tsv(*rows: list[str]) -> str:
    return "\n".join("\t".join(r) for r in rows)


def _row(**cells: str) -> list[str]:
    return [cells.get(c, "") for c in MODEL_COLUMNS]


def test_parse_repairs_and_warnings() -> None:
    good = _row(references="Cit.", verbatimScientificName="Canis lupus", verbatimTraitValue="1.5")
    second = _row(verbatimScientificName="Canis lupus", verbatimTraitValue="2.5")
    text = "<think>\n\n</think>\n\n```tsv\n" + _tsv(list(MODEL_COLUMNS), good, second) + "\n```"
    p = parse_model_tsv(text)
    assert p.header_ok and len(p.records) == 2
    assert p.records[1]["references"] == "Cit."  # forward-filled
    assert any("code fences" in w for w in p.warnings)

    short = parse_model_tsv(_tsv(list(MODEL_COLUMNS), good[:5]))  # too few cells
    assert short.malformed_rows == 1 and short.records[0]["verbatimTraitValue"] == ""

    cut = parse_model_tsv(_tsv(list(MODEL_COLUMNS), good, second[:3]), complete=False)
    assert len(cut.records) == 1  # partial last row dropped

    no_header = parse_model_tsv(_tsv(good))
    assert not no_header.header_ok and no_header.records[0]["verbatimTraitValue"] == "1.5"


def test_parse_maps_reordered_header_by_name() -> None:
    cols = ["verbatimTraitValue", "verbatimScientificName", "author", "references"]
    p = parse_model_tsv(_tsv(cols, ["3.3", "Mus spretus", "X", "Cit."]))
    assert p.records[0]["verbatimTraitValue"] == "3.3"
    assert p.records[0]["verbatimScientificName"] == "Mus spretus"
    assert "author" not in p.records[0]  # never taken from the model


def test_document_columns_are_known() -> None:
    assert set(DOCUMENT_COLUMNS) <= set(MODEL_COLUMNS)
