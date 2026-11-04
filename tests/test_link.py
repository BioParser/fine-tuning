"""Tests for bioparser-link-pdfs. The files can have any name: the text and the name decide."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest
from conftest import FILLER, ets_row, make_pdf, write_ets

pytest.importorskip("pypdf")
pytest.importorskip("httpx")  # The link module uses the report functions of the fetch command

from bioparser_ft.fetch.cli import REPORT_COLUMNS
from bioparser_ft.fetch.link import Decision, build_parser, doi_in_file_name, run

CIT_BATS = "Doe, J. and Roe, R., 2020. Echolocation calls of cave bats in northern Somewhere. Journal of Mammalogy, 101(2), pp.10-20."
CIT_MICE = "Smith, A., 2021. Skull morphology of field mice from Elsewhere. Zootaxa, 5000(1), pp.1-5."
CIT_SHREWS = "Kim, B. and Lee, C., 2023. Body size variation of island shrews across the Pacific. Mammalia, 87(1), pp.1-9."
TITLE_SHREWS = "Body size variation of island shrews across the Pacific"
FROGS = "Calls of tree frogs in tropical rain forests and their seasonal variation." * 4


def _setup(tmp: Path, *, report: bool = True) -> argparse.Namespace:
    tsv = tmp / "ets_tsv"
    tsv.mkdir()
    write_ets(tsv / "ETS_Doe and Roe, 2020.tsv", [ets_row(references=CIT_BATS)])
    write_ets(tsv / "ETS_Smith, 2021.tsv", [ets_row(references=CIT_MICE)])
    write_ets(tsv / "ETS_Kim and Lee, 2023.tsv", [ets_row(references=CIT_SHREWS)])
    (tmp / "pdfs" / "_unverified").mkdir(parents=True)
    if report:  # The fetch command found these DOIs with Crossref
        with (tmp / "pdf_sources.tsv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, REPORT_COLUMNS, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerow(
                {"key": "Smith, 2021", "status": "not_found", "confidence": "high", "doi": "10.1000/mice"}
            )
            w.writerow(
                {"key": "Kim and Lee, 2023", "status": "not_found", "confidence": "high",
                 "doi": "10.1093/jmammal/gyac068", "matched_title": TITLE_SHREWS}
            )  # fmt: skip
    return _args(tmp)


def _args(tmp: Path, *extra: str) -> argparse.Namespace:
    return build_parser().parse_args([
        "--tsv-dir", str(tmp / "ets_tsv"), "--pdf-dir", str(tmp / "pdfs"),
        "--report", str(tmp / "pdf_sources.tsv"), "--overrides", str(tmp / "none.tsv"), *extra,
    ])  # fmt: skip


def _put(tmp: Path, name: str, text: str, folder: str = "_unverified") -> Path:
    path = tmp / "pdfs" / folder / name
    path.write_bytes(make_pdf(text))
    return path


def _by_name(decisions: list[Decision]) -> dict[str, Decision]:
    return {d.path.name: d for d in decisions}


def test_links_by_doi_name_and_title(tmp_path: Path) -> None:
    _setup(tmp_path)
    _put(tmp_path, "67640adb47e5f1734609627.pdf", "Received 2021. https://doi.org/10.1000/mice" + FILLER)
    _put(tmp_path, "gyac068.pdf", TITLE_SHREWS + " Kim B and Lee C" + FILLER)
    wiley = "Journal of Mammalogy - 2020 - Doe - Echolocation calls of cave bats in northern Somewhere.pdf"
    _put(tmp_path, wiley, FILLER)  # Only the file name gives the title
    _put(tmp_path, "unrelated.pdf", FROGS)

    dry = _by_name(run(_args(tmp_path)))
    assert {n: d.action for n, d in dry.items()} == {
        "67640adb47e5f1734609627.pdf": "move",
        "gyac068.pdf": "move",
        wiley: "move",
        "unrelated.pdf": "review",
    }
    assert dry["67640adb47e5f1734609627.pdf"].evidence.doi_in_text
    assert dry["gyac068.pdf"].evidence.doi_in_name
    assert dry["unrelated.pdf"].target is None
    assert not (tmp_path / "pdfs" / "Smith, 2021.pdf").exists()  # A dry run moves no file

    run(_args(tmp_path, "--apply"))
    pdfs = tmp_path / "pdfs"
    assert sorted(p.name for p in pdfs.glob("*.pdf")) == [
        "Doe and Roe, 2020.pdf",
        "Kim and Lee, 2023.pdf",
        "Smith, 2021.pdf",
    ]
    assert [p.name for p in (pdfs / "_unverified").iterdir()] == ["unrelated.pdf"]
    with (tmp_path / "pdf_sources.tsv").open(encoding="utf-8") as fh:
        rows = {r["key"]: r for r in csv.DictReader(fh, delimiter="\t")}
    assert rows["Smith, 2021"]["status"] == "downloaded" and rows["Smith, 2021"]["pdf_source"] == "manual"
    assert rows["Smith, 2021"]["pdf_url"] == "67640adb47e5f1734609627.pdf"
    assert len(rows["Smith, 2021"]["sha256"]) == 64 and rows["Smith, 2021"]["verified"] == "yes"
    assert rows["Doe and Roe, 2020"]["status"] == "downloaded"  # A new row for a paper without a row


def test_never_replaces_a_pdf_in_pdfs(tmp_path: Path) -> None:
    _setup(tmp_path)
    existing = _put(tmp_path, "Smith, 2021.pdf", "the first copy" + FILLER, folder=".")
    _put(tmp_path, "copy.pdf", "https://doi.org/10.1000/mice" + FILLER)
    decisions = _by_name(run(_args(tmp_path, "--apply")))
    assert decisions["copy.pdf"].action == "duplicate"
    assert (tmp_path / "pdfs" / "_unverified" / "copy.pdf").exists()
    assert existing.read_bytes() == make_pdf("the first copy" + FILLER)


def test_two_files_for_one_paper(tmp_path: Path) -> None:
    _setup(tmp_path)
    _put(
        tmp_path,
        "a.pdf",
        "https://doi.org/10.1000/mice Skull morphology of field mice from Elsewhere" + FILLER,
    )
    _put(tmp_path, "b.pdf", "Skull morphology of field mice from Elsewhere. Smith A. 2021" + FILLER)
    decisions = _by_name(run(_args(tmp_path)))
    assert decisions["a.pdf"].action == "move"  # The DOI is better evidence than the title
    assert decisions["b.pdf"].action == "duplicate" and "a.pdf" in decisions["b.pdf"].reason


def test_similar_titles_go_to_review(tmp_path: Path) -> None:
    """Two papers with the same title, author and year: a person must decide."""
    tsv = tmp_path / "ets_tsv"
    tsv.mkdir()
    for n in ("1", "2"):
        cit = f"Park, D., 2022. Bats of the northern islands: morphology and diet. J. Bats, {n}(1), pp.1-2."
        write_ets(tsv / f"ETS_Park, 2022 part {n}.tsv", [ets_row(references=cit)])
    (tmp_path / "pdfs" / "_unverified").mkdir(parents=True)
    _put(tmp_path, "park.pdf", "Bats of the northern islands: morphology and diet. Park D. 2022" + FILLER)
    (d,) = run(_args(tmp_path))
    assert d.action == "review" and "two papers agree" in d.reason


def test_scanned_pdf_goes_to_review(tmp_path: Path) -> None:
    _setup(tmp_path)
    _put(tmp_path, "gyac068.pdf", "x")  # The DOI is in the name, but the file has no text layer
    (d,) = run(_args(tmp_path))
    assert d.action == "review" and d.reason.startswith("no text layer")
    assert d.target is not None and d.target.paper.key == "Kim and Lee, 2023"  # The best guess


def test_misnamed_file_in_pdfs_and_not_a_pdf(tmp_path: Path) -> None:
    _setup(tmp_path)
    _put(tmp_path, "download (3).pdf", "https://doi.org/10.1000/mice" + FILLER, folder=".")
    (tmp_path / "pdfs" / "_unverified" / "error.pdf").write_text("<html>Access denied</html>")
    decisions = _by_name(run(_args(tmp_path, "--apply")))
    assert decisions["download (3).pdf"].action == "move"
    assert (tmp_path / "pdfs" / "Smith, 2021.pdf").exists()
    assert decisions["error.pdf"].action == "skip" and "not a PDF" in decisions["error.pdf"].reason


def test_link_after_review(tmp_path: Path) -> None:
    _setup(tmp_path)
    _put(tmp_path, "scan.pdf", "x")
    with pytest.raises(SystemExit, match="Similar names"):
        run(_args(tmp_path, "--link", "scan.pdf=Kim and Le, 2023"))
    (d,) = run(_args(tmp_path, "--apply", "--link", "scan.pdf=ETS_Kim and Lee, 2023.tsv"))
    assert d.action == "move" and d.forced
    assert (tmp_path / "pdfs" / "Kim and Lee, 2023.pdf").exists()
    with (tmp_path / "pdf_sources.tsv").open(encoding="utf-8") as fh:
        rows = {r["key"]: r for r in csv.DictReader(fh, delimiter="\t")}
    assert rows["Kim and Lee, 2023"]["verified"] == "manual"


def test_no_tsv_files(tmp_path: Path) -> None:
    (tmp_path / "ets_tsv").mkdir()
    with pytest.raises(SystemExit, match="no ETS TSV files"):
        run(_args(tmp_path))


@pytest.mark.parametrize(
    ("doi", "name", "expected"),
    [
        ("10.1093/jmammal/gyac068", "gyac068.pdf", True),
        ("10.3161/15081109ACC2022.24.2.001", "15081109ACC2022.24.2.001.pdf", True),
        ("10.1093/jmammal/gyac068", "10.1093_jmammal_gyac068.pdf", True),
        ("10.1016/j.mambio.2023.01.004", "1-s2.0-S161650472300004X-main.pdf", False),
        ("10.1093/jmammal/gyac068", "gyac06.pdf", False),
        ("10.1000/ab12", "ab12.pdf", False),  # Too short to identify a file
    ],
)
def test_doi_in_file_name(doi: str, name: str, expected: bool) -> None:
    assert doi_in_file_name(doi, name) is expected
