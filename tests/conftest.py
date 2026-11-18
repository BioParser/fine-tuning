"""Synthetic fixtures that reproduce the edge cases found in the client's ETS exports.

No client data lives in the repository. tests/test_real_data.py runs the same
checks on the real files when BIOPARSER_ETS_DIR points at them.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pytest

from bioparser_ft.schema import ETS_COLUMNS

CITATION_A = "Doe, J. and Roe, R., 2020. Bats of Somewhere. Journal of Mammalogy, 101(2), pp.10-20."
CITATION_B = "Smith, A., 2021. Mice of Elsewhere. Zootaxa, 5000(1), pp.1-5."
ORCID = "0000-0001-9627-8821"


FILLER = " The study reports measurements of skull and body size for many specimens." * 6


def make_pdf(text: str | list[str], pad: int = 6000) -> bytes:
    """Return a minimal PDF file with a text layer that pypdf can read. A list gives one page for each text."""
    texts = [text] if isinstance(text, str) else text
    font = 3 + 2 * len(texts)  # Objects: catalog, pages, (page, content) for each page, font
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(texts)))
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(texts)} >>".encode(),
    ]
    for i, page_text in enumerate(texts):
        content = f"BT /F1 11 Tf 50 700 Td ({page_text}) Tj ET".encode("latin-1")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font} 0 R >> >> >>".encode()
        )
        objs.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n%" + b"x" * pad + b"\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def ets_row(**cells: str) -> list[str]:
    return [cells.get(c, "") for c in ETS_COLUMNS]


def write_ets(path: Path, rows: list[list[str]]) -> Path:
    """Write like Excel does: TAB-delimited, CRLF, quotes only where needed."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(ETS_COLUMNS)
    writer.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")
    return path


def paper_a_rows() -> list[list[str]]:
    common = {"references": CITATION_A, "author": ORCID, "verbatimLocality": "Cave X, Region Y"}
    return [
        ets_row(
            **common,
            verbatimScientificName="Myotis ater",
            taxonRank="species",
            verbatimTraitName="Forearm length (FA)",
            verbatimTraitUnit="mm ",
            individualCount="12",
            measurementValue_min="36.5",
            measurementValue_max="39.1",
            dispersion="0.83",
            statisticalMethod="standard deviation",
            verbatimTraitValue="37.42",
            sex="males",
            lifeStage="adult",
            associatedReferences="Original study",
        ),
        ets_row(
            **common,
            verbatimScientificName="Myotis ater",
            taxonRank="Species",
            verbatimTraitName="Greatest Length of Skull\t/Breadth Across Molars",  # embedded TAB
            verbatimTraitUnit="unitless",
            individualCount="12",
            verbatimTraitValue="2.31",
            sex="Famale",
            associatedReferences="Darwin et al., 1830\r\nWallace, 1858",
        ),  # embedded CRLF
        ets_row(
            **common,
            verbatimScientificName="Myotis cf. muricola",
            taxonRank="Species",
            verbatimTraitName="Body mass",
            verbatimTraitUnit="g",
            verbatimTraitValue="5.85",
            measurementAccuracy=" 0.1 g",
            sex="both",
            associatedReferences="Original study",
        ),
    ]


def paper_a_markdown() -> str:
    return (
        "# Bats of Somewhere\n\n![](images/page_0_image_1.jpg)\n\n"
        "Myotis ater and Myotis cf. muricola were collected in Cave X.\n\n"
        "<table><tr><td>Species</td><td>FA</td><td>GLS/BAM</td><td>Mass</td></tr>"
        "<tr><td>M. ater</td><td>37.42 ± 0.83 (36.5–39.1), n = 12</td><td>2.31</td><td></td></tr>"
        "<tr><td>M. cf. muricola</td><td></td><td></td><td>5.85</td></tr></table>\n"
    )


def paper_a_mineru_markdown() -> str:
    """The start of the markdown.md that MinerU 4.2.3 (mineru-kit parse --format zip --tier basic)
    wrote for a synthetic PDF file with the table of paper A. MinerU 4.1.6 wrote the same text.
    Different from paper_a_markdown(): a pipe table, an ASCII hyphen in the range and inline LaTeX."""
    return (
        "# Bats of Somewhere: morphometrics of two Myotis species\n\n"
        "J. Doe and R. Roe\n\n"
        "## Abstract\n\n"
        "Myotis ater and Myotis cf. muricola were collected in Cave X, Region Y, during 2018 and 2019. "
        "We report external and cranial measurements of adult males and compare them with published "
        "values for the region. Body mass was measured to the nearest 0.1 g with a spring balance.\n\n"
        "## Results\n\n"
        "Forearm length (FA) and the ratio of the greatest length of skull to the breadth across molars "
        "(GLS/BAM) are given in Table 1. The coefficient of variation was computed as "
        "$\\mathsf{CV} = 100 \\times \\mathsf{SD} / \\bar{\\mathsf{x}}.$\n\n"
        "Table 1. External measurements of the two species (mean ± SD (range), n).\n\n"
        "| Species | FA (mm) | GLS/BAM | Mass (g) |\n"
        "| --- | --- | --- | --- |\n"
        "| M. ater | 37.42 ± 0.83 (36.5-39.1), n = 12 | 2.31 |  |\n"
        "| M. cf. muricola |  |  | 5.85 |\n\n"
        "![](images/page_0_chart_8.jpg)\n\n"
        "Figure 1. Number of captures per survey night in Cave X."
    )


def paper_b_rows() -> list[list[str]]:
    return [
        ets_row(
            references=CITATION_B,
            author=ORCID,
            verbatimScientificName="Mus musculus domesticus",
            taxonRank="Subspecies",
            verbatimTraitName="Head and body length (HB)",
            verbatimTraitUnit="mm",
            individualCount="40",
            verbatimTraitValue="81.7",
            dispersion="4.4",
            statisticalMethod="SD",
            associatedReferences="Original study",
        ),
        ets_row(
            references=CITATION_B,
            author=ORCID,
            verbatimScientificName="Mus spretus",
            taxonRank="species",
            verbatimTraitName="Tail length (T)",
            verbatimTraitUnit="mm",
            verbatimTraitValue="66.2",
            associatedReferences="Original study",
        ),
    ]


def paper_b_markdown() -> str:
    return "Mus musculus domesticus HB 81.7 ± 4.4 mm (n = 40). Mus spretus T 66.2 mm.\n"


@pytest.fixture
def ets_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ets_tsv"
    d.mkdir()
    write_ets(d / "ETS_Doe and Roe, 2020.tsv", paper_a_rows()[:2])
    write_ets(d / "ETS_Doe and Roe, 2020 part 2.tsv", paper_a_rows()[2:])  # same paper, 2nd file
    write_ets(d / "ETS_Smith, 2021.tsv", paper_b_rows())
    write_ets(d / "ETS_Mislinked, 2022.tsv", paper_b_rows())  # paired with an unrelated paper below
    write_ets(d / "ETS_Nobody, 1999.tsv", paper_b_rows())  # no parsed PDF at all
    return d


@pytest.fixture
def parsed_dir(tmp_path: Path) -> Path:
    d = tmp_path / "parsed"
    d.mkdir()
    # MinerU --format zip layout (verified for mineru-kit 4.1.6 and 4.2.3)
    with zipfile.ZipFile(d / "Doe and Roe, 2020.zip", "w") as zf:
        zf.writestr("markdown.md", paper_a_markdown())
        zf.writestr("middle_json.json", "{}")
        zf.writestr("structured_content.json", "{}")
        zf.writestr("images/page_0_image_1.jpg", b"\xff\xd8")
    (d / "Smith, 2021.md").write_text(paper_b_markdown(), encoding="utf-8")  # --format markdown
    (d / "Mislinked, 2022.md").write_text("An unrelated paper about frogs, 12.5 and 99.1.", encoding="utf-8")
    return d


@pytest.fixture(params=["tsv", "csv"])
def manifest(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    if request.param == "tsv":  # default format: names contain commas
        p = tmp_path / "manifest.tsv"
        p.write_text(
            "tsv\tmarkdown\nETS_Doe and Roe, 2020 part 2.tsv\tDoe and Roe, 2020.zip\n", encoding="utf-8"
        )
    else:  # Excel "Save as CSV" quotes names with commas
        p = tmp_path / "manifest.csv"
        p.write_text(
            'tsv,markdown\n"ETS_Doe and Roe, 2020 part 2.tsv","Doe and Roe, 2020.zip"\n', encoding="utf-8"
        )
    return p
