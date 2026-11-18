"""The API uses clean_markdown and normalize_cell. They must stay fast on hostile input."""

from __future__ import annotations

import time

import pytest

from bioparser_ft.markdown import clean_markdown
from bioparser_ft.tsv import normalize_cell

LIMIT_S = 1.0  # Linear code needs some milliseconds for 200 KB. Quadratic code needs minutes.


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(" " * 200_000 + "x", id="spaces"),
        pytest.param("![" * 100_000, id="image-alt"),
        pytest.param("![](" * 50_000, id="image-path"),
        pytest.param("# " + "1" * 200_000 + "x", id="heading-number"),
        pytest.param("# " + " " * 200_000, id="heading-spaces"),
        pytest.param("# References\n" + "x\n" * 100_000, id="long-back-matter"),
        pytest.param("$" + "1 " * 100_000 + "$", id="math-digits"),
        pytest.param("$" + "1" * 200_000, id="math-unclosed"),
    ],
)
def test_clean_markdown_takes_linear_time(text: str) -> None:
    start = time.perf_counter()
    clean_markdown(text)
    assert time.perf_counter() - start < LIMIT_S


def test_normalize_cell_takes_linear_time() -> None:
    start = time.perf_counter()
    assert normalize_cell("a" + " " * 200_000 + "b") == "a b"
    assert time.perf_counter() - start < LIMIT_S


def test_normalize_cell_joins_line_breaks() -> None:
    assert normalize_cell("a \u2028\n b\tc") == "a | b c"


def test_clean_markdown_output() -> None:
    text = "# T\r\n\r\n![](images/a.jpg)  \n\n\n\nText \t\n![Fig. 2](images/b.jpg)\n"
    assert clean_markdown(text) == "# T\n\nText\nFig. 2"


BACK_MATTER_ARTICLE = """# Results

Forearm 37.4 mm.

# Acknowledgements

We thank the museum.

## 6. LITERATURE CITED

Smith, J. 2020. Bats of Elsewhere. Zootaxa 1: 1-10.

APPENDIX 1. Specimens examined

<table><tr><td>M. ater</td><td>12.5</td></tr></table>

# Conflict of interest

None.

## Supplementary data

<table><tr><td>Mass</td><td>5.85</td></tr></table>"""


def test_clean_markdown_removes_back_matter() -> None:
    """References and administrative sections go. The appendix after the references stays."""
    assert clean_markdown(BACK_MATTER_ARTICLE) == (
        "# Results\n\nForearm 37.4 mm.\n\n"
        "APPENDIX 1. Specimens examined\n\n<table><tr><td>M. ater</td><td>12.5</td></tr></table>\n\n"
        "## Supplementary data\n\n<table><tr><td>Mass</td><td>5.85</td></tr></table>"
    )
    assert "Smith, J. 2020" in clean_markdown(BACK_MATTER_ARTICLE, drop_back_matter=False)


@pytest.mark.parametrize(
    "heading",
    ["# References", "## REFERENCES", "# 7. References cited", "# Literature Cited", "## VII. Bibliography",
     "# Referências bibliográficas", "# Literatura citada", "## **Acknowledgments**", "# Funding",
     "# Competing interests", "# Authors’ contributions", "# Data availability statement"],
)  # fmt: skip
def test_back_matter_headings(heading: str) -> None:
    assert clean_markdown(f"# Methods\n\nText\n\n{heading}\n\nGone") == "# Methods\n\nText"


def test_clean_markdown_joins_the_digits_in_formulas() -> None:
    text = r"FA $3 7 . 4 2 \pm 0 . 8 3$ mm, $1 0 ^ { - 3 }$, $p < 0 . 0 5$; 1 2 mm and 3 . 4 outside"
    assert (
        clean_markdown(text)
        == r"FA $37.42 \pm 0.83$ mm, $10 ^ { - 3 }$, $p < 0.05$; 1 2 mm and 3 . 4 outside"
    )


@pytest.mark.parametrize("heading", ["# Literature review", "# Results and references", "# Appendix 2"])
def test_other_headings_stay(heading: str) -> None:
    assert clean_markdown(f"{heading}\n\nKept") == f"{heading}\n\nKept"
