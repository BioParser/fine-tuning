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
