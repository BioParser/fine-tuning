"""Deterministic cleaning of PDF-parser output (MinerU markdown).

Used by schema.build_messages(), i.e. identically at training and inference time.
Only lossless-for-the-task changes are made: HTML tables, <sup>/<sub> and LaTeX are
kept because trait values live in them. The reference list and the administrative back
matter (acknowledgements, funding, ...) are removed: they are about a fifth of the tokens
of an article, and on the delivered data they contain 1 of 10,020 labelled numbers.
"""

from __future__ import annotations

import re
import unicodedata

# C0/C1 control characters except TAB (\x09) and LF (\x0a); CR is normalised below.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Zero-width characters that only waste tokens or break word matching.
_ZERO_WIDTH = re.compile(r"[​-‍⁠﻿]")
# ![alt](images/page_0_image_9.jpg): the image itself is not available to a text model.
# The alt text cannot contain "[", and the path cannot contain "(". For this reason, each search
# stops at the next "[" or "(", and the time is linear also for hostile input.
_IMAGE = re.compile(r"!\[([^\[\]]*)\]\([^()]*\)")
_BLANK_RUNS = re.compile(r"\n{3,}")
# MinerU writes the numbers in a formula with spaces between the digits, for example
# "$3 7 . 4 2 \pm 0 . 8 3$". LaTeX ignores spaces in math mode, so "$37.42 \pm 0.83$" is the same
# formula: the number is easy to copy, it uses fewer tokens, and the coverage check finds it.
_MATH = re.compile(r"\$([^$\n]+)\$")
_MATH_DIGIT_SPACE = re.compile(r"(?<=\d) (?=[\d.])|(?<=\.) (?=\d)")

# MinerU writes the section titles as Markdown headings ("# " or "## ").
_HEADING = re.compile(r"#{1,6}[ \t]+(.+)")
# "7. ", "7.1 " or "VII. " before the title. A roman numeral needs its period: the "Li" of
# "Literature Cited" and the "C" of "Conflict of interest" are not numbers.
_NUMBERING = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.)\s+")
# The sections without trait data. "Appendix" and "Supplementary ..." are not in the list:
# they often contain the measurement tables.
_BACK_MATTER = re.compile(
    r"references?(?: cited)?|literature(?: cited)?|cited literature|works cited|bibliography"
    r"|refer[eê]ncias?(?: bibliogr[aá]ficas)?|literatura citada|bibliograf[ií]a"
    r"|acknowledge?ments?|agradecimientos|agradecimentos"
    r"|funding(?: information)?|conflicts? of interests?|competing interests?"
    r"|declarations? of (?:competing )?interests?|authors?[’']? contributions?|participation of authors"
    r"|credit authorship contribution statement|ethics(?: statement)?|ethical approval"
    r"|data availability(?: statement)?|data accessibility|data resources|orcid|copyright|published by"
)
# MinerU does not always write a heading for an appendix after the references, for example
# "APPENDIX 1. List of ..." and then the tables. The removal stops at such a line.
_KEEP_FROM = re.compile(r"<table|(?:appendix|table|supplementary)\b", re.IGNORECASE)


def _section_name(line: str) -> str | None:
    """Return the normalized title of a heading line, or None for a line that is not a heading."""
    m = _HEADING.fullmatch(line)
    if m is None:
        return None
    name = _NUMBERING.sub("", m.group(1).strip(" *_"), count=1)
    return " ".join(name.casefold().rstrip(" .:*_").split())


def remove_back_matter(text: str) -> str:
    """Remove the reference list and the administrative sections, from the heading to the next heading."""
    kept: list[str] = []
    removing = False
    for line in text.split("\n"):
        name = _section_name(line)
        if name is not None:
            removing = _BACK_MATTER.fullmatch(name) is not None
        elif removing and _KEEP_FROM.match(line):
            removing = False
        if not removing:
            kept.append(line)
    return "\n".join(kept)


def join_math_digits(text: str) -> str:
    """Remove the spaces between the digits of the numbers in inline formulas ($...$)."""
    return _MATH.sub(lambda m: "$" + _MATH_DIGIT_SPACE.sub("", m.group(1)) + "$", text)


def clean_markdown(text: str, *, drop_images: bool = True, drop_back_matter: bool = True) -> str:
    """Normalise Unicode/newlines, drop image links (keep alt text) and the back matter, squeeze blank lines."""
    t = unicodedata.normalize("NFC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _ZERO_WIDTH.sub("", _CONTROL.sub("", t))
    if drop_images:
        t = _IMAGE.sub(lambda m: m.group(1).strip(), t)
    # Remove the spaces and TABs at the end of each line. Do not use a regex such as [ \t]+\n:
    # its time is quadratic on a long run of spaces without a line break.
    t = "\n".join(line.rstrip(" \t") for line in t.split("\n"))
    t = join_math_digits(t)
    if drop_back_matter:
        t = remove_back_matter(t)
    t = _BLANK_RUNS.sub("\n\n", t)
    return t.strip()
