"""This module groups the ETS TSV files by article and parses their Google Scholar citations."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..tsv import read_ets_tsv

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+")
# Google Scholar Harvard format: "Surname, I., Surname, I. and Surname, I., 2022. Title. Journal, 1(2), pp.3-4."
_HARVARD = re.compile(r"^(?P<authors>.+?),\s*(?P<year>(?:1[5-9]|20)\d{2})[a-z]?\.\s+(?P<rest>.+)$", re.S)
# APA format: "Roy, P., Das, A. and Biswas, J. K. (2023). Title. Journal, 5 (2): 33-45."
_APA = re.compile(
    r"^(?P<authors>.{3,400}?)\s*\((?P<year>(?:1[5-9]|20)\d{2})[a-z]?\)\.?\s+(?P<rest>.+)$", re.S
)
# Google Scholar format without a year, for an article in press. Examples:
# "Molinari, J., Title. Journal." and "A, B. and C, D.E., Title."
_INITIALS = r"(?:[^\W\d_]{1,2}\.\s?-?){1,4}"
_LAST_AUTHOR = re.compile(rf"\band\s+[^,]{{1,60}},\s*{_INITIALS},\s+")
_SINGLE_AUTHOR = re.compile(rf"^[^,]{{1,60}},\s*{_INITIALS},\s+")
_YEAR = re.compile(r"(?<!\d)((?:1[5-9]|20)\d{2})(?!\d)")
# A taxonomic authority such as "(Linnaeus, 1758)" does not give the year of the article.
_AUTHORITY = re.compile(r"\(\s*[^\W\d_][^()]{0,60}?,\s*(?:1[5-9]|20)\d{2}\s*\)")
_QUOTED = re.compile(r"[\"“]([^\"”]{15,})[\"”]")  # Chicago format: the title is in quotation marks


def _looks_like_names(sentence: str) -> bool:
    """Return True if the sentence contains only names, for example 'Zaib-Ur-Rahman and Iftikhar Ahmad'.

    Google Scholar sometimes puts author names in the title position. A sentence with only
    names has capitalized words and the word "and". A real title also has lowercase words.
    """
    words = sentence.replace(",", " ").split()
    return "and" in words and len(words) <= 10 and all(w[0].isupper() for w in words if w != "and")


_THESIS = re.compile(r"\((?:Doctoral|Master'?s?|PhD|Ph\.D\.)[^)]*\)", re.I)
# This pattern finds the start of the source part after the title:
# a journal with its volume, the pages, a book or a thesis.
_SOURCE = re.compile(r"^In\s|,\s*\d+\s?\(\d+\)|,\s*pp?\.\s?\w|,\s*pp?$|^[^.]{0,120},\s*\d+")
_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


@dataclass
class Paper:
    key: str  # The PDF file stem: the shortest TSV stem without the "ETS_" prefix
    citation: str
    tsv_files: list[str]
    first_author: str = ""
    year: int | None = None
    title: str = ""  # Our best guess from the citation. The search and the PDF check use it.
    is_thesis: bool = False
    doi: str | None = None  # From the citation or from doi_overrides.tsv
    url: str | None = None  # From doi_overrides.tsv: a landing page or a PDF
    overridden: bool = False  # True if the DOI or the URL comes from doi_overrides.tsv
    notes: list[str] = field(default_factory=list)

    @property
    def pdf_name(self) -> str:
        return safe_name(self.key) + ".pdf"


def safe_name(stem: str) -> str:
    """Return a file name that is safe on Linux, macOS and Windows. Unicode letters do not change."""
    return _UNSAFE.sub("_", stem).strip(" .") or "unnamed"


def clean_doi(doi: str) -> str:
    doi = doi.strip().removeprefix("https://doi.org/").removeprefix("http://dx.doi.org/")
    return doi.rstrip(".,;)]").lower()


def _title_from_rest(rest: str) -> tuple[str, bool]:
    """Return (title, is_thesis). The title is the text before the source part.

    The function examines the text one sentence at a time. For this reason, it does not cut
    a title that contains a "?" or a second sentence, for example
    "How to weigh a fossil mammal? South American notoungulates as a case study ...".
    """
    rest = rest.strip()
    while (first := re.split(r"\.\s+", rest, maxsplit=1)) and len(first) == 2 and _looks_like_names(first[0]):
        rest = first[1]  # Remove the author names that Google Scholar put before the title
    if m := _THESIS.search(rest):
        return rest[: m.start()].strip(" ."), True
    for m in re.finditer(r"[.?!]\s+", rest):
        tail = rest[m.end() :]
        first_sentence = re.split(r"\.\s+", tail, maxsplit=1)[0]
        if _SOURCE.search(first_sentence):
            end = m.start() + (1 if rest[m.start()] in "?!" else 0)
            return rest[:end].strip(), False
    return re.split(r"\.\s+", rest, maxsplit=1)[0].strip(" ."), False


def _first_author(authors: str) -> str:
    """Return the surname of the first author, for example "Dahal D.R" gives "Dahal"."""
    first = authors.split(",")[0].strip()
    if re.search(r"(?:^|\s)[^\W\d_]{1,2}\.", first):  # The initials come after the surname, without a comma
        return first.split()[0]
    return first


def parse_citation(citation: str) -> tuple[str, int | None, str, bool, str | None]:
    """Return (first author surname, year, title guess, is_thesis, doi).

    The function reads these citation formats from the client data:

    - Google Scholar Harvard format: 165 of the 175 citations
    - Google Scholar format without a year
    - APA format, with the year as "(2023)."
    - Chicago format, with the title in quotation marks
    """
    text = " ".join(citation.split())
    doi_match = DOI_RE.search(text)
    doi = clean_doi(doi_match.group(0)) if doi_match else None
    for pattern in (_HARVARD, _APA):
        if m := pattern.match(text):
            title, thesis = _title_from_rest(m["rest"])
            return _first_author(m["authors"]), int(m["year"]), title, thesis, doi
    years_text = _AUTHORITY.sub(" ", text)  # The year of a taxonomic authority is not the publication year
    in_parens = re.findall(r"\(((?:1[5-9]|20)\d{2})\)", years_text)
    years = in_parens or _YEAR.findall(years_text)
    year = int(years[-1]) if years else None
    if quoted := _QUOTED.search(text):
        return _first_author(text), year, quoted.group(1).strip(" ."), bool(_THESIS.search(text)), doi
    m = _LAST_AUTHOR.search(text[:600]) or _SINGLE_AUTHOR.match(text)
    if m:
        title, thesis = _title_from_rest(text[m.end() :])
        return _first_author(text), year, title, thesis, doi
    title = max(re.split(r"\.\s+", text), key=len)
    return _first_author(text), year, title, bool(_THESIS.search(text)), doi


def _stem_key(name: str) -> str:
    stem = name[:-4] if name.lower().endswith(".tsv") else name
    stem = stem[4:] if stem[:4].lower() == "ets_" else stem
    return " ".join(stem.split()).casefold()


def load_overrides(path: Path | None) -> dict[str, dict[str, str]]:
    """Read doi_overrides.tsv. The file is tab-separated, because the file names contain commas.

    The columns are tsv, doi, url and note. The value in `tsv` is the TSV file name or stem,
    with or without the "ETS_" prefix.
    """
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = csv.DictReader(fh, delimiter="\t")
        if not rows.fieldnames or "tsv" not in rows.fieldnames:
            raise ValueError(f"{path}: the file must have a 'tsv' column and a 'doi' or 'url' column")
        out = {}
        for row in rows:
            if (name := (row.get("tsv") or "").strip()) and not name.startswith("#"):
                out[_stem_key(name)] = {k: (row.get(k) or "").strip() for k in ("doi", "url", "note")}
        return out


def group_papers(tsv_dir: Path, overrides_path: Path | None = None) -> list[Paper]:
    """Return one Paper for each article. TSV files with the same citation or override share one PDF."""
    overrides = load_overrides(overrides_path)
    groups: dict[str, list[tuple[Path, str]]] = {}
    for path in sorted(tsv_dir.glob("*.tsv")):
        records = read_ets_tsv(path).records
        citation = records[0]["references"] if records else ""
        ov = overrides.get(_stem_key(path.name), {})
        if ov.get("doi") or ov.get("url"):
            gkey = "override:" + (clean_doi(ov["doi"]) if ov.get("doi") else ov["url"])
        elif citation:
            gkey = "citation:" + " ".join(citation.split()).casefold()
        else:
            gkey = "file:" + path.name
        groups.setdefault(gkey, []).append((path, citation))

    papers = []
    for gkey, members in groups.items():
        stems = [p.stem[4:] if p.stem[:4].lower() == "ets_" else p.stem for p, _ in members]
        key = min(stems, key=lambda s: (len(s), s))
        citation = next((c for _, c in members if c), "")
        first_author, year, title, thesis, doi = parse_citation(citation)
        paper = Paper(key, citation, [p.name for p, _ in members], first_author, year, title, thesis, doi)
        if gkey.startswith("override:"):
            ov = overrides[_stem_key(members[0][0].name)]
            paper.doi = clean_doi(ov["doi"]) if ov.get("doi") else paper.doi
            paper.url = ov.get("url") or None
            paper.overridden = True
            paper.notes.append("override" + (f": {ov['note']}" if ov.get("note") else ""))
        bases = {re.sub(r"(\s+part\s*\d+|_\d+)$", "", _stem_key(s)) for s in stems}
        if len(bases) > 1:
            paper.notes.append(
                f"{len(stems)} TSV files have the same citation. If they are different articles, "
                "add a row for one of them to doi_overrides.tsv"
            )
        papers.append(paper)
    return sorted(papers, key=lambda p: p.key.casefold())
