"""Does a parsed PDF actually contain the values in its ETS labels? (stdlib only)

A training pair is only useful if the answer is in the input. Low coverage means
the TSV is linked to the wrong PDF, the parser lost the tables (e.g. tables
rendered as images), or the values come from supplementary files that were not
parsed. Such pairs teach the model to hallucinate and should not be trained on.

Measured on the three MinerU samples from the original notebook: 100% value coverage.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

VALUE_COLUMNS = (
    "verbatimTraitValue",
    "measurementValue_min",
    "measurementValue_max",
    "dispersion",
    "individualCount",
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
MAX_NUMBER_CHARS = 30  # A longer digit run (an ID, a sequence or OCR noise) is not a label value
_WORD = re.compile(r"[a-z]+")
_NAME_STOPWORDS = frozenset({"cf", "aff", "sp", "spp", "nov", "nr", "n", "var", "ssp", "subsp"})


def _canonical_numbers(token: str) -> set[str]:
    """'5.20' -> {'5.2'}; '1,234' -> {'1234', '1.234'} (thousands or decimal comma)."""
    variants = {token}
    if "," in token:
        variants.add(token.replace(",", ""))
        if token.count(",") == 1 and "." not in token:
            variants.add(token.replace(",", "."))
    out: set[str] = set()
    for v in variants:
        try:
            out.add(format(Decimal(v).normalize(), "f"))
        except InvalidOperation:
            continue
    return out


def numbers_in(text: str, *, min_value: int = 10) -> set[str]:
    """Canonical numbers in text. Integers < min_value are skipped: small counts such
    as "3" occur in any article and would make the check meaningless."""
    out: set[str] = set()
    for token in _NUMBER.findall(text):
        if len(token) > MAX_NUMBER_CHARS:
            continue  # int() below stops with an error above 4,300 digits
        for canon in _canonical_numbers(token):
            if "." in canon or int(canon) >= min_value:
                out.add(canon)
    return out


def _name_key(name: str) -> str:
    """Most specific lower-case epithet of a name (genus if there is none)."""
    words: list[str] = re.findall(r"[A-Za-z]+", name)
    epithets = [w.casefold() for w in words[1:] if w.islower() and w.casefold() not in _NAME_STOPWORDS]
    if epithets:
        return epithets[-1]
    return words[0].casefold() if words else ""


@dataclass
class Coverage:
    values_total: int
    values_found: int
    names_total: int
    names_found: int
    missing_values: list[str] = field(default_factory=list)  # sample, for the report
    missing_names: list[str] = field(default_factory=list)

    @property
    def value_ratio(self) -> float | None:
        return self.values_found / self.values_total if self.values_total else None

    @property
    def name_ratio(self) -> float | None:
        return self.names_found / self.names_total if self.names_total else None


def label_coverage(markdown: str, records: Iterable[Mapping[str, str]], sample: int = 10) -> Coverage:
    records = list(records)
    md_numbers = numbers_in(markdown)
    gold_numbers: set[str] = set()
    for rec in records:
        for col in VALUE_COLUMNS:
            gold_numbers |= numbers_in(rec.get(col, ""))
    found = gold_numbers & md_numbers

    md_words = set(_WORD.findall(markdown.casefold()))
    names = {k for r in records if (k := _name_key(r.get("verbatimScientificName", "")))}
    names_found = names & md_words

    def by_magnitude(values: set[str]) -> list[str]:
        return sorted(values, key=lambda s: (-len(s), s))[:sample]

    return Coverage(
        values_total=len(gold_numbers),
        values_found=len(found),
        names_total=len(names),
        names_found=len(names_found),
        missing_values=by_magnitude(gold_numbers - found),
        missing_names=sorted(names - names_found)[:sample],
    )
