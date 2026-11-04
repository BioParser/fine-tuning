"""This module decides if a search result is the cited article. It compares the title, the year and the first author."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

_STOP = frozenset({
    "the", "and", "for", "with", "from", "into", "its", "their", "this", "that", "are", "was",
    "were", "has", "have", "not", "but", "del", "los", "las", "des", "une", "der", "die", "das",
    "und", "von", "dos", "por", "para", "con", "una",
})  # fmt: skip


def norm(text: str) -> str:
    """Make the text lowercase, remove the accents, and keep only letters and digits."""
    decomposed = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text).split())


def tokens(text: str) -> list[str]:
    return [t for t in norm(text).split() if len(t) >= 3 and t not in _STOP]


@dataclass
class Verdict:
    confidence: str  # "high", "medium" or "low"
    title_recall: float  # The part of the title words of the result that are also in the citation
    title_precision: float  # The part of the words of the title guess that are also in the result title
    year_ok: bool
    author_ok: bool

    @property
    def accepted(self) -> bool:
        return self.confidence in {"high", "medium"}

    def summary(self) -> str:
        return (
            f"{self.confidence} (title {self.title_recall:.2f}/{self.title_precision:.2f}, "
            f"year {'ok' if self.year_ok else 'no'}, author {'ok' if self.author_ok else 'no'})"
        )


CONFIDENCE_ORDER = {"high": 2, "medium": 1, "low": 0}


def verify(
    citation: str,
    title_guess: str,
    first_author: str,
    year: int | None,
    cand_title: str,
    cand_year: int | None,
    cand_authors: Iterable[str],
) -> Verdict:
    """Compare one search result with the citation and return a Verdict.

    The full citation text is the reference, so an exact title from the citation is not
    necessary. The recall uses the full citation. For this reason, an error in the title guess has
    a small effect. The precision uses the title guess. For this reason, a short general
    title of a result, for example "Chiroptera", cannot get a high score.
    """
    cand = tokens(cand_title)
    cit = set(tokens(citation))
    guess = tokens(title_guess)
    recall = sum(t in cit for t in cand) / len(cand) if cand else 0.0
    cand_set = set(cand)
    precision = sum(t in cand_set for t in guess) / len(guess) if guess else recall
    # The online year and the print year of an article can be different
    year_ok = year is None or cand_year is None or abs(cand_year - year) <= 1
    fa = norm(first_author)
    author_ok = bool(fa) and any(fa in norm(a) for a in cand_authors if a)
    if len(cand) < 3:
        confidence = "low"
    elif recall >= 0.9 and precision >= 0.7 and year_ok and author_ok:
        confidence = "high"
    elif recall >= 0.8 and precision >= 0.5 and (year_ok or author_ok):
        confidence = "medium"
    else:
        confidence = "low"
    return Verdict(confidence, recall, precision, year_ok, author_ok)


def title_recall_in_text(title: str, text: str) -> float:
    """Return the part of the title words that are in the text of a downloaded PDF file."""
    words = tokens(title)
    if not words:
        return 0.0
    present = set(tokens(text))
    return sum(w in present for w in words) / len(words)
