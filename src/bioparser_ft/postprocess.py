"""Model TSV -> ETS records (JSON-ready dicts) for the FastAPI service. Stdlib only.

Typical use in the API::

    from bioparser_ft.postprocess import to_ets_records
    result = to_ets_records(generated_text, complete=finished_with_eos, references=citation_or_None)
    return {"records": result.records, "warnings": result.warnings}

Deterministic fixes applied here instead of trusting a 0.8B model:
* ``author`` is always "BioParser" (client rule; never generated);
* ``references`` forward-filled from the first row, or replaced by a citation the
  caller already has (client: it comes from Google Scholar, not from the PDF);
* empty ``taxonRank`` inferred from the name (client rule, see infer_taxon_rank);
* "H. sapiens" expanded to "Homo sapiens" when the genus is unambiguous in the output;
* controlled vocabularies canonicalised exactly as in the training labels.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field

from .schema import CONSTANT_COLUMNS, ETS_COLUMNS
from .tsv import Record, canonicalize, parse_model_tsv

_PARENTHESES = re.compile(r"\([^)]*\)")  # "(Subgenus)" or "(Linnaeus, 1758)"
_QUALIFIERS = frozenset({"cf.", "cf", "aff.", "aff", "nr.", "nr", "?"})
_NEW_SPECIES = re.compile(r"\b(?:sp\.?\s*nov|n\.\s*sp|sp\.?\s*n)\b\.?", re.IGNORECASE)
_SP_PLACEHOLDER = re.compile(r"^spp?\.?$", re.IGNORECASE)
_SP_NUMBERED = re.compile(r"^spp?\.?\d+[a-z]?$", re.IGNORECASE)  # "Oligoryzomys sp1" morphospecies
# The client also lists -ina (subtribe); it is not applied because real genera end in
# -ina (Blarina, Murina are labelled Genus in the delivered data).
_RANK_SUFFIXES = (("oidea", "Superfamily"), ("idae", "Family"), ("inae", "Subfamily"), ("ini", "Tribe"))


def infer_taxon_rank(name: str) -> str:
    """Client rule: 1 word Genus, 2 Species, 3 Subspecies; suffixes for higher ranks.

    Refined on the delivered labels (98.8% -> 99.9% agreement): open nomenclature
    ("Myotis cf. ater" is a species, "Rhinolophus sp. nov." and "Soriculus sp. 3"
    are species, "Berylmys sp." is a genus record) and trailing authorship
    ("Canis lupus Linnaeus, 1758"). Returns "" when the name gives no clue.
    """
    if not name.strip():
        return ""
    if _NEW_SPECIES.search(name):
        return "Species"
    words = [w for w in _PARENTHESES.sub(" ", name).split() if w.lower() not in _QUALIFIERS]
    if not words:
        return ""
    if len(words) >= 2 and _SP_NUMBERED.match(words[1]):
        return "Species"
    if len(words) >= 2 and _SP_PLACEHOLDER.match(words[1]):
        return "Species" if len(words) > 2 else "Genus"
    epithets = [words[0]]
    for w in words[1:]:
        if w[:1].isupper() or any(ch.isdigit() for ch in w) or w in {"&", "and", "et", "ex"}:
            break  # authorship begins; epithets are lower case
        epithets.append(w)
    if len(epithets) == 1:
        lowered = epithets[0].lower()
        for suffix, rank in _RANK_SUFFIXES:
            if lowered.endswith(suffix):
                return rank
        return "Genus"
    return {2: "Species", 3: "Subspecies"}.get(len(epithets), "")


_ABBREVIATED = re.compile(r"^([A-Z])\.\s*([a-z].*)$")
_GENUS = re.compile(r"^[A-Z][a-z]+$")


def expand_abbreviated_genera(
    records: Sequence[MutableMapping[str, str]], column: str = "verbatimScientificName"
) -> int:
    """Client rule "H. sapiens" -> "Homo sapiens", using genera spelled out in the same output.

    Only expands when exactly one genus with that initial occurs, so it never guesses.
    Returns the number of changed cells.
    """
    by_initial: dict[str, set[str]] = {}
    for rec in records:
        first = rec.get(column, "").split(" ", 1)[0]
        if _GENUS.match(first):
            by_initial.setdefault(first[0], set()).add(first)
    changed = 0
    for rec in records:
        m = _ABBREVIATED.match(rec.get(column, ""))
        if m and len(candidates := by_initial.get(m[1], set())) == 1:
            rec[column] = f"{next(iter(candidates))} {m[2]}"
            changed += 1
    return changed


@dataclass
class EtsResult:
    records: list[Record]
    warnings: list[str] = field(default_factory=list)
    complete: bool = True  # False: generation was cut by max_new_tokens

    def to_json(self) -> str:
        return json.dumps(
            {"records": self.records, "warnings": self.warnings, "complete": self.complete},
            ensure_ascii=False,
        )


def to_ets_records(
    model_output: str,
    *,
    references: str | None = None,
    complete: bool = True,
    fill_taxon_rank: bool = True,
    expand_genera: bool = True,
) -> EtsResult:
    """Turn raw generated text into full 20-column ETS records (ETS column order).

    references: a verified citation (e.g. from the client's Google Scholar lookup or a
        DOI resolver); when given it replaces whatever the model generated.
    complete:   pass False when generation stopped without EOS (see infer.Generation).
    """
    parsed = parse_model_tsv(model_output, complete=complete)
    warnings = list(parsed.warnings)
    rows: list[Record] = parsed.records
    if expand_genera and (n := expand_abbreviated_genera(rows)):
        warnings.append(f"expanded {n} abbreviated genus names")

    out: list[Record] = []
    rank_disagreements = 0
    for rec in rows:
        full: Record = {}
        for col in ETS_COLUMNS:
            full[col] = CONSTANT_COLUMNS.get(col) or canonicalize(col, rec.get(col, ""))
        if references is not None:
            full["references"] = references
        inferred = infer_taxon_rank(full["verbatimScientificName"])
        if not full["taxonRank"] and fill_taxon_rank:
            full["taxonRank"] = inferred
        elif inferred and full["taxonRank"] != inferred:
            rank_disagreements += 1
        out.append(full)

    if rank_disagreements:
        warnings.append(f"{rank_disagreements} rows: taxonRank differs from the name-based rule")
    if out and not out[0]["references"]:
        warnings.append("no citation in output; pass references= to fill it")
    if not out:
        warnings.append("no records parsed")
    return EtsResult(records=out, warnings=warnings, complete=complete)


def records_from_json(payload: str | bytes) -> list[Record]:
    """Inverse of EtsResult.to_json()['records'] with column validation (for API tests)."""
    data = json.loads(payload)
    records = data["records"] if isinstance(data, Mapping) else data
    return [{c: str(r.get(c, "")) for c in ETS_COLUMNS} for r in records]
