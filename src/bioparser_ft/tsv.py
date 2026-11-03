"""Client ETS TSV reader, model-TSV writer and model-TSV parser (stdlib only).

Two TSV dialects exist in this project:

1. Client ETS files (the labels) are Excel-style exports: a cell may be quoted and
   contain TABs and CRLF line breaks, e.g. ``"Greatest Length of Skull<TAB>/Breadth
   Across Molars"`` (Claudio et al. 2023) or several references separated by CRLF
   (Menendez et al. 2022). They must be read with the quote-aware ``csv`` module;
   ``line.split("\\t")`` breaks 150 rows of the delivered data.
2. Model TSV (training target and model output): one record per line, no quoting,
   cells never contain TAB or newline. Easy to generate token by token and to parse
   even when the model makes mistakes. ``normalize_cell`` maps (1) onto (2).
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .schema import (
    CANONICAL_VALUES,
    DOCUMENT_COLUMNS,
    ETS_COLUMNS,
    LIST_SEPARATOR,
    MODEL_COLUMNS,
)

Record = dict[str, str]

MAX_TSV_BYTES = 64 * 1024 * 1024  # largest delivered file is 2.8 MB
SUSPICIOUS_CELL_CHARS = 4000  # longer usually means an unbalanced quote swallowed rows
MAX_WARNINGS = 50  # per file / per parse; keeps reports readable

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")  # keeps \t \n \r
_ZERO_WIDTH = re.compile(r"[​-‍⁠﻿]")
_LINEBREAKS = re.compile(r"\s*[\r\n  ]+\s*")
_INLINE_WS = re.compile(r"\s+")  # Unicode-aware: TAB, NBSP (801 cells), thin spaces, ...


def normalize_cell(value: str | None) -> str:
    """Make a cell safe for the model TSV dialect and remove invisible noise.

    NFC; drop control/zero-width chars; strip (2,275 delivered cells had stray
    leading/trailing spaces such as "mm " or " 0.1 mm"); join internal line breaks
    with " | "; collapse all other whitespace, incl. TAB and NBSP, to one space.
    NFKC is deliberately not used: it would rewrite units such as "mm²" -> "mm2".
    """
    if not value:
        return ""
    v = unicodedata.normalize("NFC", value)
    v = _ZERO_WIDTH.sub("", _CONTROL.sub("", v)).strip()
    v = _LINEBREAKS.sub(LIST_SEPARATOR, v)
    return _INLINE_WS.sub(" ", v).strip()


def canonicalize(column: str, value: str) -> str:
    """Map controlled-vocabulary variants (case, plurals, known typos) to one form."""
    table = CANONICAL_VALUES.get(column)
    if not table or not value:
        return value
    return table.get(value.casefold(), value)


def normalize_record(raw: Mapping[str, str], columns: Sequence[str] = ETS_COLUMNS) -> Record:
    return {c: canonicalize(c, normalize_cell(raw.get(c, ""))) for c in columns}


# --------------------------------------------------------------------------- client files


@dataclass
class EtsTable:
    path: Path
    records: list[Record]
    warnings: list[str] = field(default_factory=list)


def read_ets_tsv(path: str | Path) -> EtsTable:
    """Read one client ETS export (quote-aware), mapping columns by header name."""
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_TSV_BYTES:
        raise ValueError(f"{path.name}: {size} bytes exceeds MAX_TSV_BYTES")
    raw = path.read_bytes()
    warnings: list[str] = []
    try:
        text = raw.decode("utf-8-sig")  # tolerate a BOM from Excel
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
        warnings.append("not valid UTF-8; decoded as cp1252 (check special characters)")

    # newline="" + csv: quoted cells may contain CR/LF and TAB (Excel export).
    reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t", quotechar='"', strict=False)
    try:
        rows = list(reader)
    except csv.Error as exc:  # e.g. "field larger than field limit": an unclosed quote
        raise ValueError(f"{path.name}: unparseable TSV ({exc})") from exc
    if not rows:
        raise ValueError(f"{path.name}: empty file")

    header = [normalize_cell(h) for h in rows[0]]
    index: dict[str, int] = {}
    for i, name in enumerate(header):
        if name in ETS_COLUMNS and name not in index:
            index[name] = i
        elif name:
            warnings.append(f"unknown or duplicate column {name!r} ignored")
    missing = [c for c in ETS_COLUMNS if c not in index]
    if missing:
        warnings.append(f"missing columns filled with empty values: {missing}")
    if not index:
        raise ValueError(f"{path.name}: no ETS columns in header {header[:5]}")

    records: list[Record] = []
    for n, row in enumerate(rows[1:], start=1):  # n = record number (not physical line)
        if not any(cell.strip() for cell in row):
            continue
        if len(row) > len(header) and any(c.strip() for c in row[len(header) :]):
            warnings.append(f"record {n}: {len(row)} cells for {len(header)} columns, extra dropped")
        raw_rec = {c: (row[i] if i < len(row) else "") for c, i in index.items()}
        long = [c for c, v in raw_rec.items() if len(v) > SUSPICIOUS_CELL_CHARS]
        if long:
            warnings.append(f"record {n}: very long cell in {long} (unbalanced quote?)")
        records.append(normalize_record(raw_rec))
    return EtsTable(path=path, records=records, warnings=warnings[:MAX_WARNINGS])


# --------------------------------------------------------------------------- model dialect


def to_model_tsv(records: Sequence[Mapping[str, str]], columns: Sequence[str] = MODEL_COLUMNS) -> str:
    """Serialise records as the training target: header + one line per record.

    Document-level columns (``references``) are written only when they change, i.e.
    normally on the first data row; ``parse_model_tsv`` forward-fills them. No
    trailing newline: the training text ends with the EOS token directly after the
    last cell (TRL appends EOS only when the text does not already end with it).
    """
    lines = ["\t".join(columns)]
    last: dict[str, str] = {}
    for rec in records:
        cells = []
        for col in columns:
            value = normalize_cell(rec.get(col, ""))
            if col in DOCUMENT_COLUMNS:
                if value and value == last.get(col):
                    value = ""  # same as the row above -> "ditto"
                elif value:
                    last[col] = value
            cells.append(value)
        lines.append("\t".join(cells))
    return "\n".join(lines)


def to_ets_tsv(records: Sequence[Mapping[str, str]]) -> str:
    """Write records in the client's full 20-column ETS layout (plain TSV, LF)."""
    lines = ["\t".join(ETS_COLUMNS)]
    lines += ["\t".join(normalize_cell(r.get(c, "")) for c in ETS_COLUMNS) for r in records]
    return "\n".join(lines) + "\n"


@dataclass
class ParsedTsv:
    records: list[Record]
    warnings: list[str]
    header_ok: bool
    malformed_rows: int


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^\s*```")
REPETITION_WARN_RUN = 25  # identical consecutive rows usually mean a degenerate loop


def parse_model_tsv(text: str, columns: Sequence[str] = MODEL_COLUMNS, *, complete: bool = True) -> ParsedTsv:
    """Parse model output into records with ``columns`` keys; never raises on bad output.

    ``complete=False`` means generation stopped at max_new_tokens (no EOS), so the
    last line is probably cut mid-row and is dropped. Repairs are reported in
    ``warnings``: missing header, header in another order (mapped by name), rows with
    the wrong number of cells (padded/truncated), code fences, think blocks.
    """
    warnings: list[str] = []
    text = _THINK_BLOCK.sub("", text)
    if "<think>" in text:
        warnings.append("unterminated <think> block; text after it ignored")
        text = text.split("<think>", 1)[0]
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    if any(_FENCE.match(ln) for ln in lines):
        warnings.append("code fences removed")
    lines = [ln for ln in lines if ln.strip() and not _FENCE.match(ln)]
    if not complete and lines:
        lines.pop()
        warnings.append("generation hit max_new_tokens; dropped the last (partial) row")

    expected = list(columns)
    names = expected
    header_ok = False
    if lines:
        first = [c.strip() for c in lines[0].split("\t")]
        if first == expected:
            header_ok = True
            lines = lines[1:]
        elif sum(c in ETS_COLUMNS for c in first) >= max(3, len(first) // 2):
            names = first  # a header with another order/column set: map by name
            unknown = [c for c in first if c not in expected]
            warnings.append(f"header differs from expected columns; mapped by name (ignored: {unknown})")
            lines = lines[1:]
        else:
            warnings.append("no header line; assuming the expected column order")

    records: list[Record] = []
    malformed = 0
    run, prev = 0, None
    for n, line in enumerate(lines, start=1):
        cells = line.split("\t")
        if [c.strip() for c in cells] == names:
            continue  # header repeated mid-output
        while len(cells) > len(names) and not cells[-1].strip():
            cells.pop()  # trailing empty cells are harmless
        if len(cells) != len(names):
            malformed += 1
            if malformed <= MAX_WARNINGS:
                warnings.append(f"row {n}: {len(cells)} cells, expected {len(names)}")
            cells = (cells + [""] * len(names))[: len(names)]
        rec = dict.fromkeys(expected, "")
        for name, cell in zip(names, cells, strict=True):
            if name in rec:
                rec[name] = normalize_cell(cell)
        run = run + 1 if line == prev else 1
        prev = line
        if run == REPETITION_WARN_RUN:
            warnings.append(f"row {n}: {run} identical consecutive rows (repetition loop?)")
        records.append(rec)

    last: dict[str, str] = {}
    for rec in records:  # forward-fill "ditto" cells of document-level columns
        for col in DOCUMENT_COLUMNS:
            if col not in rec:
                continue
            if rec[col]:
                last[col] = rec[col]
            elif col in last:
                rec[col] = last[col]
    return ParsedTsv(records=records, warnings=warnings, header_ok=header_ok, malformed_rows=malformed)
