"""Pair client ETS TSV files with parsed PDFs (stdlib only).

Replaces the old "rename everything to 001-999 by hand" step. Matching is by
normalised file stem, so ``ETS_Abdel-Hamid and Alqahtani, 2022.tsv`` pairs with
``Abdel-Hamid and Alqahtani, 2022.zip`` (MinerU ``--format zip``) or ``.md``
(``--format markdown``). Anything else goes in a tab-separated manifest
(``<TAB>`` = tab character; names contain commas, so not CSV):

    tsv<TAB>markdown
    ETS_Abraham et al., 2022 part 2.tsv<TAB>Abraham et al., 2022.zip

Several TSVs that point at the same parsed PDF are merged into one example.
Parsed files are untrusted input: zip members are read into memory with a size cap
and never extracted to disk (no zip-slip).
"""

from __future__ import annotations

import csv
import difflib
import unicodedata
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

MAX_MARKDOWN_BYTES = 32 * 1024 * 1024
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
MINERU_ZIP_MEMBER = "markdown.md"  # verified for mineru-kit 4.1.6 --format zip
# A damaged zip file gives one of these errors: a bad CRC, damaged compressed data, a missing
# member, an encrypted member or an unknown compression method.
ZIP_ERRORS = (zipfile.BadZipFile, zlib.error, EOFError, KeyError, RuntimeError, NotImplementedError)

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_QUOTES = {ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"'}


def paper_key(stem: str) -> str:
    """Normalise a file stem for matching: NFC, ASCII dashes/quotes, no ETS_ prefix,
    collapsed whitespace, casefolded. ("López‐Aguirre" uses U+2010, not "-".)"""
    s = unicodedata.normalize("NFC", stem).translate(_DASHES).translate(_QUOTES)
    if s[:4].casefold() == "ets_":
        s = s[4:]
    return " ".join(s.split()).casefold()


@dataclass(frozen=True)
class MarkdownSource:
    path: Path
    member: str | None = None  # zip member name, None for plain files

    @property
    def key(self) -> str:
        stem = self.path.parent.name if self.path.name == MINERU_ZIP_MEMBER else self.path.stem
        return paper_key(stem)

    def read_text(self, max_bytes: int = MAX_MARKDOWN_BYTES) -> str:
        """Return the text. Raises OSError or ValueError if the file is too large or damaged."""
        if self.member is None:
            if self.path.stat().st_size > max_bytes:
                raise ValueError(f"{self.path.name}: larger than {max_bytes} bytes")
            data = self.path.read_bytes()
        else:
            try:
                with zipfile.ZipFile(self.path) as zf, zf.open(self.member) as fh:
                    data = fh.read(max_bytes + 1)  # never trust the declared size
            except ZIP_ERRORS as exc:
                raise ValueError(f"{self.path.name}:{self.member}: damaged zip file ({exc})") from exc
            if len(data) > max_bytes:
                raise ValueError(f"{self.path.name}:{self.member}: larger than {max_bytes} bytes")
        return data.decode("utf-8", errors="replace")


def _zip_markdown_member(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
    except zipfile.BadZipFile:
        return None
    if MINERU_ZIP_MEMBER in names:
        return MINERU_ZIP_MEMBER
    md = sorted(n for n in names if n.lower().endswith(".md") and "/" not in n)
    return md[0] if len(md) == 1 else None


def discover_markdown(root: Path) -> tuple[dict[str, MarkdownSource], list[str]]:
    """All parsed documents under root: *.md/*.markdown/*.txt files and MinerU zips."""
    sources: dict[str, MarkdownSource] = {}
    warnings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        suffix = path.suffix.lower()
        if suffix in MARKDOWN_SUFFIXES:
            src = MarkdownSource(path)
        elif suffix == ".zip":
            member = _zip_markdown_member(path)
            if member is None:
                warnings.append(f"{path.name}: zip without a single markdown file, skipped")
                continue
            src = MarkdownSource(path, member)
        else:
            continue
        if src.key in sources:
            warnings.append(f"{path.name}: same key as {sources[src.key].path.name}, kept the first")
            continue
        sources[src.key] = src
    return sources, warnings


@dataclass
class Pair:
    markdown: MarkdownSource
    tsv_paths: list[Path]


@dataclass
class LinkResult:
    pairs: list[Pair]
    unmatched_tsv: list[Path] = field(default_factory=list)
    unmatched_markdown: list[MarkdownSource] = field(default_factory=list)
    suggestions: dict[str, str] = field(default_factory=dict)  # tsv name -> close markdown name
    warnings: list[str] = field(default_factory=list)


def read_manifest(path: Path) -> dict[str, str]:
    """Manifest with columns tsv, markdown (file names, not paths).

    Tab-separated by default, because the client's file names contain commas
    ("Abdel-Hamid and Alqahtani, 2022.tsv"); a .csv file is read as comma-separated
    and then needs quoted names (Excel's "Save as CSV" does that).
    """
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if not reader.fieldnames or not {"tsv", "markdown"} <= set(reader.fieldnames):
            raise ValueError(f"{path}: manifest needs the columns 'tsv' and 'markdown'")
        return {row["tsv"].strip(): row["markdown"].strip() for row in reader if row.get("tsv")}


def link(tsv_dir: Path, markdown_dir: Path, manifest: Path | None = None) -> LinkResult:
    sources, warnings = discover_markdown(markdown_dir)
    by_name = {s.path.name: s for s in sources.values()}
    explicit = read_manifest(manifest) if manifest else {}

    grouped: dict[str, Pair] = {}
    unmatched: list[Path] = []
    for tsv in sorted(tsv_dir.glob("*.tsv")):
        if tsv.name in explicit:
            src = by_name.get(explicit[tsv.name]) or sources.get(paper_key(Path(explicit[tsv.name]).stem))
            if src is None:
                warnings.append(f"manifest: {explicit[tsv.name]!r} for {tsv.name} not found")
        else:
            src = sources.get(paper_key(tsv.stem))
        if src is None:
            unmatched.append(tsv)
            continue
        grouped.setdefault(src.key, Pair(src, [])).tsv_paths.append(tsv)

    for pair in grouped.values():  # "X, 2020" before "X, 2020 part 2"; "_02" before "_10"
        pair.tsv_paths.sort(key=lambda p: (len(p.stem), p.stem))
    unmatched_md = [s for k, s in sources.items() if k not in grouped]
    # Suggest against all parsed files: a "part 2" TSV belongs to an already-matched PDF.
    suggestions = {}
    for tsv in unmatched:
        close = difflib.get_close_matches(paper_key(tsv.stem), list(sources), n=1, cutoff=0.8)
        if close:
            suggestions[tsv.name] = sources[close[0]].path.name
    return LinkResult(
        pairs=sorted(grouped.values(), key=lambda p: p.markdown.key),
        unmatched_tsv=unmatched,
        unmatched_markdown=unmatched_md,
        suggestions=suggestions,
        warnings=warnings,
    )
