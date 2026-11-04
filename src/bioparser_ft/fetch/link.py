"""This module links the PDF files that a person downloaded to the ETS TSV files of the papers.

A file from EZproxy or from a browser can have any name, for example gyac068.pdf or
1-s2.0-S004452312300089X-main.pdf. The module reads the first pages of each file and compares
them with each paper: the DOI, the title, the first author and the year. When a file agrees with
one paper that has no PDF file, the module moves the file to fine_tuning_data/pdfs/<paper>.pdf.
Then MinerU and bioparser-build-dataset link it to the TSV files of the paper.

    uv run bioparser-link-pdfs           # Show the result for each file. No file moves.
    uv run bioparser-link-pdfs --apply   # Move the files that agree with one paper.
    uv run bioparser-link-pdfs --apply --link "67640adb47e5f1734609627.pdf=Smith et al., 2023"

The module reads the files in pdfs/_unverified/ and the PDF files in pdfs/ that do not have the
name of a paper. It never replaces a file in pdfs/.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..linking import paper_key
from .citations import Paper, clean_doi, group_papers
from .cli import read_report, write_report_rows
from .matching import norm, title_recall_in_text, tokens
from .pdfcheck import is_pdf

TEXT_PAGES = 3  # The title, the authors and the DOI are on the first pages
MIN_TEXT_CHARS = 200  # A file with less text is a scanned document without a text layer
TITLE_OK = 0.8  # The part of the title words that must be in the file, with the author or the year
MIN_TITLE_WORDS = 4  # A shorter title is too general for a decision
MIN_ID_CHARS = 6  # The shortest part of a DOI that can identify a file name
CONFIDENT_REPORT_LEVELS = ("high", "medium", "given")


@dataclass
class Target:
    """A paper, with the information that the comparison uses."""

    paper: Paper
    doi: str | None
    titles: list[str]
    has_pdf: bool


@dataclass(frozen=True)
class Evidence:
    """The result of the comparison of one file with one paper."""

    doi_in_text: bool = False
    # The file name is the end of the DOI, for example gyac068.pdf for 10.1093/jmammal/gyac068
    doi_in_name: bool = False
    title: float = 0.0  # The part of the title words in the text or in the file name
    title_words: int = 0
    author: bool = False
    year: bool = False

    @property
    def confident(self) -> bool:
        if self.doi_in_text:
            return True
        if self.doi_in_name:
            return self.title >= 0.5 or self.author
        return self.title >= TITLE_OK and self.title_words >= MIN_TITLE_WORDS and (self.author or self.year)

    @property
    def possible(self) -> bool:
        """True if the paper is a good guess for a person, but not good enough for a decision."""
        return self.doi_in_name or self.title >= 0.4 or (self.author and self.year)

    @property
    def rank(self) -> tuple[bool, bool, bool, float, bool, bool]:
        return (
            self.confident,
            self.doi_in_text,
            self.doi_in_name,
            round(self.title, 2),
            self.author,
            self.year,
        )

    def describe(self) -> str:
        found = [("DOI in the text", self.doi_in_text), ("DOI in the file name", self.doi_in_name)]
        parts = [name for name, ok in found if ok]
        parts.append(f"title {self.title:.2f}")
        parts += [name for name, ok in (("first author", self.author), ("year", self.year)) if ok]
        return ", ".join(parts)


@dataclass
class Decision:
    path: Path
    action: str  # "move", "duplicate", "review" or "skip"
    target: Target | None = None
    evidence: Evidence = Evidence()
    reason: str = ""
    pages: int = 0
    forced: bool = False


# ------------------------------------------------------------------------ comparison
def read_text(data: bytes) -> tuple[str, int]:
    """Return the text of the first pages and the number of pages. A damaged file gives ("", 0)."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")  # Many publisher PDF files use an empty user password
        pages = len(reader.pages)
        text = " ".join((reader.pages[i].extract_text() or "") for i in range(min(TEXT_PAGES, pages)))
    except Exception:  # A damaged or unusual PDF file
        return "", 0
    return text, pages


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def doi_in_file_name(doi: str, name: str) -> bool:
    """Return True if the file name is the DOI or its end, for example 15081109ACC2022.24.2.001.pdf."""
    file_id = _alnum(Path(name).stem)
    if len(file_id) < MIN_ID_CHARS:
        return False
    doi = doi.lower()
    last = _alnum(doi.rsplit("/", 1)[-1])
    ids = {_alnum(doi), _alnum(doi.split("/", 1)[-1]), last}
    return file_id in ids or (len(last) > MIN_ID_CHARS and file_id.endswith(last))


def evidence(target: Target, text: str, name: str) -> Evidence:
    """Compare one file, its text and its name, with one paper."""
    doi_in_text = doi_in_name = False
    if target.doi:
        doi_in_text = target.doi.lower() in re.sub(r"\s+", "", text).lower()
        doi_in_name = doi_in_file_name(target.doi, name)
    # Some file names contain the title, for example "Austral Ecology - 2015 - Arman - Dietary ...pdf"
    words = f"{text} {Path(name).stem.replace('_', ' ')}"
    title, title_words = 0.0, 0
    for t in target.titles:
        n = len(tokens(t))
        if n and (r := title_recall_in_text(t, words), n) > (title, title_words):
            title, title_words = r, n
    padded = f" {norm(words)} "
    first = norm(target.paper.first_author)
    author = bool(first) and f" {first} " in padded
    year = target.paper.year is not None and f" {target.paper.year} " in padded
    return Evidence(doi_in_text, doi_in_name, title, title_words, author, year)


def decide(path: Path, targets: Sequence[Target]) -> Decision:
    """Compare one file with all papers, and decide what to do with it."""
    data = path.read_bytes()
    if not is_pdf(data):
        return Decision(path, "skip", reason="not a PDF file")
    text, pages = read_text(data)
    if pages == 0:
        return Decision(path, "skip", reason="pypdf cannot read the file")
    scored = sorted(
        ((evidence(t, text, path.name), t) for t in targets), key=lambda s: s[0].rank, reverse=True
    )
    best, target = scored[0]
    second = scored[1][0] if len(scored) > 1 else Evidence()
    scan = "" if len(text.strip()) >= MIN_TEXT_CHARS else "no text layer (a scanned document?): "
    if not best.confident:
        guess = target if best.possible else None
        return Decision(path, "review", guess, best, scan + "no paper agrees enough", pages)
    if second.confident and second.rank[:3] == best.rank[:3] and abs(second.title - best.title) < 0.1:
        other = scored[1][1].paper.key
        return Decision(
            path, "review", target, best, f"two papers agree: {target.paper.key} and {other}", pages
        )
    if target.has_pdf:
        return Decision(path, "duplicate", target, best, f"pdfs/{target.paper.pdf_name} exists", pages)
    return Decision(path, "move", target, best, pages=pages)


def _one_file_per_paper(decisions: list[Decision]) -> None:
    """If two files agree with one paper, move only the file with the best evidence."""
    best: dict[str, Decision] = {}
    for d in sorted(decisions, key=lambda d: (d.forced, d.evidence.rank), reverse=True):
        if d.action != "move" or d.target is None:
            continue
        key = d.target.paper.key
        if key in best:
            d.action, d.reason = "duplicate", f"{best[key].path.name} agrees better with this paper"
        else:
            best[key] = d


# ------------------------------------------------------------------------ inputs
def find_target(name: str, targets: Sequence[Target]) -> Target | None:
    """Find a paper by its name, for example "Smith et al., 2023", "ETS_Smith et al., 2023.tsv" or a PDF name."""
    stem = name[:-4] if name.lower().endswith((".pdf", ".tsv")) else name
    key = paper_key(stem)
    for t in targets:
        if any(paper_key(n) == key for n in (t.paper.key, *(Path(f).stem for f in t.paper.tsv_files))):
            return t
    return None


def load_targets(args: argparse.Namespace) -> list[Target]:
    papers = group_papers(args.tsv_dir, args.overrides)
    if not papers:
        raise SystemExit(f"There are no ETS TSV files in {args.tsv_dir}. Put the client files there first.")
    report = read_report(args.report)
    targets = []
    for p in papers:
        row = report.get(p.key, {})
        titles = [p.title]
        # The title from Crossref or OpenAlex is better than the title from the citation
        if row.get("matched_title") and row.get("confidence") in CONFIDENT_REPORT_LEVELS:
            titles.append(row["matched_title"])
        doi = p.doi or row.get("doi") or None
        targets.append(
            Target(p, clean_doi(doi) if doi else None, [t for t in dict.fromkeys(titles) if t], False)
        )
    for t in targets:
        t.has_pdf = (args.pdf_dir / t.paper.pdf_name).is_file()
    return targets


def _pdfs_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def find_sources(explicit: Sequence[Path] | None, pdf_dir: Path, targets: Sequence[Target]) -> list[Path]:
    if explicit:
        found: list[Path] = []
        for p in explicit:
            found += _pdfs_in(p) if p.is_dir() else [p]
        return list(dict.fromkeys(found))
    names = {t.paper.pdf_name for t in targets}
    loose = [p for p in _pdfs_in(pdf_dir) if p.name not in names]
    return _pdfs_in(pdf_dir / "_unverified") + loose


def parse_links(values: Sequence[str]) -> dict[str, str]:
    """Parse --link "FILE=PAPER" values. Without "=PAPER", the file name is the paper."""
    links = {}
    for v in values:
        file, _, paper = v.partition("=")
        links[Path(file.strip()).name] = paper.strip() or Path(file.strip()).stem
    return links


# ------------------------------------------------------------------------ output
def _print_decisions(decisions: Sequence[Decision]) -> None:
    for d in decisions:
        print(f"{d.action.upper():10}{d.path.name}")
        if d.action == "skip":
            print(f"          {d.reason}")
        elif d.action == "review":
            if d.target is None:
                print(f"          {d.reason}. Examine the file. Then use --link FILE=PAPER.")
            else:
                print(
                    f"          best guess: {d.target.paper.key} ({d.evidence.describe()}). {d.reason}.\n"
                    f'          If the file is correct, use: --link "{d.path.name}={d.target.paper.key}"'
                )
        elif d.target is not None:
            arrow = f"-> pdfs/{d.target.paper.pdf_name}"
            why = "--link" if d.forced else d.evidence.describe()
            print(f"          {arrow} ({why})" + (f". {d.reason}" if d.reason else ""))


def _apply(decisions: Sequence[Decision], pdf_dir: Path, report: Path) -> int:
    rows = read_report(report)
    moved = 0
    for d in decisions:
        if d.action != "move" or d.target is None:
            continue
        paper = d.target.paper
        dest = pdf_dir / paper.pdf_name
        if dest.exists():  # The file came after the start of the command: do not replace it
            print(f"not moved: {dest} exists")
            continue
        shutil.move(str(d.path), dest)
        moved += 1
        data = dest.read_bytes()
        row = rows.setdefault(
            paper.key,
            {"key": paper.key, "tsv_files": " | ".join(paper.tsv_files), "citation": paper.citation},
        )
        note = f"bioparser-link-pdfs: {d.path.name} ({'--link' if d.forced else d.evidence.describe()})"
        old = row.get("notes", "")
        row.update(
            status="downloaded",
            pdf_source="manual",
            pdf_url=d.path.name,
            verified="yes" if d.evidence.confident else "manual",
            title_recall=f"{d.evidence.title:.2f}",
            pages=str(d.pages),
            bytes=str(len(data)),
            sha256=hashlib.sha256(data).hexdigest(),
            notes=f"{old} ; {note}" if old else note,
        )
    if moved:
        write_report_rows(report, rows.values())
    return moved


# ------------------------------------------------------------------------ main
def run(args: argparse.Namespace) -> list[Decision]:
    targets = load_targets(args)
    sources = find_sources(args.sources, args.pdf_dir, targets)
    links = parse_links(args.link)
    unknown = sorted(set(links) - {p.name for p in sources})
    if unknown:
        raise SystemExit(f"--link: these files are not in the input folders: {unknown}")
    decisions: list[Decision] = []
    for path in sources:
        if path.name not in links:
            decisions.append(decide(path, targets))
            continue
        target = find_target(links[path.name], targets)
        if target is None:
            close = difflib.get_close_matches(links[path.name], [t.paper.key for t in targets], n=3)
            raise SystemExit(f"--link: no paper {links[path.name]!r}. Similar names: {close}")
        text, pages = read_text(path.read_bytes())
        ev = evidence(target, text, path.name)
        action, reason = (
            ("duplicate", f"pdfs/{target.paper.pdf_name} exists") if target.has_pdf else ("move", "")
        )
        decisions.append(Decision(path, action, target, ev, reason, pages, forced=True))
    _one_file_per_paper(decisions)

    if not sources:
        print("There are no PDF files to link.")
    _print_decisions(decisions)
    counts = {a: sum(d.action == a for d in decisions) for a in ("move", "review", "duplicate", "skip")}
    print(
        f"\nFiles: {len(decisions)}. Move: {counts['move']}. Review: {counts['review']}. "
        f"Duplicate: {counts['duplicate']}. Not usable: {counts['skip']}."
    )
    moving = {d.target.paper.key for d in decisions if d.action == "move" and d.target is not None}
    if args.apply:
        moved = _apply(decisions, args.pdf_dir, args.report)
        print(f"Moved {moved} files to {args.pdf_dir}/. The report is {args.report}.")
    elif counts["move"]:
        print("This was a dry run: no file moved. To move the files, add --apply.")
    missing = [t.paper.key for t in targets if not t.has_pdf and t.paper.key not in moving]
    print(f"Papers without a PDF file after this step: {len(missing)}. For example: {missing[:8]}")
    if args.apply and counts["move"]:
        print("Next steps: scripts/parse_pdfs.sh, then uv run bioparser-build-dataset.")
    return decisions


def build_parser() -> argparse.ArgumentParser:
    d = Path("fine_tuning_data")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv-dir", type=Path, default=d / "ets_tsv")
    ap.add_argument("--pdf-dir", type=Path, default=d / "pdfs")
    ap.add_argument(
        "--from",
        dest="sources",
        type=Path,
        action="append",
        help="a PDF file or a folder with PDF files. You can use this option more than one time. "
        "Default: pdfs/_unverified/ and the PDF files in pdfs/ that do not have the name of a paper.",
    )
    ap.add_argument("--overrides", type=Path, default=d / "doi_overrides.tsv")
    ap.add_argument("--report", type=Path, default=d / "pdf_sources.tsv")
    ap.add_argument(
        "--apply", action="store_true", help="move the files. Without this option, no file moves."
    )
    ap.add_argument(
        "--link",
        action="append",
        default=[],
        metavar="FILE=PAPER",
        help='link a file after you examined it, for example "gyac068.pdf=Kim et al., 2023". '
        "Without =PAPER, the name of the file is the paper. You can use this option more than one time.",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
