"""Download the article PDF file for each client ETS TSV file.

Do these steps in this sequence:

    uv run bioparser-fetch-pdfs --email you@helsinki.fi --resolve-only   # 1. Examine the DOI matches
    uv run bioparser-fetch-pdfs --email you@helsinki.fi                  # 2. Open-access APIs and landing pages
    uv run bioparser-fetch-pdfs --email you@helsinki.fi --browser        # 3. Add the headless browser
    uv run bioparser-fetch-pdfs --email you@helsinki.fi --manual \\
        --proxy-prefix "https://login.libproxy.helsinki.fi/login?url="  # 4. You click or log in, the code saves

Do these steps outside the network of the university. Only step 4 uses the library proxy
and your login. The library terms do not let programs download articles automatically.

A second run is fast. The cache keeps the API answers, and the code keeps the PDF files
from the earlier runs.

Results:
  fine_tuning_data/pdfs/<paper>.pdf          PDF files that contain the title or the DOI
  fine_tuning_data/pdfs/_unverified/         PDF files without the title or the DOI: examine them
  fine_tuning_data/pdf_sources.tsv           one row for each paper: status, DOI, source, URL, sha256
  fine_tuning_data/manifest.tsv              the other TSV files of a paper -> <paper>.zip

If a DOI is incorrect or not found, add a row to fine_tuning_data/doi_overrides.tsv.
The columns are tsv, doi, url and note.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from ..linking import paper_key
from .citations import Paper, group_papers, safe_name
from .landing import pdf_links_from_html
from .net import PoliteClient
from .pdfcheck import PdfCheck, check_pdf, is_pdf
from .scholarly import Candidate, Resolution, Scholarly, make_candidate

REPORT_COLUMNS = (
    "key", "status", "confidence", "doi", "resolved_by", "pdf_source", "pdf_url", "verified",
    "title_recall", "pages", "bytes", "sha256", "tsv_files", "citation", "matched_title", "notes",
)  # fmt: skip
MISSING = ("not_found", "unverified", "unresolved")


@dataclass
class Got:
    data: bytes
    url: str
    source: str
    check: PdfCheck


@dataclass
class Outcome:
    paper: Paper
    resolution: Resolution
    status: str = ""
    got: Got | None = None
    notes: list[str] = field(default_factory=list)


def _better(a: Got | None, b: Got | None) -> Got | None:
    if a is None:
        return b
    if b is None:
        return a
    return b if b.check.verified and not a.check.verified else a


def fetch_candidate(
    client: PoliteClient, cand: Candidate, title: str, doi: str | None, tried: set[str], notes: list[str]
) -> Got | None:
    """Send a GET request for one candidate.

    If the answer is HTML, for example a landing page or a viewer page, the function parses
    it one time for PDF links. The function examines each PDF file that it finds.
    """
    fetched = client.get_web(cand.url)
    if fetched is None:
        notes.append(f"{cand.source}: no response")
        return None
    if fetched.status >= 400:
        notes.append(f"{cand.source}: HTTP {fetched.status}")
        return None
    if is_pdf(fetched.body):
        return Got(fetched.body, fetched.url, cand.source, check_pdf(fetched.body, title, doi))
    if "html" not in fetched.content_type and fetched.body.lstrip()[:1] != b"<":
        notes.append(f"{cand.source}: {fetched.content_type or 'unknown content'}")
        return None
    best = None
    for link in pdf_links_from_html(fetched.body, fetched.url):
        if link in tried:
            continue
        tried.add(link)
        # Some sites send the PDF only with the Referer header
        page = client.get_web(link, referer=fetched.url)
        if page is None or page.status >= 400 or not is_pdf(page.body):
            continue
        source = cand.source if cand.kind == "pdf" else "landing"
        best = _better(best, Got(page.body, page.url, source, check_pdf(page.body, title, doi)))
        if best.check.verified:
            break
    if best is None:
        notes.append(f"{cand.source}: no PDF link on {fetched.url[:90]}")
    return best


def fetch_http(
    client: PoliteClient, sch: Scholarly, paper: Paper, res: Resolution
) -> tuple[Got | None, list[str]]:
    """Use the candidates one batch at a time, and stop at the first verified PDF.

    Then the code does not call the APIs of the next batches.
    """
    title = res.title or paper.title
    notes: list[str] = []
    tried: set[str] = set()

    def batches() -> Iterator[list[Candidate]]:
        if res.candidates:
            yield sorted(res.candidates)
        if res.doi:
            yield from sch.oa_candidates(res.doi)
        elif res.landing:
            yield [c for u in res.landing if (c := make_candidate(u, "landing", kind="landing", stage=2))]

    best = None
    for batch in batches():
        for cand in batch:
            if cand.url in tried:
                continue
            tried.add(cand.url)
            best = _better(best, fetch_candidate(client, cand, title, res.doi, tried, notes))
            if best and best.check.verified:
                return best, notes
    return best, notes


def _save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)  # An atomic write: an interruption does not leave a partial PDF file


def _apply(oc: Outcome, got: Got | None, out_dir: Path) -> None:
    if got is None:
        oc.status = oc.status or "not_found"
        return
    if got.check.verified:
        _save(out_dir / oc.paper.pdf_name, got.data)
        (out_dir / "_unverified" / oc.paper.pdf_name).unlink(missing_ok=True)
        oc.status, oc.got = "downloaded", got
    elif oc.got is None or not oc.got.check.verified:
        _save(out_dir / "_unverified" / oc.paper.pdf_name, got.data)
        oc.status, oc.got = "unverified", got
        oc.notes.append(f"unverified: {got.check.reason}")


def _landing_url(oc: Outcome, *, manual: bool, proxy_prefix: str | None) -> str | None:
    r = oc.resolution
    url = (
        (f"https://doi.org/{r.doi}" if r.doi else None)
        or oc.paper.url
        or (r.landing[0] if r.landing else None)
    )
    if url:
        return (proxy_prefix or "") + url
    if manual and oc.paper.citation:  # No DOI found: let the person search for the article
        return "https://scholar.google.com/scholar?q=" + quote(oc.paper.citation[:300])
    return None


def browser_pass(outcomes: list[Outcome], args: argparse.Namespace, *, manual: bool) -> None:
    from .browser import BrowserFetcher

    pending = [o for o in outcomes if o.status in MISSING]
    if not pending:
        return
    print(f"\n{'manual' if manual else 'headless browser'} pass: {len(pending)} papers")
    if not manual and args.proxy_prefix:
        print("  The headless pass does not use --proxy-prefix. Only the --manual pass uses it.")
    # The library terms do not let programs download articles automatically. For this reason, only the
    # manual pass uses the library proxy and the saved profile with your login. The headless pass
    # starts with a new, empty profile, so it has no login cookies.
    proxy_prefix = args.proxy_prefix if manual else None
    with contextlib.ExitStack() as stack:
        if manual:
            profile_dir = args.browser_profile
        else:
            tmp = tempfile.TemporaryDirectory(prefix="bioparser-headless-", ignore_cleanup_errors=True)
            profile_dir = Path(stack.enter_context(tmp))
        bf = stack.enter_context(
            BrowserFetcher(
                profile_dir=profile_dir,
                headless=not manual,
                executable_path=args.browser_executable,
                channel=args.browser_channel,
            )
        )
        for i, oc in enumerate(pending, 1):
            url = _landing_url(oc, manual=manual, proxy_prefix=proxy_prefix)
            if not url:
                continue
            label = f"[{i}/{len(pending)}] {oc.paper.key}"
            raw = bf.fetch_manual(url, label, args.manual_timeout) if manual else bf.fetch_auto(url)
            if raw is None:
                if not manual:
                    print(f"{label}: the browser found no PDF")
                continue
            data, src = raw
            title = oc.resolution.title or oc.paper.title
            before = oc.status
            _apply(
                oc,
                Got(data, src, "manual" if manual else "browser", check_pdf(data, title, oc.resolution.doi)),
                args.out_dir,
            )
            print(f"{label}: {before} -> {oc.status}")


# ------------------------------------------------------------------------ results
def _cell(value: object) -> str:
    return re.sub(r"[\t\r\n]+", " ", str(value)) if value is not None else ""


def _read_report(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["key"]: row for row in csv.DictReader(fh, delimiter="\t")}


def write_report(path: Path, outcomes: list[Outcome]) -> None:
    previous = _read_report(path)
    rows = []
    for oc in outcomes:
        if oc.status == "exists" and oc.paper.key in previous:
            rows.append(previous[oc.paper.key] | {"status": "exists"})
            continue
        r, g = oc.resolution, oc.got
        rows.append({
            "key": oc.paper.key, "status": oc.status, "confidence": r.confidence, "doi": r.doi or "",
            "resolved_by": r.source, "pdf_source": g.source if g else "", "pdf_url": g.url if g else "",
            "verified": ("yes" if g.check.verified else "no") if g else "",
            "title_recall": f"{g.check.title_recall:.2f}" if g and g.check.title_recall is not None else "",
            "pages": g.check.pages if g else "", "bytes": len(g.data) if g else "",
            "sha256": hashlib.sha256(g.data).hexdigest() if g else "",
            "tsv_files": " | ".join(oc.paper.tsv_files), "citation": oc.paper.citation,
            "matched_title": r.title, "notes": " ; ".join(dict.fromkeys(oc.notes)),
        })  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(REPORT_COLUMNS)
        for row in rows:
            w.writerow([_cell(row.get(c, "")) for c in REPORT_COLUMNS])


def update_manifest(path: Path, papers: list[Paper]) -> int:
    """Map the other TSV files of a paper to its parsed PDF, <paper>.zip.

    The function keeps the current rows of the manifest, which can contain your manual changes.
    """
    rows: dict[str, str] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows = {r["tsv"]: r["markdown"] for r in csv.DictReader(fh, delimiter="\t") if r.get("tsv")}
    added = 0
    for paper in papers:
        target = safe_name(paper.key) + ".zip"
        for tsv in paper.tsv_files:
            if tsv in rows or paper_key(Path(tsv).stem) == paper_key(Path(target).stem):
                continue  # The manifest has a row for this file, or the file names agree
            rows[tsv] = target
            added += 1
    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            fh.write("tsv\tmarkdown\n")
            for tsv, md in sorted(rows.items()):
                fh.write(f"{tsv}\t{md}\n")
    return added


def summarise(outcomes: list[Outcome], report: Path) -> None:
    counts = Counter(o.status for o in outcomes)
    print(f"\n{len(outcomes)} papers: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    by_source = Counter(o.got.source for o in outcomes if o.got and o.status == "downloaded")
    if by_source:
        print("downloaded from: " + ", ".join(f"{k} {v}" for k, v in by_source.most_common()))
    missing = [o for o in outcomes if o.status in MISSING]
    for o in missing[:20]:
        where = f"https://doi.org/{o.resolution.doi}" if o.resolution.doi else "no DOI found"
        print(f"  {o.status:10} {o.paper.key[:55]:55} {where}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
    print(f"report: {report}")
    if missing:
        print(
            "next steps: use --browser, then --manual with --proxy-prefix for library access. "
            "Add incorrect or unresolved matches to doi_overrides.tsv. Ask the client for the other PDF files."
        )


# ------------------------------------------------------------------------ main
def run(args: argparse.Namespace, client: PoliteClient | None = None) -> list[Outcome]:
    """Run the full procedure. The tests give a `client` with an httpx.MockTransport."""
    papers = group_papers(args.tsv_dir, args.overrides)
    if args.only:
        papers = [p for p in papers if any(o.casefold() in p.key.casefold() for o in args.only)]
    papers = papers[: args.limit] if args.limit else papers
    email = args.email or os.environ.get("BIOPARSER_CONTACT_EMAIL")
    if not email:
        print(
            "NOTE: there is no --email. The code does not use Unpaywall, the largest open-access index. "
            "Crossref and OpenAlex use their slower common pool."
        )
    if client is None:
        client = PoliteClient(email=email, cache_dir=args.cache_dir, delay=args.delay, refresh=args.refresh)
    sch = Scholarly(client, email=email)
    outcomes: list[Outcome] = []
    try:
        for i, paper in enumerate(papers, 1):
            target = args.out_dir / paper.pdf_name
            if target.exists() and not args.refresh:
                with target.open("rb") as fh:
                    if is_pdf(fh.read(1024)):
                        outcomes.append(Outcome(paper, Resolution(), "exists"))
                        continue
            res = sch.resolve(paper)
            oc = Outcome(paper, res, notes=paper.notes + res.notes)
            if res.confidence == "medium":
                oc.notes.append(f"medium-confidence match: examine '{res.title[:80]}'")
            if not (res.doi or res.candidates or res.landing) or (
                args.min_confidence == "high" and res.confidence == "medium"
            ):
                oc.status = "unresolved"
            elif args.resolve_only:
                oc.status = "resolved"
            else:
                got, notes = fetch_http(client, sch, paper, res)
                oc.notes += notes[-8:]
                _apply(oc, got, args.out_dir)
            outcomes.append(oc)
            src = oc.got.source if oc.got else res.source or "-"
            print(
                f"[{i}/{len(papers)}] {oc.status:10} {res.confidence:6} {src:15} {paper.key[:60]}", flush=True
            )
        if not args.resolve_only:
            if args.browser:
                browser_pass(outcomes, args, manual=False)
            if args.manual:
                browser_pass(outcomes, args, manual=True)
    finally:
        client.close()
        write_report(args.report, outcomes)
        if (added := update_manifest(args.manifest, papers)) and not args.resolve_only:
            print(
                f"manifest: the code added {added} rows for papers with more than one TSV file to {args.manifest}"
            )
    summarise(outcomes, args.report)
    return outcomes


def build_parser() -> argparse.ArgumentParser:
    d = Path("fine_tuning_data")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv-dir", type=Path, default=d / "ets_tsv")
    ap.add_argument("--out-dir", type=Path, default=d / "pdfs")
    ap.add_argument("--report", type=Path, default=d / "pdf_sources.tsv")
    ap.add_argument("--overrides", type=Path, default=d / "doi_overrides.tsv")
    ap.add_argument("--manifest", type=Path, default=d / "manifest.tsv")
    ap.add_argument("--cache-dir", type=Path, default=d / ".fetch_cache")
    ap.add_argument(
        "--email",
        help="your contact email for Unpaywall, Crossref and OpenAlex. You can also set $BIOPARSER_CONTACT_EMAIL.",
    )
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between two requests to the same host")
    ap.add_argument(
        "--resolve-only",
        action="store_true",
        help="find the DOIs of the citations, but do not download files",
    )
    ap.add_argument("--min-confidence", choices=("medium", "high"), default="medium")
    ap.add_argument(
        "--only",
        action="append",
        help="use only the papers with this text in the name. You can use this option more than one time.",
    )
    ap.add_argument("--limit", type=int, help="use only the first N papers")
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="do not use the API cache or the PDF files from the earlier runs",
    )
    ap.add_argument(
        "--browser", action="store_true", help="use a headless browser for pages that use JavaScript"
    )
    ap.add_argument(
        "--manual",
        action="store_true",
        help="use a visible browser: you click or log in, and the code saves the file",
    )
    ap.add_argument(
        "--manual-timeout",
        type=float,
        default=300,
        help="the time limit in seconds for each paper in the --manual mode",
    )
    ap.add_argument(
        "--proxy-prefix",
        help="the EZproxy prefix of your library. Only the --manual pass uses it. "
        'University of Helsinki: "https://login.libproxy.helsinki.fi/login?url="',
    )
    ap.add_argument(
        "--browser-profile",
        type=Path,
        default=d / ".browser_profile",
        help="the browser profile on the disk. It keeps your library login and it contains cookies, so keep it private.",
    )
    ap.add_argument("--browser-channel", help="use an installed browser: chrome or msedge")
    ap.add_argument("--browser-executable", help="the path to a Chromium or Chrome binary")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
