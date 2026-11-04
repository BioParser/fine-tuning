"""Download the article PDF file for each client ETS TSV file.

Do these steps in this sequence:

    uv run bioparser-fetch-pdfs --email you@helsinki.fi --resolve-only   # 1. Examine the DOI matches
    uv run bioparser-fetch-pdfs --email you@helsinki.fi                  # 2. Open-access APIs and landing pages
    uv run bioparser-fetch-pdfs --email you@helsinki.fi --browser        # 3. Add the headless browser
    uv run bioparser-fetch-pdfs --email you@helsinki.fi --manual \\
        --proxy-prefix "https://login.libproxy.helsinki.fi/login?url="  # 4. You click or log in, the code saves

Do these steps outside the network of the university. Only step 4 uses the library proxy
and your login. The library terms do not let programs download articles automatically.

The command works on 6 papers at the same time (--workers), and the headless pass uses
3 browsers (--browser-workers). Each host keeps its own rate limit.

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
import queue
import re
import tempfile
import threading
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "key", "status", "confidence", "oa_status", "doi", "resolved_by", "pdf_source", "pdf_url", "verified",
    "title_recall", "pages", "bytes", "sha256", "tsv_files", "citation", "matched_title", "notes",
)  # fmt: skip
MISSING = ("not_found", "unverified", "unresolved")
_PRINT_LOCK = threading.Lock()


def _say(text: str) -> None:
    """Print one message. The lock keeps the lines of different threads apart."""
    with _PRINT_LOCK:
        print(text, flush=True)


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
    oa_status: str = ""  # From Unpaywall or OpenAlex: gold, diamond, hybrid, bronze, green or closed


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
        if best is not None and best.check.verified:
            break
    if best is None:
        notes.append(f"{cand.source}: no PDF link on {fetched.url[:90]}")
    return best


def fetch_http(
    client: PoliteClient, sch: Scholarly, paper: Paper, res: Resolution, info: dict[str, str] | None = None
) -> tuple[Got | None, list[str]]:
    """Use the candidates one batch at a time, and stop at the first verified PDF.

    Then the code does not call the APIs of the next batches. The function writes the
    open-access status to info["oa_status"].
    """
    title = res.title or paper.title
    notes: list[str] = []
    tried: set[str] = set()

    def batches() -> Iterator[list[Candidate]]:
        if res.candidates:
            yield sorted(res.candidates)
        if res.doi:
            yield from sch.oa_candidates(res.doi, info)
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


def browser_pass(
    outcomes: list[Outcome], args: argparse.Namespace, *, manual: bool, stop: threading.Event | None = None
) -> None:
    pending = [o for o in outcomes if o.status in MISSING]
    if not pending:
        return
    # A person works with one window at a time. The headless pass can use more browsers.
    workers = 1 if manual else max(1, min(args.browser_workers, len(pending)))
    mode = "manual" if manual else "headless browser"
    more = f", {workers} browsers at the same time" if workers > 1 else ""
    _say(f"\n{mode} pass: {len(pending)} papers{more}")
    if not manual and args.proxy_prefix:
        _say("  The headless pass does not use --proxy-prefix. Only the --manual pass uses it.")
    stop = stop or threading.Event()
    todo: queue.SimpleQueue[tuple[int, Outcome]] = queue.SimpleQueue()
    for item in enumerate(pending, 1):
        todo.put(item)
    if workers == 1:
        _browser_worker(todo, len(pending), args, manual=manual, stop=stop)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="browser") as pool:
        futures = [
            pool.submit(_browser_worker, todo, len(pending), args, manual=False, stop=stop)
            for _ in range(workers)
        ]
        try:
            for f in futures:
                f.result()  # An error in a browser thread comes to the main thread here
        except BaseException:
            stop.set()  # The other browsers stop after their current paper
            raise


def _browser_worker(
    todo: queue.SimpleQueue[tuple[int, Outcome]],
    total: int,
    args: argparse.Namespace,
    *,
    manual: bool,
    stop: threading.Event,
) -> None:
    """Get the papers from the queue, one at a time, with one browser. Each thread has its own browser."""
    from .browser import BrowserFetcher

    # The library terms do not let programs download articles automatically. For this reason, only the
    # manual pass uses the library proxy and the saved profile with your login. Each headless browser
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
        while not stop.is_set():
            try:
                i, oc = todo.get_nowait()
            except queue.Empty:
                return
            url = _landing_url(oc, manual=manual, proxy_prefix=proxy_prefix)
            if not url:
                continue
            label = f"[{i}/{total}] {oc.paper.key}"
            raw = bf.fetch_manual(url, label, args.manual_timeout) if manual else bf.fetch_auto(url)
            if raw is None:
                if not manual:
                    _say(f"{label}: the browser found no PDF")
                continue
            data, src = raw
            title = oc.resolution.title or oc.paper.title
            before = oc.status
            _apply(
                oc,
                Got(data, src, "manual" if manual else "browser", check_pdf(data, title, oc.resolution.doi)),
                args.out_dir,
            )
            _say(f"{label}: {before} -> {oc.status}")


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
    rows: list[Mapping[str, object]] = []
    for oc in outcomes:
        if oc.status == "exists" and oc.paper.key in previous:
            rows.append(previous[oc.paper.key] | {"status": "exists"})
            continue
        r, g = oc.resolution, oc.got
        rows.append({
            "key": oc.paper.key, "status": oc.status, "confidence": r.confidence,
            "oa_status": oc.oa_status, "doi": r.doi or "",
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
    if missing:
        closed = sum(o.oa_status == "closed" for o in missing)
        free = sum(o.oa_status not in ("", "closed") for o in missing)
        print(
            f"not found: {closed} closed access (only --manual with --proxy-prefix can get them), "
            f"{free} open access (a free copy exists, but the download failed), "
            f"{len(missing) - closed - free} unknown (no DOI, or no answer from Unpaywall and OpenAlex)"
        )
    for o in missing[:20]:
        where = f"https://doi.org/{o.resolution.doi}" if o.resolution.doi else "no DOI found"
        print(f"  {o.status:10} oa:{o.oa_status or '-':7} {o.paper.key[:55]:55} {where}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
    print(f"report: {report}")
    if missing:
        print(
            "next steps: use --browser, then --manual with --proxy-prefix for library access. "
            "Add incorrect or unresolved matches to doi_overrides.tsv. Ask the client for the other PDF files."
        )


# ------------------------------------------------------------------------ main
def process_paper(paper: Paper, client: PoliteClient, sch: Scholarly, args: argparse.Namespace) -> Outcome:
    """Find the DOI and an open-access PDF file for one paper. The threads of the pool call this function."""
    target = args.out_dir / paper.pdf_name
    if target.exists() and not args.refresh:
        with target.open("rb") as fh:
            if is_pdf(fh.read(1024)):
                return Outcome(paper, Resolution(), "exists")
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
        info: dict[str, str] = {}
        got, notes = fetch_http(client, sch, paper, res, info)
        oc.oa_status = info.get("oa_status", "")
        oc.notes += notes[-8:]
        _apply(oc, got, args.out_dir)
    return oc


def _process_safely(
    paper: Paper, client: PoliteClient, sch: Scholarly, args: argparse.Namespace, stop: threading.Event
) -> Outcome | None:
    if stop.is_set():
        return None
    try:
        return process_paper(paper, client, sch, args)
    except Exception as exc:  # An error in one paper must not stop the full run
        return Outcome(paper, Resolution(), "not_found", notes=[f"error: {exc!r}"])


def _progress(done: int, total: int, oc: Outcome) -> None:
    if oc.status == "exists":
        _say(f"[{done}/{total}] exists     {oc.paper.key[:60]} (the PDF file is there from an earlier run)")
        return
    r = oc.resolution
    pdf = oc.got.source if oc.got else "-"
    _say(
        f"[{done}/{total}] {oc.status:10} {r.confidence:6} doi:{r.source or '-':15} "
        f"pdf:{pdf:17} oa:{oc.oa_status or '-':7} {oc.paper.key[:60]}"
    )


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
    workers = max(1, args.workers)
    # The paper threads wait for the API threads. For this reason, they use two different pools:
    # one pool cannot wait for itself (no deadlock).
    paper_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="paper")
    api_pool = ThreadPoolExecutor(max_workers=3 * workers, thread_name_prefix="api") if workers > 1 else None
    sch = Scholarly(client, email=email, pool=api_pool)
    stop = threading.Event()
    results: dict[int, Outcome] = {}
    try:
        futures = {
            paper_pool.submit(_process_safely, paper, client, sch, args, stop): i
            for i, paper in enumerate(papers)
        }
        for done, fut in enumerate(as_completed(futures), 1):
            if (oc := fut.result()) is not None:
                results[futures[fut]] = oc
                _progress(done, len(papers), oc)
        outcomes = [results[i] for i in sorted(results)]  # The sequence of the TSV files
        if not args.resolve_only:
            if args.browser:
                browser_pass(outcomes, args, manual=False, stop=stop)
            if args.manual:
                browser_pass(outcomes, args, manual=True, stop=stop)
    except BaseException:
        stop.set()  # For example Ctrl+C: the threads stop after their current request
        raise
    finally:
        paper_pool.shutdown(wait=False, cancel_futures=True)
        if api_pool is not None:
            api_pool.shutdown(wait=False, cancel_futures=True)
        client.close()
        outcomes = [results[i] for i in sorted(results)]
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
    ap.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="seconds between two requests to the same web host. The scholarly APIs use their own limits.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=6,
        help="the number of papers at the same time. Each host keeps its own rate limit.",
    )
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
        "--browser-workers",
        type=int,
        default=3,
        help="the number of headless browsers at the same time. The --manual pass uses one browser.",
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
