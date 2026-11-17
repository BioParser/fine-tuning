"""CLI: build train/val/test JSONL from client ETS TSVs + parsed PDFs (stdlib only).

    uv run bioparser-build-dataset \\
        --tsv-dir fine_tuning_data/ets_tsv --markdown-dir fine_tuning_data/parsed

Per article (one parsed PDF + all TSVs linked to it) this

1. pairs the TSVs with the parsed PDFs: by file name, by manifest.tsv, and by citation (a TSV
   without a parsed PDF joins the article whose TSVs have the same ``references`` value). Two
   parsed files with the same text are one article (one PDF under two names),
2. reads the TSVs quote-aware and normalises cells/vocabularies (tsv.read_ets_tsv), uses one
   citation for the article and removes the duplicate rows (e.g. a "part 2" file that repeats rows),
3. checks that the parsed PDF contains the labelled values (coverage.label_coverage)
   and excludes pairs below --min-coverage (wrong PDF, lost tables, supplement data),
4. removes the rows that have no value in the parsed text (coverage.row_support): the model
   cannot learn to copy a value that is not in its input. --keep-unsupported-rows keeps them,
5. serialises the target in the model TSV dialect (tsv.to_model_tsv),
6. assigns train/val/test by a hash of the citation, so multi-file papers
   (e.g. "Abraham et al., 2022" + "... part 2") never straddle splits and adding
   new papers later does not reshuffle existing ones,
7. writes chat-format JSONL plus build_report.json (exclusions, unmatched files).

Token-length filtering happens at training time with the real tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .coverage import label_coverage, numbers_in, row_support
from .linking import Pair, link
from .markdown import clean_markdown
from .schema import MODEL_COLUMNS, PROMPT_VERSION, build_messages
from .tsv import Record, read_ets_tsv, to_model_tsv

SPLITS = ("train", "val", "test")
MIN_VALUES_TO_JUDGE = 5  # distinct labelled numbers needed before value coverage decides


def stable_fraction(key: str, salt: str) -> float:
    """Deterministic pseudo-random number in [0, 1) for a key (sha256-based)."""
    digest = hashlib.sha256(f"{salt}\x00{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(
    key: str, val_fraction: float, test_fraction: float, salt: str, *, folds: int = 0, fold: int = 0
) -> str:
    """Return "train", "val" or "test" for an article.

    With folds = K (cross-validation), the articles go into K folds. Fold `fold` is the test
    split, and test_fraction is not used. After K builds with fold = 0 ... K - 1, each article
    was in the test split one time. val_fraction of the other articles is the val split.
    """
    u = stable_fraction(key, salt)
    if folds:
        if int(u * folds) == fold:
            return "test"
        return "val" if stable_fraction(key, salt + "/val") < val_fraction else "train"
    if u < test_fraction:
        return "test"
    if u < test_fraction + val_fraction:
        return "val"
    return "train"


def _citation_key(records: Sequence[Record]) -> str:
    return " ".join(records[0]["references"].split()).casefold() if records else ""


def _citation_of(tsv: Path) -> str:
    """Return the citation key of a TSV file, or "" if the file cannot be read."""
    try:
        return _citation_key(read_ets_tsv(tsv).records)
    except (OSError, ValueError):
        return ""


def _sort_tsvs(paths: list[Path]) -> None:
    paths.sort(key=lambda p: (len(p.stem), p.stem))  # The order of linking.link: "X" before "X part 2"


def join_by_citation(pairs: Sequence[Pair], unmatched: Sequence[Path]) -> tuple[list[Path], list[str]]:
    """Add each unmatched TSV file to the article whose TSV files have the same citation.

    The client writes the same citation in all the TSV files of one article, for example
    "Abraham et al., 2022" and "Abraham et al., 2022 part 2". Returns the TSV files that stay
    unmatched, and a message for each file that joined an article or that has two candidates.
    """
    owners: dict[str, set[int]] = defaultdict(set)
    for i, pair in enumerate(pairs):
        for tsv in pair.tsv_paths:
            if citation := _citation_of(tsv):
                owners[citation].add(i)
    still: list[Path] = []
    notes: list[str] = []
    for tsv in unmatched:
        citation = _citation_of(tsv)
        found = owners.get(citation, set()) if citation else set()
        if len(found) == 1:
            pair = pairs[next(iter(found))]
            pair.tsv_paths.append(tsv)
            _sort_tsvs(pair.tsv_paths)
            notes.append(f"{tsv.name} -> {pair.markdown.path.name}")
            continue
        if found:
            names = sorted(pairs[i].markdown.path.name for i in found)
            notes.append(f"{tsv.name}: {names} have its citation. Add a row to the manifest.")
        still.append(tsv)
    return still, notes


def _dedupe(records: list[Record]) -> tuple[list[Record], int]:
    """Remove the rows that are equal to an earlier row in all the columns that the model writes."""
    seen: set[tuple[str, ...]] = set()
    unique: list[Record] = []
    for rec in records:
        key = tuple(rec.get(c, "") for c in MODEL_COLUMNS)
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique, len(records) - len(unique)


@dataclass
class ExampleInfo:
    id: str
    split: str | None
    source_markdown: str
    source_tsvs: list[str]
    n_rows: int
    n_input_chars: int
    value_coverage: float | None
    name_coverage: float | None
    missing_values_sample: list[str]
    excluded_reason: str | None
    warnings: list[str]
    duplicate_rows: int = 0  # Removed: equal to an earlier row
    unsupported_rows: int = 0  # Removed: no value of the row is in the parsed text
    partly_supported_rows: int = 0  # Kept: some values of the row are not in the parsed text


def build(
    tsv_dir: Path,
    markdown_dir: Path,
    out_dir: Path,
    *,
    manifest: Path | None = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    min_coverage: float = 0.5,
    min_name_coverage: float = 0.5,
    salt: str = "bioparser-split-v1",
    keep_unsupported_rows: bool = False,
    folds: int = 0,
    fold: int = 0,
) -> dict[str, Any]:
    if folds and not (folds >= 2 and 0 <= fold < folds):
        raise ValueError(f"fold must be 0 ... {folds - 1}, and folds must be 2 or more")
    if manifest is None:
        # The fetch command writes fine_tuning_data/manifest.tsv. Use it when it is there.
        default = tsv_dir.parent / "manifest.tsv"
        manifest = default if default.is_file() else None
    linked = link(tsv_dir, markdown_dir, manifest)
    unmatched, joined = join_by_citation(linked.pairs, linked.unmatched_tsv)
    used_markdown = {p.markdown.path.name for p in linked.pairs}
    # A suggestion for a parsed file that has its own TSV files is a different article: those files
    # have a different citation, else join_by_citation added the TSV file.
    suggestions = {
        tsv: md
        for tsv, md in linked.suggestions.items()
        if tsv in {p.name for p in unmatched} and md not in used_markdown
    }

    # Read the parsed files. Two files with the same text are one PDF under two names: one article.
    articles: list[tuple[Pair, str | None, str, str]] = []  # pair, raw text, clean text, error
    first_with_text: dict[str, Pair] = {}
    same_text: list[str] = []
    # The shortest name is the name of the article, as for the PDF files (fetch.citations.group_papers)
    for pair in sorted(linked.pairs, key=lambda p: (len(p.markdown.key), p.markdown.key)):
        try:
            raw = pair.markdown.read_text()
        except (OSError, ValueError) as exc:
            articles.append((pair, None, "", str(exc)))
            continue
        markdown = clean_markdown(raw)  # For the coverage check and the report
        first = first_with_text.setdefault(hashlib.sha256(markdown.encode()).hexdigest(), pair)
        if first is not pair:
            first.tsv_paths.extend(pair.tsv_paths)
            _sort_tsvs(first.tsv_paths)
            same_text.append(f"{pair.markdown.path.name} has the same text as {first.markdown.path.name}")
            continue
        articles.append((pair, raw, markdown, ""))

    infos: list[ExampleInfo] = []
    rows_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    citation_owner: dict[str, str] = {}
    cross_pair_citations: list[str] = []

    for pair, raw_markdown, markdown, error in articles:
        warnings: list[str] = []
        tables: list[tuple[Path, list[Record]]] = []
        for tsv in pair.tsv_paths:
            try:
                table = read_ets_tsv(tsv)
            except ValueError as exc:
                warnings.append(str(exc))
                continue
            tables.append((tsv, table.records))
            warnings += [f"{tsv.name}: {w}" for w in table.warnings]
        if raw_markdown is not None and len(tables) > 1:
            # In an article with more TSV files, each file has its own coverage check. For example,
            # one file has the values of the PDF and the other file has the values of a data set.
            kept_tables = []
            for tsv, recs in tables:
                c = label_coverage(markdown, recs)
                judged = c.value_ratio is not None and c.values_total >= MIN_VALUES_TO_JUDGE
                if judged and c.value_ratio is not None and c.value_ratio < min_coverage:
                    warnings.append(f"{tsv.name}: value coverage {c.value_ratio:.0%}: rows not used")
                    continue
                kept_tables.append((tsv, recs))
            tables = kept_tables or tables  # No file passes: the article check below excludes the article
        records = [rec for _, recs in tables for rec in recs]

        example_id = pair.markdown.key
        if raw_markdown is None:
            # One damaged or too large parsed file must not stop the build of all the other examples
            infos.append(
                ExampleInfo(
                    id=example_id,
                    split=None,
                    source_markdown=pair.markdown.path.name,
                    source_tsvs=[p.name for p in pair.tsv_paths],
                    n_rows=len(records),
                    n_input_chars=0,
                    value_coverage=None,
                    name_coverage=None,
                    missing_values_sample=[],
                    excluded_reason=f"cannot read the parsed file: {error}",
                    warnings=warnings,
                )
            )
            continue
        # One article has one citation. TSV files with different citations come together only
        # through the manifest or a parsed file with the same text. The citation of most rows is used.
        citations = Counter(r["references"] for r in records if r["references"])
        if len(citations) > 1:
            citation = citations.most_common(1)[0][0]
            warnings.append(f"{len(citations)} citations for one article: the citation of most rows is used")
            for rec in records:
                rec["references"] = citation
        records, duplicate_rows = _dedupe(records)

        cov = label_coverage(markdown, records)
        reason = None
        if not records:
            reason = "no label rows"
        elif cov.value_ratio is not None and cov.values_total >= MIN_VALUES_TO_JUDGE:
            if cov.value_ratio < min_coverage:
                reason = f"value coverage {cov.value_ratio:.0%} < {min_coverage:.0%}"
        # Names are a weaker signal (curators may update taxonomy), so they decide only
        # when there are too few numbers to judge.
        elif cov.name_ratio is not None and cov.name_ratio < min_name_coverage:
            reason = f"name coverage {cov.name_ratio:.0%} < {min_name_coverage:.0%}"

        unsupported = partly = 0
        if reason is None:
            text_numbers = numbers_in(markdown)
            support = [row_support(rec, text_numbers) for rec in records]
            partly = support.count("some")
            if not keep_unsupported_rows:
                kept = [rec for rec, s in zip(records, support, strict=True) if s != "none"]
                unsupported = len(records) - len(kept)
                records = kept
                if not records:
                    reason = "no label row has a value in the parsed text"

        citation = _citation_key(records)
        if citation:
            owner = citation_owner.setdefault(citation, example_id)
            if owner != example_id:
                cross_pair_citations.append(
                    f"{example_id!r} and {owner!r} have the same references value (check the labels)"
                )
        split = (
            None
            if reason
            else assign_split(
                citation or example_id, val_fraction, test_fraction, salt, folds=folds, fold=fold
            )
        )
        infos.append(
            ExampleInfo(
                id=example_id,
                split=split,
                source_markdown=pair.markdown.path.name,
                source_tsvs=[p.name for p in pair.tsv_paths],
                n_rows=len(records),
                n_input_chars=len(markdown),
                value_coverage=cov.value_ratio,
                name_coverage=cov.name_ratio,
                missing_values_sample=cov.missing_values,
                excluded_reason=reason,
                warnings=warnings,
                duplicate_rows=duplicate_rows,
                unsupported_rows=unsupported,
                partly_supported_rows=partly,
            )
        )
        if split is None:
            continue
        rows_by_split[split].append(
            {
                "id": example_id,
                "split": split,
                "prompt_version": PROMPT_VERSION,
                "source_markdown": pair.markdown.path.name,
                "source_tsvs": [p.name for p in pair.tsv_paths],
                "references": records[0]["references"],
                "n_rows": len(records),
                "value_coverage": cov.value_ratio,
                # build_messages cleans the text one time, as at inference. A second clean can give
                # a different text: for example, NFC runs before the code removes zero-width characters.
                "messages": build_messages(raw_markdown, to_model_tsv(records)),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for row in sorted(rows, key=lambda r: r["id"]):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    excluded = [i for i in infos if i.excluded_reason]
    used = [i for i in infos if i.split]
    report = {
        "prompt_version": PROMPT_VERSION,
        "manifest": str(manifest) if manifest else None,
        "folds": {"folds": folds, "fold": fold} if folds else None,
        "counts": {s: len(r) for s, r in rows_by_split.items()}
        | {
            "excluded": len(excluded),
            "unmatched_tsv": len(unmatched),
            "joined_by_citation": len(linked.unmatched_tsv) - len(unmatched),
            "same_text": len(same_text),
            "duplicate_rows": sum(i.duplicate_rows for i in used),
            "unsupported_rows": sum(i.unsupported_rows for i in used),
            "partly_supported_rows": sum(i.partly_supported_rows for i in used),
        },
        "excluded": {i.id: i.excluded_reason for i in excluded},
        "unmatched_tsv": [p.name for p in unmatched],
        "unmatched_markdown": [s.path.name for s in linked.unmatched_markdown],
        "joined_by_citation": joined,
        "same_text": same_text,
        "manifest_suggestions": suggestions,
        "warnings": linked.warnings + cross_pair_citations,
        "examples": [asdict(i) for i in infos],
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def _print_summary(report: dict[str, Any], out_dir: Path) -> None:
    c = report["counts"]
    if report.get("manifest"):
        print(f"manifest: {report['manifest']}")
    print(
        f"train={c['train']} val={c['val']} test={c['test']} excluded={c['excluded']} "
        f"unmatched_tsv={c['unmatched_tsv']}  ->  {out_dir}"
    )
    print(
        f"  TSVs joined by citation: {c['joined_by_citation']}; parsed files with the same text: "
        f"{c['same_text']}"
    )
    print(
        f"  rows removed: {c['duplicate_rows']} duplicates, {c['unsupported_rows']} without a value in the "
        f"parsed text; {c['partly_supported_rows']} rows have some values that are not in the text"
    )
    by_reason: dict[str, list[str]] = defaultdict(list)
    for ex_id, reason in report["excluded"].items():
        by_reason[" ".join(reason.split()[:2])].append(ex_id)  # "value coverage", "no label", ...
    for reason, ids in by_reason.items():
        print(f"  excluded ({reason}): {len(ids)}  e.g. {ids[:3]}")
    if report["unmatched_tsv"]:
        print(
            f"  {len(report['unmatched_tsv'])} TSVs without a parsed PDF, e.g. {report['unmatched_tsv'][:3]}"
        )
    if report["manifest_suggestions"]:
        print(f"  manifest suggestions (verify!): {list(report['manifest_suggestions'].items())[:3]}")
    for w in report["warnings"][:10]:
        print(f"  warning: {w}")
    print("  details: build_report.json")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv-dir", type=Path, default=Path("fine_tuning_data/ets_tsv"))
    ap.add_argument("--markdown-dir", type=Path, default=Path("fine_tuning_data/parsed"))
    ap.add_argument(
        "--manifest",
        type=Path,
        help="tab-separated file with the columns tsv and markdown, for files with different names. "
        "Default: manifest.tsv next to the --tsv-dir folder, if the file is there.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=0.5,
        help="min share of labelled numbers found in the parsed PDF (0 disables)",
    )
    ap.add_argument("--min-name-coverage", type=float, default=0.5)
    ap.add_argument(
        "--keep-unsupported-rows",
        action="store_true",
        help="keep the label rows that have no value in the parsed text (default: remove them)",
    )
    ap.add_argument("--salt", default="bioparser-split-v1", help="change only to draw a new split")
    ap.add_argument(
        "--folds",
        type=int,
        default=0,
        metavar="K",
        help="cross-validation: put the articles into K folds, and use fold --fold as the test split "
        "(--test-fraction is then not used). Build and train K times, with --fold 0 ... K-1.",
    )
    ap.add_argument("--fold", type=int, default=0, metavar="I", help="the test fold, 0 ... K-1 (default: 0)")
    args = ap.parse_args(argv)
    if args.folds and not (args.folds >= 2 and 0 <= args.fold < args.folds):
        ap.error("--folds must be 2 or more, and --fold must be 0 ... K-1")
    report = build(
        args.tsv_dir,
        args.markdown_dir,
        args.out_dir,
        manifest=args.manifest,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        min_coverage=args.min_coverage,
        min_name_coverage=args.min_name_coverage,
        salt=args.salt,
        keep_unsupported_rows=args.keep_unsupported_rows,
        folds=args.folds,
        fold=args.fold,
    )
    _print_summary(report, args.out_dir)


if __name__ == "__main__":
    main()
