"""CLI: build train/val/test JSONL from client ETS TSVs + parsed PDFs (stdlib only).

    uv run bioparser-build-dataset \\
        --tsv-dir fine_tuning_data/ets_tsv --markdown-dir fine_tuning_data/parsed

Per article (one parsed PDF + all TSVs linked to it) this

1. reads the TSVs quote-aware and normalises cells/vocabularies (tsv.read_ets_tsv),
2. checks that the parsed PDF contains the labelled values (coverage.label_coverage)
   and excludes pairs below --min-coverage (wrong PDF, lost tables, supplement data),
3. serialises the target in the model TSV dialect (tsv.to_model_tsv),
4. assigns train/val/test by a hash of the citation, so multi-file papers
   (e.g. "Abraham et al., 2022" + "... part 2") never straddle splits and adding
   new papers later does not reshuffle existing ones,
5. writes chat-format JSONL plus build_report.json (exclusions, unmatched files).

Token-length filtering happens at training time with the real tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .coverage import label_coverage
from .linking import link
from .markdown import clean_markdown
from .schema import PROMPT_VERSION, build_messages
from .tsv import Record, read_ets_tsv, to_model_tsv

SPLITS = ("train", "val", "test")
MIN_VALUES_TO_JUDGE = 5  # distinct labelled numbers needed before value coverage decides


def stable_fraction(key: str, salt: str) -> float:
    """Deterministic pseudo-random number in [0, 1) for a key (sha256-based)."""
    digest = hashlib.sha256(f"{salt}\x00{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(key: str, val_fraction: float, test_fraction: float, salt: str) -> str:
    u = stable_fraction(key, salt)
    if u < test_fraction:
        return "test"
    if u < test_fraction + val_fraction:
        return "val"
    return "train"


def _citation_key(records: Sequence[Record]) -> str:
    return " ".join(records[0]["references"].split()).casefold() if records else ""


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
) -> dict[str, Any]:
    if manifest is None:
        # The fetch command writes fine_tuning_data/manifest.tsv. Use it when it is there.
        default = tsv_dir.parent / "manifest.tsv"
        manifest = default if default.is_file() else None
    linked = link(tsv_dir, markdown_dir, manifest)
    infos: list[ExampleInfo] = []
    rows_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    citation_owner: dict[str, str] = {}
    cross_pair_citations: list[str] = []

    for pair in linked.pairs:
        warnings: list[str] = []
        records: list[Record] = []
        for tsv in pair.tsv_paths:
            try:
                table = read_ets_tsv(tsv)
            except ValueError as exc:
                warnings.append(str(exc))
                continue
            records += table.records
            warnings += [f"{tsv.name}: {w}" for w in table.warnings]

        example_id = pair.markdown.key
        raw_markdown = pair.markdown.read_text()
        markdown = clean_markdown(raw_markdown)  # For the coverage check and the report
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

        citation = _citation_key(records)
        if citation:
            owner = citation_owner.setdefault(citation, example_id)
            if owner != example_id:
                cross_pair_citations.append(
                    f"{example_id!r} and {owner!r} have the same references value (check the labels)"
                )
        split = None if reason else assign_split(citation or example_id, val_fraction, test_fraction, salt)
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
    report = {
        "prompt_version": PROMPT_VERSION,
        "manifest": str(manifest) if manifest else None,
        "counts": {s: len(r) for s, r in rows_by_split.items()}
        | {"excluded": len(excluded), "unmatched_tsv": len(linked.unmatched_tsv)},
        "excluded": {i.id: i.excluded_reason for i in excluded},
        "unmatched_tsv": [p.name for p in linked.unmatched_tsv],
        "unmatched_markdown": [s.path.name for s in linked.unmatched_markdown],
        "manifest_suggestions": linked.suggestions,
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
    ap.add_argument("--salt", default="bioparser-split-v1", help="change only to draw a new split")
    args = ap.parse_args(argv)
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
    )
    _print_summary(report, args.out_dir)


if __name__ == "__main__":
    main()
