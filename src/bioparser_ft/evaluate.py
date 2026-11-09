"""Field-level evaluation of generated TSV against the gold labels (stdlib only).

Training loss says little about extraction quality; these metrics do. All are
computed on normalised values (case, whitespace, "5.20" == "5.2"):

* row   - exact match of whole records (all generated columns except references)
* cell  - (taxon, trait, column, value) facts for non-empty cells: robust to row
          order and to one wrong cell spoiling a whole row
* value - cell metric restricted to the numeric columns (the core of the task)
* references_exact - citation identical
Counts are summed over documents before computing P/R/F1 (micro average).

CLI:  bioparser-evaluate --dataset data/processed/test.jsonl --predictions preds.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .coverage import VALUE_COLUMNS
from .schema import DOCUMENT_COLUMNS, MODEL_COLUMNS
from .tsv import parse_model_tsv

SCORED_COLUMNS = tuple(c for c in MODEL_COLUMNS if c not in DOCUMENT_COLUMNS)
KEY_COLUMNS = ("verbatimScientificName", "verbatimTraitName")
METRICS = ("row", "cell", "value")


def _norm(value: str) -> str:
    v = " ".join(value.split()).casefold()
    try:
        d = Decimal(v)
        # Model output can contain "1e1000000". Then normalize() raises Overflow, which is not an
        # InvalidOperation, and "f" can write one million digits. Compare these values as text.
        if not d.is_finite() or abs(d.adjusted()) > 100:
            return v
        return format(d.normalize(), "f")
    except ArithmeticError:  # The base class of all decimal errors
        return v


def _prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    precision = tp / n_pred if n_pred else float(n_gold == 0)
    recall = tp / n_gold if n_gold else float(n_pred == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _facts(records: Iterable[Mapping[str, str]], columns: Sequence[str]) -> Counter[tuple[str, ...]]:
    facts: Counter[tuple[str, ...]] = Counter()
    for r in records:
        key = tuple(_norm(r.get(c, "")) for c in KEY_COLUMNS)
        for col in columns:
            if col not in KEY_COLUMNS and (v := _norm(r.get(col, ""))):
                facts[(*key, col, v)] += 1
    return facts


def score_document(pred: Sequence[Mapping[str, str]], gold: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Counts (tp, n_pred, n_gold) per metric for one article."""
    rows_p = Counter(tuple(_norm(r.get(c, "")) for c in SCORED_COLUMNS) for r in pred)
    rows_g = Counter(tuple(_norm(r.get(c, "")) for c in SCORED_COLUMNS) for r in gold)
    out: dict[str, Any] = {}
    for name, p, g in (
        ("row", rows_p, rows_g),
        ("cell", _facts(pred, SCORED_COLUMNS), _facts(gold, SCORED_COLUMNS)),
        ("value", _facts(pred, VALUE_COLUMNS), _facts(gold, VALUE_COLUMNS)),
    ):
        out[name] = (sum((p & g).values()), sum(p.values()), sum(g.values()))
    ref_p = _norm(pred[0].get("references", "")) if pred else ""
    ref_g = _norm(gold[0].get("references", "")) if gold else ""
    out["references_exact"] = bool(ref_g) and ref_p == ref_g
    out["rows_pred"], out["rows_gold"] = len(pred), len(gold)
    return out


def aggregate(doc_scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"documents": len(doc_scores)}
    for name in METRICS:
        tp = sum(d[name][0] for d in doc_scores)
        n_pred = sum(d[name][1] for d in doc_scores)
        n_gold = sum(d[name][2] for d in doc_scores)
        result[name] = _prf(tp, n_pred, n_gold)
    if doc_scores:
        result["references_exact"] = sum(d["references_exact"] for d in doc_scores) / len(doc_scores)
        result["rows_pred"] = sum(d["rows_pred"] for d in doc_scores)
        result["rows_gold"] = sum(d["rows_gold"] for d in doc_scores)
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def score_output(output: str, gold_tsv: str, *, complete: bool = True) -> dict[str, Any]:
    """Counts for one generated TSV table and the gold TSV table of the same article."""
    parsed = parse_model_tsv(output, complete=complete)
    doc = score_document(parsed.records, parse_model_tsv(gold_tsv).records)
    doc["complete"] = complete
    doc["parse_warnings"] = len(parsed.warnings)
    return doc


def evaluate_files(dataset: Path, predictions: Path) -> dict[str, Any]:
    """Score the predictions of a dataset split.

    An article without a prediction counts as an empty table. If the code ignored it, a run that
    stopped early would get a better score. Raises ValueError if no prediction id is in the dataset.
    """
    gold = {ex["id"]: ex for ex in _read_jsonl(dataset)}
    preds: dict[str, dict[str, Any]] = {}
    unknown: set[str] = set()
    duplicates: set[str] = set()
    for pred in _read_jsonl(predictions):
        if pred["id"] not in gold:
            unknown.add(pred["id"])
        elif pred["id"] in preds:
            duplicates.add(pred["id"])
        preds[pred["id"]] = pred  # The last line wins, for example after a second run appended to the file
    if not gold.keys() & preds.keys():
        # Without this check, aggregate() gives the score 1.0 for 0 documents
        raise ValueError(f"no prediction id in {predictions} is in {dataset}: examine the two files")
    per_doc = {}
    for ex_id, ex in gold.items():
        pred = preds.get(ex_id, {"output": "", "complete": True})
        per_doc[ex_id] = score_output(
            pred["output"], ex["messages"][-1]["content"], complete=pred.get("complete", True)
        )
    summary = aggregate(list(per_doc.values()))
    summary["missing_predictions"] = sorted(gold.keys() - preds.keys())
    summary["unknown_predictions"] = sorted(unknown)
    summary["duplicate_predictions"] = sorted(duplicates)
    summary["incomplete_generations"] = sum(not d["complete"] for d in per_doc.values())
    return {"summary": summary, "documents": per_doc}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="split JSONL from bioparser-build-dataset")
    ap.add_argument("--predictions", type=Path, required=True, help="JSONL from bioparser-predict")
    ap.add_argument("--out", type=Path, help="write the full report as JSON")
    args = ap.parse_args(argv)
    try:
        report = evaluate_files(args.dataset, args.predictions)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report["summary"], indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
