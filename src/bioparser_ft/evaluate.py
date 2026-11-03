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
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .coverage import VALUE_COLUMNS
from .schema import DOCUMENT_COLUMNS, MODEL_COLUMNS
from .tsv import parse_model_tsv

SCORED_COLUMNS = tuple(c for c in MODEL_COLUMNS if c not in DOCUMENT_COLUMNS)
KEY_COLUMNS = ("verbatimScientificName", "verbatimTraitName")
METRICS = ("row", "cell", "value")


def _norm(value: str) -> str:
    v = " ".join(value.split()).casefold()
    try:
        return format(Decimal(v).normalize(), "f")
    except InvalidOperation:
        return v


def _prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    precision = tp / n_pred if n_pred else float(n_gold == 0)
    recall = tp / n_gold if n_gold else float(n_pred == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _facts(records: Iterable[Mapping[str, str]], columns: Sequence[str]) -> Counter:
    facts: Counter = Counter()
    for r in records:
        key = tuple(_norm(r.get(c, "")) for c in KEY_COLUMNS)
        for col in columns:
            if col not in KEY_COLUMNS and (v := _norm(r.get(col, ""))):
                facts[(*key, col, v)] += 1
    return facts


def score_document(pred: Sequence[Mapping[str, str]], gold: Sequence[Mapping[str, str]]) -> dict:
    """Counts (tp, n_pred, n_gold) per metric for one article."""
    rows_p = Counter(tuple(_norm(r.get(c, "")) for c in SCORED_COLUMNS) for r in pred)
    rows_g = Counter(tuple(_norm(r.get(c, "")) for c in SCORED_COLUMNS) for r in gold)
    out: dict = {}
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


def aggregate(doc_scores: Sequence[dict]) -> dict:
    result: dict = {"documents": len(doc_scores)}
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


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate_files(dataset: Path, predictions: Path) -> dict:
    gold = {ex["id"]: ex for ex in _read_jsonl(dataset)}
    docs, per_doc = [], {}
    for pred in _read_jsonl(predictions):
        ex = gold.get(pred["id"])
        if ex is None:
            continue
        gold_records = parse_model_tsv(ex["messages"][-1]["content"]).records
        parsed = parse_model_tsv(pred["output"], complete=pred.get("complete", True))
        doc = score_document(parsed.records, gold_records)
        doc["complete"] = pred.get("complete", True)
        doc["parse_warnings"] = len(parsed.warnings)
        docs.append(doc)
        per_doc[pred["id"]] = doc
    summary = aggregate(docs)
    summary["missing_predictions"] = sorted(set(gold) - set(per_doc))
    summary["incomplete_generations"] = sum(not d["complete"] for d in docs)
    return {"summary": summary, "documents": per_doc}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="split JSONL from bioparser-build-dataset")
    ap.add_argument("--predictions", type=Path, required=True, help="JSONL from bioparser-predict")
    ap.add_argument("--out", type=Path, help="write the full report as JSON")
    args = ap.parse_args(argv)
    report = evaluate_files(args.dataset, args.predictions)
    print(json.dumps(report["summary"], indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
