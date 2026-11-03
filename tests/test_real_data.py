"""Checks on the real client export; skipped unless BIOPARSER_ETS_DIR is set.

BIOPARSER_ETS_DIR=fine_tuning_data/ets_tsv uv run pytest tests/test_real_data.py -s
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from bioparser_ft.postprocess import infer_taxon_rank
from bioparser_ft.schema import MODEL_COLUMNS
from bioparser_ft.tsv import parse_model_tsv, read_ets_tsv, to_model_tsv

ETS_DIR = os.environ.get("BIOPARSER_ETS_DIR")
pytestmark = pytest.mark.skipif(not ETS_DIR, reason="set BIOPARSER_ETS_DIR to the client TSV folder")


@pytest.fixture(scope="module")
def tables():
    paths = sorted(Path(ETS_DIR).glob("*.tsv"))
    assert paths, f"no .tsv files in {ETS_DIR}"
    return [read_ets_tsv(p) for p in paths]


def test_every_file_reads_and_round_trips(tables):
    total = 0
    for t in tables:
        parsed = parse_model_tsv(to_model_tsv(t.records))
        assert parsed.header_ok and parsed.malformed_rows == 0, t.path.name
        assert parsed.records == [{c: r[c] for c in MODEL_COLUMNS} for r in t.records], t.path.name
        total += len(t.records)
    print(f"\n{len(tables)} files, {total} records round-trip losslessly")


def test_no_cell_breaks_the_model_dialect(tables):
    for t in tables:
        for r in t.records:
            assert not any("\t" in v or "\n" in v or "\r" in v for v in r.values()), t.path.name


def test_taxon_rank_rule_agreement(tables):
    agree = total = 0
    misses: Counter = Counter()
    for t in tables:
        for r in t.records:
            if not r["taxonRank"]:
                continue
            total += 1
            inferred = infer_taxon_rank(r["verbatimScientificName"])
            agree += inferred == r["taxonRank"]
            if inferred != r["taxonRank"]:
                misses[(r["verbatimScientificName"], r["taxonRank"], inferred)] += 1
    print(
        f"\ntaxonRank rule agreement {agree}/{total} = {agree / total:.3%}; top misses {misses.most_common(5)}"
    )
    assert agree / total > 0.995


def test_vocabularies_are_canonical(tables):
    seen = {c: Counter() for c in ("taxonRank", "sex", "lifeStage", "statisticalMethod")}
    for t in tables:
        for r in t.records:
            for c in seen:
                seen[c][r[c]] += 1
    print("\n" + "\n".join(f"{c}: {dict(v)}" for c, v in seen.items()))
    assert set(seen["taxonRank"]) <= {"", "Genus", "Species", "Subspecies"}
    assert set(seen["sex"]) <= {"", "Male", "Female", "Mixed", "Unknown"}
