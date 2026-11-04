from __future__ import annotations

import json

import pytest

from bioparser_ft.postprocess import (
    expand_abbreviated_genera,
    infer_taxon_rank,
    records_from_json,
    to_ets_records,
)
from bioparser_ft.schema import ETS_COLUMNS, MODEL_COLUMNS


@pytest.mark.parametrize(
    ("name", "rank"),
    [
        ("Canis", "Genus"),
        ("Canis lupus", "Species"),
        ("Canis lupus familiaris", "Subspecies"),
        ("Myotis cf. ater", "Species"),
        ("Rhinolophus sp. nov.", "Species"),
        ("Calomyscus darvishi sp. nov.", "Species"),
        ("Pudella carlae n. sp.", "Species"),
        ("Soriculus sp. 3", "Species"),
        ("Oligoryzomys sp1", "Species"),
        ("Berylmys sp.", "Genus"),
        ("Genetta sp", "Genus"),
        ("Pipistrellus kuhlii Cretzschmar, 1830", "Species"),
        ("Panthera leo (Linnaeus, 1758)", "Species"),
        ("Mus (Coelomys) mayori", "Species"),
        ("Vespertilionidae", "Family"),
        ("Murinae", "Subfamily"),
        ("Muroidea", "Superfamily"),
        ("Blarina", "Genus"),  # -ina genus, not a subtribe
        ("", ""),
    ],
)
def test_infer_taxon_rank(name: str, rank: str) -> None:
    assert infer_taxon_rank(name) == rank


def test_expand_abbreviated_genera() -> None:
    recs = [{"verbatimScientificName": n} for n in ("Myotis ater", "M. ater", "Murina sp.", "R. ferrum")]
    # "M." is ambiguous (Myotis, Murina) -> untouched; "R." has no spelled-out genus
    assert expand_abbreviated_genera(recs) == 0
    recs = [{"verbatimScientificName": n} for n in ("Homo sapiens", "H. neanderthalensis")]
    assert expand_abbreviated_genera(recs) == 1
    assert recs[1]["verbatimScientificName"] == "Homo neanderthalensis"


def _model_output(*rows: dict[str, str]) -> str:
    lines = ["\t".join(MODEL_COLUMNS)]
    lines += ["\t".join(r.get(c, "") for c in MODEL_COLUMNS) for r in rows]
    return "\n".join(lines)


def test_to_ets_records_fills_deterministic_columns() -> None:
    text = _model_output(
        {
            "references": "Cit.",
            "verbatimScientificName": "Homo sapiens",
            "taxonRank": "species",
            "sex": "males",
        },
        {"verbatimScientificName": "H. erectus", "verbatimTraitValue": "1.2"},
    )
    res = to_ets_records(text)
    assert [list(r) for r in res.records] == [list(ETS_COLUMNS)] * 2  # full layout, ordered
    a, b = res.records
    assert a["author"] == b["author"] == "BioParser"
    assert b["references"] == "Cit."  # forward-filled
    assert a["taxonRank"] == "Species" and a["sex"] == "Male"  # canonicalised
    assert (
        b["verbatimScientificName"] == "Homo erectus" and b["taxonRank"] == "Species"
    )  # expanded + inferred
    assert res.complete

    override = to_ets_records(text, references="Verified citation.")
    assert {r["references"] for r in override.records} == {"Verified citation."}


def test_to_ets_records_bad_output_never_raises() -> None:
    res = to_ets_records("I could not find any traits.", complete=False)
    assert isinstance(res.records, list)
    assert res.warnings and not res.complete


def test_json_round_trip() -> None:
    res = to_ets_records(_model_output({"references": "C.", "verbatimScientificName": "Mus spretus"}))
    payload = json.loads(res.to_json())
    assert payload["complete"] is True and payload["records"][0]["author"] == "BioParser"
    assert records_from_json(res.to_json()) == res.records
