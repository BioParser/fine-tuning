from __future__ import annotations

import json
import zipfile
from pathlib import Path

from conftest import (
    ets_row,
    paper_a_markdown,
    paper_a_mineru_markdown,
    paper_a_rows,
    paper_b_markdown,
    paper_b_rows,
    write_ets,
)

from bioparser_ft.build_dataset import assign_split, build
from bioparser_ft.coverage import label_coverage, numbers_in, row_support
from bioparser_ft.linking import MarkdownSource, discover_markdown, link, paper_key
from bioparser_ft.markdown import clean_markdown
from bioparser_ft.schema import ETS_COLUMNS, MODEL_HEADER, SYSTEM_PROMPT
from bioparser_ft.tsv import normalize_record, parse_model_tsv


def test_paper_key_normalisation() -> None:
    assert paper_key("ETS_López‐Aguirre et al., 2023") == paper_key("López-Aguirre et al.,  2023")
    assert paper_key("ETS_Abi-Said, M.R. and El Haddad, E., 2023") == "abi-said, m.r. and el haddad, e., 2023"


def test_discover_markdown_layouts(parsed_dir: Path) -> None:
    extracted = parsed_dir / "Extracted, 2024"
    extracted.mkdir()
    (extracted / "markdown.md").write_text("x", encoding="utf-8")  # unzipped MinerU folder
    sources, warnings = discover_markdown(parsed_dir)
    assert {"doe and roe, 2020", "smith, 2021", "mislinked, 2022", "extracted, 2024"} <= set(sources)
    assert sources["doe and roe, 2020"].member == "markdown.md"
    assert "Cave X" in sources["doe and roe, 2020"].read_text()
    assert not warnings


def test_zip_member_size_cap(tmp_path: Path) -> None:
    z = tmp_path / "big.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("markdown.md", "0" * 10_000)
    try:
        MarkdownSource(z, "markdown.md").read_text(max_bytes=1000)
    except ValueError:
        pass
    else:
        raise AssertionError("size cap not enforced")


def test_link_with_manifest(ets_dir: Path, parsed_dir: Path, manifest: Path) -> None:
    res = link(ets_dir, parsed_dir, manifest)
    pairs = {p.markdown.key: [t.name for t in p.tsv_paths] for p in res.pairs}
    assert pairs["doe and roe, 2020"] == ["ETS_Doe and Roe, 2020.tsv", "ETS_Doe and Roe, 2020 part 2.tsv"]
    assert [p.name for p in res.unmatched_tsv] == ["ETS_Nobody, 1999.tsv"]
    no_manifest = link(ets_dir, parsed_dir)
    assert "ETS_Doe and Roe, 2020 part 2.tsv" in {p.name for p in no_manifest.unmatched_tsv}
    assert no_manifest.suggestions["ETS_Doe and Roe, 2020 part 2.tsv"] == "Doe and Roe, 2020.zip"


def test_coverage_detects_wrong_pdf() -> None:
    records = [normalize_record(dict(zip(ETS_COLUMNS, r, strict=True))) for r in paper_a_rows()]
    good = label_coverage(paper_a_markdown(), records)
    assert good.value_ratio == 1.0 and good.name_ratio == 1.0
    bad = label_coverage("A paper about frogs: 12.7 and 3.3", records)
    assert bad.value_ratio == 0.0 and bad.missing_values


def test_mineru_4_2_zip(tmp_path: Path) -> None:
    """The members and the Markdown of a real MinerU 4.2.3 --format zip file give full coverage."""
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    with zipfile.ZipFile(parsed / "Doe and Roe, 2020.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("images/page_0_chart_8.jpg", b"\xff\xd8")
        zf.writestr("images/page_0_table_7.jpg", b"\xff\xd8")
        zf.writestr("markdown.md", paper_a_mineru_markdown())
        zf.writestr("middle_json.json", '{"metadata": {"producer": {"name": "mineru", "version": "4.2.3"}}}')
        zf.writestr("structured_content.json", "{}")
    sources, warnings = discover_markdown(parsed)
    assert not warnings
    assert sources["doe and roe, 2020"].member == "markdown.md"
    text = clean_markdown(sources["doe and roe, 2020"].read_text())
    assert "![" not in text and "$\\mathsf{CV}" in text  # the image link goes, the LaTeX stays
    records = [normalize_record(dict(zip(ETS_COLUMNS, r, strict=True))) for r in paper_a_rows()]
    cov = label_coverage(text, records)
    assert (cov.value_ratio, cov.name_ratio) == (1.0, 1.0)


def test_numbers_in() -> None:
    assert numbers_in("n = 3, FA 37.40 mm, 1,5 g, 1,234 kg") >= {"37.4", "1.5", "1234"}
    assert "3" not in numbers_in("n = 3")  # small integers carry no signal


def test_numbers_in_ignores_long_digit_runs() -> None:
    assert numbers_in("ID " + "1" * 5000 + ", FA 37.40 mm") == {"37.4"}


def test_build_end_to_end(ets_dir: Path, parsed_dir: Path, manifest: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    report = build(ets_dir, parsed_dir, out, manifest=manifest, val_fraction=0.0, test_fraction=0.0)
    assert report["counts"]["train"] == 2  # Doe (2 TSVs merged) + Smith
    assert set(report["excluded"]) == {"mislinked, 2022"}  # coverage check caught the wrong PDF
    assert report["unmatched_tsv"] == ["ETS_Nobody, 1999.tsv"]
    assert any("same references value" in w for w in report["warnings"])  # Smith vs Mislinked labels

    rows = [json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    doe = next(r for r in rows if r["id"] == "doe and roe, 2020")
    system, user, assistant = doe["messages"]
    assert system["content"] == SYSTEM_PROMPT
    assert user["content"].startswith("<article>") and "![](" not in user["content"]  # images dropped
    assert assistant["content"].split("\n")[0] == MODEL_HEADER
    assert len(parse_model_tsv(assistant["content"]).records) == 3 == doe["n_rows"]


def test_split_is_stable_and_grouped() -> None:
    a = assign_split("doe, j., 2020. title.", 0.2, 0.2, "salt")
    assert all(assign_split("doe, j., 2020. title.", 0.2, 0.2, "salt") == a for _ in range(5))
    splits = [assign_split(f"paper {i}", 0.1, 0.1, "salt") for i in range(2000)]
    assert 0.07 < splits.count("test") / 2000 < 0.13


def test_cross_validation_folds() -> None:
    """Each article is in the test split of exactly one fold."""
    keys = [f"paper {i}" for i in range(500)]
    splits = [[assign_split(k, 0.1, 0.0, "salt", folds=5, fold=f) for f in range(5)] for k in keys]
    assert all(s.count("test") == 1 for s in splits)
    in_fold_0 = [s[0] for s in splits]
    assert 0.15 < in_fold_0.count("test") / 500 < 0.25 and 0.05 < in_fold_0.count("val") / 500 < 0.12


def test_build_uses_default_manifest(ets_dir: Path, parsed_dir: Path, tmp_path: Path) -> None:
    """The fetch command writes manifest.tsv next to ets_tsv/. The builder uses it without --manifest."""
    (ets_dir.parent / "manifest.tsv").write_text(
        "tsv\tmarkdown\nETS_Doe and Roe, 2020 part 2.tsv\tDoe and Roe, 2020.zip\n", encoding="utf-8"
    )
    report = build(ets_dir, parsed_dir, tmp_path / "processed", val_fraction=0.0, test_fraction=0.0)
    assert report["manifest"].endswith("manifest.tsv")
    assert report["unmatched_tsv"] == ["ETS_Nobody, 1999.tsv"]  # the "part 2" file has its pair


def test_build_excludes_a_damaged_zip_and_continues(ets_dir: Path, parsed_dir: Path, tmp_path: Path) -> None:
    broken = parsed_dir / "Nobody, 1999.zip"  # Pairs with ETS_Nobody, 1999.tsv by name
    with zipfile.ZipFile(broken, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("markdown.md", "Mus spretus T 66.2 mm. " * 20)
    data = bytearray(broken.read_bytes())
    data[data.index(b"Mus spretus")] ^= 0xFF  # The CRC-32 of the member is now wrong
    broken.write_bytes(bytes(data))

    report = build(ets_dir, parsed_dir, tmp_path / "processed", val_fraction=0.0, test_fraction=0.0)
    assert report["excluded"]["nobody, 1999"].startswith("cannot read the parsed file")
    assert report["counts"]["train"] >= 1  # The other examples are still in the dataset


def test_build_without_manifest_file(ets_dir: Path, parsed_dir: Path, tmp_path: Path) -> None:
    """The "part 2" file has the citation of "Doe and Roe, 2020": it joins that article without a manifest."""
    report = build(ets_dir, parsed_dir, tmp_path / "processed", val_fraction=0.0, test_fraction=0.0)
    assert report["manifest"] is None
    assert report["unmatched_tsv"] == ["ETS_Nobody, 1999.tsv"]  # Smith and Mislinked have its citation
    assert report["counts"]["joined_by_citation"] == 1
    assert report["joined_by_citation"][0] == "ETS_Doe and Roe, 2020 part 2.tsv -> Doe and Roe, 2020.zip"
    assert "Add a row to the manifest" in report["joined_by_citation"][1]
    doe = next(e for e in report["examples"] if e["id"] == "doe and roe, 2020")
    assert doe["n_rows"] == 3 and len(doe["source_tsvs"]) == 2


def test_suggestions_need_the_same_author_and_year(tmp_path: Path) -> None:
    tsv, parsed = tmp_path / "ets_tsv", tmp_path / "parsed"
    tsv.mkdir()
    parsed.mkdir()
    for name in ("Chen et al., 2023", "Doe et al., 2021"):
        write_ets(tsv / f"ETS_{name}.tsv", paper_b_rows())
    for name in ("Benda et al., 2023", "Doe et al., 2022"):
        (parsed / f"{name}.md").write_text("x", encoding="utf-8")
    assert link(tsv, parsed).suggestions == {}  # Not "Chen" -> "Benda", not 2021 -> 2022


def test_row_support() -> None:
    text = numbers_in("FA 37.42 ± 0.83 mm (36.5–39.1), n = 12")
    assert row_support({"verbatimTraitValue": "37.42", "dispersion": "0.83"}, text) == "all"
    assert row_support({"verbatimTraitValue": "37.42", "dispersion": "0.91"}, text) == "some"
    assert row_support({"measurementValue_min": "12.5", "measurementValue_max": "14.1"}, text) == "none"
    assert row_support({"verbatimTraitValue": "brown", "individualCount": "12"}, text) == "unchecked"


def test_build_aligns_the_labels_with_the_text(ets_dir: Path, parsed_dir: Path, tmp_path: Path) -> None:
    """Duplicate rows and rows without a value in the text go. A parsed file with the same text is one article."""
    rows = paper_b_rows()
    absent = ets_row(**{**dict(zip(ETS_COLUMNS, rows[1], strict=True)), "verbatimTraitValue": "71.9"})
    # A duplicate row and a row with a value that is not in the text
    write_ets(ets_dir / "ETS_Smith, 2021.tsv", [*rows, rows[0], absent])
    # The same PDF under another name, with another citation
    (parsed_dir / "Smith copy, 2021.md").write_text(paper_b_markdown(), encoding="utf-8")
    cells = [dict(zip(ETS_COLUMNS, r, strict=True)) for r in rows]
    write_ets(
        ets_dir / "ETS_Smith copy, 2021.tsv",
        [ets_row(**{**c, "references": "Smith, A. Mice."}) for c in cells],
    )

    out = tmp_path / "processed"
    report = build(ets_dir, parsed_dir, out, val_fraction=0.0, test_fraction=0.0)
    smith = next(e for e in report["examples"] if e["id"] == "smith, 2021")
    assert report["same_text"] == ["Smith copy, 2021.md has the same text as Smith, 2021.md"]
    assert smith["source_tsvs"] == ["ETS_Smith, 2021.tsv", "ETS_Smith copy, 2021.tsv"]
    assert any("2 citations for one article" in w for w in smith["warnings"])
    assert (smith["duplicate_rows"], smith["unsupported_rows"], smith["n_rows"]) == (3, 1, 2)
    train = [json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    target = parse_model_tsv(next(r for r in train if r["id"] == "smith, 2021")["messages"][-1]["content"])
    assert {r["references"] for r in target.records} == {paper_b_rows()[0][0]}
    assert "71.9" not in {r["verbatimTraitValue"] for r in target.records}

    kept = build(
        ets_dir,
        parsed_dir,
        tmp_path / "kept",
        val_fraction=0.0,
        test_fraction=0.0,
        keep_unsupported_rows=True,
    )
    assert next(e for e in kept["examples"] if e["id"] == "smith, 2021")["n_rows"] == 3
