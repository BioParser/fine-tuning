from __future__ import annotations

import json
import zipfile
from pathlib import Path

from conftest import paper_a_markdown, paper_a_rows

from bioparser_ft.build_dataset import assign_split, build
from bioparser_ft.coverage import label_coverage, numbers_in
from bioparser_ft.linking import MarkdownSource, discover_markdown, link, paper_key
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


def test_numbers_in() -> None:
    assert numbers_in("n = 3, FA 37.40 mm, 1,5 g, 1,234 kg") >= {"37.4", "1.5", "1234"}
    assert "3" not in numbers_in("n = 3")  # small integers carry no signal


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


def test_build_uses_default_manifest(ets_dir: Path, parsed_dir: Path, tmp_path: Path) -> None:
    """The fetch command writes manifest.tsv next to ets_tsv/. The builder uses it without --manifest."""
    (ets_dir.parent / "manifest.tsv").write_text(
        "tsv\tmarkdown\nETS_Doe and Roe, 2020 part 2.tsv\tDoe and Roe, 2020.zip\n", encoding="utf-8"
    )
    report = build(ets_dir, parsed_dir, tmp_path / "processed", val_fraction=0.0, test_fraction=0.0)
    assert report["manifest"].endswith("manifest.tsv")
    assert report["unmatched_tsv"] == ["ETS_Nobody, 1999.tsv"]  # the "part 2" file has its pair


def test_build_without_manifest_file(ets_dir: Path, parsed_dir: Path, tmp_path: Path) -> None:
    report = build(ets_dir, parsed_dir, tmp_path / "processed", val_fraction=0.0, test_fraction=0.0)
    assert report["manifest"] is None
    assert "ETS_Doe and Roe, 2020 part 2.tsv" in report["unmatched_tsv"]
