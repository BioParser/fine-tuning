"""check_prompt_version and check_base_model need no GPU stack, so CI runs these tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bioparser_ft.infer import check_base_model, check_prompt_version
from bioparser_ft.schema import MODEL_COLUMNS, PROMPT_VERSION, build_messages


def test_check_prompt_version(tmp_path: Path) -> None:
    meta = tmp_path / "bioparser_ft.json"
    meta.write_text(json.dumps({"prompt_version": PROMPT_VERSION}), encoding="utf-8")
    check_prompt_version(tmp_path)  # The same version: no error
    meta.write_text(json.dumps({"prompt_version": "ets-tsv-v0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt_version"):
        check_prompt_version(tmp_path)


def test_check_prompt_version_without_metadata(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    check_prompt_version(tmp_path)  # No file, for example a Hub id: a warning, not an error
    assert "Cannot examine the prompt version" in caplog.text


def _model_with_commit(commit: str | None) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(_commit_hash=commit))


def test_check_base_model(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "bioparser_ft.json").write_text(json.dumps({"base_model_commit": "aaa"}), encoding="utf-8")
    assert check_base_model(_model_with_commit("aaa"), tmp_path)
    assert check_base_model(_model_with_commit(None), tmp_path)  # A local base model has no commit
    assert check_base_model(_model_with_commit("bbb"), tmp_path / "no-metadata")
    assert not caplog.text
    assert not check_base_model(_model_with_commit("bbb"), tmp_path)  # The Hub gave a newer commit
    assert "is not the commit of the training" in caplog.text


def test_predict_dataset_shows_running_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The GPU parts are stubs: the test examines the loop, the output file and the metrics."""
    from bioparser_ft import infer

    gold = (
        "\t".join(MODEL_COLUMNS)
        + "\n"
        + "\t".join("1.5" if c == "verbatimTraitValue" else "" for c in MODEL_COLUMNS)
    )
    ds = tmp_path / "test.jsonl"
    rows = [
        {"id": "right", "messages": build_messages("x", gold)},
        {"id": "empty", "messages": build_messages("y", gold)},
        {"id": "no-gold", "messages": build_messages("z")},  # Not scored: no assistant turn
    ]
    ds.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    outputs = iter([gold, "", gold])

    def fake_generate(model: object, processor: object, messages: object, **kw: object) -> infer.Generation:
        return infer.Generation(next(outputs), True, prompt_tokens=10, new_tokens=5, seconds=0.1)

    monkeypatch.setattr(infer, "load_for_inference", lambda *a, **kw: (None, None))
    monkeypatch.setattr(infer, "generate_from_messages", fake_generate)
    out = tmp_path / "preds.jsonl"
    infer.main(["--model", "m", "--dataset", str(ds), "--out", str(out)])

    assert [json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()] == [
        "right",
        "empty",
        "no-gold",
    ]
    text = capsys.readouterr().out
    assert "all articles: cell P 1.000 F1 1.000" in text  # After the first article
    assert "all articles: cell P 1.000 F1 0.667" in text  # The empty table finds nothing
    assert '"exact_match": 0.5' in text and '"documents": 2' in text  # The final summary
