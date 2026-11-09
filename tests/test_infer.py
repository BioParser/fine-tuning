"""check_prompt_version and check_base_model need no GPU stack, so CI runs these tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bioparser_ft.infer import check_base_model, check_prompt_version
from bioparser_ft.schema import PROMPT_VERSION


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
