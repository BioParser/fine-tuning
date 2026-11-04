"""check_prompt_version needs no GPU stack, so CI runs these tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bioparser_ft.infer import check_prompt_version
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
