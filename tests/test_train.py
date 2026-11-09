"""The parts of train.py that need no GPU stack. CI runs these tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from bioparser_ft.train import TrainConfig, last_checkpoint


def test_last_checkpoint(tmp_path: Path) -> None:
    assert last_checkpoint(tmp_path / "not-there") is None
    assert last_checkpoint(tmp_path) is None
    for name in ("checkpoint-9", "checkpoint-10", "checkpoint-x", "runs"):
        (tmp_path / name).mkdir()
    (tmp_path / "checkpoint-99").write_text("a file, not a folder", encoding="utf-8")
    assert last_checkpoint(tmp_path) == tmp_path / "checkpoint-10"  # 10 > 9: a number, not a string


def test_train_config_refuses_a_warmup_fraction_of_one_or_more() -> None:
    TrainConfig(warmup_fraction=0.0)
    with pytest.raises(ValueError, match="warmup_fraction"):
        TrainConfig(warmup_fraction=1.0)  # transformers would read 1.0 as 1 step, not 100 %
