"""The parts of train.py that need no GPU stack. CI runs these tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from bioparser_ft import distributed, train
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


def test_per_device_grad_accum() -> None:
    assert distributed.per_device_grad_accum(8, 1) == 8
    assert distributed.per_device_grad_accum(8, 2) == 4  # Effective batch 8, as on 1 GPU
    with pytest.raises(ValueError, match="not a multiple of 2 GPUs"):
        distributed.per_device_grad_accum(3, 2)
    with pytest.raises(ValueError, match="grad-accum"):
        TrainConfig(grad_accum=3, num_gpus=2)


def test_torchrun_command() -> None:
    cmd = distributed.torchrun_command(2, "bioparser_ft.train", ["--num-gpus", "2", "--epochs", "1"])
    assert cmd[1:] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "bioparser_ft.train",
        "--num-gpus",
        "2",
        "--epochs",
        "1",
    ]


def test_main_starts_torchrun_for_two_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[tuple[int, str, list[str]]] = []

    def fake_launch(num_gpus: int, module: str, args: list[str]) -> int:
        started.append((num_gpus, module, list(args)))
        return 0

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(distributed, "launch", fake_launch)
    monkeypatch.setattr(train, "run", lambda cfg: pytest.fail("the parent process must not train"))
    with pytest.raises(SystemExit) as exit_info:
        train.main(["--num-gpus", "2", "--epochs", "1"])
    assert exit_info.value.code == 0
    assert started == [(2, "bioparser_ft.train", ["--num-gpus", "2", "--epochs", "1"])]


def test_main_in_a_torchrun_process_trains(monkeypatch: pytest.MonkeyPatch) -> None:
    configs: list[TrainConfig] = []
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(distributed, "launch", lambda *a: pytest.fail("a worker must not start torchrun"))
    monkeypatch.setattr(train, "run", configs.append)
    train.main(["--num-gpus", "2"])
    train.main([])  # torchrun was started by hand, without --num-gpus
    assert [c.num_gpus for c in configs] == [2, 2]
    with pytest.raises(SystemExit):
        train.main(["--num-gpus", "1"])  # torchrun started 2 processes


def test_main_refuses_a_grad_accum_that_is_not_a_multiple(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(distributed, "launch", lambda *a: pytest.fail("no torchrun for bad options"))
    with pytest.raises(SystemExit) as exit_info:
        train.main(["--num-gpus", "2", "--grad-accum", "3"])
    assert exit_info.value.code == 2  # The argparse error code
