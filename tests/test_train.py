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


def test_balanced_shards() -> None:
    shards = distributed.balanced_shards([10.0, 1.0, 9.0, 2.0, 5.0], 2)
    assert sorted(i for s in shards for i in s) == [0, 1, 2, 3, 4]  # Each article one time
    loads = [sum([10.0, 1.0, 9.0, 2.0, 5.0][i] for i in s) for s in shards]
    assert max(loads) - min(loads) <= 1.0  # 14 and 13: the GPUs wait little for each other
    assert distributed.balanced_shards([3.0, 1.0], 1) == [[0, 1]]
    assert distributed.balanced_shards([], 2) == [[], []]


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.toml")), ids=lambda p: p.name)
def test_the_profiles_in_configs_are_valid(path: Path) -> None:
    profile = train.read_profile(path)
    cfg = TrainConfig(**profile)  # TrainConfig checks the values, for example --grad-accum and the GPUs
    assert cfg.model_revision and len(cfg.model_revision) == 40  # The same base weights in each run
    assert "4b" in str(cfg.output_dir).lower() or "4B" not in cfg.model_name  # No mix of model outputs


def test_read_profile_refuses_errors(tmp_path: Path) -> None:
    p = tmp_path / "x.toml"
    p.write_text("epochs = 3\nmax_seq_length = 4096\nextra = {group_by_length = true}\n", encoding="utf-8")
    assert train.read_profile(p) == {
        "epochs": 3.0,
        "max_seq_length": 4096,
        "extra": {"group_by_length": True},
    }
    for text, message in [
        ("epoch = 3", "not profile settings"),  # A typing error must not use the default silently
        ("resume = true", "not profile settings"),  # A property of one run
        ('max_seq_length = "4096"', "is not a int"),
        ("tiled_mlp = 1", "is not a bool"),
        ("lora_r = true", "is not a int"),
        ("max_seq_length = ", "x.toml"),  # Not TOML
    ]:
        p.write_text(text + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            train.read_profile(p)


def test_main_uses_the_profile_and_the_options_override_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "gpu.toml"
    p.write_text(
        'model_name = "Qwen/Qwen3.5-4B"\nmax_seq_length = 4096\nlora_r = 32\n'
        "warmup_fraction = 0.1\nmerge_16bit = true\neval = false\n",
        encoding="utf-8",
    )
    configs: list[TrainConfig] = []
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(train, "run", configs.append)
    train.main(["--config", str(p), "--lora-r", "8", "--no-merge-16bit"])
    (cfg,) = configs
    assert (cfg.model_name, cfg.max_seq_length, cfg.warmup_fraction) == ("Qwen/Qwen3.5-4B", 4096, 0.1)
    assert (cfg.lora_r, cfg.merge_16bit, cfg.eval) == (8, False, False)  # The options override the profile
    assert cfg.profile == str(p)
    p.write_text("lora_rank = 32\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="lora_rank"):
        train.main(["--config", str(p)])


def test_a_profile_with_two_gpus_starts_torchrun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "two.toml"
    p.write_text("num_gpus = 2\ngrad_accum = 8\n", encoding="utf-8")
    started: list[tuple[int, list[str]]] = []

    def fake_launch(num_gpus: int, module: str, args: list[str]) -> int:
        started.append((num_gpus, list(args)))
        return 0

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(distributed, "launch", fake_launch)
    with pytest.raises(SystemExit):
        train.main(["--config", str(p)])
    assert started == [(2, ["--config", str(p)])]  # Each GPU process reads the same profile


def test_train_config_validates_the_metric_options() -> None:
    TrainConfig(best_metric="loss", eval_docs=0)
    with pytest.raises(ValueError, match="best_metric"):
        TrainConfig(best_metric="cell_f2")
    with pytest.raises(ValueError, match="eval_docs"):
        TrainConfig(eval_docs=-1)
