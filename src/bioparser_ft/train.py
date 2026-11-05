"""CLI: LoRA fine-tuning of Qwen3.5 with Unsloth (GPU).  `uv run bioparser-train --help`

Memory on small GPUs (developed against an RTX 2060 Max-Q, 6 GB):
* use_gradient_checkpointing="unsloth" offloads checkpointed activations to CPU RAM
  (asynchronous, double-buffered), so long sequences fit; this is the
  "Unsloth: Will smartly offload gradients to save VRAM!" message;
* offload_embedding="auto" (Unsloth default) moves the 248K-vocab input embedding
  to RAM when the training reserve for max_seq_length does not fit in VRAM;
* --tiled-mlp chunks MLP activations along the sequence for very long context.
Precision: GPUs without bf16 (RTX 20xx, T4) train Qwen3.5 in float32. Unsloth
forces it: the Gated-DeltaNet layers give NaN gradients in float16. 4-bit QLoRA
is not recommended for Qwen3.5, so the base stays 16-bit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .chat import (
    INSTRUCTION_PART,
    RESPONSE_PART,
    check_label_masking,
    check_prompt,
    prepare_training_texts,
)
from .schema import MODEL_COLUMNS, PROMPT_VERSION

# Qwen3.5 = 3 Gated-DeltaNet (linear attention) layers per full-attention layer. The
# original notebook targeted only q/k/v/o (+MLP), which exist only in the 6 full-
# attention layers, so 18 of 24 layers had no attention adapters. Same list as
# Unsloth's own Qwen3.5 recipe. (in_proj_a / in_proj_b are tiny gate projections.)
# fmt: off
LORA_TARGET_MODULES: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",  # full attention (6 of 24 layers)
        "in_proj_qkv", "in_proj_z", "out_proj",  # Gated DeltaNet (18 of 24 layers)
        "gate_proj", "up_proj", "down_proj",     # MLP (all layers)
        )
# fmt: on


@dataclass
class TrainConfig:
    dataset_dir: Path = Path("data/processed")
    output_dir: Path = Path("outputs/qwen3.5-0.8b-ets")

    # Model
    model_name: str = "Qwen/Qwen3.5-0.8B"
    model_revision: str | None = None  # pin of model
    max_seq_length: int = 65536  # still need a bit longer to fit all PDFs
    # still few long PDFs are dropped if exceeds msl

    # LoRA:   W' = W + (alpha/r) BA
    #
    # r, rank 16: trainable parameters 32,768
    # r, rank 64: trainable parameters 131,072
    #
    # r = 16, alpha = 32 =>   32/16=2 stronger update
    #
    # r = 64, alpha = 64 =>   64/64=1 but larger adapter
    #
    lora_r: int = 16  # lower rank -> less trainable parameters -> worse fine-tu
    lora_alpha: int = 16  # how strongly the LoRA update affects the weights
    lora_dropout: float = 0.0  # 0 enables Unsloth's fast LoRA kernels
    target_modules: tuple[str, ...] = LORA_TARGET_MODULES

    # Training
    epochs: float = 3.0
    learning_rate: float = 2e-4
    batch_size: int = 1  # long sequences; use grad_accum for the effective batch
    grad_accum: int = 8
    warmup_fraction: float = 0.05
    weight_decay: float = 0.01
    seed: int = 3407
    tiled_mlp: bool = True  # False if enough compute
    eval: bool = True
    merge_16bit: bool = False
    push_to_hub: str | None = None  # private repo id; token from $HF_TOKEN or `hf auth login`
    extra: dict[str, Any] = field(default_factory=dict)


def read_split(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    path = dataset_dir / f"{split}.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    stale = {r.get("prompt_version") for r in rows} - {PROMPT_VERSION}
    if stale:
        raise ValueError(f"{path} built with prompt_version {stale}, code is {PROMPT_VERSION}: rebuild it")
    return rows


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _package_versions() -> dict[str, str | None]:
    """Return the versions of the training stack. A person needs them to make the same model again."""
    versions: dict[str, str | None] = {}
    for name in ("unsloth", "unsloth-zoo", "torch", "transformers", "trl", "peft", "datasets"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def lora_report(model: Any) -> Counter[str]:
    """Number of LoRA-wrapped modules per leaf name (verifies the targets really attached)."""
    counts: Counter[str] = Counter()
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is not None and hasattr(lora_a, "keys") and len(lora_a):
            counts[name.rsplit(".", 1)[-1]] += 1
    return counts


def load_model(cfg: TrainConfig) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel  # must precede transformers/trl imports

    model, processor = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        revision=cfg.model_revision,
        max_seq_length=cfg.max_seq_length,  # also sizes Unsloth's embedding-offload decision
        dtype=None,  # auto: bf16 if supported, else float32 for Qwen3.5
        load_in_4bit=False,  # 16-bit LoRA
        use_gradient_checkpointing="unsloth",  # activation offload to CPU RAM
        unsloth_tiled_mlp=cfg.tiled_mlp,
        random_state=cfg.seed,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=list(cfg.target_modules),
        finetune_vision_layers=False,  # text-only task: leave the vision tower frozen
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    return model, processor


def build_trainer(
    cfg: TrainConfig,
    model: Any,
    processor: Any,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> Any:
    from unsloth.chat_templates import train_on_responses_only  # noqa: I001 - unsloth must patch trl first
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    steps_per_epoch = math.ceil(len(train_rows) / (cfg.batch_size * cfg.grad_accum))
    total_steps = max(1, math.ceil(steps_per_epoch * cfg.epochs))
    has_eval = cfg.eval and bool(val_rows)
    args = SFTConfig(
        output_dir=str(cfg.output_dir / "checkpoints"),
        dataset_text_field="text",
        max_length=cfg.max_seq_length,  # TRL default is 1024 and truncates silently
        packing=False,
        dataset_num_proc=1,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        warmup_steps=max(1, round(cfg.warmup_fraction * total_steps)),
        lr_scheduler_type="linear",
        weight_decay=cfg.weight_decay,
        optim="adamw_8bit",
        logging_steps=1,
        eval_strategy="epoch" if has_eval else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        seed=cfg.seed,
        report_to="none",
        **cfg.extra,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,  # Unsloth accepts the processor here (as in its notebooks)
        train_dataset=Dataset.from_list([{"text": r["text"]} for r in train_rows]),
        eval_dataset=Dataset.from_list([{"text": r["text"]} for r in val_rows]) if has_eval else None,
        args=args,
    )
    # Loss only on the assistant TSV. Without this the loss is dominated by
    # reproducing the paper text (thousands of prompt tokens per example).
    return train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)


def push_private(folder: Path, repo_id: str) -> None:
    """Upload to a PRIVATE hub repo. The adapter was trained on client data and can
    regurgitate it, so a public repo is refused."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))  # None -> cached `hf auth login` token
    api.create_repo(repo_id, private=True, exist_ok=True)
    if not api.repo_info(repo_id).private:
        raise RuntimeError(f"{repo_id} exists and is public; refusing to upload client-derived weights")
    api.upload_folder(repo_id=repo_id, folder_path=str(folder))


def run(cfg: TrainConfig) -> Path:
    # Cheap checks first: fail before a multi-minute model load.
    train_raw, val_raw = read_split(cfg.dataset_dir, "train"), read_split(cfg.dataset_dir, "val")
    if not train_raw:
        raise SystemExit(f"no training data in {cfg.dataset_dir}; run bioparser-build-dataset first")
    if cfg.model_revision is None:
        print(f"WARNING: no --model-revision. The Hub gives the newest commit of {cfg.model_name}.")

    model, processor = load_model(cfg)
    import torch  # after unsloth

    attached = lora_report(model)
    print(f"LoRA modules per target: {dict(attached)}")
    if missing := [t for t in cfg.target_modules if not attached.get(t)]:
        print(f"WARNING: no adapters attached to {missing}")

    if problems := check_prompt(processor, train_raw[0]["messages"]):
        raise SystemExit(f"prompt rendering problems: {problems}")
    train_rows, dropped = prepare_training_texts(processor, train_raw, cfg.max_seq_length)
    val_rows, val_dropped = prepare_training_texts(processor, val_raw, cfg.max_seq_length)
    print(f"train: {len(train_rows)} kept, {len(dropped)} longer than {cfg.max_seq_length} tokens dropped")
    print(f"val:   {len(val_rows)} kept, {len(val_dropped)} dropped")
    if not train_rows:
        raise SystemExit("every training example is longer than max_seq_length")

    trainer = build_trainer(cfg, model, processor, train_rows, val_rows)
    if problems := check_label_masking(trainer.train_dataset, processor):
        raise SystemExit(f"label masking problems: {problems}")

    stats = trainer.train()
    if torch.cuda.is_available():
        print(f"peak reserved VRAM: {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
    return save_artifacts(cfg, model, processor, train_rows, val_rows, dropped + val_dropped, stats.metrics)


def save_artifacts(
    cfg: TrainConfig,
    model: Any,
    processor: Any,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    dropped: list[tuple[str, int]],
    metrics: dict[str, Any],
) -> Path:
    """Save adapter + processor + provenance (prompt version, dataset hashes, ids)."""
    adapter_dir = cfg.output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    meta = {
        "bioparser_ft": __version__,
        "prompt_version": PROMPT_VERSION,
        "model_columns": list(MODEL_COLUMNS),
        # The commit that the Hub gave. transformers sets config._commit_hash, also without a revision.
        "base_model_commit": getattr(getattr(model, "config", None), "_commit_hash", None),
        "packages": _package_versions(),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "train_ids": [r["id"] for r in train_rows],
        "val_ids": [r["id"] for r in val_rows],
        "dropped_too_long": dropped,
        "dataset_sha256": {s: _sha256(cfg.dataset_dir / f"{s}.jsonl") for s in ("train", "val", "test")},
        "train_metrics": metrics,
    }
    (adapter_dir / "bioparser_ft.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"adapter saved to {adapter_dir}")

    if cfg.merge_16bit:  # standalone weights for vLLM / transformers serving
        merged = cfg.output_dir / "merged_16bit"
        model.save_pretrained_merged(str(merged), processor, save_method="merged_16bit")
        # The serving code reads this file and examines the prompt version (infer.check_prompt_version).
        (merged / "bioparser_ft.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"merged model saved to {merged}")
    if cfg.push_to_hub:
        push_private(adapter_dir, cfg.push_to_hub)
        print(f"adapter uploaded to private repo {cfg.push_to_hub}")
    return adapter_dir


def main(argv: Sequence[str] | None = None) -> None:
    d = TrainConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=d.dataset_dir)
    ap.add_argument("--output-dir", type=Path, default=d.output_dir)
    ap.add_argument("--model-name", default=d.model_name)
    ap.add_argument(
        "--model-revision",
        default=d.model_revision,
        help="commit hash of the base model on the Hub. With it, each run trains on the same weights.",
    )
    ap.add_argument("--max-seq-length", type=int, default=d.max_seq_length)
    ap.add_argument("--lora-r", type=int, default=d.lora_r)
    ap.add_argument("--lora-alpha", type=int, default=d.lora_alpha)
    ap.add_argument("--epochs", type=float, default=d.epochs)
    ap.add_argument("--learning-rate", type=float, default=d.learning_rate)
    ap.add_argument("--grad-accum", type=int, default=d.grad_accum)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument(
        "--tiled-mlp",
        action=argparse.BooleanOptionalAction,
        default=d.tiled_mlp,
        help="tiled MLP: less activation memory at long context. On by default; --no-tiled-mlp turns it off",
    )
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--merge-16bit", action="store_true", help="also save merged 16-bit weights")
    ap.add_argument("--push-to-hub", metavar="REPO_ID", help="upload adapter to a PRIVATE hub repo")
    a = ap.parse_args(argv)
    run(
        TrainConfig(
            dataset_dir=a.dataset_dir,
            output_dir=a.output_dir,
            model_name=a.model_name,
            model_revision=a.model_revision,
            max_seq_length=a.max_seq_length,
            lora_r=a.lora_r,
            lora_alpha=a.lora_alpha,
            epochs=a.epochs,
            learning_rate=a.learning_rate,
            grad_accum=a.grad_accum,
            seed=a.seed,
            tiled_mlp=a.tiled_mlp,
            eval=not a.no_eval,
            merge_16bit=a.merge_16bit,
            push_to_hub=a.push_to_hub,
        )
    )


if __name__ == "__main__":
    main()
