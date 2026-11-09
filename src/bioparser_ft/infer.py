"""Generation with a fine-tuned adapter (GPU) and the bioparser-predict CLI.

    # one parsed PDF -> ETS JSON (what the FastAPI endpoint does)
    uv run bioparser-predict --model outputs/qwen3.5-0.8b-ets/lora_adapter \\
        --markdown paper.md --out paper.json
    # a dataset split -> predictions JSONL for bioparser-evaluate
    uv run bioparser-predict --model outputs/qwen3.5-0.8b-ets/lora_adapter \\
        --dataset data/processed/test.jsonl --out outputs/preds_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .chat import render_prompt, text_tokenizer
from .schema import PROMPT_VERSION, build_messages

log = logging.getLogger(__name__)


@dataclass
class Generation:
    text: str
    complete: bool  # ended with EOS; False = cut by the token budget
    prompt_tokens: int
    new_tokens: int
    seconds: float


def read_training_metadata(model_path: str | Path) -> dict[str, Any] | None:
    """Return the bioparser_ft.json of train.save_artifacts, or None if the folder has no such file."""
    meta_path = Path(model_path) / "bioparser_ft.json"
    if not meta_path.is_file():  # For example, a Hub id
        return None
    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta


def check_prompt_version(model_path: str | Path) -> None:
    """Stop if the model was trained with a different PROMPT_VERSION.

    With a different version, the model gets a prompt that it did not learn. Then the output is
    worse, but no error occurs. train.save_artifacts writes the version to bioparser_ft.json.
    """
    meta = read_training_metadata(model_path)
    if meta is None:
        log.warning(
            "Cannot examine the prompt version: %s does not exist", Path(model_path) / "bioparser_ft.json"
        )
        return
    trained = meta.get("prompt_version")
    if trained != PROMPT_VERSION:
        raise ValueError(
            f"{model_path} was trained with prompt_version {trained!r}, but the code uses "
            f"{PROMPT_VERSION!r}. Train the model again, or use the code of its version."
        )


def check_base_model(model: Any, model_path: str | Path) -> bool:
    """Warn if the loaded base weights are not the base weights of the training. Return False then.

    For an adapter folder, Unsloth loads the base model by name and ignores `revision` for it, so it
    gets the newest commit on the Hub. If that commit changed after the training, the adapter
    operates on other weights: the output can be worse, but no error occurs. A merged model has
    its own weights, and a local base model has no commit, so the check does nothing for them.
    """
    trained = (read_training_metadata(model_path) or {}).get("base_model_commit")
    loaded = getattr(getattr(model, "config", None), "_commit_hash", None)
    if trained and loaded and trained != loaded:
        log.warning(
            "The base model commit %s is not the commit of the training (%s). Use the merged model of "
            "bioparser-train --merge-16bit, or train the adapter again on this commit.",
            loaded,
            trained,
        )
        return False
    return True


def load_for_inference(model_path: str | Path, *, max_seq_length: int = 32768) -> tuple[Any, Any]:
    """Load base + adapter exactly as trained: 16-bit LoRA, NOT 4-bit.

    (The original notebook trained 16-bit and reloaded with load_in_4bit=True,
    i.e. served the adapter on differently quantised weights.)
    """
    check_prompt_version(model_path)  # A fast check before the slow model load
    from unsloth import FastLanguageModel  # import before transformers so patches apply

    model, processor = FastLanguageModel.from_pretrained(
        model_name=str(model_path), max_seq_length=max_seq_length, load_in_4bit=False
    )
    check_base_model(model, model_path)
    FastLanguageModel.for_inference(model)
    return model, processor


def _stop_token_ids(tok: Any) -> list[int]:
    ids = {tok.eos_token_id}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != getattr(tok, "unk_token_id", None):
        ids.add(im_end)
    return sorted(i for i in ids if i is not None)


def generate_from_messages(
    model: Any,
    processor: Any,
    messages: Sequence[dict[str, str]],
    *,
    max_new_tokens: int = 8192,
    max_seq_length: int | None = None,
    streamer: Any = None,
) -> Generation:
    """Greedy decoding (deterministic; sampling only adds noise to table extraction)."""
    import torch

    tok = text_tokenizer(processor)
    prompt = render_prompt(processor, [m for m in messages if m["role"] != "assistant"])
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    n_prompt = int(enc["input_ids"].shape[1])
    budget = max_new_tokens if max_seq_length is None else min(max_new_tokens, max_seq_length - n_prompt)
    if budget <= 0:
        return Generation("", False, n_prompt, 0, 0.0)
    stop = _stop_token_ids(tok)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=budget,
            do_sample=False,
            use_cache=True,
            eos_token_id=stop,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else stop[0],
            streamer=streamer,
        )
    new = out[0, n_prompt:].tolist()
    return Generation(
        text=tok.decode(new, skip_special_tokens=True),
        complete=bool(new) and new[-1] in stop,
        prompt_tokens=n_prompt,
        new_tokens=len(new),
        seconds=time.perf_counter() - start,
    )


def generate_tsv(model: Any, processor: Any, markdown: str, **kwargs: Any) -> Generation:
    return generate_from_messages(model, processor, build_messages(markdown), **kwargs)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="adapter dir (or merged model dir / hub id)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--markdown", type=Path, help="one parsed PDF (.md)")
    src.add_argument("--dataset", type=Path, help="split JSONL from bioparser-build-dataset")
    ap.add_argument("--references", help="verified Harvard citation; overrides the generated one")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--max-seq-length", type=int, default=32768, help="prompt + output token cap")
    ap.add_argument("--out", type=Path, help="output file (default: stdout for --markdown)")
    args = ap.parse_args(argv)

    from .postprocess import to_ets_records

    model, processor = load_for_inference(args.model, max_seq_length=args.max_seq_length)
    kw = {"max_new_tokens": args.max_new_tokens, "max_seq_length": args.max_seq_length}
    if args.markdown:
        gen = generate_tsv(model, processor, args.markdown.read_text(encoding="utf-8"), **kw)
        result = to_ets_records(gen.text, references=args.references, complete=gen.complete)
        payload = result.to_json()
        if args.out:
            args.out.write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload + "\n")
        return

    if args.out is None:
        ap.error("--out is required with --dataset")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.dataset.open(encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line)
            gen = generate_from_messages(model, processor, ex["messages"], **kw)
            row = {"id": ex["id"], "output": gen.text} | {k: v for k, v in asdict(gen).items() if k != "text"}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"{ex['id']}: {gen.new_tokens} tokens in {gen.seconds:.0f}s complete={gen.complete}")


if __name__ == "__main__":
    main()
