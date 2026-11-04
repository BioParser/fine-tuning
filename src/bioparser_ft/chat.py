"""Prompt/training-text rendering shared by training, inference and serving.

The training text and the inference prompt must come from the same function, or
the model is trained on one format and queried with another. This module needs
only a tokenizer/processor object (no torch), so it is unit-testable with a stub.

Format (Qwen ChatML, non-thinking; the same prefix Unsloth's own Qwen3.5 code uses)::

    <|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n
    <|im_start|>user\\n<article>...</article><|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{TSV}<|im_end|>
                             '-- prompt ends here --'   '-- trained span --'

Serving: build the prompt with render_prompt() and call a raw completion endpoint,
or use a chat endpoint with chat_template_kwargs={"enable_thinking": False}.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .schema import MODEL_HEADER, SYSTEM_PROMPT

NO_THINK_PREFIX = "<think>\n\n</think>\n\n"
OPEN_THINK = "<think>\n"
INSTRUCTION_PART = "<|im_start|>user\n"  # markers for unsloth's train_on_responses_only
RESPONSE_PART = "<|im_start|>assistant\n"

Message = Mapping[str, str]


def text_tokenizer(processing_class: Any) -> Any:
    """Unsloth returns a multimodal processor (Qwen3VLProcessor) for Qwen3.5, because
    the checkpoint is Qwen3_5ForConditionalGeneration. Text APIs (eos_token, decode,
    calling on plain text) live on its ``.tokenizer``."""
    return getattr(processing_class, "tokenizer", processing_class)


def render_prompt(processing_class: Any, messages: Sequence[Message]) -> str:
    """System + user turns + non-thinking assistant prefix, as a string."""
    prompt_messages = [dict(m) for m in messages if m["role"] != "assistant"]
    errors: list[str] = []
    text: str | None = None
    for owner in (text_tokenizer(processing_class), processing_class):
        if not getattr(owner, "chat_template", None):
            continue
        try:
            out = owner.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            text = out[0] if isinstance(out, list) else out
            break
        except Exception as exc:  # try the next owner; all failures are reported
            errors.append(f"{type(owner).__name__}: {exc!r}")
    if text is None:
        raise RuntimeError(f"no usable chat template on {type(processing_class).__name__}: {errors}")
    # Template versions differ: some ignore enable_thinking=False and open a thinking
    # block, some emit no think block. Normalise to the explicit empty block.
    if text.endswith(OPEN_THINK):
        text = text[: -len(OPEN_THINK)]
    if not text.endswith(NO_THINK_PREFIX):
        text += NO_THINK_PREFIX
    return text


def render_training_text(processing_class: Any, messages: Sequence[Message]) -> str:
    """Prompt + target TSV + EOS. Ends exactly with EOS, so TRL does not append a second one."""
    if messages[-1]["role"] != "assistant":
        raise ValueError("training messages must end with the assistant turn")
    eos: str = text_tokenizer(processing_class).eos_token
    return render_prompt(processing_class, messages) + messages[-1]["content"] + eos


def count_tokens(processing_class: Any, texts: Sequence[str]) -> list[int]:
    tok = text_tokenizer(processing_class)
    return [len(ids) for ids in tok(list(texts), add_special_tokens=False)["input_ids"]]


def prepare_training_texts(
    processing_class: Any, rows: Sequence[Mapping[str, Any]], max_seq_length: int
) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
    """Render dataset rows and drop (never truncate) those longer than max_seq_length.

    Truncation cuts the end of the sequence, i.e. the TSV answer: the original
    notebook (max_seq_length=2048, 14k-character prompt) trained on zero answer tokens.
    Returns (kept rows with id/text/n_tokens, dropped (id, n_tokens)).
    """
    texts = [render_training_text(processing_class, r["messages"]) for r in rows]
    lengths = count_tokens(processing_class, texts)
    kept, dropped = [], []
    for row, text, n in zip(rows, texts, lengths, strict=True):
        if n <= max_seq_length:
            kept.append({"id": row["id"], "text": text, "n_tokens": n})
        else:
            dropped.append((row["id"], n))
    return kept, dropped


def check_prompt(processing_class: Any, messages: Sequence[Message]) -> list[str]:
    """Human-readable problems with the rendered prompt; empty list means OK."""
    text = render_prompt(processing_class, messages)
    problems = []
    if RESPONSE_PART not in text:
        problems.append("no '<|im_start|>assistant' header: not a ChatML template?")
    for m in messages:
        if m["role"] != "assistant" and m["content"][:200] not in text:
            problems.append(f"{m['role']} message not found verbatim in the prompt")
    return problems


def check_label_masking(dataset: Any, processing_class: Any, n: int = 3) -> list[str]:
    """After train_on_responses_only: the trained span must be the TSV only."""
    tok = text_tokenizer(processing_class)
    problems = []
    fully_masked = sum(all(t == -100 for t in ex["labels"]) for ex in dataset)
    if fully_masked:
        problems.append(f"{fully_masked} examples have no trainable tokens (marker not found?)")
    for i in range(min(n, len(dataset))):
        kept = [t for t in dataset[i]["labels"] if t != -100]
        decoded = tok.decode(kept)
        if MODEL_HEADER not in decoded:
            problems.append(f"example {i}: TSV header is not in the trained span")
        if SYSTEM_PROMPT[:60] in decoded or "<article>" in decoded:
            problems.append(f"example {i}: prompt tokens are being trained")
    return problems
