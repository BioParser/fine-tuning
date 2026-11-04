from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from bioparser_ft.chat import (
    NO_THINK_PREFIX,
    check_prompt,
    prepare_training_texts,
    render_prompt,
    render_training_text,
)
from bioparser_ft.evaluate import aggregate, evaluate_files, score_document
from bioparser_ft.schema import MODEL_COLUMNS, build_messages


def _rec(**kw: str) -> dict[str, str]:
    return {c: kw.get(c, "") for c in MODEL_COLUMNS}


GOLD = [
    _rec(
        references="C.",
        verbatimScientificName="Mus spretus",
        verbatimTraitName="T",
        verbatimTraitValue="66.2",
    ),
    _rec(
        references="C.",
        verbatimScientificName="Mus spretus",
        verbatimTraitName="HB",
        verbatimTraitValue="81.70",
    ),
]


def test_scores() -> None:
    perfect = aggregate([score_document(list(reversed(GOLD)), GOLD)])  # order does not matter
    assert perfect["row"]["f1"] == perfect["cell"]["f1"] == perfect["value"]["f1"] == 1.0
    assert perfect["references_exact"] == 1.0
    wrong = [dict(GOLD[0]), dict(GOLD[1], verbatimTraitValue="81.8")]
    s = aggregate([score_document(wrong, GOLD)])
    assert s["value"]["f1"] == 0.5 and s["row"]["f1"] == 0.5
    assert aggregate([score_document([], GOLD)])["cell"]["recall"] == 0.0
    numeric = aggregate([score_document([dict(GOLD[0]), dict(GOLD[1], verbatimTraitValue="81.7")], GOLD)])
    assert numeric["value"]["f1"] == 1.0  # "81.70" == "81.7"


def test_evaluate_files(tmp_path: Path) -> None:
    tsv = "\n".join(["\t".join(MODEL_COLUMNS)] + ["\t".join(r[c] for c in MODEL_COLUMNS) for r in GOLD])
    ds = tmp_path / "test.jsonl"
    ds.write_text(json.dumps({"id": "p1", "messages": build_messages("x", tsv)}) + "\n", encoding="utf-8")
    preds = tmp_path / "preds.jsonl"
    preds.write_text(json.dumps({"id": "p1", "output": tsv, "complete": True}) + "\n", encoding="utf-8")
    report = evaluate_files(ds, preds)
    assert report["summary"]["row"]["f1"] == 1.0 and not report["summary"]["missing_predictions"]


class StubTokenizer:
    """Minimal ChatML tokenizer; `honours_flag=False` mimics a template that ignores
    enable_thinking=False and opens a thinking block."""

    eos_token = "<|im_end|>"

    def __init__(self, honours_flag: bool = True):
        self.chat_template = "stub"
        self.honours_flag = honours_flag

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
    ) -> str:
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if enable_thinking or not self.honours_flag:
                text += "<think>\n"
            else:
                text += NO_THINK_PREFIX
        return text

    def __call__(self, texts: Sequence[str], add_special_tokens: bool = False) -> dict[str, list[list[str]]]:
        return {"input_ids": [t.split() for t in texts]}


class StubProcessor:  # what FastLanguageModel returns for Qwen3.5
    def __init__(self, tok: StubTokenizer):
        self.tokenizer = tok
        self.chat_template = None


@pytest.mark.parametrize("honours_flag", [True, False])
def test_render_prompt_is_non_thinking(honours_flag: bool) -> None:
    proc = StubProcessor(StubTokenizer(honours_flag))
    msgs = build_messages("# Paper\nMus spretus 66.2 mm", "ANSWER_SENTINEL")
    prompt = render_prompt(proc, msgs)
    assert prompt.endswith("<|im_start|>assistant\n" + NO_THINK_PREFIX)
    assert "ANSWER_SENTINEL" not in prompt  # assistant turn excluded
    assert not check_prompt(proc, msgs[:2])
    text = render_training_text(proc, msgs)
    assert text == prompt + "ANSWER_SENTINEL<|im_end|>"  # ends exactly with EOS


def test_prepare_training_texts_drops_not_truncates() -> None:
    proc = StubProcessor(StubTokenizer())
    rows = [
        {"id": "short", "messages": build_messages("a", "T")},
        {"id": "long", "messages": build_messages("word " * 5000, "T")},
    ]
    kept, dropped = prepare_training_texts(proc, rows, max_seq_length=2000)
    assert [r["id"] for r in kept] == ["short"] and dropped[0][0] == "long"


def test_render_with_real_transformers_template() -> None:
    """Runs where transformers is installed (the training env): real apply_chat_template."""
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    raw = Tokenizer(models.WordLevel({"<unk>": 0, "<|im_end|>": 1}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, eos_token="<|im_end|>", unk_token="<unk>"
    )
    tok.chat_template = (  # Qwen-style: empty think block only when enable_thinking is False
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n"
        "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n"
        "{% endif %}{% endif %}"
    )
    prompt = render_prompt(tok, build_messages("Mus spretus", None))
    assert prompt.endswith("<|im_start|>assistant\n" + NO_THINK_PREFIX)
    assert prompt.count(NO_THINK_PREFIX) == 1
