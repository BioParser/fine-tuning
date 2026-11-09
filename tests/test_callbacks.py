"""ExtractionMetricsCallback with a stub model. The test needs torch and transformers, so CI skips it."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from bioparser_ft.callbacks import ExtractionMetricsCallback  # noqa: E402
from bioparser_ft.schema import MODEL_COLUMNS, SYSTEM_PROMPT, build_messages  # noqa: E402

GOLD = "\n".join(
    [
        "\t".join(MODEL_COLUMNS),
        "\t".join(
            {"verbatimScientificName": "Mus spretus", "verbatimTraitValue": "66.2"}.get(c, "")
            for c in MODEL_COLUMNS
        ),
    ]
)


def _tokenizer() -> Any:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    raw = Tokenizer(models.BPE())
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()  # Byte level: each text comes back without a change
    specials = ["<|im_start|>", "<|im_end|>", "<pad>"]
    raw.train_from_iterator(
        [SYSTEM_PROMPT, GOLD],
        trainers.BpeTrainer(
            vocab_size=400, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        eos_token="<|im_end|>",
        pad_token="<pad>",
        model_input_names=["input_ids", "attention_mask"],
    )
    tok.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
    )
    return tok


class StubModel:
    """generate() writes the next answer of a list after the prompt, then EOS."""

    device = torch.device("cpu")

    def __init__(self, tok: Any, answers: list[str]) -> None:
        self.tok, self.answers, self.training = tok, answers, True

    def eval(self) -> StubModel:
        self.training = False
        return self

    def train(self) -> StubModel:
        self.training = True
        return self

    def generate(self, input_ids: Any, max_new_tokens: int, **kwargs: Any) -> Any:
        assert not torch.is_inference_mode_enabled()  # Training needs no_grad, not inference mode
        new = self.tok(self.answers.pop(0), add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]
        return torch.cat([input_ids, torch.tensor([new[:max_new_tokens]])], dim=1)


class StubTrainer:
    def __init__(self, model: StubModel) -> None:
        self.model = model
        self.logs: list[tuple[dict[str, float], bool]] = []

    def log(self, logs: dict[str, float]) -> None:
        self.logs.append((logs, self.model.training))


def test_on_evaluate_adds_the_extraction_metrics() -> None:
    tok = _tokenizer()
    model = StubModel(tok, [GOLD, ""])  # The first table is correct, the second table is empty
    trainer = StubTrainer(model)
    examples = [
        {"id": "a", "messages": build_messages("Mus spretus 66.2", GOLD)},
        {"id": "b", "messages": build_messages("Mus spretus 66.2 mm", GOLD)},
        {"id": "no-gold", "messages": build_messages("x")},  # Not used: no assistant turn
    ]
    callback = ExtractionMetricsCallback(trainer, tok, examples, max_new_tokens=500)
    metrics = {"eval_loss": 1.0}

    callback.on_evaluate(None, SimpleNamespace(epoch=1.0), None, metrics=metrics, model=model)

    assert metrics["eval_loss"] == 1.0
    assert metrics["eval_cell_precision"] == 1.0 and metrics["eval_cell_f1"] == pytest.approx(2 / 3)
    assert metrics["eval_cell_accuracy"] == 0.5 and metrics["eval_exact_match"] == 0.5
    assert metrics["eval_gen_documents"] == 2.0 and metrics["eval_gen_incomplete"] == 0.0
    (logged, training_during_log), *_ = trainer.logs
    assert logged["eval_cell_f1"] == metrics["eval_cell_f1"]
    assert not training_during_log  # TRL's SFTTrainer.log() must see an evaluation
    assert model.training  # The callback switched the model back to training
