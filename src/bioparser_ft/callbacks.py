"""Extraction metrics on the validation articles after each epoch (needs transformers and torch).

The training loss and eval_loss do not show if the model writes correct tables. After each
evaluation of the Trainer, ExtractionMetricsCallback lets the model write the TSV table of each
validation article with greedy decoding, as bioparser-predict does. Then it calculates the
metrics of evaluate.py and adds them to the log and to the evaluation metrics: eval_cell_f1,
eval_row_accuracy, eval_exact_match and the other values of evaluate.scalar_metrics. The
Trainer can then select the best checkpoint with one of them (metric_for_best_model).

The callback does not calculate the token accuracy of teacher forcing. That value needs the full
logits (sequence length x 248K vocabulary for each example), and Unsloth does not keep them, to
save memory.

With more than one GPU, each process writes the tables of a part of the articles. Then each
process gets the counts of all the processes, so all the processes calculate the same metrics and
select the same checkpoint.

Import this module only after unsloth, because it imports transformers.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from transformers import TrainerCallback

from . import distributed
from .evaluate import aggregate, format_summary, gold_tsv, scalar_metrics, score_output
from .infer import Generation, generate_from_messages

ModelHook = Callable[[Any], object]


def _plain_inference(model: Any) -> object:
    return model.eval()


def _plain_training(model: Any) -> object:
    return model.train()


class ExtractionMetricsCallback(TrainerCallback):  # type: ignore[misc]  # mypy sees transformers as Any
    """Calculate the extraction metrics of evaluate.py on the validation articles after each evaluation.

    for_inference and for_training switch the model between generation and training. For Unsloth,
    use FastLanguageModel.for_inference and FastLanguageModel.for_training: they turn the gradient
    checkpointing off for the generation and on again after it.
    """

    def __init__(
        self,
        trainer: Any,
        processor: Any,
        examples: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int = 8192,
        max_seq_length: int | None = None,
        for_inference: ModelHook = _plain_inference,
        for_training: ModelHook = _plain_training,
    ) -> None:
        self.trainer = trainer
        self.processor = processor
        self.examples = [ex for ex in examples if gold_tsv(ex) is not None]
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length
        self.for_inference = for_inference
        self.for_training = for_training
        # The generation time is approximately proportional to the length of the table
        costs = [float(len(gold_tsv(ex) or "")) for ex in self.examples]
        self.shard = distributed.balanced_shards(costs, distributed.world_size())[distributed.rank()]

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, float] | None = None,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        if not self.examples or model is None:
            return
        start = time.perf_counter()
        self.for_inference(model)
        try:
            local = [(i, self._score(model, i)) for i in self.shard]
            scored = sorted(
                (item for part in distributed.gather_objects(local) for item in part),
                key=lambda item: item[0],
            )
            docs = [doc for _, doc in scored]
            summary = aggregate(docs)
            values = {f"eval_{key}": value for key, value in scalar_metrics(summary).items()}
            values["eval_gen_documents"] = float(len(docs))
            values["eval_gen_incomplete"] = float(sum(not doc["complete"] for doc in docs))
            values["eval_gen_seconds"] = time.perf_counter() - start
            if metrics is not None:
                metrics.update(values)  # The Trainer selects the best checkpoint with these metrics
            if distributed.is_main_process():
                print(f"validation, epoch {state.epoch or 0:.2f}: {format_summary(summary)}", flush=True)
            # A copy, because log() adds keys. The model is still in eval mode, so TRL's
            # SFTTrainer.log() does not add (and clear) the training metrics of the current step.
            self.trainer.log(dict(values))
        finally:
            self.for_training(model)

    def _score(self, model: Any, index: int) -> dict[str, Any]:
        example = self.examples[index]
        gold = gold_tsv(example) or ""
        try:
            gen = generate_from_messages(
                model,
                self.processor,
                example["messages"],
                max_new_tokens=self.max_new_tokens,
                max_seq_length=self.max_seq_length,
                inference_mode=False,  # Tensors of inference mode cannot go into the next training step
            )
        except torch.cuda.OutOfMemoryError:
            # Do not stop a training of many hours at the end of an epoch. The article counts as an
            # empty table, so the metrics are lower, and eval_gen_incomplete shows it.
            torch.cuda.empty_cache()
            print(f"WARNING: GPU {distributed.rank()}: no memory to generate {example['id']}", flush=True)
            gen = Generation("", False, 0, 0, 0.0)
        if distributed.is_main_process():
            print(f"  {example['id']}: {gen.new_tokens} tokens in {gen.seconds:.0f}s", flush=True)
        return score_output(gen.text, gold, complete=gen.complete)
