# fine-tuning

This repository fine-tunes the model `Qwen/Qwen3.5-0.8B` for
[BioParser](https://github.com/BioParser/BioParser) with Unsloth and LoRA.

The model reads a scientific article that MinerU parsed from a PDF file. Then the model writes
the trait data of the article as a TSV table in the client ETS format.

```
PDF --MinerU--> markdown --clean_markdown--> chat prompt --Qwen3.5+LoRA--> TSV --to_ets_records--> JSON --> FastAPI
             one training example = (parsed PDF, client ETS TSV)
```

## Installation

You need an NVIDIA GPU and [uv](https://docs.astral.sh/uv/). The smallest GPU that we tested is an
RTX 2060 Max-Q with 6 GB of VRAM. Run this command in the repository folder:

```bash
uv sync
uv run unsloth install-kernels causal_conv1d
```

The command installs the tested package versions from `uv.lock`.

## Fine-tune the model

For a short list of all commands, refer to the [Quickstart](fine_tuning_data/README.md#quickstart).

1. Put the client ETS files in `fine_tuning_data/ets_tsv/`.
2. Put one PDF file for each article in `fine_tuning_data/pdfs/`. The commands
   `bioparser-fetch-pdfs` and `bioparser-link-pdfs` help you. For the procedure, refer to
   [Step 1](fine_tuning_data/README.md#step-1-get-the-pdf-files).

   **CAUTION:** Do not run `bioparser-fetch-pdfs` on the network of the university. On this
   network, the command also downloads articles that are not open access. The terms of the
   library do not let programs do this.

3. Parse the PDF files with MinerU:

   ```bash
   scripts/parse_pdfs.sh
   ```

   The script writes one zip file for each PDF file to `fine_tuning_data/parsed/`. Use the MinerU
   version of production. The variable `MINERU_VERSION` sets the version.

4. Make the dataset:

   ```bash
   uv run bioparser-build-dataset
   ```

   The command writes `train.jsonl`, `val.jsonl` and `test.jsonl` to `data/processed/`. Examine the
   pairs that the command did not use in `data/processed/build_report.json`.

5. Train the model:

   ```bash
   uv run bioparser-train
   ```

   The command saves the LoRA adapter in `outputs/qwen3.5-0.8b-ets/lora_adapter/`. You can also
   use the notebook `notebooks/Qwen3_5_0_8B.ipynb`. The notebook uses the same functions.

6. Use the adapter on the test articles:

   ```bash
   uv run bioparser-predict --model outputs/qwen3.5-0.8b-ets/lora_adapter --dataset data/processed/test.jsonl --out outputs/preds_test.jsonl
   ```

7. Calculate the precision, the recall and the F1 score for rows, cells and values:

   ```bash
   uv run bioparser-evaluate --dataset data/processed/test.jsonl --predictions outputs/preds_test.jsonl
   ```

### Training options

| Option | Default | Description |
|---|---|---|
| `--model-revision HASH` | The newest commit | The commit of the base model on Hugging Face. Use it to train on the same weights in each run. |
| `--max-seq-length N` | 16384 | The maximum number of tokens for the prompt, the article and the TSV table. |
| `--tiled-mlp` | Off | Use less GPU memory for long sequences, for example 32,768 tokens. |
| `--merge-16bit` | Off | Also save a merged 16-bit model for vLLM. |
| `--push-to-hub REPO` | Off | Upload the adapter to a private Hugging Face repository. |

For all options, run `uv run bioparser-train --help`.

### GPU memory

- GPUs without bf16, for example RTX 20xx and T4, train Qwen3.5 in float32.
- When the VRAM is too small, Unsloth moves data to the CPU RAM. The training command sets these
  options.
- A typical example has 6,000 to 15,000 tokens. The command removes the examples that are longer
  than `--max-seq-length`. It does not cut an example, because a cut removes the answer.

## Output format

The file `src/bioparser_ft/schema.py` sets the output format. The training, the inference and the
API use this file.

- The model writes 19 of the 20 ETS columns. The function `to_ets_records` adds the column
  `author` with the value `BioParser`.
- The model writes the citation (`references`) only on the first row. `to_ets_records` copies it
  to the other rows. If the API has a correct citation, give it with `references=`.
- `to_ets_records` also uses fixed rules. It finds an empty `taxonRank` from the name. It writes
  the full genus name for `H. sapiens`. It changes forms such as `males` to `Male`.

**CAUTION:** If you change the columns, the prompt or the text cleaning, increase `PROMPT_VERSION`
in `src/bioparser_ft/schema.py`. Then make the dataset again and train the model again. The tools
stop with an error when the dataset or the model has a different `PROMPT_VERSION`.

## Use the model in an API

The modules `schema`, `markdown`, `tsv` and `postprocess` use only the Python standard library. The
FastAPI service can import these modules or keep a copy of them. If the service keeps a copy, copy
the files again after each change in this repository.

```python
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bioparser_ft.infer import generate_tsv, load_for_inference
from bioparser_ft.postprocess import to_ets_records

app = FastAPI()
# Each worker process loads its own copy of the model. Use one worker for each GPU.
model, processor = load_for_inference("outputs/qwen3.5-0.8b-ets/lora_adapter")
gpu = threading.Lock()  # One generation at a time on one GPU. For more throughput, use vLLM.
MAX_MARKDOWN_CHARS = 400_000  # Approximately 100,000 tokens: more than the context of the model


class ParseRequest(BaseModel):
    # MinerU output for one PDF file. Use the MinerU version of the training data.
    markdown: str = Field(max_length=MAX_MARKDOWN_CHARS)
    references: str | None = Field(default=None, max_length=2_000)  # A correct Harvard citation


@app.post("/parse")
def parse(req: ParseRequest) -> dict:  # A sync function: FastAPI runs it in a worker thread.
    # Do not wait for the GPU without a limit. Each request that waits keeps one of the 40 threads.
    if not gpu.acquire(timeout=1.0):
        raise HTTPException(503, "The GPU is busy. Try again later.", headers={"Retry-After": "60"})
    try:
        gen = generate_tsv(model, processor, req.markdown, max_new_tokens=8192, max_seq_length=32768)
    finally:
        gpu.release()
    if gen.new_tokens == 0 and not gen.complete:  # The prompt is longer than max_seq_length
        raise HTTPException(413, "The article is too long for the model.")
    result = to_ets_records(gen.text, references=req.references, complete=gen.complete)
    # {"records": [...20 ETS columns...], "warnings": [...], "complete": bool}
    return {"records": result.records, "warnings": result.warnings, "complete": result.complete}
```

**CAUTION:** Set the read timeout of the client and of the reverse proxy to more than the longest
generation. One generation can take minutes on a small GPU. If the client sends the request again,
the GPU does the same work two times.

For vLLM, save a merged model with `bioparser-train --merge-16bit`. Then send a prompt from
`bioparser_ft.chat.render_prompt` to a raw completion endpoint. This prompt is the same as the
prompt of the training.

**CAUTION:** Do not use the vLLM chat endpoint before you compare its prompt tokens with the tokens
of `render_prompt`. Some chat templates ignore `enable_thinking: false`.

## Data protection

**CAUTION:** Do not commit client data to git. The client TSV files, the PDF files, the parsed
text, the datasets and the model weights are confidential. The file `.gitignore` and the Git hooks
stop these files.

**CAUTION:** Do not upload the model weights to a public repository. Model weights can show the
client data again. The option `--push-to-hub` stops with an error if the repository is public.

**CAUTION:** Do not use the MinerU option `--remote`. This option sends the PDF files to
mineru.net. The script `scripts/parse_pdfs.sh` parses the files on your computer.

Keep the Hugging Face token in `$HF_TOKEN`, or use `hf auth login`. Do not write the token in the
code.

## Tests and checks

Run the tests:

```bash
uv run pytest
```

Run the tests with the client files:

```bash
BIOPARSER_ETS_DIR=fine_tuning_data/ets_tsv uv run pytest tests/test_real_data.py -s
```

Install the Git hooks one time:

```bash
uv run pre-commit install
```

Before each commit, the hooks run ruff and mypy, and they stop client data, model weights and
tokens. CI runs the same checks and the tests. CI has no GPU. For this reason, test changes to the
training code on a computer with a GPU.
