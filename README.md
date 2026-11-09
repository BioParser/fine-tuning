# fine-tuning

This repository fine-tunes the model `Qwen/Qwen3.5-0.8B` for
[BioParser](https://github.com/BioParser/BioParser) with Unsloth and LoRA.

The model reads a scientific article that MinerU parsed from a PDF file. Then the model writes
the trait data of the article as a TSV table in the client ETS format.

```
PDF --MinerU--> markdown --clean_markdown--> chat prompt --Qwen3.5+LoRA--> TSV --to_ets_records--> JSON --> FastAPI
             one training example = (parsed PDF, client ETS TSV)
```

**NOTE:** Run all commands in the root folder of the repository, not in a subfolder. The
commands use paths that start in the root folder.

## Requirements

- An NVIDIA GPU. The smallest GPU that we tested is an RTX 2060 Max-Q with 6 GB of VRAM.
- An NVIDIA driver for CUDA 13.0, that is, the driver version 580 or newer. `uv.lock` contains
  torch 2.10 for CUDA 13.0.
- [uv](https://docs.astral.sh/uv/). uv installs Python 3.13 and all packages.

## Installation

Do these steps one time:

```bash
# Install the packages of uv.lock.
uv sync

# Install the CUDA kernels of causal_conv1d for the Qwen3.5 layers.
uv run unsloth install-kernels causal_conv1d

# Install the browser for bioparser-fetch-pdfs.
uv run playwright install chromium

# Download the MinerU models for scripts/parse_pdfs.sh.
uvx --from "mineru==4.1.6" mineru-models-download --tier basic
```

**NOTE:** `uv sync` removes `causal_conv1d`, because `uv.lock` does not contain it. After each
`uv sync`, install the kernels again. Or use `uv sync --inexact`, which does not remove packages.

If you install `causal_conv1d`, Unsloth uses it. Without it, Unsloth uses its own Triton
kernels. If no prebuilt wheel agrees with your system, `install-kernels` compiles the kernels.
This can take a long time.

## Quickstart

Before you start:

1. Put the client ETS files in `fine_tuning_data/ets_tsv/`.
2. In step 1, write your email address in place of `you@helsinki.fi`. The scholarly APIs ask for
   an email address. The command sends it only to these APIs, not to the publishers.

**CAUTION:** Do step 1 outside the network of the university. On the university network, the
automatic parts of step 1 also download articles that are not open access. The terms of the
library do not let programs do this. Refer to
[Library access with EZproxy](fine_tuning_data/README.md#library-access-with-ezproxy).

```bash
# 1. Get the PDF files. The command uses open-access sources first, then a headless browser.
#    At the end, it opens a visible browser with the library proxy, and you get the other files.
uv run bioparser-fetch-pdfs --email you@helsinki.fi --browser --manual \
    --proxy-prefix "https://login.libproxy.helsinki.fi/login?url="

# 1b. Optional: link the PDF files that you downloaded yourself. Put them in
#     fine_tuning_data/pdfs/_unverified/ first. Without --apply, the command only shows the result.
uv run bioparser-link-pdfs --apply

# 2. Parse the PDF files with MinerU.
scripts/parse_pdfs.sh

# 3. Merge the parsed PDF files with the ETS files, and make the dataset.
uv run bioparser-build-dataset

# 4. Fine-tune the model on the GPU.
uv run bioparser-train

# 5. Optional: use the model on the test articles, and calculate the scores.
uv run bioparser-predict --model outputs/qwen3.5-0.8b-ets/lora_adapter \
    --dataset data/processed/test.jsonl --out outputs/preds_test.jsonl
uv run bioparser-evaluate --dataset data/processed/test.jsonl --predictions outputs/preds_test.jsonl
```

| Step | Result | Examine | More information |
|---|---|---|---|
| 1 | `fine_tuning_data/pdfs/`, `pdf_sources.tsv` and `manifest.tsv` | The papers without a PDF file | [Step 1](fine_tuning_data/README.md#step-1-get-the-pdf-files) |
| 2 | `fine_tuning_data/parsed/*.zip` | The error messages of MinerU | [Step 2](fine_tuning_data/README.md#step-2-parse-the-pdf-files) |
| 3 | `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl` and `build_report.json` | The pairs that the command did not use | [Step 3](fine_tuning_data/README.md#step-3-make-the-dataset) |
| 4 | `outputs/qwen3.5-0.8b-ets/lora_adapter/` | The training loss and the eval loss | [Training](#training) |
| 5 | `outputs/preds_test.jsonl` and the scores | The precision, the recall and the F1 score for rows, cells and values | |

- The last part of step 1 waits for you. For each article, get the PDF file in the browser
  window, or push Enter to go to the next article.
- The first run of step 4 downloads the model `Qwen/Qwen3.5-0.8B` from Hugging Face.
- You can use the notebook `notebooks/Qwen3_5_0_8B.ipynb` in place of step 4. Start JupyterLab
  with `uv run jupyter lab`. The notebook uses the same functions as the commands.
- When you do a step again, also do all the steps after it.

## Training

The command `uv run bioparser-train` trains a LoRA adapter with Unsloth. It saves the adapter in
`outputs/qwen3.5-0.8b-ets/lora_adapter/`.

| Option | Default | Description |
|---|---|---|
| `--max-seq-length N` | 65536 | The maximum number of tokens for the prompt, the article and the TSV table |
| `--tiled-mlp`, `--no-tiled-mlp` | On | Tiled MLP uses less GPU memory for long sequences. On a GPU with much memory, use `--no-tiled-mlp`. |
| `--model-revision HASH` | The newest commit | The commit of the base model on Hugging Face. Use it to train on the same weights in each run. |
| `--resume` | Off | Continue an interrupted training from the last checkpoint in `outputs/qwen3.5-0.8b-ets/checkpoints/`. Use the same dataset and options. |
| `--merge-16bit` | Off | Also save a merged 16-bit model for vLLM |
| `--push-to-hub REPO` | Off | Upload the adapter to a private Hugging Face repository |

For all options, run `uv run bioparser-train --help`.

GPU memory:

- GPUs without bf16, for example the RTX 20xx and the T4, train Qwen3.5 in float32.
- When the VRAM is too small, Unsloth moves data to the CPU RAM. The training command sets these
  options.
- The command removes the examples that are longer than `--max-seq-length`. It does not cut an
  example, because a cut removes the answer.

## Output format

The file `src/bioparser_ft/schema.py` sets the output format. The training, the inference and the
API use this file.

- The model writes 19 of the 20 ETS columns. The function `to_ets_records` adds the column
  `author` with the value `BioParser`.
- The model writes the citation, the column `references`, only on the first row. `to_ets_records` copies it
  to the other rows. If the API has a correct citation, give it with `references=`.
- `to_ets_records` also uses fixed rules. It finds an empty `taxonRank` from the name. It changes
  forms such as `males` to `Male`. It writes the full genus name for an abbreviation such as
  `H. sapiens`, if only one genus in the output starts with this letter.

**CAUTION:** If you change the columns, the prompt or the rules that clean the text, increase
`PROMPT_VERSION` in `src/bioparser_ft/schema.py`. Then make the dataset again and train the model again. The tools
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

The example uses 32,768 tokens. The training uses examples with up to 65,536 tokens. For longer
articles, give a larger `max_seq_length` to `load_for_inference` and to `generate_tsv`, if the
GPU has enough memory.

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

Before each commit, the hooks run ruff and mypy. They also stop client data, model weights,
tokens and notebooks with outputs. CI runs the same checks and the tests. CI has no GPU. For this
reason, test changes to the training code on a computer with a GPU.
