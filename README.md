# fine-tuning

This repository contains the tools to fine-tune the model `Qwen/Qwen3.5-0.8B` for
[BioParser](https://github.com/BioParser/BioParser). The tools use Unsloth and LoRA.

The model reads a scientific article that MinerU parsed from a PDF file. Then the model writes
the trait data of the article as a TSV table in the client ETS format. The FastAPI service
changes the TSV table to JSON with the function `bioparser_ft.postprocess.to_ets_records`.

```
PDF --MinerU--> markdown --clean_markdown--> chat prompt --Qwen3.5+LoRA--> TSV --to_ets_records--> JSON --> FastAPI
             one training example = (parsed PDF, client ETS TSV)
```

## Installation

1. Install [uv](https://docs.astral.sh/uv/).
2. Go to the repository folder.
3. Run this command:

   ```bash
   uv sync
   ```

The command `uv sync` installs the packages from `uv.lock`. It also installs the package
`bioparser_ft` of this repository in editable mode.

The file `uv.lock` contains the package versions that we tested on the RTX 2060 GPU
with 6 GB of VRAM:

- Unsloth 2026.11.1
- transformers 5.17.0 and TRL 0.22.2. The RTX 2060 run used transformers 5.2.0. Version 5.17.0
  fixes 4 security advisories. Test it on the GPU before you merge a change to the stack.
- torch 2.14.1+cu130, Triton 3.8.0 and xformers 0.0.35
- Python 3.13

**CAUTION:** Do not use `pip install` in the `.venv` environment. Packages from pip are not in
`uv.lock`. Then uv cannot make the same environment again.

To add or change a package, do the steps that follow:

1. Change `pyproject.toml`.
2. Run `uv lock`.
3. Run `uv sync`.

**NOTE:** The command `unsloth install-kernels` is not necessary for Qwen3.5. Unsloth contains
the Gated-DeltaNet Triton kernels. Kernel packages from other sources are not in `uv.lock`, and
`uv sync` removes them. The command `uv sync --inexact` keeps them.

## Get the PDF files

The command `bioparser-fetch-pdfs` downloads the article PDF for each client TSV file. Most
client citations do not contain a DOI. For this reason, the command first finds the DOI from the
citation text. Then it finds a PDF file for the DOI.

The command uses these sources, in this sequence:

1. Open-access indexes: Unpaywall, OpenAlex, Europe PMC, Semantic Scholar, DataCite and CORE
2. The PDF links on the landing page of the publisher
3. With the option `--browser`: a headless browser that clicks the PDF link
4. With the option `--manual`: a visible browser. You click the link or log in, and the command
   saves the PDF file. Only this option uses the library proxy of `--proxy-prefix`.

The command makes sure that each PDF file contains the title or the DOI of the article. It works
on 6 papers at the same time, and each host keeps its own rate limit. It keeps the API answers in
`fine_tuning_data/.fetch_cache/`, so a second run is fast. For the meaning of the output and the
speed options, refer to [The output of the command](fine_tuning_data/README.md#the-output-of-the-command).

**CAUTION:** Do these steps outside the network of the university. On the university network,
the automatic steps also download articles that are not open access. The terms of the library do
not let programs do this. For more information, refer to
[Library access with EZproxy](fine_tuning_data/README.md#library-access-with-ezproxy).

To get the PDF files, do the steps that follow:

1. Find the DOIs only, and examine the matches in `fine_tuning_data/pdf_sources.tsv`:

   ```bash
   uv run bioparser-fetch-pdfs --email you@helsinki.fi --resolve-only
   ```

2. Download the PDF files:

   ```bash
   uv run bioparser-fetch-pdfs --email you@helsinki.fi
   ```

3. Install the browser for the options `--browser` and `--manual`. Do this step one time only:

   ```bash
   uv run playwright install chromium
   ```

4. Download the PDF files from pages that use JavaScript:

   ```bash
   uv run bioparser-fetch-pdfs --email you@helsinki.fi --browser
   ```

5. Get the other PDF files manually. To use the access of the University of Helsinki, add the
   EZproxy prefix of the library:

   ```bash
   uv run bioparser-fetch-pdfs --email you@helsinki.fi --manual \
       --proxy-prefix "https://login.libproxy.helsinki.fi/login?url="
   ```

6. If you downloaded PDF files yourself, put them in `fine_tuning_data/pdfs/_unverified/`. Then
   link them to the papers. The command reads each file, and it moves the files that agree with
   one paper to `fine_tuning_data/pdfs/`. Without `--apply`, it only shows the result:

   ```bash
   uv run bioparser-link-pdfs --apply
   ```

   For more information, refer to
   [Link the PDF files that you downloaded yourself](fine_tuning_data/README.md#link-the-pdf-files-that-you-downloaded-yourself).

The command writes these results:

- `fine_tuning_data/pdfs/`: the PDF files that contain the title or the DOI of the article
- `fine_tuning_data/pdfs/_unverified/`: PDF files without the title or the DOI. Examine them
  manually.
- `fine_tuning_data/pdf_sources.tsv`: one row for each article, with the status, the DOI, the
  source and the SHA-256 hash
- `fine_tuning_data/manifest.tsv`: new rows for articles with more than one TSV file

If a DOI is incorrect or not found, add a row to `fine_tuning_data/doi_overrides.tsv`. This file is
tab-separated and has the columns `tsv`, `doi`, `url` and `note`. Then run the command again.

**CAUTION:** Keep the folder `fine_tuning_data/.browser_profile/` private. It contains the cookies
of your library login. When you have all the PDF files, delete the folder.

**NOTE:** The command does not use shadow libraries. It does not go around paywalls or CAPTCHAs.
For articles that are not open access, other sources are your library, an interlibrary loan, the
authors and the client. The client has the PDF files that they used for the labels.

## From client data to a trained model

For a short list of all commands, refer to the Quickstart in
[fine_tuning_data/README.md](fine_tuning_data/README.md#quickstart).

1. Put the client data in the folder `fine_tuning_data/`, as
   [fine_tuning_data/README.md](fine_tuning_data/README.md) shows.
2. Get the PDF files with `bioparser-fetch-pdfs`, as the section above shows.
3. Parse the PDF files with the MinerU version of production:

   ```bash
   scripts/parse_pdfs.sh
   ```

   The script writes one zip file for each PDF file to `fine_tuning_data/parsed/`.
4. Make the dataset:

   ```bash
   uv run bioparser-build-dataset
   ```

   The command writes `train.jsonl`, `val.jsonl` and `test.jsonl` to `data/processed/`. It also
   examines each pair of files and writes the results to `build_report.json`.
5. Train the model with this command:

   ```bash
   uv run bioparser-train
   ```

   You can also use the notebook. Start JupyterLab with `uv run jupyter lab`. Then open
   `notebooks/Qwen3_5_0_8B.ipynb`. The command and the notebook use the same functions in
   `src/bioparser_ft/`. The tools save the adapter in `outputs/qwen3.5-0.8b-ets/lora_adapter/`.
6. Use the adapter on the test articles:

   ```bash
   uv run bioparser-predict --model outputs/qwen3.5-0.8b-ets/lora_adapter --dataset data/processed/test.jsonl --out outputs/preds_test.jsonl
   ```

7. Calculate the scores:

   ```bash
   uv run bioparser-evaluate --dataset data/processed/test.jsonl --predictions outputs/preds_test.jsonl
   ```

   The command gives the precision, the recall and the F1 score for rows, cells and values.

## Output format

The file `src/bioparser_ft/schema.py` contains the output format. The training tools, the
inference tools and the API all use this file.

- **19 columns.** The model writes all ETS columns, but not the column `author`.
- **Column `author`.** This column always contains `BioParser`. This is a client rule. The
  function `to_ets_records` adds the value. The client labels contain the ORCID of the curator in
  this column. For this reason, the training data does not contain the column `author`.
- **Column `references`.** This column contains the Harvard citation from Google Scholar. The
  client added this citation. It is not text from the PDF file. The model writes the citation
  only on the first data row. The function `to_ets_records` copies it to the other rows.
- **Correct citation from the API.** If the API has a correct citation, give it to
  `to_ets_records` with the parameter `references=`.
- **Shorter output.** A citation on each row adds approximately 60 tokens to each row. With only
  one citation and without the column `author`, the output for the client data is approximately
  59% shorter.
- **Model TSV format.** Each line contains one record. The format has no quotation marks. A cell
  never contains a TAB character or a line break.
- **Client TSV files.** The client files are Excel exports. Some cells in these files have
  quotation marks, TAB characters and line breaks. The function `tsv.read_ets_tsv` reads these
  cells correctly. It joins the lines of a cell with ` | `, the list separator of Darwin Core.
- **Fixed rules after the model.** The function `to_ets_records` also uses fixed rules, because
  these rules are more accurate than a small model:
  - If `taxonRank` is empty, the function finds the rank from the name. The rule comes from the
    client. It gives the same result as the client labels for 99.98% of the records.
  - The function changes `H. sapiens` to `Homo sapiens` when the other rows show the full genus
    name.
  - The function changes different forms of a value to one form, for example `males` to `Male`.

### Changes to the output format

If you change the columns, the prompt or the normalization, do the steps that follow:

1. Increase the value of `PROMPT_VERSION` in `src/bioparser_ft/schema.py`.
2. Make the dataset again.
3. Train the model again.

The tools stop with an error if the dataset has an old `PROMPT_VERSION`.

## FastAPI integration

The modules `schema`, `markdown`, `tsv` and `postprocess` use only the Python standard library.
The FastAPI service can import these modules or keep a copy of them. If the service keeps a copy,
copy the files again after each change in this repository.

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

**CAUTION:** One generation can take minutes on a small GPU. Set the read timeout of the client and
of the reverse proxy to more than the longest generation. If not, the client sends the request
again, and the GPU does the same work two times.

For vLLM, export a merged 16-bit model with `bioparser-train --merge-16bit`. Then use one of the
two methods that follow:

- Send a prompt from `bioparser_ft.chat.render_prompt` to a raw completion endpoint. This method
  gives the same prompt text as the training. Use this method if you are not sure.
- Use the chat endpoint with `chat_template_kwargs={"enable_thinking": false}`. Some versions of
  the chat template ignore this flag, or write no empty think block. `render_prompt` corrects
  this, but the vLLM chat endpoint does not. Before you use this method, compare the prompt tokens
  of vLLM (for example, from its `/tokenize` endpoint) with the tokens of `render_prompt`.

## GPU notes: RTX 2060 Max-Q with 6 GB
> [!NOTE] 
> The RTX 2060 Max-Q with 6 GB is a low-end Turing GPU. We use it to test the minimum requirements.
- **Data type.** Turing GPUs do not have bf16. On these GPUs, Unsloth trains Qwen3.5 in float32,
  because the Gated-DeltaNet layers give NaN gradients in float16. Unsloth does not recommend
  4-bit QLoRA for Qwen3.5.
- **CPU RAM offload.** When the VRAM is too small, Unsloth moves data to the CPU RAM:
  - `use_gradient_checkpointing="unsloth"` moves the checkpointed activations to the CPU RAM.
  - `offload_embedding="auto"` moves the input embedding for the 248K-token vocabulary to the
    CPU RAM.
  - The function `train.load_model` sets the two options. It also gives `max_seq_length` to the
    loader, because Unsloth calculates the offload from this value.
- **Sequence length.** The default value of `max_seq_length` is 16,384 tokens. Typical examples
  have approximately 6,000 to 15,000 tokens:
  - System prompt: approximately 770 tokens
  - Article: 4,000 to 12,000 tokens
  - TSV table: 1,000 to 6,000 tokens

  The tools remove examples that are longer than the limit. They never cut an example, because a
  cut removes the answer. For 32,768 tokens, also use the option `--tiled-mlp`.
- **LoRA modules.** The LoRA adapters are on these modules:
  - `in_proj_qkv`, `in_proj_z` and `out_proj` in the 18 Gated-DeltaNet layers
  - `q_proj`, `k_proj`, `v_proj` and `o_proj` in the 6 full-attention layers
  - `gate_proj`, `up_proj` and `down_proj` in the MLP of all 24 layers

## Data protection and security

**CAUTION:** Do not commit client data to git. The client TSV files, the PDF files, the parsed
text, the datasets and the model weights are confidential. Git does not add these files, because
the file `.gitignore` contains them.

**CAUTION:** Do not upload model weights to a public repository. Model weights from client data
can show this data again. The option `--push-to-hub` uploads only to private repositories. If the
repository is public, the option stops with an error.

**CAUTION:** Do not use the MinerU option `--remote` with client PDF files. This option sends the
PDF files to mineru.net. The script `scripts/parse_pdfs.sh` uses `--tier basic`, which operates
only on your computer.

- Keep the Hugging Face token in `$HF_TOKEN`, or use `hf auth login`. Do not write the token in
  the code.
- Parsed zip files can contain dangerous content. For this reason, the tools read only
  `markdown.md`, in memory and with a size limit. The tools do not extract files to the disk.
- Two Dependabot alerts have no fix that agrees with Unsloth 2026.11.1, the newest version.
  Unsloth needs `datasets<5`, and `datasets` 4.8.5 needs `fsspec<=2026.2.0`. The code does not
  use the vulnerable functions:
  - `datasets` (GHSA-379c-qx7v-6h59): the folder-based dataset builders. The code uses only
    `Dataset.from_list`.
  - `fsspec` (GHSA-27vj-qcqg-25rc): the `ReferenceFileSystem` (URLs that start with
    `reference://`). The code does not use fsspec directly.

  When Unsloth accepts `datasets>=5.0.1`, update `datasets` and `fsspec`.
- Commit notebooks without outputs. The test `tests/test_notebook_clean.py` makes sure that the
  notebooks have no outputs.
- The Git hooks and CI stop a commit with client data, PDF files, model weights or tokens. For
  more information, refer to [Checks before a commit and in CI](#checks-before-a-commit-and-in-ci).

## Tests

1. To run the tests with the test data of the repository, use this command:

   ```bash
   uv run pytest
   ```

2. To also run the tests with the real client files, use this command:

   ```bash
   BIOPARSER_ETS_DIR=fine_tuning_data/ets_tsv uv run pytest tests/test_real_data.py -s
   ```

The tests with client files make sure that the tools read all 181 TSV files with 50,060 records.
They also make sure that each record goes through the model TSV format without changes. They
also compare the `taxonRank` rule with the client labels.

## Checks before a commit and in CI

The checks are the same as in the [bioparser](https://github.com/bioparser/bioparser) repository.
Install the Git hooks one time:

```bash
uv run pre-commit install
```

Before each commit, the hooks do these checks:

- `uv-lock` makes sure that `uv.lock` agrees with `pyproject.toml`.
- `ruff check`, `ruff format` and `mypy` examine the Python files. mypy uses the strict mode.
- The notebooks must have no outputs.
- Git must not get client data, PDF files, model weights, Hugging Face tokens or private keys.

To do all checks on all files, use this command:

```bash
uv run pre-commit run --all-files
```

The GitHub Actions workflow `.github/workflows/ci.yml` does these checks for each pull request
and for each push to `main`:

| Job | Checks |
|---|---|
| Lint and format | `ruff check`, `ruff format --check`, and the checks for client data and tokens |
| Type check | `mypy` in strict mode |
| Unit tests | `pytest` on Ubuntu x64, Ubuntu ARM and Windows. Only an error on Ubuntu x64 stops the run. |
| Browser tests | The tests of `bioparser-fetch-pdfs` with Chromium |

CI does not install the GPU training stack, for example torch and Unsloth. The GitHub runners
have no GPU, and the stack uses many GB of disk space. For this reason, CI installs only the groups `dev` and `fetch`, and
the tests import the package from `src/`. Test changes to the training code on a computer with a
GPU.

Dependabot updates the GitHub Actions, the hooks and the Python packages each week. It does not
update the GPU training stack, because CI cannot test it. CI uses uv 0.13.4. To get the same
version or a newer one, use `uv self update`.
