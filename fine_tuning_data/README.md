# fine_tuning_data/

This folder contains the client data. **Git does not track the files in this folder. It
tracks only this README and `manifest.example.tsv`.** The files are confidential client data
and copyrighted articles. Do not commit them.

## Quickstart

Do these steps in the root folder of the repository. Each step uses the results of the step
before it. The commands use the default paths. For this reason, no more options are necessary.

Before you start:

- Put the client ETS files in `fine_tuning_data/ets_tsv/`.
- In step 1, write your email address in place of `you@helsinki.fi`. The scholarly APIs ask
  for an email address. The command sends it only to these APIs, not to the publishers.

**CAUTION:** Do step 1 outside the network of the university. On the university network, the
automatic parts of step 1 also download articles that are not open access. The terms of the
library do not let programs do this. For more information, refer to
[Library access with EZproxy](#library-access-with-ezproxy).

```bash
# 0. Install the packages, the browser and the MinerU models. Do this step one time only.
uv sync
uv run playwright install chromium
uvx --from "mineru==4.1.6" mineru-models-download --tier basic

# 1. Get the PDF files. The command uses open-access sources first, then a headless browser.
#    At the end, it opens a visible browser with the library proxy, and you get the other files.
uv run bioparser-fetch-pdfs --email you@helsinki.fi --browser --manual \
    --proxy-prefix "https://login.libproxy.helsinki.fi/login?url="

# 1b. Optional: if you downloaded PDF files yourself, put them in fine_tuning_data/pdfs/_unverified/.
#     Then link them to the papers. Without --apply, the command only shows the result.
uv run bioparser-link-pdfs --apply

# 2. Parse the PDF files with MinerU.
scripts/parse_pdfs.sh

# 3. Merge the parsed PDF files with the ETS files, and make the dataset.
uv run bioparser-build-dataset

# 4. Fine-tune the model on the GPU.
uv run bioparser-train

# 5. Optional: use the model on the test articles, and calculate the scores.
uv run bioparser-predict --model outputs/qwen3.5-0.8b-ets/lora_adapter --dataset data/processed/test.jsonl --out outputs/preds_test.jsonl
uv run bioparser-evaluate --dataset data/processed/test.jsonl --predictions outputs/preds_test.jsonl
```

| Step | Result | Examine |
|---|---|---|
| 1 | `fine_tuning_data/pdfs/`, `pdf_sources.tsv` and `manifest.tsv` | The rows with `not_found`, `unverified` or `unresolved` in `pdf_sources.tsv` |
| 2 | `fine_tuning_data/parsed/*.zip` | The error messages of MinerU |
| 3 | `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl` and `build_report.json` | The excluded pairs in `build_report.json` |
| 4 | `outputs/qwen3.5-0.8b-ets/lora_adapter/` | The training loss and the eval loss |
| 5 | `outputs/preds_test.jsonl` and the scores | The F1 values for rows, cells and values |

- The last part of step 1 waits for you. For each article, get the PDF file in the browser
  window, or push Enter to go to the next article. The time limit for each article is 300
  seconds. To change it, use `--manual-timeout`.
- Step 1 writes the papers with more than one TSV file to `manifest.tsv`. Step 3 uses this file
  automatically.
- Step 4 uses the GPU. The first run downloads the model `Qwen/Qwen3.5-0.8B` from Hugging Face.
- You can use the notebook in place of step 4. Start JupyterLab with `uv run jupyter lab`, and
  then open `notebooks/Qwen3_5_0_8B.ipynb`.
- When you do a step again, also do all the steps after it.

The sections below give more information about each step.

## Folder layout

```
fine_tuning_data/
├── ets_tsv/            the client ETS files, for example "ETS_Abdel-Hamid and Alqahtani, 2022.tsv"
├── pdfs/               the articles, for example "Abdel-Hamid and Alqahtani, 2022.pdf"
│   └── _unverified/    PDF files without the title or the DOI, and your own downloads: bioparser-link-pdfs
├── parsed/             the MinerU output: one "<pdf-stem>.zip" or ".md" file for each PDF file
├── manifest.tsv        pairs with different file names (the fetch command adds rows)
├── doi_overrides.tsv   optional: your corrections of DOI matches (columns: tsv, doi, url, note)
├── pdf_sources.tsv     the report of the fetch command: status, DOI, source URL and sha256
├── .fetch_cache/       the API answers in a cache, so a second run is fast
└── .browser_profile/   the browser profile of --manual. It contains your university login: keep it private.
```

## Step 1. Get the PDF files

Step 1 gets one PDF file for each paper. The command of the Quickstart does all of step 1 in
one run. You can also do the parts one at a time. Each part uses the results of the part
before it.

**CAUTION:** Do step 1 outside the network of the university. Refer to
[Library access with EZproxy](#library-access-with-ezproxy).

```bash
# a. Find a DOI for each citation, but do not download the files.
uv run bioparser-fetch-pdfs --email you@helsinki.fi --resolve-only

# b. Download the free copies from the APIs and from the pages of the publishers.
uv run bioparser-fetch-pdfs --email you@helsinki.fi

# c. Also use a headless browser for the pages that use JavaScript.
uv run bioparser-fetch-pdfs --email you@helsinki.fi --browser

# d. Get the other files yourself in a visible browser, through the library proxy.
uv run bioparser-fetch-pdfs --email you@helsinki.fi --manual \
    --proxy-prefix "https://login.libproxy.helsinki.fi/login?url="
```

A second run is fast, because the command keeps the API answers and the PDF files.

### After each run

1. In `pdf_sources.tsv`, examine the rows with the confidence `medium`. If a DOI is not correct,
   write the correct DOI in `doi_overrides.tsv`. Then do the run again.
2. For each `unresolved` paper, find the DOI or a URL yourself. Write it in `doi_overrides.tsv`.
3. Examine the files in `pdfs/_unverified/`. Then link them to the papers. Refer to
   [Link the PDF files that you downloaded yourself](#link-the-pdf-files-that-you-downloaded-yourself).

`doi_overrides.tsv` is tab-separated. Write a DOI or a URL in each row:

```
tsv	doi	url	note
ETS_Example et al., 2020.tsv	10.1000/example		found manually
```

Two citations of the client have problems:

- The citation of `Ancillotto et al., 2023` is not correct. Add a row to `doi_overrides.tsv`.
- `Morenikeji et al., 2022` and `Morenikeji et al., 2023` have the same citation. Ask the client
  if they are one article.

### How the command finds a PDF file

Only 2 of the 175 citations contain a DOI. For this reason, the command first finds the DOI. It
searches Crossref, OpenAlex, Semantic Scholar, DataCite and CORE. It compares each result with
the title, the year and the first author of the citation. Then it gets the PDF file from the
open-access indexes or from the page of the publisher.

Each PDF file must contain the title or the DOI of the paper. If the file does not contain them,
the command puts it in `pdfs/_unverified/`. For the papers with more than one TSV file, the
command adds rows to `manifest.tsv`. Step 3 uses these rows.

### The output of the command

The command writes one line for each paper:

```
[1/175] not_found  high   doi:crossref        pdf:-                 oa:closed  Example et al., 2020
```

The line shows the status, the DOI confidence, the DOI source, the PDF source and the
open-access status. The command works on many papers at the same time. For this reason, the
sequence of the lines can change.

| Status | Description | What to do |
|---|---|---|
| `downloaded` | The PDF file is in `pdfs/`, and it contains the title or the DOI. | Nothing |
| `exists` | The PDF file is in `pdfs/` from an earlier run. | Nothing |
| `unverified` | The PDF file does not contain the title or the DOI. It is in `pdfs/_unverified/`. | Examine it. Then use `bioparser-link-pdfs`. |
| `not_found` and `oa:closed` | There is no free copy. | Get it in part d. |
| `not_found` and a different `oa` | A free copy exists, but the site stopped the download. | Part c tries again. Then use part d. |
| `unresolved` | No DOI agrees with the citation. | Write the DOI in `doi_overrides.tsv`. |

The confidence of the DOI match:

- `high`: the title, the year and the first author agree.
- `medium`: most of the title agrees, and the year or the first author agrees. Examine these
  matches.
- `none`: no search result agrees with the citation.
- `given`: the DOI comes from the citation or from `doi_overrides.tsv`.

The exact rules are in `src/bioparser_ft/fetch/matching.py`.

The open-access status:

- `gold`, `diamond`, `hybrid` or `bronze`: the publisher gives a free copy.
- `green`: a repository has a free copy.
- `closed`: there is no free copy.
- `-`: there is no information.

### Library access with EZproxy

Many articles are not open access. The University of Helsinki pays for access to them. The
publishers give this access to the IP addresses of the university. EZproxy is the proxy server
of the library. You log in to EZproxy with your University of Helsinki account. Then EZproxy gets
the articles for you from an IP address of the university.

This is the EZproxy prefix of the University of Helsinki:

```
https://login.libproxy.helsinki.fi/login?url=
```

Put the prefix in front of the URL of an article, for example:

```
https://login.libproxy.helsinki.fi/login?url=https://doi.org/10.1000/example
```

**CAUTION:** The library does not let programs download articles. If a user does not obey this
rule, the publisher can block the access for all users of the University of Helsinki. Obey these
rules:

- Do step 1 outside the network of the university. Do not use the campus network, eduroam on
  the campus or the university VPN with the allroute option. On these networks, the automatic
  parts b and c also get the articles that the university pays for.
- Use only your own user account. Do not save your password in the browser of the command.
- Keep the folder `.browser_profile/` private, because it contains your login. Delete it after
  step 1.
- If you are not sure that the license lets you use the articles as training data, ask the
  library.

Only part d uses the proxy and your login. The headless browser of part c starts with an empty
profile, so it does not have your login.

Part d operates like this:

1. The command opens the next article in a visible browser, through the proxy. If the command
   did not find the article, it opens a Google Scholar search.
2. If EZproxy shows the login page, log in. The browser keeps the login for the next articles.
3. Click "PDF" or "Download PDF" on the page of the publisher. The command saves the file with
   the correct name.
4. If you cannot get the article, push Enter in the terminal. The command goes to the next
   article. The time limit for each article is 300 seconds.

| Problem | What to do |
|---|---|
| EZproxy shows an error page. The library has no proxy access to this publisher. | Push Enter, and use a different source. |
| The publisher asks you to pay. The library does not pay for this journal. | Push Enter, and use a different source. |
| The prefix does not operate. The library changed it. | Find the new prefix on the page "Remote access to electronic resources" of Helsinki University Library. |

Other sources for an article are the client, an interlibrary loan from Helsinki University
Library, and the authors. The client has the PDF files that they used for the labels.

### Link the PDF files that you downloaded yourself

A file from EZproxy or from a browser can have any name, for example `gyac068.pdf`. Put these
files in `pdfs/_unverified/`. Then use these commands:

```bash
# Show the result for each file. No file moves.
uv run bioparser-link-pdfs

# Move the files that agree with one paper to pdfs/<paper>.pdf.
uv run bioparser-link-pdfs --apply
```

The command compares the first pages and the name of each file with each paper: the DOI, the
title, the first author and the year. It also examines the PDF files in `pdfs/` that do not have
the name of a paper. It never replaces a file in `pdfs/`.

| Result | Description |
|---|---|
| `MOVE` | The file agrees with one paper that has no PDF file. |
| `REVIEW` | No paper agrees enough, or two papers agree. The command shows the best guess. |
| `DUPLICATE` | The paper has a PDF file, or a different file agrees better with the paper. |
| `SKIP` | The file is not a PDF file, or pypdf cannot read it. |

Examine each `REVIEW` file. If the best guess is correct, link the file yourself:

```bash
uv run bioparser-link-pdfs --apply --link "67640adb47e5f1734609627.pdf=Smith et al., 2023"
```

Sometimes two papers agree with one file, for example `Froidevaux et al., 2023 - 02` and
`Jérémy et al., 2023_02`. These are possibly one article with two citations. Link the file to one
paper. Then add a row to `manifest.tsv` for the TSV files of the other paper.

### Options

Each host keeps its own rate limit, so the command does not send too many requests to one
server. The scholarly APIs get 5 requests each second or less.

| Option | Default | Description |
|---|---|---|
| `--workers` | 6 | The number of papers at the same time |
| `--browser-workers` | 3 | The number of headless browsers in part c. On a slow computer, use 1. |
| `--delay` | 1.0 | The time in seconds between two requests to the same publisher |
| `--manual-timeout` | 300 | The time limit in seconds for each article in part d |
| `--no-browser-sandbox` | off | Start Chromium without its sandbox. Use it only if Chromium cannot start. |

## Step 2. Parse the PDF files

```bash
scripts/parse_pdfs.sh fine_tuning_data/pdfs fine_tuning_data/parsed
```

The two folders are the default values. Without arguments, `scripts/parse_pdfs.sh` uses the
same folders. The script runs `mineru-kit parse ... --format zip --tier basic` with MinerU
4.1.6 in a separate `uvx` environment. MinerU parses the files on your computer.

**CAUTION:** Do not add the option `--remote` to MinerU. This option sends the client PDF files
to the servers of MinerU.

Set `MINERU_VERSION` to the MinerU version of the production system. The model learns the
Markdown format of this version. Before the first parse, download the MinerU models for the same
version. Do this one time only:

```bash
uvx --from "mineru==4.1.6" mineru-models-download --tier basic
```

The old notebook had three manual steps. You do not do them now:

- Number the files 001–999. The dataset builder finds the pairs by the file names.
- Extract `markdown.md` from the zip files. The dataset builder reads `markdown.md` directly
  from the zip file.
- Filter the files by the page count. The training removes the examples that are too long. It
  uses the exact token count.

## Step 3. Make the dataset

```bash
uv run bioparser-build-dataset
```

The command writes `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl` and
`build_report.json`.

### Pairs of TSV files and parsed PDF files

The builder makes a pair from `ETS_<name>.tsv` and `<name>.zip`, `<name>.md` or
`<name>/markdown.md`. Upper case and lower case, Unicode dash types and repeated spaces do not
change the result.

For other names, add a row to `manifest.tsv`. The builder uses this file automatically. This
file is **tab-separated**, because the file names contain commas:

```
tsv	markdown
ETS_Abraham et al., 2022 part 2.tsv	Abraham et al., 2022.zip
```

The builder merges all TSV files of one parsed PDF file into one training example. The build
report shows the files without a pair, and it gives possible pairs. Examine each possible pair
before you add it.

### Coverage of the labels

For each pair, the builder finds which numbers of the labels are also in the parsed text. The
report shows this result as `value_coverage`. The builder removes the pairs below
`--min-coverage`. The default value is 50%. Usually, these pairs have an incorrect PDF file,
tables that the parser lost, or values from supplementary material. The 3 MinerU samples from
the old notebook have 100% coverage.

## Step 4. Fine-tune the model

The command `uv run bioparser-train` trains a LoRA adapter for `Qwen/Qwen3.5-0.8B` with Unsloth.
The [root README](../README.md) gives the options, the GPU notes and the evaluation.

## Known problems in the labels of the 2026-09-14 delivery (181 TSV files, 175 papers)

- The column `author` contains the ORCID of the curator. The client wants "BioParser". The
  function `to_ets_records` adds this value.
- `Morenikeji et al., 2022` and `Morenikeji et al., 2023` have the same `references` value. Ask
  the client which value is correct.
- The code changes these variants to one form: `species`/`Species`, `Famale`, `males`,
  `both`/`sexes combined` → `Mixed`, `standard deviation` → `SD`.
- Some files have incorrect cells, for example `lifeStage = "55"`, and numbers in
  `measurementDeterminedBy`.
- The output size changes very much from paper to paper. The median is approximately 4,000
  tokens. But some compilations, for example Lundgren 2023, Wimberly 2023_2 and McNab 2008, have
  thousands of rows. They are longer than the maximum context length of the training. For this
  reason, the training does not use them.
