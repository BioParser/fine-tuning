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
│   └── _unverified/    PDF files without the title or the DOI: examine them manually
├── parsed/             the MinerU output: one "<pdf-stem>.zip" or ".md" file for each PDF file
├── manifest.tsv        pairs with different file names (the fetch command adds rows)
├── doi_overrides.tsv   optional: your corrections of DOI matches (columns: tsv, doi, url, note)
├── pdf_sources.tsv     the report of the fetch command: status, DOI, source URL and sha256
├── .fetch_cache/       the API answers in a cache, so a second run is fast
└── .browser_profile/   the browser profile of --manual. It contains your university login: keep it private.
```

## Step 1. Get the PDF files

Step 1 has four parts. You can do the parts one at a time. Each part uses the results of the
part before it. The command of the Quickstart does all four parts in one run.

```bash
# a. Find a DOI for each citation. Then examine the matches in pdf_sources.tsv.
uv run bioparser-fetch-pdfs --email you@helsinki.fi --resolve-only

# b. Download the open-access files from the APIs and from the landing pages.
uv run bioparser-fetch-pdfs --email you@helsinki.fi

# c. Also use a headless browser for the pages that use JavaScript.
uv run bioparser-fetch-pdfs --email you@helsinki.fi --browser

# d. Get the other files yourself in a visible browser, with the library proxy.
uv run bioparser-fetch-pdfs --email you@helsinki.fi --manual \
    --proxy-prefix "https://login.libproxy.helsinki.fi/login?url="
```

### How the command finds the PDF files

Only 2 of the 175 citations contain a DOI. For this reason, the command first finds a DOI for
each citation. It uses Crossref, OpenAlex, Semantic Scholar, DataCite and CORE. It compares
each match with the title, the year and the first author of the citation.

Examine the medium-confidence matches in `pdf_sources.tsv`. If a match is not correct, write
the correct DOI or URL in `doi_overrides.tsv`. This file is tab-separated:

```
tsv	doi	url	note
ETS_Example et al., 2020.tsv	10.1000/example		found manually
```

Each downloaded PDF file must contain the title or the DOI of the article. If the file does not
contain them, the command puts the file in `pdfs/_unverified/`. Examine these files manually.

165 of the 175 client citations use the Harvard format of Google Scholar. The command also reads
the other formats: Google Scholar without a year, APA and Chicago. The citation of
"Ancillotto et al., 2023" is not correct. For this citation, add a row to `doi_overrides.tsv`.

The command also adds rows to `manifest.tsv` for the papers with more than one TSV file:
Abraham 2022, Jérémy 2023, Morenikeji 2022/2023 and Wimberly 2023. Morenikeji 2022 and
Morenikeji 2023 have the same citation. Ask the client if they are one article.

### Library access with EZproxy

Many of the articles are not open access. The University of Helsinki pays for access to them.
The publishers give this access to the IP addresses of the university network. Outside this
network, you get the same access through EZproxy.

EZproxy is the proxy server of the library. It operates like this:

1. You open a proxy link. A proxy link is the EZproxy prefix and then the URL of the article.
2. EZproxy asks you to log in with your University of Helsinki user account.
3. EZproxy sends your requests to the publisher from an IP address of the university.
4. The publisher sees the university, and it gives you the article.

EZproxy also changes the address of the publisher site. For example, `www.sciencedirect.com`
becomes `www-sciencedirect-com.libproxy.helsinki.fi`. This address shows that the page comes
through the proxy.

The EZproxy prefix of the University of Helsinki is:

```
https://login.libproxy.helsinki.fi/login?url=
```

This is an example of a proxy link:

```
https://login.libproxy.helsinki.fi/login?url=https://doi.org/10.1000/example
```

#### How the command uses EZproxy

Only part d, with `--manual`, uses the prefix:

1. The command opens the proxy link of the next article in a visible browser. If the command
   did not find the article, it opens a Google Scholar search for the citation.
2. If EZproxy shows the login page, log in with your University of Helsinki user name and
   password. The browser keeps the login for the next articles.
3. Get the PDF file. Usually, you click "PDF" or "Download PDF" on the page of the publisher.
4. The command saves the file with the correct name. Then it opens the next article.
5. If you cannot get the article, push Enter in the terminal. The command goes to the next
   article.

#### The rules for EZproxy

The terms of use of the library let you save reasonable amounts of material for your own study,
teaching and research. They do not let you download material systematically with bots or other
automatic processes. If a user does not obey the terms, the publisher can block the access for
all users of the University of Helsinki.

For this reason, the command follows these rules:

- Only the manual part uses the proxy and your login. In this part, you download each file
  yourself.
- The automatic parts b and c do not use the proxy. The headless browser starts with a new,
  empty profile. For this reason, it does not have your login.

You must also follow these rules:

- Do step 1 outside the network of the university. Do not use the campus network, eduroam on
  the campus or the university VPN with the allroute option. On these networks, the publishers
  also give access to the automatic parts.
- Use only your own user account. Do not give your password to other persons.
- Do not save your password in the browser of the command.
- Keep the folder `.browser_profile/` private. It contains the cookies of your university login.
  When you have all the PDF files, delete the folder.
- The licenses also control how you can use the articles, for example as training data for a
  model. If you are not sure, ask the library before you start.

#### If EZproxy does not operate

- EZproxy shows an error page: the library does not give proxy access to this publisher. Push
  Enter, and use a different source.
- The publisher asks you to pay: the library does not pay for this journal. Push Enter, and use
  a different source.
- The prefix does not operate at all: the library can change the prefix. Find the current
  prefix on the page "Remote access to electronic resources" of Helsinki University Library.

Other sources for the articles that you cannot get:

- The client. The client has the PDF files that they used for the labels.
- An interlibrary loan from Helsinki University Library.
- The authors of the article. Many authors send a copy if you ask.

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
