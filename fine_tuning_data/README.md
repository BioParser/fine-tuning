# fine_tuning_data/

Client data lives here. **Everything in this folder except this README and
`manifest.example.tsv` is git-ignored.** The files are confidential client data and
copyrighted articles; do not commit them.

```
fine_tuning_data/
├── ets_tsv/        client ETS exports, e.g. "ETS_Abdel-Hamid and Alqahtani, 2022.tsv"
├── pdfs/           the matching articles, e.g. "Abdel-Hamid and Alqahtani, 2022.pdf"
├── parsed/         MinerU output, one "<pdf-stem>.zip" (or .md) per PDF
└── manifest.tsv    optional: pairs whose file names do not match (see below)
```

## 1. Parse the PDFs

```bash
scripts/parse_pdfs.sh fine_tuning_data/pdfs fine_tuning_data/parsed
```

This runs `mineru-kit parse ... --format zip --tier basic` with MinerU 4.1.6 in an isolated
`uvx` environment. Set `MINERU_VERSION` to the version production uses. The model learns
that version's markdown dialect.

Not needed any more: numbering the files 001–999 and unzipping `markdown.md` by hand. The
dataset builder pairs files by name and reads `markdown.md` straight from the zip. Also not
needed: the page-count filter. Over-long examples are dropped at training time by exact token
count.

## 2. Pair TSVs with parsed PDFs

`ETS_<name>.tsv` pairs with `<name>.zip`, `<name>.md` or `<name>/markdown.md`. Matching
ignores case, Unicode dash variants and repeated spaces. For anything else, create
`manifest.tsv`. It is **tab-separated** because the file names contain commas:

```
tsv	markdown
ETS_Abraham et al., 2022 part 2.tsv	Abraham et al., 2022.zip
```

Several TSVs that point to one parsed PDF are merged into one training example. The build
report lists unmatched files and suggests likely pairs. Check each suggestion before adding it.

## 3. Build the dataset

```bash
uv run bioparser-build-dataset   # writes data/processed/{train,val,test}.jsonl + build_report.json
```

For each pair, the builder checks that the labelled numbers really occur in the parsed text
(`value_coverage`). Pairs below `--min-coverage` (default 50%) are excluded. They are
typically a wrong PDF, tables the parser lost, or values from supplementary material. The
three MinerU samples from the original notebook have 100% coverage.

## Known label issues in the 2026-09-14 delivery (181 TSVs, 175 papers)

* `author` contains the curator's ORCID. The client wants "BioParser"; it is set in post-processing.
* `Morenikeji et al., 2022` and `... 2023` carry the same `references` value. Check which is right.
* Case and spelling variants, which are normalised automatically: `species`/`Species`, `Famale`,
  `males`, `both`/`sexes combined` → `Mixed`, `standard deviation` → `SD`.
* Single bad cells, e.g. `lifeStage = "55"`, and numbers in `measurementDeterminedBy` in a few files.
* Output size varies widely: median ~4k tokens, but a few compilations (e.g. Lundgren 2023,
  Wimberly 2023_2, McNab 2008) have thousands of rows. They exceed any trainable context
  length and are dropped.
