This folder includes everything needed to form the MinerU 4.0 parsed-TSV output fine-tuning dataset.

The used [MinerU](https://opendatalab.github.io/MinerU/) version should match the version used in production.

## Steps
- Fetch the PDFs and the matching target TSVs and number them 001-999.
- Include them in the "PDFs" and "TSVs" folders under the current folder.
- Parse the PDFs using MinerU 4.0 (below).
- Assemble the dataset (bottom of the page).

## Parse PDFs

The parsing is done using MinerU 4.0 on the **basic** tier

### Script
Parse the PDFs into the zipped format (the output includes all of the PDF images and parsed markdown, middle json, model output and structured content. systemd-inhibit is used to stop idling on sleep).
- The parsed data will be outputted into folder "output" under the current folder.
```
systemd-inhibit --what=idle --who="mineru" --why="PDF parsing" mineru-kit parse ./PDFs/ -o ./output --format zip --tier basic
```

#### Extract the needed files into a folder
Extract the markdown files from the parsed zip files.
```
mkdir -p ./markdown_input

for z in ./output/*.zip; do
    name=$(basename "$z" .zip);
    unzip -p "$z" "*markdown.md" > "./markdown_input/${name}.md";
done
```

## Creating the dataset (Alpaca format)
See [create_dataset_markdown.py](./create_dataset_markdown.py) for a script to use for assembling the final dataset. [create_dataset_markdown_filtered.py](./create_dataset_markdown_filtered.py) can be used to filter the files based on their page count.