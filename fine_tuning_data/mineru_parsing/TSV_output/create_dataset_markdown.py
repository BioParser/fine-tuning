# Script to form the dataset (Alpaca format)

import csv
import json
import os

# prompt used across all entries
PROMPT = """
You're an expert parser. Your goal is to parse TSV fields from the input text. A single input text file can have multiple 
different species and/or multiple individuals from the same species. The output should include a TSV header and the following
fields should be filled:

references: Harvard citation format of the article
verbatimScientificName: FULL scientific name
taxonRank:
verbatimTraitName:
verbatimTraitUnit:
individualCount:
measurementValue_min:
measurementValue_max:
dispersion:
statisticalMethod
verbatimTraitValue:
sex:
lifeStage:
measurementMethod:
measurementRemarks:
measurementAccuracy:
measurementDeterminedBy:
verbatimLocality:
author: BioParser
associatedReferences:
"""

# the files are matched using their numbering
# folder for the parsed input markdown files
MARKDOWN_DIR = "./markdown_input"
# folder for the TSV outputs
TSV_DIR = "./TSVs"

# output file name
OUTPUT_NAME = "training_data_markdown"

# fetch input file names (markdown files) and sort them
def file_names():
    names = []
    for file in os.scandir(MARKDOWN_DIR):
        names.append(file.name[:-3])
    names.sort()
    
    return names

# create a single JSON Alpaca format entry
def create_alpaca_entry(file_prefix):
    with open(MARKDOWN_DIR + "/" + file_prefix + ".md", "r") as file:
        markdown_data = file.read()
    with open(TSV_DIR + "/" + file_prefix + ".tsv", newline='') as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        tsv_data = []
        for row in tsv_reader:
            tsv_data.append(row)

    return {
        "instruction": str(PROMPT),
        "input": str(markdown_data),
        "output": str(tsv_data)
    }

# create the full Alpaca dataset
def create_alpaca_dataset(file_prefixes):
    alpaca_data = [create_alpaca_entry(file_prefix) for file_prefix in file_prefixes]

    with open(OUTPUT_NAME + ".jsonl", "w", encoding="utf-8") as file:
        json.dump(alpaca_data, file, ensure_ascii=False, indent=2)

    print(f"Processed {len(alpaca_data)} entries and saved to {OUTPUT_NAME + ".jsonl"}")

if __name__ == "__main__":
    create_alpaca_dataset(file_names())
