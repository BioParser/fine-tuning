# Notebooks
The `Qwen...original_tutorial.ipynb` was the original test notebook. It is kept
as a reference. Its install cell is a Markdown cell, so that nobody runs it: the commands install
packages that are not in `uv.lock`, and one of them has security advisories. Also the notebook
does not follow the repo development.

The test data of the tutorial is client data: 3 parsed articles and their TSV labels. For this
reason, it is not in git. The notebook loads it from `fine_tuning_data/tutorial_test_data.json`.
Get this file from the team. The notebook has no saved outputs, because the outputs contained
client data and local paths.

# Quick start 
```
uv run jupyter lab
```
You will get a link similar to this that you can open in your browser:
```
http://localhost:8888/lab?token=...
http://127.0.0.1:8888/lab?token=...
```
