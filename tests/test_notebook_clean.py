"""Committed notebooks must have no outputs and no pasted data.

Outputs would contain client data (parsed articles, generated TSVs) and local paths. A very
long cell is usually pasted data, for example the parsed articles and the TSV labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))
# The longest cell of Qwen3_5_0_8B.ipynb has approximately 1,000 characters. A parsed article has
# 20,000 characters or more.
MAX_CELL_CHARS = 10_000
# The original Unsloth tutorial is a reference. It keeps its install cell (refer to notebooks/README.md).
REFERENCE_NOTEBOOKS = {"Qwen3_5_0_8B_original_tutorial.ipynb"}


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_outputs(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    dirty = [i for i, c in enumerate(nb["cells"]) if c.get("outputs") or c.get("execution_count")]
    assert not dirty, f"{path.name}: clear outputs before committing (cells {dirty})"
    assert "widgets" not in nb.get("metadata", {}), f"{path.name}: widget state in metadata"


@pytest.mark.parametrize(
    "path", [p for p in NOTEBOOKS if p.name not in REFERENCE_NOTEBOOKS], ids=lambda p: p.name
)
def test_notebook_does_not_install_packages(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code":
            assert "pip install" not in src, "install with `uv sync`, not from the notebook"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_pasted_data(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    long_cells = [i for i, c in enumerate(nb["cells"]) if len("".join(c["source"])) > MAX_CELL_CHARS]
    assert not long_cells, (
        f"{path.name}: cells {long_cells} have more than {MAX_CELL_CHARS} characters. "
        "Load the data from fine_tuning_data/, and do not paste it into the notebook."
    )
