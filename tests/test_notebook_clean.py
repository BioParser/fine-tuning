"""Committed notebooks must have no outputs: outputs would contain client data
(parsed articles, generated TSVs) and local paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_outputs(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    dirty = [i for i, c in enumerate(nb["cells"]) if c.get("outputs") or c.get("execution_count")]
    assert not dirty, f"{path.name}: clear outputs before committing (cells {dirty})"
    assert "widgets" not in nb.get("metadata", {}), f"{path.name}: widget state in metadata"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_does_not_install_packages(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code":
            assert "pip install" not in src, "install with `uv sync`, not from the notebook"
