"""Kaggle notebooks (§16 M4) — validate structure + referenced symbols, offline.

The notebooks run on Kaggle, not here, so this only asserts they are valid
nbformat-4 JSON and that every ``vitreous`` symbol they declare (in notebook
metadata) resolves in the installed public API. It also checks the single
``DATASET`` dataset-swap knob is present.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

# packages/core/tests/test_notebooks.py -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
KAGGLE_DIR = REPO_ROOT / "kaggle"
NOTEBOOKS = ["train.ipynb", "precompute.ipynb", "sae.ipynb"]


@pytest.fixture(params=NOTEBOOKS)
def notebook(request):
    path = KAGGLE_DIR / request.param
    assert path.exists(), f"missing notebook {path}"
    return request.param, json.loads(path.read_text(encoding="utf-8"))


def test_valid_nbformat_4(notebook):
    name, nb = notebook
    assert nb["nbformat"] == 4, f"{name} is not nbformat 4"
    assert isinstance(nb.get("nbformat_minor"), int)
    assert nb["cells"], f"{name} has no cells"
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("code", "markdown")
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            # nbformat-4 code cells need these keys.
            assert "outputs" in cell and "execution_count" in cell


def test_symbols_resolve_in_public_api(notebook):
    name, nb = notebook
    symbols = nb["metadata"]["vitreous"]["symbols"]
    assert symbols, f"{name} declares no vitreous symbols"
    for sym in symbols:
        module_name, attr = sym.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        assert hasattr(mod, attr), f"{name}: symbol {sym} not found in public API"


def test_dataset_swap_knob_present(notebook):
    name, nb = notebook
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'DATASET = "eurosat"' in code, f"{name} missing the DATASET swap knob"
    # The pip-install cell wires up the vitreous package from the repo branch.
    assert "pip" in code and "vitreous" in code, f"{name} missing the install cell"


def test_all_three_notebooks_exist():
    for n in NOTEBOOKS:
        assert (KAGGLE_DIR / n).exists()
